// R14 gate 2b probe: online-daemon single-request batch path vs single-stream generation path.
//   Both paths share the daemon prefill (gemma4_prefill_batch_slot, slot 0 of a Bmax-slot batch state).
//   batch: gemma4_decode_batch W=1 ragged (exactly what run_online_daemon does for one request).
//   fast : gemma4_generate on a G4DecodeState view of the same slot KV (G4_CUDA_GRAPH selects OFF/v1/v2).
//   Every step's full logits and argmax token are compared byte-for-byte; logits are also written
//   (<out>.batch.f32 / <out>.fast.f32, exclusive create) for external cmp.
//   Usage: r14_path_equiv <weights> <prompt.bin> <out_prefix> [steps=128] [maxlen=2048] [Bmax=8]
#include "runtime/knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

static std::vector<int> read_ids(const char *p){
    FILE *f=fopen(p,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",p);exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<int> v(b/4); if(fread(v.data(),4,v.size(),f)!=v.size()) exit(1); fclose(f); return v;
}
static FILE *excl(const std::string &p){ FILE *f=fopen(p.c_str(),"wbx"); if(!f){fprintf(stderr,"cannot create %s\n",p.c_str());exit(1);} return f; }
static int argmax(const float *p,int V){ int b=0; float bv=p[0]; for(int j=1;j<V;j++) if(p[j]>bv){bv=p[j];b=j;} return b; }

struct FastCapture { int V; std::vector<float> host; std::vector<std::vector<float>> *rows; std::vector<int> *toks; };
static int on_step(const float *d_logits,int token,int step,void *opaque){
    FastCapture *c=(FastCapture*)opaque; (void)step;
    CK(cudaMemcpy(c->host.data(),d_logits,(size_t)c->V*4,cudaMemcpyDeviceToHost));
    c->rows->push_back(c->host); c->toks->push_back(token);
    return 0;
}

int main(int argc,char **argv){
    if(argc<4){ fprintf(stderr,"usage: %s <weights> <prompt.bin> <out_prefix> [steps] [maxlen] [Bmax]\n",argv[0]); return 2; }
    std::string WD=argv[1]; std::vector<int> prompt=read_ids(argv[2]); std::string out=argv[3];
    const int steps=argc>4?atoi(argv[4]):128, maxlen=argc>5?atoi(argv[5]):2048, Bmax=argc>6?atoi(argv[6]):8;
    const int uq=2, P=(int)prompt.size();
    if(P+steps+1>maxlen){ fprintf(stderr,"prompt %d + steps %d exceeds maxlen %d\n",P,steps,maxlen); return 2; }
    ModelConfig c=load_config_gemma4((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, hd_s=c.head_dim, hd_f=c.global_head_dim?c.global_head_dim:hd_s;
    float *cf,*sf,*cs,*ss;
    CK(cudaMalloc(&cf,(size_t)maxlen*hd_f*4)); CK(cudaMalloc(&sf,(size_t)maxlen*hd_f*4));
    CK(cudaMalloc(&cs,(size_t)maxlen*hd_s*4)); CK(cudaMalloc(&ss,(size_t)maxlen*hd_s*4));
    if(gemma4_gen_rope_tables((WD+"/manifest.json").c_str(),maxlen,hd_f,cf,sf,hd_s,cs,ss)!=0) return 1;
    G4Weights w=load_weights_gemma4(WD.c_str(),c,1,0);
    G4BatchState bs=alloc_g4_batch_state(c,Bmax,maxlen);
    int *d_pf,*d_tok,*d_pos; float *d_logit;
    CK(cudaMalloc(&d_pf,(size_t)maxlen*4)); CK(cudaMalloc(&d_tok,Bmax*4)); CK(cudaMalloc(&d_pos,Bmax*4));
    CK(cudaMalloc(&d_logit,(size_t)Bmax*V*4));
    std::vector<float> host(V);

    auto prefill=[&]()->int{
        CK(cudaMemcpy(d_pf,prompt.data(),(size_t)P*4,cudaMemcpyHostToDevice));
        gemma4_prefill_batch_slot(c,w,bs,0,d_pf,P,cf,sf,cs,ss,d_logit,uq);
        CK(cudaMemcpy(host.data(),d_logit,(size_t)V*4,cudaMemcpyDeviceToHost));
        return argmax(host.data(),V);
    };

    // batch path (daemon W=1)
    std::vector<std::vector<float>> brow; std::vector<int> btok;
    int first_b=prefill(), last=first_b, pos=P;
    for(int s=0;s<steps;s++){
        bs.B=1;
        CK(cudaMemcpy(d_tok,&last,4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_pos,&pos,4,cudaMemcpyHostToDevice));
        gemma4_decode_batch(c,w,bs,d_tok,cf,sf,cs,ss,d_logit,0,uq,d_pos,&pos);
        CK(cudaMemcpy(host.data(),d_logit,(size_t)V*4,cudaMemcpyDeviceToHost));
        brow.push_back(host); last=argmax(host.data(),V); btok.push_back(last); pos++;
    }

    // fast path: G4DecodeState view of slot 0
    int first_f=prefill();
    G4DecodeState view{}; view.n_past=P; view.max_len=maxlen; view.L=bs.L; view.kv_cap=bs.kv_cap;
    std::vector<float*> kv(2*(size_t)bs.L); view.Kc=kv.data(); view.Vc=kv.data()+bs.L;
    for(int l=0;l<bs.L;l++){ view.Kc[l]=bs.Kc[l]; view.Vc[l]=bs.Vc[l]; }   // slot 0 = base pointer
    std::vector<std::vector<float>> frow; std::vector<int> ftok;
    FastCapture cap{V,std::vector<float>(V),&frow,&ftok};
    G4GenerationStats gs{};
    int accepted=gemma4_generate(c,w,view,first_f,steps,-1,cf,sf,cs,ss,uq,on_step,&cap,&gs);

    const char *mode=knob_raw("G4_CUDA_GRAPH");
    int first_div=-1; double maxd=0; int tok_div=-1;
    for(int s=0;s<steps && s<(int)frow.size();s++){
        if(memcmp(brow[s].data(),frow[s].data(),(size_t)V*4)!=0 && first_div<0) first_div=s;
        for(int j=0;j<V;j++){ double d=fabs((double)brow[s][j]-frow[s][j]); if(d>maxd) maxd=d; }
        if(btok[s]!=ftok[s] && tok_div<0) tok_div=s;
    }
    FILE *fb=excl(out+".batch.f32"), *ff=excl(out+".fast.f32");
    for(auto &r:brow) fwrite(r.data(),4,V,fb); for(auto &r:frow) fwrite(r.data(),4,V,ff); fclose(fb); fclose(ff);
    FILE *tb=excl(out+".batch.i32"), *tf=excl(out+".fast.i32");
    fwrite(btok.data(),4,btok.size(),tb); fwrite(ftok.data(),4,ftok.size(),tf); fclose(tb); fclose(tf);
    const bool pass=first_b==first_f && accepted==steps && first_div<0 && tok_div<0 && (int)frow.size()==steps;
    printf("R14_PATH_EQUIV mode=%s prompt_tokens=%d steps=%d accepted=%d first_batch=%d first_fast=%d "
           "first_logit_divergence=%d first_token_divergence=%d max_abs_diff=%.9g planned_captures=%llu "
           "instantiations=%llu update_failures=%llu speculative=%llu result=%s\n",
           mode?mode:"unset",P,steps,accepted,first_b,first_f,first_div,tok_div,maxd,
           gs.planned_captures,gs.instantiations,gs.update_failures,gs.speculative_steps,pass?"PASS":"FAIL");
    return pass?0:3;
}
