// R17 probe (from tools/r16_spec_pos.cu, adds the drafter precision argument):
// R16 PART C probe: plain decode (gemma4_generate, default launch mode = Graph v2) vs speculative decode
// with the Gemma 4 assistant drafter (fixed K), same prompt, same number of generated tokens.
//   Speculative loop = tests/test_gemma4_spec_e2e.cu spec_run with tau<0 (fixed K), copied verbatim in
//   structure: K drafter steps on the target's shared KV, one verify gemma4_decode(S=K+1, commit=0,
//   amort=1), accept the matching prefix + 1, carry prenorm row m into the drafter.
//   Each configuration runs twice (warm-up, then measured). Output per measured run:
//   R17_SPEC prompt=<file> P=<prompt tokens> mode=<plain|K=k> tokens=<N> ms=<wall> tps=<N/s>
//            cycles=<n> acc=<N/cycles-1> draft_ms verify_ms other_ms match=<spec tokens == plain tokens>
// Usage: r17_spec_pos <target weights> <assistant weights> <ast_uq 0=fp32|2=W4A8> <N> <K list e.g. 1,2,3,4> <prompt.bin>...
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"
#include "../models/gemma4_assistant.h"
#include "../kernels/embedding.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/argmax.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }
static std::vector<int> rii(const std::string &p){
    FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"open %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); rewind(f); std::vector<int> v(b/4);
    if(fread(v.data(),4,v.size(),f)!=v.size()){fprintf(stderr,"read %s\n",p.c_str());exit(1);} fclose(f); return v;
}

struct Collect { std::vector<int> *toks; static int call(const float*,int token,int,void *o){ ((Collect*)o)->toks->push_back(token); return 0; } };

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IOLBF,0);
    if(argc<7){ fprintf(stderr,"usage: %s <target W> <assistant W> <ast_uq> <N> <K list> <prompt.bin>...\n",argv[0]); return 1; }
    const std::string TW=argv[1], AW=argv[2]; const int ast_uq=atoi(argv[3]), N=atoi(argv[4]);
    std::vector<int> Ks; { std::string s(argv[5]); size_t p=0; while(p<=s.size()){ size_t q=s.find(',',p); if(q==std::string::npos)q=s.size(); if(q>p)Ks.push_back(atoi(s.substr(p,q-p).c_str())); p=q+1; } }
    int KMAX=1; for(int k:Ks) if(k>KMAX) KMAX=k;
    std::vector<std::string> prompts(argv+6,argv+argc);
    int Pmax=0; std::vector<std::vector<int>> pids;
    for(auto &p:prompts){ pids.push_back(rii(p)); if((int)pids.back().size()>Pmax) Pmax=pids.back().size(); }
    const int total=Pmax+N+KMAX+16;

    ModelConfig c=load_config_gemma4((TW+"/manifest.json").c_str());
    AstConfig ca=load_config_assistant((AW+"/manifest.json").c_str());
    const int V=c.vocab_size, H=c.hidden_size, hd_f=c.global_head_dim, hd_s=c.head_dim, uq=2;
    int lf=-1, ls=-1; for(int l=0;l<c.num_layers;l++){ if(c.layer_is_full[l]) lf=l; else ls=l; }
    float *d_cf,*d_sf,*d_cs,*d_ss;
    CK(cudaMalloc(&d_cf,(size_t)total*hd_f*4)); CK(cudaMalloc(&d_sf,(size_t)total*hd_f*4));
    CK(cudaMalloc(&d_cs,(size_t)total*hd_s*4)); CK(cudaMalloc(&d_ss,(size_t)total*hd_s*4));
    if(gemma4_gen_rope_tables((TW+"/manifest.json").c_str(),total,hd_f,d_cf,d_sf,hd_s,d_cs,d_ss)){ fprintf(stderr,"rope\n"); return 1; }
    G4Weights w=load_weights_gemma4(TW.c_str(),c,1,0);
    AstWeights wa=load_weights_assistant(AW.c_str(),ca,ast_uq);
    int *d_tok; CK(cudaMalloc(&d_tok,(size_t)(Pmax+KMAX+1)*4));
    float *d_lg; CK(cudaMalloc(&d_lg,(size_t)(KMAX+1)*V*4));
    float *d_pn; CK(cudaMalloc(&d_pn,(size_t)(Pmax>KMAX+1?Pmax:KMAX+1)*H*4));
    float *d_in; CK(cudaMalloc(&d_in,(size_t)2*ca.backbone*4));
    float *d_alg; CK(cudaMalloc(&d_alg,(size_t)ca.vocab_size*4));
    float *d_ho; CK(cudaMalloc(&d_ho,(size_t)ca.backbone*4));
    int *d_am; CK(cudaMalloc(&d_am,(size_t)(KMAX+1)*4));
    printf("[r17_spec_pos] N=%d K=%s total=%d target uq=2 drafter %s\n",N,argv[5],total,ast_uq==2?"W4A8":"fp32");

    auto prefill=[&](G4DecodeState &st,const std::vector<int> &ids,int &anchor){
        CK(cudaMemcpy(d_tok,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
        gemma4_decode(c,w,st,d_tok,(int)ids.size(),d_cf,d_sf,d_cs,d_ss,d_lg,1,uq,1,1,d_pn);
        launch_argmax(d_lg,V,d_am); CK(cudaMemcpy(&anchor,d_am,4,cudaMemcpyDeviceToHost));
    };
    auto plain_run=[&](const std::vector<int> &ids,std::vector<int> &toks,double &msec){
        G4DecodeState st=alloc_g4_decode_state(c,total);
        int cur; prefill(st,ids,cur); CK(cudaDeviceSynchronize());
        toks.assign(1,cur); Collect col{&toks}; G4GenerationStats gs;
        auto t0=clk::now();
        gemma4_generate(c,w,st,cur,N,-1,d_cf,d_sf,d_cs,d_ss,uq,&Collect::call,&col,&gs,0);
        msec=ms_since(t0);
        free_g4_decode_state(st);
    };
    auto spec_run=[&](const std::vector<int> &ids,int K,std::vector<int> &toks,double &msec,int &cycles,double &ms_draft,double &ms_verify){
        G4DecodeState st=alloc_g4_decode_state(c,total);
        const int P=(int)ids.size();
        int cur; prefill(st,ids,cur);
        int L=P;
        CK(cudaMemcpy(d_in+ca.backbone,d_pn+(size_t)(P-1)*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
        CK(cudaDeviceSynchronize());
        toks.assign(1,cur); cycles=0; ms_draft=0; ms_verify=0;
        std::vector<int> chunk(K+1), am(K+1);
        auto t0=clk::now();
        while((int)toks.size()<N+1 && L+K+1<total){
            auto td=clk::now(); int dt=cur;
            for(int j=0;j<K;j++){
                CK(cudaMemcpy(d_tok,&dt,4,cudaMemcpyHostToDevice));
                launch_embedding(d_tok,w.embed,d_in,1,ca.backbone);
                if(c.embed_scale>0.f) launch_scale(d_in,d_in,c.embed_scale,(size_t)ca.backbone);
                gemma4_assistant_step(ca,wa,d_in,st.Kc[lf],st.Vc[lf],L,st.Kc[ls],st.Vc[ls],L,
                                      d_cf+(size_t)L*hd_f,d_sf+(size_t)L*hd_f,d_cs+(size_t)L*hd_s,d_ss+(size_t)L*hd_s,d_alg,d_ho,ast_uq);
                launch_argmax(d_alg,ca.vocab_size,d_am); CK(cudaMemcpy(&dt,d_am,4,cudaMemcpyDeviceToHost));
                chunk[j+1]=dt;
                CK(cudaMemcpy(d_in+ca.backbone,d_ho,(size_t)ca.backbone*4,cudaMemcpyDeviceToDevice));
            }
            ms_draft+=ms_since(td);
            auto tv=clk::now();
            chunk[0]=cur;
            CK(cudaMemcpy(d_tok,chunk.data(),(size_t)(K+1)*4,cudaMemcpyHostToDevice));
            st.n_past=L;
            gemma4_decode(c,w,st,d_tok,K+1,d_cf,d_sf,d_cs,d_ss,d_lg,0,uq,0,1,d_pn);
            for(int i=0;i<=K;i++) launch_argmax(d_lg+(size_t)i*V,V,d_am+i);
            CK(cudaMemcpy(am.data(),d_am,(size_t)(K+1)*4,cudaMemcpyDeviceToHost));
            ms_verify+=ms_since(tv);
            int m=0; for(int i=1;i<=K;i++){ if(am[i-1]==chunk[i]) m++; else break; }
            for(int i=1;i<=m && (int)toks.size()<N+1;i++) toks.push_back(chunk[i]);
            if((int)toks.size()<N+1) toks.push_back(am[m]);
            CK(cudaMemcpy(d_in+ca.backbone,d_pn+(size_t)m*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
            cur=am[m]; L+=m+1; cycles++;
        }
        msec=ms_since(t0);
        free_g4_decode_state(st);
    };

    int mismatches=0;
    for(size_t pi=0;pi<prompts.size();pi++){
        const auto &ids=pids[pi]; std::vector<int> ptoks,stoks; double pms;
        plain_run(ids,ptoks,pms); plain_run(ids,ptoks,pms);
        printf("R17_SPEC prompt=%s P=%zu mode=plain tokens=%zu ms=%.6f tps=%.6f cycles=%d acc=0 draft_ms=0 verify_ms=0 other_ms=%.6f match=1\n",
               prompts[pi].c_str(),ids.size(),ptoks.size()-1,pms,(ptoks.size()-1)/(pms/1e3),N,pms);
        for(int K:Ks){
            double sms,md,mv; int cyc;
            spec_run(ids,K,stoks,sms,cyc,md,mv); spec_run(ids,K,stoks,sms,cyc,md,mv);
            const bool match=stoks==ptoks; if(!match) mismatches++;
            printf("R17_SPEC prompt=%s P=%zu mode=K=%d tokens=%zu ms=%.6f tps=%.6f cycles=%d acc=%.6f draft_ms=%.6f verify_ms=%.6f other_ms=%.6f match=%d\n",
                   prompts[pi].c_str(),ids.size(),K,stoks.size()-1,sms,(stoks.size()-1)/(sms/1e3),cyc,(double)(stoks.size()-1)/cyc-1.0,md,mv,sms-md-mv,match?1:0);
        }
    }
    printf("[r17_spec_pos] done mismatches=%d\n",mismatches);
    return mismatches?2:0;
}
