// Separate gates for the opt-in dense Gemma MMQ path; no changes to legacy tests.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <cuda_runtime.h>
#include "../../runtime/decode_state_gemma4.h"
#include "../../runtime/manifest_gemma4.h"
#include "../../runtime/rope_gemma4.h"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2); } } while(0)
static void require(bool ok,const char *why) { if(!ok) { fprintf(stderr,"FAIL %s\n",why); exit(1); } }
template<class T> static std::vector<T> read_exact(const std::string &p,size_t count) {
    FILE *f=fopen(p.c_str(),"rb"); require(f!=nullptr,p.c_str());
    require(fseek(f,0,SEEK_END)==0,"file seek"); long bytes=ftell(f);
    require(bytes>=0 && (size_t)bytes==count*sizeof(T),p.c_str()); rewind(f);
    std::vector<T> out(count); require(fread(out.data(),sizeof(T),count,f)==count,"short file read"); fclose(f); return out;
}
template<class T> static T *device(size_t n) { T *p=nullptr; CK(cudaMalloc(&p,n*sizeof(T))); return p; }
static void write_new(const std::string &path,const void *data,size_t bytes) {
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    require(fd>=0,path.c_str());
    FILE *f=fdopen(fd,"wb"); require(f!=nullptr,"output fdopen");
    require(fwrite(data,1,bytes,f)==bytes,"short output write");
    require(fclose(f)==0,"output close");
}
static void stamp(const char *mode,const char *phase) {
    double epoch=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    printf("INTERVAL mode=%s phase=%s epoch=%.9f\n",mode,phase,epoch); fflush(stdout);
}
struct Error { double maxabs=0, refmax=0, l1=0, refl1=0, squares=0; };
static Error compare(const std::vector<float>& a,const std::vector<float>& ref) {
    require(a.size()==ref.size(),"compare sizes"); Error e;
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(ref[i]),"nonfinite result/reference");
        double d=std::abs((double)a[i]-ref[i]); e.maxabs=std::max(e.maxabs,d);
        e.refmax=std::max(e.refmax,std::abs((double)ref[i])); e.l1+=d; e.refl1+=std::abs((double)ref[i]); e.squares+=d*d;
    } return e;
}
static int argmax(const float *x,int n) { int best=0; for(int j=1;j<n;++j) if(x[j]>x[best]) best=j; return best; }
static void finite(const std::vector<float>& v) { for(float x:v) require(std::isfinite(x),"nonfinite output"); }
static bool all_finite(const std::vector<float>& v) { for(float x:v) if(!std::isfinite(x)) return false; return true; }
static std::vector<float> download_raw(const float *p,size_t n) { std::vector<float> v(n); CK(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost)); return v; }
static std::vector<float> download(const float *p,size_t n) { auto v=download_raw(p,n); finite(v); return v; }

int main(int argc,char **argv) {
    require(argc==4 || argc==5,"usage: prefill WEIGHTS GOLDEN golden|determinism|chunk|dump [NEW_OUTDIR]");
    const std::string wd=argv[1], gd=std::string(argv[2])+"/", mode=argv[3];
    require(mode=="golden" || mode=="determinism" || mode=="chunk" || mode=="dump","unknown mode");
    require(argc==(mode=="dump"?5:4),"only dump requires NEW_OUTDIR");
    const char *env=getenv("G4_MMQ_PREFILL");
    const bool mmq=!env || strcmp(env,"0")!=0;
    const std::string out=mode=="dump"?argv[4]:"";
    require(getenv("MMQ256") && strcmp(getenv("MMQ256"),"0")==0,"MMQ256 must equal 0");
    ModelConfig c=load_config_gemma4((wd+"/manifest.json").c_str());
    require(c.num_experts==0 && c.num_layers>=8 && (c.scale_dtype==0 || c.scale_dtype==1),"dense scale_dtype 0/1 model only");
    printf("PATH=%s G4_MMQ_PREFILL=%s SCALE_DTYPE=%d uq=2 amort=1 MMQ256=0 coverage=dispatch_request_not_kernel_trace\n",mmq?"MMQ":"SB",env?env:"<unset>",c.scale_dtype);
    const int H=c.hidden_size,V=c.vocab_size,P=mode=="chunk"?1024:16;
    const int hf=c.global_head_dim?c.global_head_dim:c.head_dim, hs=c.head_dim;
    const int kf=c.num_global_kv_heads?c.num_global_kv_heads:c.num_kv_heads;
    auto prefix=read_exact<int>(gd+"token_ids.bin",5);
    std::vector<int> tokens(P); for(int i=0;i<P;++i) tokens[i]=prefix[i%5];
    for(int t:tokens) require(t>=0 && t<V,"token outside vocabulary");
    size_t weights=(size_t)V*H*4+(size_t)H*4, kv=0;
    const size_t scale_bytes=c.scale_dtype==1?2:4;
    auto qbytes=[&](size_t k,size_t n) { require(k%32==0,"K must be divisible by 32"); return k*n/2+k*n/32*scale_bytes; };
    weights+=qbytes(H,V);
    for(int l=0;l<c.num_layers;++l) {
        bool full=c.layer_is_full?c.layer_is_full[l]:true;
        size_t hd=full?hf:hs, kd=(full?kf:c.num_kv_heads)*hd, qd=c.num_q_heads*hd;
        weights+=4*((size_t)4*H+2*hd)+2*qbytes(H,qd)+qbytes(H,kd)*(full && c.attention_k_eq_v?1:2)+3*qbytes(H,c.intermediate_size);
        kv+=(size_t)P*kd*8;
    }
    size_t qmax=(size_t)c.num_q_heads*std::max(hf,hs), kvmax=std::max((size_t)kf*hf,(size_t)c.num_kv_heads*hs);
    size_t activation=(size_t)P*(6*(size_t)H+3*qmax+4*kvmax+3*(size_t)c.intermediate_size)*4;
    size_t quant=(size_t)P*std::max({H,c.intermediate_size,(int)qmax})*5/4;
    size_t logits=(size_t)(mode=="chunk"?1:P)*V*4, rope=(size_t)P*(hf+hs)*8;
    // Includes transient host embedding load, host result copies, and a 2GiB context/allocator reserve.
    size_t budget=weights+kv+activation+quant+rope+4*logits+(size_t)P*H*8+(size_t)V*H*4+(size_t(2)<<30);
    printf("MEMORY weights=%zu kv=%zu activation=%zu quant=%zu rope=%zu logits=%zu estimated_peak_with_host_and_2GiB_reserve=%zu\n",weights,kv,activation,quant,rope,logits,budget);
    require(budget<=(size_t(40)<<30),"estimate exceeds external 40GiB guard budget"); fflush(stdout);
    if(mode=="dump") require(mkdir(out.c_str(),0700)==0,"dump output directory must be new; parent must exist");
    stamp(mode.c_str(),"begin");
    G4Weights w=load_weights_gemma4(wd.c_str(),c,1,0);
    G4DecodeState st=alloc_g4_decode_state(c,P);
    int *ids=device<int>(P); CK(cudaMemcpy(ids,tokens.data(),P*4,cudaMemcpyHostToDevice));
    float *cf=device<float>((size_t)P*hf),*sf=device<float>((size_t)P*hf);
    float *cs=device<float>((size_t)P*hs),*ss=device<float>((size_t)P*hs);
    require(gemma4_gen_rope_tables((wd+"/manifest.json").c_str(),P,hf,cf,sf,hs,cs,ss)==0,"serving RoPE generation"); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    if(mode=="golden" || mode=="dump") {
        const char *files[]={"rope_full_cos.bin","rope_full_sin.bin","rope_sliding_cos.bin","rope_sliding_sin.bin"};
        float *ptrs[]={cf,sf,cs,ss}; int dims[]={hf,hf,hs,hs};
        for(int i=0;i<4;++i) {
            auto ref=read_exact<float>(gd+files[i],(size_t)5*dims[i]); finite(ref);
            CK(cudaMemcpy(ptrs[i],ref.data(),ref.size()*4,cudaMemcpyHostToDevice));
            auto actual=download(ptrs[i],ref.size()); require(memcmp(actual.data(),ref.data(),ref.size()*4)==0,"golden RoPE prefix byte identity");
        }
        printf("PREFIX original_tokens=5 original_rope_rows=5 appended_rows=11 serving_rope_generated_then_prefix_preserved=1\n");
        if(mode=="dump") {
            write_new(out+"/token_ids.bin",tokens.data(),tokens.size()*sizeof(int));
            for(int i=0;i<4;++i) {
                auto actual=download(ptrs[i],(size_t)P*dims[i]);
                write_new(out+"/"+files[i],actual.data(),actual.size()*sizeof(float));
            }
        }
    }
    float *lg=device<float>(logits/4), *pn=device<float>((size_t)P*H);
    auto clear=[&]() { st.n_past=0; for(int l=0;l<st.L;++l) {
        bool full=c.layer_is_full?c.layer_is_full[l]:true; size_t n=(size_t)P*(full?kf:c.num_kv_heads)*(full?hf:hs)*4;
        CK(cudaMemset(st.Kc[l],0,n)); CK(cudaMemset(st.Vc[l],0,n));
    }};
    auto run=[&](const ModelConfig &cfg,int offset,int n,int last,int commit) {
        gemma4_decode(cfg,w,st,ids+offset,n,cf,sf,cs,ss,lg,commit,2,last,1,pn);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    };
    bool passed=true;
    if(mode=="golden") {
        for(int d=0;d<=8;++d) {
            ModelConfig limited=c; limited.num_layers=d; clear(); run(limited,0,P,0,0);
            auto all=download_raw(pn,(size_t)P*H); std::vector<float> got(all.begin(),all.begin()+(size_t)5*H);
            auto ref=read_exact<float>(gd+"q4fp32_hs_"+std::to_string(d)+".bin",(size_t)5*H);
            finite(ref);
            bool finite_ok=all_finite(all);
            auto depth_logits=download_raw(lg,(size_t)P*V);
            finite_ok=all_finite(depth_logits) && finite_ok;
            Error e; double rel=std::numeric_limits<double>::infinity();
            if(all_finite(got)) { e=compare(got,ref); rel=e.maxabs/(e.refmax+1e-9); }
            else { e.maxabs=std::numeric_limits<double>::infinity(); for(float x:ref) e.refmax=std::max(e.refmax,std::abs((double)x)); }
            bool depth_pass=finite_ok && rel<1.5e-2; passed=depth_pass && passed;
            printf("GOLDEN depth=%d prefix_rows=5 executed_rows=16 maxabs=%.17g refmax=%.17g normalized_maxabs=%.17g threshold=0.015 finite=%d pass=%d\n",d,e.maxabs,e.refmax,rel,finite_ok,depth_pass); fflush(stdout);
        }
        clear(); run(c,0,P,0,0); auto got=download_raw(lg,(size_t)P*V);
        auto ref=read_exact<float>(gd+"q4_logits.bin",(size_t)5*V); finite(ref);
        int matches=0; for(int i=0;i<5;++i) {
            const float *row=got.data()+(size_t)i*V; bool row_finite=true;
            for(int j=0;j<V;++j) row_finite=row_finite && std::isfinite(row[j]);
            int a=row_finite?argmax(row,V):-1,b=argmax(ref.data()+(size_t)i*V,V);
            matches+=a==b; printf("GOLDEN_ARGMAX position=%d actual=%d reference=%d finite=%d\n",i,a,b,row_finite);
        }
        bool finite_ok=all_finite(got); passed=(matches==5 && finite_ok) && passed;
        printf("GOLDEN full_depth=%d matches=%d/5 all16finite=%d scope=HF_causal_prefix_only\n",c.num_layers,matches,finite_ok);
    } else if(mode=="dump") {
        for(int d=0;d<=c.num_layers;++d) {
            ModelConfig limited=c; limited.num_layers=d; clear(); run(limited,0,P,0,0);
            auto hidden=download(pn,(size_t)P*H);
            write_new(out+"/hs_d"+std::to_string(d)+".bin",hidden.data(),hidden.size()*sizeof(float));
            auto logits_out=download(lg,(size_t)P*V);
            if(d==c.num_layers) write_new(out+"/logits.bin",logits_out.data(),logits_out.size()*sizeof(float));
            printf("DUMP depth=%d rows=%d hidden_width=%d finite=1\n",d,P,H); fflush(stdout);
        }
        std::string metadata="{\n  \"complete\": true,\n  \"path\": \""+std::string(mmq?"MMQ":"SB")+"\",\n"
            "  \"scale_dtype\": "+std::to_string(c.scale_dtype)+",\n"
            "  \"rows\": 16,\n  \"golden_prefix_rows\": 5,\n"
            "  \"hidden_width\": "+std::to_string(H)+",\n"
            "  \"vocab_size\": "+std::to_string(V)+",\n"
            "  \"layers\": "+std::to_string(c.num_layers)+",\n"
            "  \"full_head_dim\": "+std::to_string(hf)+",\n"
            "  \"sliding_head_dim\": "+std::to_string(hs)+",\n"
            "  \"hidden_files\": \"hs_d0.bin through hs_dL.bin; float32 raw prenorm residual [16,H]\",\n"
            "  \"logits_file\": \"logits.bin; float32 [16,V]\",\n"
            "  \"rope_source\": \"serving generation, first5 rows replaced with exact golden\",\n"
            "  \"validation\": \"finite capture only, not HF golden PASS\"\n}\n";
        write_new(out+"/metadata.json",metadata.data(),metadata.size());
    } else if(mode=="determinism") {
        std::vector<float> ref;
        for(int i=0;i<30;++i) {
            clear(); run(c,0,P,0,0); auto got=download(lg,(size_t)P*V);
            if(i==0) ref=got;
            size_t mismatch=0; for(size_t j=0;j<got.size();++j) mismatch+=memcmp(&got[j],&ref[j],4)!=0;
            printf("DETERMINISM repetition=%d/30 rows=%d mismatched_float_bits=%zu finite=1\n",i+1,P,mismatch); fflush(stdout);
            require(mismatch==0,"30-run determinism");
        }
    } else {
        clear(); run(c,0,1024,1,1); require(st.n_past==1024,"whole n_past"); auto ref=download(lg,V);
        clear(); run(c,0,512,1,1); finite(download(lg,V)); require(st.n_past==512,"first chunk n_past");
        run(c,512,512,1,1); require(st.n_past==1024,"second chunk n_past"); auto got=download(lg,V);
        Error e=compare(got,ref); require(e.refl1>0,"zero reference L1");
        int a=argmax(got.data(),V),b=argmax(ref.data(),V);
        printf("CHUNK whole=1024 chunks=512+512 maxabs=%.17g rms=%.17g normalized_L1=sum_abs_difference/sum_abs_whole_reference=%.17g ref_L1=%.17g argmax=%d reference_argmax=%d match=%d finite=1\n",e.maxabs,sqrt(e.squares/V),e.l1/e.refl1,e.refl1,a,b,a==b);
        require(a==b,"chunk final argmax");
    }
    stamp(mode.c_str(),"end"); printf("RESULT mode=%s %s\n",mode.c_str(),passed?"PASS":"FAIL");
    for(float *p:{cf,sf,cs,ss,lg,pn}) CK(cudaFree(p)); CK(cudaFree(ids));
    free_g4_decode_state(st); free_weights_gemma4(w); free_config_gemma4(c); return passed?0:1;
}
