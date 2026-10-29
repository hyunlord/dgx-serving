// Gemma4 plain 서빙 튜닝 스윕 (STEP C-2). 모델-무지 serving_core + Gemma4Backend + dummy rope(현실 길이).
//   spec 無(MTP 부재) → 공간 = Bmax × regime(λ×mix). autotune.py(spec 결합)와 분리한 plain 전용 스윕.
//   측정: regime(λ) × Bmax → throughput(tok/s)·tpot95·avgB. 메모리 사전 필터(MemAvailable + KV 선형 캡).
//   ★타이밍이라 rope=엔진 생성 dummy(값 무관, golden 17행 우회 — 현실 gen 길이 U[16,192]). warm 프로토콜.
//   사용법: ./bench_serve_gemma4 <weights> [uq=2] [nreq=48] [seed=42] [pmax=32] [maxlen=384] [Bmax,...] -- [λ,...]
#include "runtime/knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/serving_core.h"
#include "../kernels/rope.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
using clk=std::chrono::steady_clock;

static size_t mem_available(){ FILE*f=fopen("/proc/meminfo","r"); long kb=0; if(f){char l[128];
    while(fgets(l,sizeof(l),f)) if(sscanf(l,"MemAvailable: %ld",&kb)==1) break; fclose(f);} return (size_t)kb*1024; }

// Gemma4 plain 서빙 backend (serve_gemma4와 동일 훅 — plain, ragged decode_step, dummy rope).
struct Gemma4Backend : ServingBackend {
    ModelConfig c; G4Weights w;
    float *d_cf=nullptr,*d_sf=nullptr,*d_cs=nullptr,*d_ss=nullptr;
    float *d_logit=nullptr; int *d_tok=nullptr; int *d_pos=nullptr; int *d_tok_pf=nullptr;
    const std::vector<int>* p_ids=nullptr;   // 토큰 스트림 참조(chunked prefill 토큰 = ids[0..P-1])
    int V=0,uq=2,maxlen=0,Bmax=0;
    bool has_spec() const override { return false; }
    size_t per_seq_bytes(int ml) const override {
        int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
        int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
        size_t b=0; for(int l=0;l<c.num_layers;l++){ int full=c.layer_is_full?c.layer_is_full[l]:1;
            int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s; b+=(size_t)ml*nkv*hd*2*4; } return b;
    }
    void* alloc_state(int B,int ml) override { G4BatchState* st=new G4BatchState(alloc_g4_batch_state(c,B,ml)); return st; }
    void free_state(void* s) override { G4BatchState* st=(G4BatchState*)s; free_g4_batch_state(*st); delete st; }
    void reset_slot(void*,int) override {}   // 위치독립 KV 덮어쓰기(serve_gemma4와 동일)
    double prefill_slot(void* s,int b,int P) override {  // chunked prefill: 슬롯 b에 P토큰(어댑터=W4A8 bit-id 게이트 통과)
        G4BatchState* st=(G4BatchState*)s;
        std::vector<int> pt(P); for(int i=0;i<P;i++) pt[i]=(*p_ids)[i % p_ids->size()];
        CK(cudaMemcpy(d_tok_pf,pt.data(),(size_t)P*4,cudaMemcpyHostToDevice));
        auto t0=clk::now();
        gemma4_prefill_batch_slot(c,w,*st,b,d_tok_pf,P,d_cf,d_sf,d_cs,d_ss,d_logit,uq);
        CK(cudaDeviceSynchronize()); auto t1=clk::now(); return std::chrono::duration<double>(t1-t0).count();
    }
    double decode_step(void* s,const int* toks,const int* pos,int B) override {
        G4BatchState* st=(G4BatchState*)s;
        CK(cudaMemcpy(d_tok,toks,B*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos,pos,B*4,cudaMemcpyHostToDevice));
        std::vector<int> hp(pos,pos+B); auto t0=clk::now();
        gemma4_decode_batch(c,w,*st,d_tok,d_cf,d_sf,d_cs,d_ss,d_logit,0,uq,d_pos,hp.data());
        CK(cudaDeviceSynchronize()); auto t1=clk::now(); return std::chrono::duration<double>(t1-t0).count();
    }
};

int main(int argc,char**argv){
    knobs_init("bench_serve_gemma4");
    setvbuf(stdout,NULL,_IONBF,0);
    std::string WD=argc>1?argv[1]:"artifacts/gemma4-31b/weights";
    int uq=argc>2?atoi(argv[2]):2; int nreq=argc>3?atoi(argv[3]):48; unsigned seed=argc>4?atoi(argv[4]):42;
    int pmax=argc>5?atoi(argv[5]):32; int maxlen=argc>6?atoi(argv[6]):384;
    std::vector<int> Bs; std::vector<double> lams; bool after=false;
    for(int i=7;i<argc;i++){ if(std::string(argv[i])=="--"){after=true;continue;} if(after) lams.push_back(atof(argv[i])); else Bs.push_back(atoi(argv[i])); }
    if(Bs.empty()) Bs={4,8,16}; if(lams.empty()) lams={1,4,16};
    Gemma4Backend be; be.uq=uq;
    be.c=load_config_gemma4((WD+"/manifest.json").c_str());
    be.V=be.c.vocab_size; be.maxlen=maxlen; int V=be.V;
    const int hd_s=be.c.head_dim, hd_f=be.c.global_head_dim?be.c.global_head_dim:hd_s;
    int Bmax=0; for(int b:Bs) if(b>Bmax)Bmax=b; be.Bmax=Bmax;
    // dummy rope [maxlen, hd] (타이밍 — 값 무관)
    CK(cudaMalloc(&be.d_cf,(size_t)maxlen*hd_f*4)); CK(cudaMalloc(&be.d_sf,(size_t)maxlen*hd_f*4));
    CK(cudaMalloc(&be.d_cs,(size_t)maxlen*hd_s*4)); CK(cudaMalloc(&be.d_ss,(size_t)maxlen*hd_s*4));
    launch_gen_rope(be.d_cf,be.d_sf,maxlen,hd_f,1e4f); launch_gen_rope(be.d_cs,be.d_ss,maxlen,hd_s,1e4f); CK(cudaDeviceSynchronize());
    be.w=load_weights_gemma4(WD.c_str(),be.c,uq!=0?1:0,uq==0?1:0);
    CK(cudaMalloc(&be.d_tok,Bmax*4)); CK(cudaMalloc(&be.d_pos,Bmax*4)); CK(cudaMalloc(&be.d_logit,(size_t)Bmax*V*4));
    CK(cudaMalloc(&be.d_tok_pf,(size_t)maxlen*4));   // chunked prefill 프롬프트 토큰(P≤pmax≤maxlen)
    // 토큰 스트림(더미, 서빙 스케줄만 — 내용 무관)
    std::vector<int> ids(256); for(int i=0;i<256;i++) ids[i]=(i*131+7)%V;
    be.p_ids=&ids;   // chunked prefill 토큰 소스

    size_t avail=mem_available(); size_t perseq=be.per_seq_bytes(maxlen);
    printf("# Gemma4 plain 서빙 스윕: uq=%d maxlen=%d pmax=%d nreq=%d seed=%u | per-seq KV=%.1fMB, MemAvailable=%.1fGB\n",
           uq,maxlen,pmax,nreq,seed,perseq/1e6,avail/1e9);
    printf("# 메모리 필터: Bmax×per-seq ≤ 0.7·MemAvailable=%.1fGB → Bmax_hi≈%ld\n",0.7*avail/1e9,(long)(0.7*avail/perseq));
    printf("\n  %-4s %-5s %-5s | %8s %8s | %7s %7s | %5s %5s\n","Bmax","mode","λ","req/s","tok/s","tpot50","tpot95","avgB","maxQ");
    printf("  %s\n",std::string(72,'-').c_str());

    for(int B:Bs){
        if((double)B*perseq > 0.7*(double)avail){ printf("  %-4d %-5s   예산 초과(스킵, 메모리 필터)\n",B,"-"); continue; }
        void* st=be.alloc_state(B,maxlen);
        // warm-up(클럭+캐시): 양 모드 1회씩(공정 비교)
        { auto tr=sc_gen_trace(8.0,12,seed,pmax); sc_run_continuous(be,st,ids,tr,B,maxlen,false); sc_run_continuous(be,st,ids,tr,B,maxlen,true); }
        for(double lam:lams){
            for(int ch=0;ch<=1;ch++){   // 0=c1tok(1토큰 ingest), 1=c-ch(chunked prefill 어댑터, W4A8 bit-id)
                auto tr=sc_gen_trace(lam,nreq,seed,pmax); Stats s=sc_run_continuous(be,st,ids,tr,B,maxlen,ch!=0);
                printf("  %-4d %-5s %-5g | %8.3f %8.1f | %7.4f %7.4f | %5.2f %5.0f\n",
                       B, ch?"c-ch":"c1tok", lam,s.thru_req,s.thru_tok,s.tpot50,s.tpot95,s.avgB,s.maxq);
            }
        }
        be.free_state(st);
    }
    printf("\n  (throughput=tok/s. tpot=per-token latency(초). avgB=평균 active 배치. plain 서빙(spec 無). W4A8 sb dispatch.)\n");
    printf("  (c1tok=프롬프트 1토큰/step ingest. c-ch=chunked prefill(어댑터, W4A8 bit-id 게이트 통과=순수 스케줄링).)\n");
    cudaFree(be.d_tok);cudaFree(be.d_pos);cudaFree(be.d_logit);cudaFree(be.d_tok_pf);cudaFree(be.d_cf);cudaFree(be.d_sf);cudaFree(be.d_cs);cudaFree(be.d_ss);
    free_weights_gemma4(be.w); free_config_gemma4(be.c); return 0;
}
