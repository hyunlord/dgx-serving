// Gemma4 full 서빙 스모크 (M4 라운드2 STEP 3). 모델-무지 serving_core.h 위에 Gemma4Backend(ServingBackend 훅)를 등록,
//   연속 배칭 파이프라인(admission 예산→slot 재사용→ragged decode→완료)을 Gemma4로 관통 — 서빙 스택 모델-중립 실증.
//   게이트: (1) 결정성(같은 trace 2회 = makespan/완료 bit-identical) (2) 완료(전 요청 done) (3) 누수(GPU free·RSS delta).
//   ★ 성능 주장 금지(스모크). golden 정확성은 test_gemma4_ragged(ragged==single bit-id)가 이미 증명 — 여기선 파이프라인만.
//   rope: gemma4 golden(gen_rope_*, 17위치)만 정확 → 짧은 길이(P+gen≤16)로 rope 근사 0. chunked=false(1토큰/step,
//     ragged 경로). chunked prefill은 gemma4 prefill_batch_slot 아날로그 부재로 이번 범위 밖(보고).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <unistd.h>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/serving_core.h"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
using clk=std::chrono::steady_clock;
static std::string WD,GD;
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static long rss_kb(){ FILE*f=fopen("/proc/self/statm","r"); long sz=0,res=0; if(f){ if(fscanf(f,"%ld %ld",&sz,&res)!=2)res=0; fclose(f);} return res*(sysconf(_SC_PAGESIZE)/1024); }

// ── Gemma4 plain 서빙 backend: qwen과 동일 인터페이스(ServingBackend 훅 5개). GDN 없음 → plain. ──
struct Gemma4Backend : ServingBackend {
    ModelConfig c; G4Weights w;
    float *d_cf=nullptr,*d_sf=nullptr,*d_cs=nullptr,*d_ss=nullptr;   // dual rope(full/sliding)
    float *d_logit=nullptr; int *d_tok=nullptr; int *d_pos=nullptr;
    int V=0,uq=2,maxlen=0;
    bool has_spec() const override { return false; }
    size_t per_seq_bytes(int ml) const override {   // KV only(GDN 없음): full=hd_f·nkv_f, sliding=hd_s·nkv_s, ×2(K,V)×4(fp32)
        int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
        int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
        size_t b=0; for(int l=0;l<c.num_layers;l++){ int full=c.layer_is_full?c.layer_is_full[l]:1;
            int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s; b+=(size_t)ml*nkv*hd*2*4; } return b;
    }
    void* alloc_state(int Bmax,int ml) override { G4BatchState* st=new G4BatchState(alloc_g4_batch_state(c,Bmax,ml)); return st; }
    void free_state(void* s) override { G4BatchState* st=(G4BatchState*)s; free_g4_batch_state(*st); delete st; }
    // Gemma4 KV=위치독립 cache. 슬롯 재사용은 새 요청이 pos 0부터 KV를 덮어쓰고 attn이 현재 pos로 bound → 잔재 무영향(no-op 안전).
    void reset_slot(void*,int) override {}
    // chunked prefill: gemma4 prefill_batch_slot 아날로그 부재 → 스모크는 chunked=false 사용, 여기 진입 시 명시적 abort(은폐 금지).
    double prefill_slot(void*,int,int) override { fprintf(stderr,"[gemma4] chunked prefill 미지원(스모크는 chunked=false) — 범위 밖\n"); exit(2); }
    double decode_step(void* s,const int* toks,const int* pos,int B) override {   // ragged 1토큰 decode(STEP 1 경로)
        G4BatchState* st=(G4BatchState*)s;
        CK(cudaMemcpy(d_tok,toks,B*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos,pos,B*4,cudaMemcpyHostToDevice));
        std::vector<int> hp(pos,pos+B); auto t0=clk::now();
        gemma4_decode_batch(c,w,*st,d_tok,d_cf,d_sf,d_cs,d_ss,d_logit,0,uq,d_pos,hp.data());   // commit=0(ragged, scheduler가 위치 관리)
        CK(cudaDeviceSynchronize()); auto t1=clk::now(); return std::chrono::duration<double>(t1-t0).count();
    }
};

// 짧은-길이 결정적 트레이스(golden rope 17위치 내: P+gen≤maxpos). Poisson 도착 + 작은 P/gen.
static std::vector<Req> tiny_trace(double lambda,int nreq,unsigned seed,int Pmax,int genMax){
    std::mt19937 rng(seed); std::exponential_distribution<double> ia(lambda);
    std::uniform_int_distribution<int> pP(2,Pmax), gG(3,genMax); std::vector<Req> v(nreq); double t=0;
    for(int i=0;i<nreq;i++){ t+=ia(rng); v[i].arrival=t; v[i].P=pP(rng); v[i].gen=gG(rng); }
    return v;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    WD=argc>1?argv[1]:"artifacts/gemma4-31b/weights"; GD=(argc>2?argv[2]:"artifacts/gemma4-31b/golden")+std::string("/");
    int uq=argc>3?atoi(argv[3]):1; int Bmax=argc>4?atoi(argv[4]):4; int nreq=argc>5?atoi(argv[5]):24; unsigned seed=argc>6?atoi(argv[6]):42;
    double lam=argc>7?atof(argv[7]):2.0;
    Gemma4Backend be; be.uq=uq;
    be.c=load_config_gemma4((WD+"/manifest.json").c_str());
    be.V=be.c.vocab_size; int V=be.V;
    const int hd_f=be.c.global_head_dim?be.c.global_head_dim:be.c.head_dim, hd_s=be.c.head_dim;
    // rope: golden(gen_rope_*, 17위치)만 정확. 행수 자동감지, maxlen=golden 행수(짧은 길이라 근사 0).
    std::vector<int> gtok=ri("gen_tokens.bin"); int rope_rows=gtok.size();   // gen_tokens 길이=rope 행수(HF 캡처)
    int maxlen=rope_rows; be.maxlen=maxlen;
    auto cf=rf("gen_rope_full_cos.bin",(size_t)rope_rows*hd_f), sf=rf("gen_rope_full_sin.bin",(size_t)rope_rows*hd_f);
    auto cs=rf("gen_rope_sliding_cos.bin",(size_t)rope_rows*hd_s), ss=rf("gen_rope_sliding_sin.bin",(size_t)rope_rows*hd_s);
    be.d_cf=up(cf); be.d_sf=up(sf); be.d_cs=up(cs); be.d_ss=up(ss);
    be.w=load_weights_gemma4(WD.c_str(),be.c,uq!=0?1:0,uq==0?1:0);
    CK(cudaMalloc(&be.d_tok,Bmax*4)); CK(cudaMalloc(&be.d_pos,Bmax*4)); CK(cudaMalloc(&be.d_logit,(size_t)Bmax*V*4));
    std::vector<int> ids=gtok;   // 토큰 스트림=golden gen_tokens(in-distribution). 정확성은 test_gemma4_ragged가 보증.
    // 트레이스: P+gen ≤ maxlen(rope 정확 범위). Pmax=4, genMax=maxlen-Pmax-1 (pos 0..maxlen-1 안).
    int Pmax=4, genMax=(maxlen-Pmax-1); if(genMax<3)genMax=3;
    printf("[serve_gemma4] uq=%d Bmax=%d nreq=%d seed=%u λ=%.1f maxlen=%d(golden rope %d행) P≤%d gen≤%d\n",
           uq,Bmax,nreq,seed,lam,maxlen,rope_rows,Pmax,genMax);

    // admission 예산 가드(backend 훅) — 시스템 다운 방지
    size_t perseq=be.per_seq_bytes(maxlen); size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB); double budget=0.7*(double)freeB;
    printf("  admission 예산: per-seq %.1fMB × Bmax %d = %.3fGB vs 0.7·free %.2fGB — %s\n",
           perseq/1e6,Bmax,(double)Bmax*perseq/1e9,budget/1e9,(double)Bmax*perseq<=budget?"OK":"초과");
    if((double)Bmax*perseq>budget){ printf("  admission guard: 예산 초과 → 중단.\n"); return 1; }

    // 누수 창: state alloc 前 baseline(weights 이미 로드됨) → 서빙 후 state free 시점. weight free는 이 창 밖(오염 방지).
    size_t gpu_free0=freeB; long rss0=rss_kb();
    void* st=be.alloc_state(Bmax,maxlen);
    // (1) 결정성: 같은 trace 2회. ★이 시뮬은 sim clock=스텝 wall 누적, admission=arrival≤simt 비교 → 본질적 wall-타이밍
    //   의존(통계적 재현). forward 수학은 bit-identical(test_gemma4_ragged 증명). 게이트=완료수 동일 + makespan 상대오차<1%.
    auto tr1=tiny_trace(lam,nreq,seed,Pmax,genMax); Stats S1=sc_run_continuous(be,st,ids,tr1,Bmax,maxlen,false);
    auto tr2=tiny_trace(lam,nreq,seed,Pmax,genMax); Stats S2=sc_run_continuous(be,st,ids,tr2,Bmax,maxlen,false);
    double mk_rel = S1.makespan>0? fabs(S1.makespan-S2.makespan)/S1.makespan : 0;
    bool det = (S1.completed==S2.completed) && (mk_rel < 0.01);
    printf("  (1) 결정성(완료수 동일+makespan 상대오차<1%%, wall 통계): run1(완료 %d, mk %.4f) run2(완료 %d, mk %.4f) rel=%.3f%% → %s\n",
           S1.completed,S1.makespan,S2.completed,S2.makespan, mk_rel*100, det?"PASS":"FAIL");
    // (2) 완료: 전 요청 done
    bool complete = (S1.completed==nreq);
    printf("  (2) 완료: %d/%d 요청 done → %s\n", S1.completed,nreq, complete?"PASS":"FAIL");
    // (3) 누수: 서빙 state alloc/free 창의 GPU free / RSS delta (weight free 제외 = 순수 서빙 누수)
    be.free_state(st);
    size_t gpu_free1=0; cudaMemGetInfo(&gpu_free1,&totB); long rss1=rss_kb();
    // 누수 = 메모리 미반환 = free 감소(양의 Δ). free 증가(음의 Δ)는 누수 아님(CUDA 할당자 transient 정착) → 단방향 게이트.
    double gpu_leak_mb=((double)gpu_free0-(double)gpu_free1)/1e6; long rss_leak_mb=(rss1-rss0)/1024;
    bool noleak = (gpu_leak_mb < 64.0) && (rss_leak_mb < 128);   // free 감소<64MB & RSS 증가<128MB
    printf("  (3) 누수(서빙 state 창, weight free 제외; 누수=free 감소만): GPU free Δ=%.1fMB, RSS Δ=%ldMB → %s (임계 GPU감소<64·RSS<128MB)\n", gpu_leak_mb,rss_leak_mb, noleak?"PASS":"FAIL");
    cudaFree(be.d_tok); cudaFree(be.d_pos); cudaFree(be.d_logit);
    cudaFree(be.d_cf); cudaFree(be.d_sf); cudaFree(be.d_cs); cudaFree(be.d_ss);
    free_weights_gemma4(be.w);

    bool pass = det && complete && noleak;
    printf("[serve_gemma4] %s (결정성+완료+무누수 — 서빙 스택 모델-중립, 성능 주장 없음)\n", pass?"PASS":"FAIL");
    free_config_gemma4(be.c); return pass?0:1;
}
