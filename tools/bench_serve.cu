// 챕터9 production scheduler 실부하 벤치: Poisson 요청 스트림 + 가변 길이 + warm-up + 정적 대비 + 부하 스윕.
//   측정: throughput(req/s·tok/s), p50/p95/p99 latency(e2e·TTFT·TPOT), avg active-B(활용 proxy), admission.
//   [M4 라운드2] 스케줄링을 serving_core.h(모델 무지)로 이관, qwen 결합을 QwenBackend(ServingBackend 훅 5개)로 묶음.
//   토큰 선택(argmax/sampling)·logit readout 제외 — GPU 병렬 op라 직교(forward가 지배). gen 길이는 trace로 사전결정.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#include "../runtime/serving_core.h"
#include "../models/qwen35_batch.h"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
using clk=std::chrono::steady_clock;
static std::string WD,GD;
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

// ── Qwen plain 서빙 backend: bench_serve 결합 6지점을 ServingBackend 훅으로(이동만, 로직 불변) ──
//   상태 소유: config·weights·rope·logit/tok device 버퍼·tiled 프롬프트. 코어는 이 backend만 접촉(③ 회피).
struct QwenBackend : ServingBackend {
    ModelConfig c; QwenWeights w; float *d_cos=nullptr,*d_sin=nullptr; float *d_logit=nullptr; int *d_tok=nullptr;
    int *g_dtiled=nullptr; int V=0,NKV=0,HD=0,uq=2,maxlen=0;
    bool has_spec() const override { return false; }   // plain(spec 사이클 없음)
    size_t per_seq_bytes(int ml) const override {       // 결합⑤: admission 예산(full=KV, gdn=state)
        int nfull=0,ngdn=0; for(int l=0;l<c.num_layers;l++) c.layer_is_full[l]?nfull++:ngdn++;
        int nh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),Kk=c.gdn_conv_kernel;
        return (size_t)nfull*ml*NKV*HD*2*2 + (size_t)ngdn*((size_t)nh*D*D+(size_t)(Kk-1)*convd)*4;
    }
    void* alloc_state(int Bmax,int ml) override {        // 결합③: batch state alloc+reset
        BatchDecodeState* st=new BatchDecodeState(alloc_batch_decode_state(c,Bmax,ml));
        st->B=Bmax; reset_batch_decode_state(c,*st); return st;
    }
    void free_state(void* s) override { BatchDecodeState* st=(BatchDecodeState*)s; free_batch_decode_state(*st); delete st; }
    void reset_slot(void* s,int b) override {            // b>=0: 슬롯 재사용 clear / b<0: 전체 reset(static)
        BatchDecodeState* st=(BatchDecodeState*)s;
        if(b>=0) reset_batch_slot(c,*st,b); else reset_batch_decode_state(c,*st);
    }
    double prefill_slot(void* s,int b,int P) override {  // 결합②: chunked prefill(1스텝, 측정)
        BatchDecodeState* st=(BatchDecodeState*)s; auto t0=clk::now();
        prefill_batch_slot(c,w,*st,b,g_dtiled,P,d_cos,d_sin,d_logit,uq,512);
        CK(cudaDeviceSynchronize()); auto t1=clk::now(); return std::chrono::duration<double>(t1-t0).count();
    }
    double decode_step(void* s,const int* toks,const int* pos,int B) override {  // 결합①: decode 1스텝
        BatchDecodeState* st=(BatchDecodeState*)s; st->B=B;   // static=Bb / continuous=Bmax
        CK(cudaMemcpy(d_tok,toks,B*4,cudaMemcpyHostToDevice));
        std::vector<int> p(pos,pos+B); auto t0=clk::now();
        qwen35_decode_batch(c,w,*st,d_tok,d_cos,d_sin,d_logit,uq,p.data());
        CK(cudaDeviceSynchronize()); auto t1=clk::now(); return std::chrono::duration<double>(t1-t0).count();
    }
};

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int uq=argc>3?atoi(argv[3]):2; int Bmax=argc>4?atoi(argv[4]):8; int nreq=argc>5?atoi(argv[5]):200; unsigned seed=argc>6?atoi(argv[6]):42;
    std::vector<double> lambdas; for(int i=7;i<argc;i++) lambdas.push_back(atof(argv[i]));
    if(lambdas.empty()) lambdas={0.5,1,2,4,8,16,32};
    int pmax=32; if(getenv("PMAX")) pmax=atoi(getenv("PMAX"));
    QwenBackend be; be.uq=uq;
    be.c=load_config_qwen((WD+"/manifest.json").c_str());
    be.V=be.c.vocab_size; be.NKV=be.c.num_kv_heads; be.HD=be.c.head_dim; int rot=be.c.rotary_dim;
    int maxlen=(pmax+224>256)?pmax+224:256; be.maxlen=maxlen; int V=be.V;
    static std::vector<int> ids=ri("token_ids.bin");
    // rope: ext 테이블(pos>=128 정확) 있으면 maxlen행, 없으면 long 128 타일(pos>128 근사).
    std::vector<float> cl, sl;
    { FILE*e=fopen((GD+"rope_cos_ext.bin").c_str(),"rb");
      if(e){ fseek(e,0,SEEK_END); long er=ftell(e)/4/rot; fclose(e);
             int use=(er>=maxlen)?maxlen:(int)er; cl=rf("rope_cos_ext.bin",(size_t)use*rot); sl=rf("rope_sin_ext.bin",(size_t)use*rot);
             printf("  rope: ext 테이블 %d행(pos>=128 정확)\n",use); }
      else { auto c0=rf("rope_cos_long.bin",(size_t)128*rot), s0=rf("rope_sin_long.bin",(size_t)128*rot);
             cl.resize((size_t)maxlen*rot); sl.resize((size_t)maxlen*rot);
             for(int i=0;i<maxlen;i++)for(int j=0;j<rot;j++){cl[(size_t)i*rot+j]=c0[(size_t)(i%128)*rot+j];sl[(size_t)i*rot+j]=s0[(size_t)(i%128)*rot+j];}
             printf("  rope: long 128행 타일(ext 없음 — pos>128 근사, gen_rope_long.py 권장)\n"); } }
    be.d_cos=up(cl); be.d_sin=up(sl);
    be.w=load_weights_qwen(WD.c_str(),be.c,1,uq==0?1:0);
    int lgrows = Bmax>512?Bmax:512;
    CK(cudaMalloc(&be.d_tok,Bmax*4)); CK(cudaMalloc(&be.d_logit,(size_t)lgrows*V*4));
    { std::vector<int> tiled(maxlen); for(int i=0;i<maxlen;i++) tiled[i]=ids[i%ids.size()];
      CK(cudaMalloc(&be.g_dtiled,(size_t)maxlen*4)); CK(cudaMemcpy(be.g_dtiled,tiled.data(),(size_t)maxlen*4,cudaMemcpyHostToDevice)); }
    // admission guard: KV 예산(alloc-time) — 결합⑤를 backend 훅으로
    size_t perseq=be.per_seq_bytes(maxlen);
    size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB); double budget=0.7*(double)freeB;
    printf("[bench_serve] uq=%d(%s) Bmax=%d nreq=%d seed=%u maxlen=%d\n",uq,uq==0?"fp32":"W4A8",Bmax,nreq,seed,maxlen);
    printf("  admission 예산: per-seq %.1fMB × Bmax %d = %.2fGB vs 0.7·free %.2fGB — %s%s\n",
           perseq/1e6,Bmax,(double)Bmax*perseq/1e9,budget/1e9,(double)Bmax*perseq<=budget?"OK":"초과",
           (double)Bmax*perseq>budget?" (Bmax↓ 필요, 중단)":"");
    if((double)Bmax*perseq>budget){ printf("  admission guard: alloc 예산 초과 → 중단(시스템 다운 방지).\n"); return 1; }
    void* st=be.alloc_state(Bmax,maxlen);
    // warm-up 실행(cold 버림)
    { auto tr=sc_gen_trace(8.0,12,seed,pmax); sc_run_continuous(be,st,ids,tr,Bmax,maxlen,true); }

    printf("\n  === 부하 스윕 (warm-up 완료분 20%% 제외, seed 고정 trace) ===\n");
    printf("  gen: short U[16,48] 70%% / long U[96,192] 30%% (평균 ~66), 프롬프트 U[8,%d]. c1tok=1토큰prefill c-ch=chunked prefill\n",pmax);
    printf("  %-6s %-4s | %8s %8s | %7s %7s %7s | %7s %7s | %5s %5s\n",
           "mode","λ","req/s","tok/s","e2e50","e2e95","e2e99","ttft95","tpot95","avgB","maxQ");
    printf("  %s\n",std::string(96,'-').c_str());
    auto row=[&](const char*nm,double lam,Stats&s){ printf("  %-6s %-4g | %8.3f %8.1f | %7.3f %7.3f %7.3f | %7.3f %7.4f | %5.2f %5.0f\n",
           nm,lam,s.thru_req,s.thru_tok,s.e2e50,s.e2e95,s.e2e99,s.ttft95,s.tpot95,s.avgB,s.maxq); };
    bool run1tok = !getenv("SKIP1TOK");
    bool runstat = !getenv("SKIPSTAT");
    for(double lam:lambdas){
        auto t2=sc_gen_trace(lam,nreq,seed,pmax); Stats Cc=sc_run_continuous(be,st,ids,t2,Bmax,maxlen,true);
        if(run1tok){ auto t1=sc_gen_trace(lam,nreq,seed,pmax); Stats C1=sc_run_continuous(be,st,ids,t1,Bmax,maxlen,false); row("c1tok",lam,C1); }
        row("c-ch",lam,Cc);
        if(runstat){ auto t3=sc_gen_trace(lam,nreq,seed,pmax); Stats T=sc_run_static(be,st,ids,t3,Bmax,maxlen); row("stat",lam,T); }
    }
    printf("\n  (latency 초 단위. 토큰선택/logit readout 제외=forward 지배. avgB=평균 active 배치=활용 proxy. maxQ=최대 큐깊이=부하.)\n");
    be.free_state(st); free_weights_qwen(be.w); free_config_qwen(be.c); return 0;
}
