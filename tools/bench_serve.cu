// 챕터9 production scheduler 실부하 벤치: Poisson 요청 스트림 + 가변 길이 + warm-up + 정적 대비 + 부하 스윕.
//   측정: throughput(req/s·tok/s), p50/p95/p99 latency(e2e·TTFT·TPOT), avg active-B(활용 proxy), admission.
//   연속(continuous) vs 정적(static) 대비 — 연속 이득 = 꼬리 지연(짧은 요청이 긴 것에 안 막힘) + slot 즉시 재사용.
//   sim clock = 실제 스텝 wall-clock 누적(std::chrono), idle면 다음 도착까지 fast-forward(GPU 놀림 반영).
//   토큰 선택(argmax/sampling)·logit readout 제외 — GPU 병렬 op라 직교(forward가 지배). gen 길이는 trace로 사전결정.
//   결정성: throughput/latency는 타이밍 의존 통계(seed 고정 trace로 재현, 여러 seed로 분산 볼 수 있음).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <deque>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#include "../models/qwen35_batch.h"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
using clk=std::chrono::steady_clock;
static std::string WD,GD;
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static double pct(std::vector<double>&v,double p){ if(v.empty())return 0; std::sort(v.begin(),v.end()); size_t i=(size_t)ceil(p*v.size()); if(i>0)i--; if(i>=v.size())i=v.size()-1; return v[i]; }

struct Req{ int P, gen; double arrival; double admit=-1, first=-1, done=-1; };
struct Stats{ double thru_req=0,thru_tok=0; double ttft50=0,ttft95=0,ttft99=0,e2e50=0,e2e95=0,e2e99=0,tpot50=0,tpot95=0,tpot99=0;
              double avgB=0,maxq=0,makespan=0; int completed=0; };

// 전역(main서 초기화) — 커널 자산
static ModelConfig c; static QwenWeights w; static float *d_cos,*d_sin; static int V,NKV,HD;
static int *d_tok; static float *d_logit; static int uq, maxlen;
static const std::vector<int> *IDS;
static int *g_dtiled=nullptr;   // maxlen 타일링 프롬프트(chunked prefill용 device 버퍼)

// 한 스텝 forward(가변 active B, ragged 위치). 반환: 스텝 wall(초). token 선택/readout 제외.
static double step_forward(BatchDecodeState&st, std::vector<int>&toks, std::vector<int>&pos){
    int B=st.B;
    CK(cudaMemcpy(d_tok,toks.data(),B*4,cudaMemcpyHostToDevice));
    auto t0=clk::now();
    qwen35_decode_batch(c,w,st,d_tok,d_cos,d_sin,d_logit,uq,pos.data());
    CK(cudaDeviceSynchronize());
    auto t1=clk::now();
    return std::chrono::duration<double>(t1-t0).count();
}

// ── continuous scheduler: 요청 도착 시점에 slot 나면 즉시 admit(재사용), 매 스텝 active만 forward ──
//   chunked=false: 프롬프트 1토큰/step ingest(현행). chunked=true: admit 시 prefill_batch_slot로 병렬 prefill(1스텝).
static Stats run_continuous(std::vector<Req> reqs,int Bmax,bool chunked){
    int R=reqs.size();
    BatchDecodeState st=alloc_batch_decode_state(c,Bmax,maxlen); st.B=Bmax;
    reset_batch_decode_state(c,st);
    struct Slot{int req=-1,pos=0,P=0,gen=0,g=0;}; std::vector<Slot> slot(Bmax);
    std::deque<int> queue; int nextArr=0, completed=0; double simt=0; long stepN=0; double sumB=0, maxq=0;
    while(completed<R){
        // 도착: arrival<=simt인 요청 큐로
        while(nextArr<R && reqs[nextArr].arrival<=simt) queue.push_back(nextArr++);
        // free slot에 admit(재사용) — reset_batch_slot(GDN 잔재 clear)
        for(int b=0;b<Bmax;b++) if(slot[b].req<0 && !queue.empty()){
            int r=queue.front(); if(reqs[r].P+reqs[r].gen>maxlen){queue.pop_front();continue;} // 길이 가드
            queue.pop_front(); slot[b]=Slot(); slot[b].req=r; slot[b].P=reqs[r].P; slot[b].gen=reqs[r].gen;
            reset_batch_slot(c,st,b); reqs[r].admit=simt;
            if(chunked){  // 병렬 prefill(1스텝, 측정) — 1토큰/step 대신. 후 pos=P부터 decode. 첫 토큰=prefill 로짓(TTFT).
                auto p0=clk::now(); prefill_batch_slot(c,w,st,b,g_dtiled,reqs[r].P,d_cos,d_sin,d_logit,uq,512);
                CK(cudaDeviceSynchronize()); auto p1=clk::now(); simt+=std::chrono::duration<double>(p1-p0).count();
                slot[b].pos=reqs[r].P; slot[b].g=1; reqs[r].first=simt;  // 첫 gen 토큰=prefill 마지막 로짓
                if(slot[b].g>=slot[b].gen){ reqs[r].done=simt; completed++; slot[b]=Slot(); } } }
        // active 수 세기
        int active=0; for(int b=0;b<Bmax;b++) if(slot[b].req>=0) active++;
        if(active==0){ // GPU idle → 다음 도착까지 fast-forward(스텝 안 돌림)
            if(nextArr<R){ simt=reqs[nextArr].arrival; continue; } else break; }
        if((double)queue.size()>maxq) maxq=queue.size();
        // 배치 구성: active=실토큰, free=dummy(무시)
        std::vector<int> toks(Bmax),pos(Bmax);
        for(int b=0;b<Bmax;b++){ Slot&s=slot[b];
            if(s.req<0){toks[b]=0;pos[b]=0;} else {toks[b]=(*IDS)[s.pos%IDS->size()];pos[b]=s.pos;} }
        double dt=step_forward(st,toks,pos); simt+=dt; sumB+=active; stepN++;
        // advance active slots
        for(int b=0;b<Bmax;b++){ Slot&s=slot[b]; if(s.req<0)continue;
            if(s.pos<s.P-1){ s.pos++; }
            else { if(s.g==0) reqs[s.req].first=simt;  // 첫 토큰(prefill 마지막 스텝 후)
                   s.g++; s.pos++;
                   if(s.g>=s.gen){ reqs[s.req].done=simt; completed++; slot[b]=Slot(); } } }
    }
    free_batch_decode_state(st);
    // 통계(warm-up 제외: 완료 순 앞 20% 버림)
    std::vector<int> order; for(int r=0;r<R;r++) if(reqs[r].done>=0) order.push_back(r);
    std::sort(order.begin(),order.end(),[&](int a,int b){return reqs[a].done<reqs[b].done;});
    int warm=(int)(order.size()*0.2); Stats S; std::vector<double> ttft,e2e,tpot;
    double t_lo=1e18,t_hi=0; long toks_steady=0;
    for(size_t k=warm;k<order.size();k++){ int r=order[k]; double a=reqs[r].arrival,f=reqs[r].first,d=reqs[r].done;
        ttft.push_back(f-a); e2e.push_back(d-a); if(reqs[r].gen>1) tpot.push_back((d-f)/(reqs[r].gen-1));
        t_lo=std::min(t_lo,a); t_hi=std::max(t_hi,d); toks_steady+=reqs[r].gen; }
    S.completed=order.size(); int nst=order.size()-warm; double win=t_hi-t_lo;
    S.thru_req=win>0?nst/win:0; S.thru_tok=win>0?toks_steady/win:0;
    S.ttft50=pct(ttft,.5);S.ttft95=pct(ttft,.95);S.ttft99=pct(ttft,.99);
    S.e2e50=pct(e2e,.5);S.e2e95=pct(e2e,.95);S.e2e99=pct(e2e,.99);
    S.tpot50=pct(tpot,.5);S.tpot95=pct(tpot,.95);S.tpot99=pct(tpot,.99);
    S.avgB=stepN?sumB/stepN:0; S.maxq=maxq; S.makespan=simt; return S;
}

// ── static scheduler: 배치를 채워(≤Bmax) launch하면 그 배치 전원 완료까지 slot 붙잡음(중간 합류 X = head-of-line) ──
static Stats run_static(std::vector<Req> reqs,int Bmax){
    int R=reqs.size();
    BatchDecodeState st=alloc_batch_decode_state(c,Bmax,maxlen); st.B=Bmax;
    reset_batch_decode_state(c,st);
    std::deque<int> queue; int nextArr=0, completed=0; double simt=0; long stepN=0; double sumB=0,maxq=0;
    while(completed<R){
        // 배치 형성: 큐에 Bmax개 모이거나(포화) 더 안 오면 있는 만큼. 없으면 다음 도착까지 fast-forward.
        while(nextArr<R && reqs[nextArr].arrival<=simt) queue.push_back(nextArr++);
        if(queue.empty()){ if(nextArr<R){simt=reqs[nextArr].arrival;continue;} else break; }
        // Bmax 미만이고 곧 도착이 있으면 채워질 때까지 대기(다음 도착으로 점프). 다 왔으면 있는 만큼 launch.
        if((int)queue.size()<Bmax && nextArr<R && reqs[nextArr].arrival> simt){ simt=reqs[nextArr].arrival; continue; }
        struct Slot{int req,pos=0,P,gen,g=0;}; std::vector<Slot> slot;
        for(int b=0;b<Bmax && !queue.empty();b++){ int r=queue.front();
            if(reqs[r].P+reqs[r].gen>maxlen){queue.pop_front();b--;continue;}
            queue.pop_front(); Slot s; s.req=r;s.P=reqs[r].P;s.gen=reqs[r].gen; slot.push_back(s); reqs[r].admit=simt; }
        if(slot.empty()) continue;
        int Bb=slot.size(); st.B=Bb; reset_batch_decode_state(c,st);
        if((double)queue.size()>maxq) maxq=queue.size();
        // 배치 전원 완료까지 (head-of-line: 짧은 요청도 긴 것 끝날 때까지 slot 점유)
        int aliveInit=Bb;
        while(true){ int alive=0; for(auto&s:slot) if(s.g<s.gen) alive++; if(alive==0) break;
            std::vector<int> toks(Bb),pos(Bb);
            for(int b=0;b<Bb;b++){ Slot&s=slot[b]; toks[b]=(*IDS)[s.pos%IDS->size()]; pos[b]=s.pos; }
            double dt=step_forward(st,toks,pos); simt+=dt; sumB+=alive; stepN++; (void)aliveInit;
            for(int b=0;b<Bb;b++){ Slot&s=slot[b]; if(s.g>=s.gen)continue;
                if(s.pos<s.P-1) s.pos++;
                else { if(s.g==0) reqs[s.req].first=simt; s.g++; s.pos++;
                       if(s.g>=s.gen){ reqs[s.req].done=simt; completed++; } } } }
    }
    free_batch_decode_state(st);
    std::vector<int> order; for(int r=0;r<R;r++) if(reqs[r].done>=0) order.push_back(r);
    std::sort(order.begin(),order.end(),[&](int a,int b){return reqs[a].done<reqs[b].done;});
    int warm=(int)(order.size()*0.2); Stats S; std::vector<double> ttft,e2e,tpot;
    double t_lo=1e18,t_hi=0; long toks_steady=0;
    for(size_t k=warm;k<order.size();k++){ int r=order[k]; double a=reqs[r].arrival,f=reqs[r].first,d=reqs[r].done;
        ttft.push_back(f-a); e2e.push_back(d-a); if(reqs[r].gen>1) tpot.push_back((d-f)/(reqs[r].gen-1));
        t_lo=std::min(t_lo,a); t_hi=std::max(t_hi,d); toks_steady+=reqs[r].gen; }
    int nst=order.size()-warm; double win=t_hi-t_lo;
    S.completed=order.size(); S.thru_req=win>0?nst/win:0; S.thru_tok=win>0?toks_steady/win:0;
    S.ttft50=pct(ttft,.5);S.ttft95=pct(ttft,.95);S.ttft99=pct(ttft,.99);
    S.e2e50=pct(e2e,.5);S.e2e95=pct(e2e,.95);S.e2e99=pct(e2e,.99);
    S.tpot50=pct(tpot,.5);S.tpot95=pct(tpot,.95);S.tpot99=pct(tpot,.99);
    S.avgB=stepN?sumB/stepN:0; S.maxq=maxq; S.makespan=simt; return S;
}

// seed 고정 trace 생성(Poisson 도착 λ + 가변 프롬프트 U[8,32] + bimodal gen[short U16-48 70% / long U96-192 30%])
static int g_pmax=32;   // 프롬프트 최대 길이(PMAX env) — 긴 프롬프트서 chunked prefill aggregate 이득 커짐
static std::vector<Req> gen_trace(double lambda,int nreq,unsigned seed){
    std::mt19937 rng(seed); std::exponential_distribution<double> ia(lambda);
    std::uniform_int_distribution<int> pP(8,g_pmax), gS(16,48), gL(96,192); std::uniform_real_distribution<double> u(0,1);
    std::vector<Req> v(nreq); double t=0;
    for(int i=0;i<nreq;i++){ t+=ia(rng); v[i].arrival=t; v[i].P=pP(rng); v[i].gen=(u(rng)<0.7)?gS(rng):gL(rng); }
    return v;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);   // 백그라운드 로그 실시간(block-buffer 방지)
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    uq=argc>3?atoi(argv[3]):2; int Bmax=argc>4?atoi(argv[4]):8; int nreq=argc>5?atoi(argv[5]):200; unsigned seed=argc>6?atoi(argv[6]):42;
    // λ 리스트: argv[7..] 또는 기본
    std::vector<double> lambdas; for(int i=7;i<argc;i++) lambdas.push_back(atof(argv[i]));
    if(lambdas.empty()) lambdas={0.5,1,2,4,8,16,32};
    if(getenv("PMAX")) g_pmax=atoi(getenv("PMAX"));   // 긴 프롬프트 테스트(chunked prefill aggregate 이득 스케일)
    c=load_config_qwen((WD+"/manifest.json").c_str());
    V=c.vocab_size; NKV=c.num_kv_heads; HD=c.head_dim; int rot=c.rotary_dim; maxlen=(g_pmax+224>256)?g_pmax+224:256;  // P+gen(≤192) 커버
    int nh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),Kk=c.gdn_conv_kernel;
    static std::vector<int> ids=ri("token_ids.bin"); IDS=&ids;
    // rope: ext 테이블(gen_rope_long.py, pos>=128 정확) 있으면 maxlen행 로드, 없으면 long(128 타일=pos>128 근사).
    std::vector<float> cl, sl;
    { FILE*e=fopen((GD+"rope_cos_ext.bin").c_str(),"rb");
      if(e){ fseek(e,0,SEEK_END); long er=ftell(e)/4/rot; fclose(e);
             int use=(er>=maxlen)?maxlen:(int)er; cl=rf("rope_cos_ext.bin",(size_t)use*rot); sl=rf("rope_sin_ext.bin",(size_t)use*rot);
             printf("  rope: ext 테이블 %d행(pos>=128 정확)\n",use); }
      else { auto c0=rf("rope_cos_long.bin",(size_t)128*rot), s0=rf("rope_sin_long.bin",(size_t)128*rot);
             cl.resize((size_t)maxlen*rot); sl.resize((size_t)maxlen*rot);
             for(int i=0;i<maxlen;i++)for(int j=0;j<rot;j++){cl[(size_t)i*rot+j]=c0[(size_t)(i%128)*rot+j];sl[(size_t)i*rot+j]=s0[(size_t)(i%128)*rot+j];}
             printf("  rope: long 128행 타일(ext 없음 — pos>128 근사, gen_rope_long.py 권장)\n"); } }
    d_cos=up(cl); d_sin=up(sl);
    w=load_weights_qwen(WD.c_str(),c,1,uq==0?1:0);
    int lgrows = Bmax>512?Bmax:512;  // prefill_batch_slot이 [Sq≤chunk_max,V] 씀 → d_logit 충분히 크게
    CK(cudaMalloc(&d_tok,Bmax*4)); CK(cudaMalloc(&d_logit,(size_t)lgrows*V*4));
    { std::vector<int> tiled(maxlen); for(int i=0;i<maxlen;i++) tiled[i]=ids[i%ids.size()];  // chunked prefill 프롬프트
      CK(cudaMalloc(&g_dtiled,(size_t)maxlen*4)); CK(cudaMemcpy(g_dtiled,tiled.data(),(size_t)maxlen*4,cudaMemcpyHostToDevice)); }
    // admission guard: KV 예산(alloc-time)
    int nfull=0,ngdn=0; for(int l=0;l<c.num_layers;l++) c.layer_is_full[l]?nfull++:ngdn++;
    size_t perseq=(size_t)nfull*maxlen*NKV*HD*2*2 + (size_t)ngdn*((size_t)nh*D*D+(size_t)(Kk-1)*convd)*4;
    size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB); double budget=0.7*(double)freeB;
    printf("[bench_serve] uq=%d(%s) Bmax=%d nreq=%d seed=%u maxlen=%d\n",uq,uq==0?"fp32":"W4A8",Bmax,nreq,seed,maxlen);
    printf("  admission 예산: per-seq %.1fMB × Bmax %d = %.2fGB vs 0.7·free %.2fGB — %s%s\n",
           perseq/1e6,Bmax,(double)Bmax*perseq/1e9,budget/1e9,(double)Bmax*perseq<=budget?"OK":"초과",
           (double)Bmax*perseq>budget?" (Bmax↓ 필요, 중단)":"");
    if((double)Bmax*perseq>budget){ printf("  admission guard: alloc 예산 초과 → 중단(시스템 다운 방지).\n"); return 1; }
    // warm-up 실행(cold 버림): 작은 trace 한 번 돌려 클럭/캐시 예열
    { auto tr=gen_trace(8.0,12,seed); run_continuous(tr,Bmax,true); }

    printf("\n  === 부하 스윕 (warm-up 완료분 20%% 제외, seed 고정 trace) ===\n");
    printf("  gen: short U[16,48] 70%% / long U[96,192] 30%% (평균 ~66), 프롬프트 U[8,32]. c1tok=1토큰prefill c-ch=chunked prefill\n");
    printf("  %-6s %-4s | %8s %8s | %7s %7s %7s | %7s %7s | %5s %5s\n",
           "mode","λ","req/s","tok/s","e2e50","e2e95","e2e99","ttft95","tpot95","avgB","maxQ");
    printf("  %s\n",std::string(96,'-').c_str());
    auto row=[&](const char*nm,double lam,Stats&s){ printf("  %-6s %-4g | %8.3f %8.1f | %7.3f %7.3f %7.3f | %7.3f %7.4f | %5.2f %5.0f\n",
           nm,lam,s.thru_req,s.thru_tok,s.e2e50,s.e2e95,s.e2e99,s.ttft95,s.tpot95,s.avgB,s.maxq); };
    for(double lam:lambdas){
        auto t1=gen_trace(lam,nreq,seed); auto t2=t1; auto t3=t1; // 같은 trace 3부
        Stats C1=run_continuous(t1,Bmax,false); Stats Cc=run_continuous(t2,Bmax,true); Stats T=run_static(t3,Bmax);
        row("c1tok",lam,C1); row("c-ch",lam,Cc); row("stat",lam,T);
    }
    printf("\n  (latency 초 단위. 토큰선택/logit readout 제외=forward 지배. avgB=평균 active 배치=활용 proxy. maxQ=최대 큐깊이=부하.)\n");
    free_weights_qwen(w); free_config_qwen(c); return 0;
}
