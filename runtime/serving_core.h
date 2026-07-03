#ifndef DGX_SERVING_CORE_H
#define DGX_SERVING_CORE_H
// 모델-무지 서빙 코어 (M4 라운드2). bench_serve.cu의 스케줄링(Poisson trace·admit/slot 재사용·continuous/static·
// warm-up·백분위 통계·admission 예산 가드)을 ServingBackend 훅 위로 이동 — 로직 변경 없음(이동만).
//   코어는 ServingBackend* 와 host 토큰 스트림(ids)만 안다. 모델 타입/커널/device 버퍼 무지(③ 회피 실증).
//   토큰선택/logit readout 제외(forward 지배, 직교) — gen 길이는 trace로 사전결정. sim clock=스텝 wall 누적.
#include <cstdio>
#include <cmath>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include "serving_backend.h"

struct Req{ int P, gen; double arrival; double admit=-1, first=-1, done=-1; };
struct Stats{ double thru_req=0,thru_tok=0; double ttft50=0,ttft95=0,ttft99=0,e2e50=0,e2e95=0,e2e99=0,tpot50=0,tpot95=0,tpot99=0;
              double avgB=0,maxq=0,makespan=0; int completed=0; };

static inline double sc_pct(std::vector<double>&v,double p){ if(v.empty())return 0; std::sort(v.begin(),v.end());
    size_t i=(size_t)ceil(p*v.size()); if(i>0)i--; if(i>=v.size())i=v.size()-1; return v[i]; }

// warm-up 20% 제외 + 백분위 집계(continuous/static 공통) — 모델 무지 순수 함수.
static inline Stats sc_finalize(std::vector<Req>&reqs,int R,long stepN,double sumB,double maxq,double simt){
    std::vector<int> order; for(int r=0;r<R;r++) if(reqs[r].done>=0) order.push_back(r);
    std::sort(order.begin(),order.end(),[&](int a,int b){return reqs[a].done<reqs[b].done;});
    int warm=(int)(order.size()*0.2); Stats S; std::vector<double> ttft,e2e,tpot;
    double t_lo=1e18,t_hi=0; long toks_steady=0;
    for(size_t k=warm;k<order.size();k++){ int r=order[k]; double a=reqs[r].arrival,f=reqs[r].first,d=reqs[r].done;
        ttft.push_back(f-a); e2e.push_back(d-a); if(reqs[r].gen>1) tpot.push_back((d-f)/(reqs[r].gen-1));
        t_lo=std::min(t_lo,a); t_hi=std::max(t_hi,d); toks_steady+=reqs[r].gen; }
    int nst=order.size()-warm; double win=t_hi-t_lo;
    S.completed=order.size(); S.thru_req=win>0?nst/win:0; S.thru_tok=win>0?toks_steady/win:0;
    S.ttft50=sc_pct(ttft,.5);S.ttft95=sc_pct(ttft,.95);S.ttft99=sc_pct(ttft,.99);
    S.e2e50=sc_pct(e2e,.5);S.e2e95=sc_pct(e2e,.95);S.e2e99=sc_pct(e2e,.99);
    S.tpot50=sc_pct(tpot,.5);S.tpot95=sc_pct(tpot,.95);S.tpot99=sc_pct(tpot,.99);
    S.avgB=stepN?sumB/stepN:0; S.maxq=maxq; S.makespan=simt; return S;
}

// ── continuous scheduler: 요청 도착 시 slot 나면 즉시 admit(재사용), 매 스텝 active만 forward ──
//   chunked=false: 프롬프트 1토큰/step ingest. chunked=true: admit 시 backend.prefill_slot로 병렬 prefill(1스텝).
static inline Stats sc_run_continuous(ServingBackend&be, void* st, const std::vector<int>&ids,
                                      std::vector<Req> reqs,int Bmax,int maxlen,bool chunked){
    int R=reqs.size();
    struct Slot{int req=-1,pos=0,P=0,gen=0,g=0;}; std::vector<Slot> slot(Bmax);
    std::deque<int> queue; int nextArr=0, completed=0; double simt=0; long stepN=0; double sumB=0, maxq=0;
    while(completed<R){
        while(nextArr<R && reqs[nextArr].arrival<=simt) queue.push_back(nextArr++);
        for(int b=0;b<Bmax;b++) if(slot[b].req<0 && !queue.empty()){
            int r=queue.front(); if(reqs[r].P+reqs[r].gen>maxlen){queue.pop_front();continue;}
            queue.pop_front(); slot[b]=Slot(); slot[b].req=r; slot[b].P=reqs[r].P; slot[b].gen=reqs[r].gen;
            be.reset_slot(st,b); reqs[r].admit=simt;
            if(chunked){  // 병렬 prefill(1스텝, 측정). 후 pos=P부터 decode. 첫 토큰=prefill 로짓(TTFT).
                simt+=be.prefill_slot(st,b,reqs[r].P);
                slot[b].pos=reqs[r].P; slot[b].g=1; reqs[r].first=simt;
                if(slot[b].g>=slot[b].gen){ reqs[r].done=simt; completed++; slot[b]=Slot(); } } }
        int active=0; for(int b=0;b<Bmax;b++) if(slot[b].req>=0) active++;
        if(active==0){ if(nextArr<R){ simt=reqs[nextArr].arrival; continue; } else break; }
        if((double)queue.size()>maxq) maxq=queue.size();
        std::vector<int> toks(Bmax),pos(Bmax);
        for(int b=0;b<Bmax;b++){ Slot&s=slot[b];
            if(s.req<0){toks[b]=0;pos[b]=0;} else {toks[b]=ids[s.pos%ids.size()];pos[b]=s.pos;} }
        double dt=be.decode_step(st,toks.data(),pos.data(),Bmax); simt+=dt; sumB+=active; stepN++;
        for(int b=0;b<Bmax;b++){ Slot&s=slot[b]; if(s.req<0)continue;
            if(s.pos<s.P-1){ s.pos++; }
            else { if(s.g==0) reqs[s.req].first=simt;
                   s.g++; s.pos++;
                   if(s.g>=s.gen){ reqs[s.req].done=simt; completed++; slot[b]=Slot(); } } }
    }
    return sc_finalize(reqs,R,stepN,sumB,maxq,simt);
}

// ── static scheduler: 배치 채워(≤Bmax) launch → 그 배치 전원 완료까지 slot 점유(head-of-line) ──
static inline Stats sc_run_static(ServingBackend&be, void* st, const std::vector<int>&ids,
                                  std::vector<Req> reqs,int Bmax,int maxlen){
    int R=reqs.size();
    std::deque<int> queue; int nextArr=0, completed=0; double simt=0; long stepN=0; double sumB=0,maxq=0;
    while(completed<R){
        while(nextArr<R && reqs[nextArr].arrival<=simt) queue.push_back(nextArr++);
        if(queue.empty()){ if(nextArr<R){simt=reqs[nextArr].arrival;continue;} else break; }
        if((int)queue.size()<Bmax && nextArr<R && reqs[nextArr].arrival> simt){ simt=reqs[nextArr].arrival; continue; }
        struct Slot{int req,pos=0,P,gen,g=0;}; std::vector<Slot> slot;
        for(int b=0;b<Bmax && !queue.empty();b++){ int r=queue.front();
            if(reqs[r].P+reqs[r].gen>maxlen){queue.pop_front();b--;continue;}
            queue.pop_front(); Slot s; s.req=r;s.P=reqs[r].P;s.gen=reqs[r].gen; slot.push_back(s); reqs[r].admit=simt; }
        if(slot.empty()) continue;
        int Bb=slot.size(); be.reset_slot(st,-1);   // static: 매 배치 전체 reset(b=-1=전체)
        if((double)queue.size()>maxq) maxq=queue.size();
        while(true){ int alive=0; for(auto&s:slot) if(s.g<s.gen) alive++; if(alive==0) break;
            std::vector<int> toks(Bb),pos(Bb);
            for(int b=0;b<Bb;b++){ Slot&s=slot[b]; toks[b]=ids[s.pos%ids.size()]; pos[b]=s.pos; }
            double dt=be.decode_step(st,toks.data(),pos.data(),Bb); simt+=dt; sumB+=alive; stepN++;
            for(int b=0;b<Bb;b++){ Slot&s=slot[b]; if(s.g>=s.gen)continue;
                if(s.pos<s.P-1) s.pos++;
                else { if(s.g==0) reqs[s.req].first=simt; s.g++; s.pos++;
                       if(s.g>=s.gen){ reqs[s.req].done=simt; completed++; } } } }
    }
    return sc_finalize(reqs,R,stepN,sumB,maxq,simt);
}

// seed 고정 trace(Poisson λ + 프롬프트 U[8,pmax] + bimodal gen[short U16-48 70% / long U96-192 30%]) — 모델 무지.
static inline std::vector<Req> sc_gen_trace(double lambda,int nreq,unsigned seed,int pmax){
    std::mt19937 rng(seed); std::exponential_distribution<double> ia(lambda);
    std::uniform_int_distribution<int> pP(8,pmax), gS(16,48), gL(96,192); std::uniform_real_distribution<double> u(0,1);
    std::vector<Req> v(nreq); double t=0;
    for(int i=0;i<nreq;i++){ t+=ia(rng); v[i].arrival=t; v[i].P=pP(rng); v[i].gen=(u(rng)<0.7)?gS(rng):gL(rng); }
    return v;
}

#endif // DGX_SERVING_CORE_H
