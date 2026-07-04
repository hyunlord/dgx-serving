// M5 온라인 연속 배칭 daemon (Gemma4 plain, greedy) — bench trace-구동 → 실시간 스트림-구동("인터랙티브 갭").
//   single-seq daemon(serve_daemon_gemma4)의 배치판: G4BatchState(Bmax 슬롯)로 동시 요청 연속 배칭.
//   ★재사용(검증됨): gemma4_prefill_batch_slot(슬롯 KV prefill, W4A8 bit-id 게이트 통과)·gemma4_decode_batch(ragged,
//     슬롯별 위치 — test_gemma4_ragged서 ragged==single bit-id 증명). 신규 = 실시간 비동기 주입 + 태그드 스트림 I/O.
//   ★정확성 불변: 온라인 배칭이 각 요청 계산 안 바꿈 — 슬롯 격리(ragged decode 슬롯 독립) → 동시 N요청 각 == 단독 bit-id.
//   프로토콜(태그드): stdin "<req_id> <max_tokens> <id0> <id1> ..."(요청, 비동기 도착) / "SHUTDOWN".
//     stdout "<req_id> <token>\n"(생성 토큰, 슬롯 인터리브·flush) / "<req_id> END\n"(요청 완료). single-writer(메인 루프).
//   Phase: gemma4 plain·greedy만·dummy rope. 성능 주장 없음(정확 온라인 동작 확립). 사용법: serve_daemon_gemma4_online <w> [uq=2] [maxlen=384] [Bmax=8]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../kernels/rope.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"[online] CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

struct Req { long id; int max_tokens; std::vector<int> ids; };
struct Slot { long req=-1; int pos=0; int gen=0; int max_tokens=0; int last=0; };  // req<0 = free

// 비동기 stdin reader → 큐(스레드 안전). 메인 배치 루프가 스텝 경계에서 drain.
static std::deque<Req> g_queue;
static std::mutex g_mtx;
static std::condition_variable g_cv;
static std::atomic<bool> g_shutdown{false};

static void reader_thread(){
    std::string line;
    while(std::getline(std::cin, line)){
        std::istringstream is(line);
        std::string first; if(!(is>>first)) continue;
        if(first=="SHUTDOWN"){ g_shutdown=true; g_cv.notify_all(); return; }
        Req r; r.id=atol(first.c_str());
        if(!(is>>r.max_tokens)) continue;
        int id; while(is>>id) r.ids.push_back(id);
        if(r.ids.empty()) continue;
        { std::lock_guard<std::mutex> lk(g_mtx); g_queue.push_back(std::move(r)); }
        g_cv.notify_all();
    }
    g_shutdown=true; g_cv.notify_all();      // stdin EOF = 종료
}

int main(int argc,char**argv){
    std::string WD = argc>1?argv[1]:"artifacts/gemma4-31b/weights";
    int uq = argc>2?atoi(argv[2]):2, maxlen = argc>3?atoi(argv[3]):384, Bmax = argc>4?atoi(argv[4]):8;

    ModelConfig c = load_config_gemma4((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, hd_s=c.head_dim, hd_f=c.global_head_dim?c.global_head_dim:c.head_dim;
    float *d_cf,*d_sf,*d_cs,*d_ss;
    CK(cudaMalloc(&d_cf,(size_t)maxlen*hd_f*4)); CK(cudaMalloc(&d_sf,(size_t)maxlen*hd_f*4));
    CK(cudaMalloc(&d_cs,(size_t)maxlen*hd_s*4)); CK(cudaMalloc(&d_ss,(size_t)maxlen*hd_s*4));
    launch_gen_rope(d_cf,d_sf,maxlen,hd_f,1e4f); launch_gen_rope(d_cs,d_ss,maxlen,hd_s,1e4f); CK(cudaDeviceSynchronize());
    G4Weights w = load_weights_gemma4(WD.c_str(), c, uq!=0?1:0, uq==0?1:0);
    G4BatchState st = alloc_g4_batch_state(c, Bmax, maxlen);
    int *d_tok,*d_pos,*d_pf; float *d_logit;
    CK(cudaMalloc(&d_tok,Bmax*4)); CK(cudaMalloc(&d_pos,Bmax*4)); CK(cudaMalloc(&d_pf,(size_t)maxlen*4));
    CK(cudaMalloc(&d_logit,(size_t)Bmax*V*4));
    std::vector<float> hlog((size_t)Bmax*V);
    auto argmax_row = [&](int b)->int{ float*p=&hlog[(size_t)b*V]; int bi=0; float bv=p[0];
        for(int j=1;j<V;j++) if(p[j]>bv){bv=p[j];bi=j;} return bi; };

    std::vector<Slot> slot(Bmax);
    std::thread reader(reader_thread);
    fprintf(stderr,"[online] ready: gemma4 uq=%d maxlen=%d Bmax=%d V=%d (연속 배칭 greedy, 태그드 ID-in/out)\n", uq,maxlen,Bmax,V); fflush(stderr);

    auto n_active = [&](){ int a=0; for(int b=0;b<Bmax;b++) if(slot[b].req>=0)a++; return a; };
    while(true){
        // 1) admit: 대기 큐 → free slot(스텝 경계). prefill(슬롯 KV) + 첫 토큰(prefill 마지막 logit argmax).
        {
            std::unique_lock<std::mutex> lk(g_mtx);
            if(g_queue.empty() && n_active()==0){
                if(g_shutdown) break;
                g_cv.wait(lk, []{ return !g_queue.empty() || g_shutdown.load(); });  // 유휴: 요청/종료 대기
                if(g_queue.empty() && g_shutdown) break;
            }
            for(int b=0;b<Bmax && !g_queue.empty();b++){
                if(slot[b].req>=0) continue;
                Req r = std::move(g_queue.front()); g_queue.pop_front();
                lk.unlock();
                int P=(int)r.ids.size(); if(P>=maxlen){P=maxlen-1; r.ids.resize(P);}
                int budget=maxlen-P; if(r.max_tokens>budget) r.max_tokens=budget;
                CK(cudaMemcpy(d_pf, r.ids.data(), (size_t)P*4, cudaMemcpyHostToDevice));
                gemma4_prefill_batch_slot(c,w,st,b,d_pf,P,d_cf,d_sf,d_cs,d_ss,d_logit,uq);  // 슬롯 b KV[0..P-1], d_logit[0:V]=마지막 logit
                CK(cudaMemcpy(hlog.data(), d_logit, (size_t)V*4, cudaMemcpyDeviceToHost));
                int first=0; { float*p=hlog.data(); float bv=p[0]; for(int j=1;j<V;j++) if(p[j]>bv){bv=p[j];first=j;} }
                slot[b] = Slot{r.id, P, 0, r.max_tokens, first};
                if(r.max_tokens>0){ printf("%ld %d\n", r.id, first); fflush(stdout); slot[b].gen=1; }
                if(slot[b].gen>=slot[b].max_tokens){ printf("%ld END\n", r.id); fflush(stdout); slot[b]=Slot(); }
                lk.lock();
            }
        }
        int A=n_active();
        if(A==0){ if(g_shutdown){ std::lock_guard<std::mutex> lk(g_mtx); if(g_queue.empty()) break; } continue; }

        // 2) decode 스텝: ragged decode_batch(Bmax 전폭, free 슬롯은 더미·출력 무시). 슬롯별 위치 pos[b]=slot[b].pos.
        std::vector<int> toks(Bmax,0), pos(Bmax,0);
        for(int b=0;b<Bmax;b++) if(slot[b].req>=0){ toks[b]=slot[b].last; pos[b]=slot[b].pos; }
        CK(cudaMemcpy(d_tok,toks.data(),Bmax*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos,pos.data(),Bmax*4,cudaMemcpyHostToDevice));
        gemma4_decode_batch(c,w,st,d_tok,d_cf,d_sf,d_cs,d_ss,d_logit,0,uq,d_pos,pos.data());  // ragged(슬롯 독립)
        CK(cudaMemcpy(hlog.data(),d_logit,(size_t)Bmax*V*4,cudaMemcpyDeviceToHost));

        // 3) 슬롯별 argmax → 스트림 → greedy 피드백 → 완료 처리(슬롯 격리: 각 슬롯 자기 logit row만)
        for(int b=0;b<Bmax;b++){ Slot&s=slot[b]; if(s.req<0) continue;
            int next=argmax_row(b);
            printf("%ld %d\n", s.req, next); fflush(stdout);
            s.last=next; s.pos++; s.gen++;
            if(s.gen>=s.max_tokens || s.pos>=maxlen-1){ printf("%ld END\n", s.req); fflush(stdout); s=Slot(); }
        }
    }
    g_shutdown=true; g_cv.notify_all();
    if(reader.joinable()) reader.detach();   // stdin blocking getline — detach(프로세스 종료 시 정리)
    cudaFree(d_tok);cudaFree(d_pos);cudaFree(d_pf);cudaFree(d_logit);cudaFree(d_cf);cudaFree(d_sf);cudaFree(d_cs);cudaFree(d_ss);
    free_g4_batch_state(st); free_weights_gemma4(w); free_config_gemma4(c);
    return 0;
}
