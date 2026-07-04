#ifndef DGX_SERVING_ONLINE_H
#define DGX_SERVING_ONLINE_H
// 모델-중립 온라인 연속 배칭 daemon 루프 (실시간 스트림-구동 greedy). serving_core.h(trace-구동)의 실시간 짝.
//   ★헌법(모델-중립 코어): 이 루프는 OnlineBackend 인터페이스로만 모델 접촉 — 모델 분기/커널/device 버퍼 무지.
//   backend가 device 전송·rope·weight·상태·logit D2H 소유 → 루프는 host 토큰/argmax/스케줄만(gemma4·qwen 공유).
//   신규(bench 대비): 실시간 비동기 주입(stdin reader 스레드+큐) + 실토큰 prefill + greedy 피드백(argmax) + 태그드 스트림 I/O.
//   ★정확성 불변: 온라인 배칭이 각 요청 계산 안 바꿈 — 슬롯 격리(ragged decode 독립). 동시 N == 단독 bit-id(test_serve_online).
//   프로토콜(태그드): stdin "<req_id> <max_tokens> <id0> ..." / "SHUTDOWN". stdout "<req_id> <tok>" / "<req_id> END"(single-writer).
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

// 모델-중립 backend 훅(실서빙용 — bench ServingBackend는 dummy-prompt·logit 미노출이라 별도).
//   backend가 모든 device op 소유. 루프는 host logit(logits_to_host 후 argmax)만 만짐.
struct OnlineBackend {
    virtual ~OnlineBackend() {}
    virtual int   vocab() const = 0;                 // V (logit 폭)
    virtual int   max_len() const = 0;               // KV 캡(슬롯 위치 상한)
    virtual void* alloc_state(int Bmax) = 0;          // batch 상태(Bmax 슬롯) alloc+reset. 반환 opaque.
    virtual void  free_state(void* st) = 0;
    // 슬롯 b 재사용 전 잔재 clear(★qwen=GDN state S/convctx clear 필수 — 없으면 이전 요청 상태 누수. gemma4=GDN 없음→no-op).
    virtual void  reset_slot(void* st, int b) {}
    // 슬롯 b에 실제 프롬프트 P토큰 prefill(KV[0..P-1] 기록) + 마지막 위치 logit을 logits[0:V]에 기록(첫 토큰 argmax 근거).
    virtual void  prefill_ids(void* st, int b, const int* h_ids, int P) = 0;
    // 활성 Bmax 슬롯 1 decode 스텝(ragged — 슬롯별 위치 h_pos[b]). logits[0:Bmax*V] 기록(슬롯별 row).
    virtual void  decode_step(void* st, const int* h_toks, const int* h_pos, int Bmax) = 0;
    // 최근 logit rows개(V씩)를 host로 D2H 복사(backend가 cuda 소유 — 루프는 순수 host). prefill 후 rows=1, decode 후 rows=Bmax.
    virtual void  logits_to_host(float* h_dst, int rows) = 0;
};

namespace serving_online_detail {
    struct Req { long id; int max_tokens; std::vector<int> ids; };
    struct Slot { long req=-1; int pos=0; int gen=0; int max_tokens=0; int last=0; };  // req<0 = free
}

// 모델-중립 온라인 daemon 루프. be=구성된 backend(weight 로드 완료), Bmax=동시 활성 슬롯 상한.
//   반환 시 daemon 종료(stdin EOF/SHUTDOWN). 호출자가 backend 정리.
inline void run_online_daemon(OnlineBackend& be, int Bmax){
    using namespace serving_online_detail;
    const int V = be.vocab(), maxlen = be.max_len();
    void* st = be.alloc_state(Bmax);
    std::vector<float> hlog((size_t)Bmax * V);
    std::vector<Slot> slot(Bmax);

    // 비동기 stdin reader(스레드 안전 큐). 메인 배치 루프가 스텝 경계에서 drain.
    std::deque<Req> queue; std::mutex mtx; std::condition_variable cv; std::atomic<bool> shutdown{false};
    std::thread reader([&]{
        std::string line;
        while(std::getline(std::cin, line)){
            std::istringstream is(line); std::string first;
            if(!(is>>first)) continue;
            if(first=="SHUTDOWN"){ shutdown=true; cv.notify_all(); return; }
            Req r; r.id=atol(first.c_str());
            if(!(is>>r.max_tokens)) continue;
            int id; while(is>>id) r.ids.push_back(id);
            if(r.ids.empty()) continue;
            { std::lock_guard<std::mutex> lk(mtx); queue.push_back(std::move(r)); }
            cv.notify_all();
        }
        shutdown=true; cv.notify_all();                   // stdin EOF
    });

    auto argmax_row = [&](int b)->int{ float*p=&hlog[(size_t)b*V]; int bi=0; float bv=p[0];
        for(int j=1;j<V;j++) if(p[j]>bv){bv=p[j];bi=j;} return bi; };
    auto n_active = [&](){ int a=0; for(int b=0;b<Bmax;b++) if(slot[b].req>=0)a++; return a; };

    while(true){
        // 1) admit: 대기 큐 → free slot(스텝 경계). prefill(슬롯 KV) + 첫 토큰(prefill 마지막 logit argmax).
        {
            std::unique_lock<std::mutex> lk(mtx);
            if(queue.empty() && n_active()==0){
                if(shutdown) break;
                cv.wait(lk, [&]{ return !queue.empty() || shutdown.load(); });  // 유휴: 요청/종료 대기
                if(queue.empty() && shutdown) break;
            }
            for(int b=0;b<Bmax && !queue.empty();b++){
                if(slot[b].req>=0) continue;
                Req r = std::move(queue.front()); queue.pop_front();
                lk.unlock();
                int P=(int)r.ids.size(); if(P>=maxlen){P=maxlen-1; r.ids.resize(P);}
                int budget=maxlen-P; if(r.max_tokens>budget) r.max_tokens=budget;
                be.reset_slot(st, b);                              // 슬롯 재사용 잔재 clear(qwen GDN state; gemma4 no-op)
                be.prefill_ids(st, b, r.ids.data(), P);            // 슬롯 b KV[0..P-1] + logits[0:V]=마지막
                be.logits_to_host(hlog.data(), 1);
                int first=0; { float*p=hlog.data(); float bv=p[0]; for(int j=1;j<V;j++) if(p[j]>bv){bv=p[j];first=j;} }
                slot[b] = Slot{r.id, P, 0, r.max_tokens, first};
                if(r.max_tokens>0){ printf("%ld %d\n", r.id, first); fflush(stdout); slot[b].gen=1; }
                if(slot[b].gen>=slot[b].max_tokens){ printf("%ld END\n", r.id); fflush(stdout); slot[b]=Slot(); }
                lk.lock();
            }
        }
        int A=n_active();
        if(A==0){ if(shutdown){ std::lock_guard<std::mutex> lk(mtx); if(queue.empty()) break; } continue; }

        // 2) decode 스텝: ragged(Bmax 전폭, free 슬롯 더미·출력 무시). 슬롯별 위치 pos[b]=slot[b].pos.
        std::vector<int> toks(Bmax,0), pos(Bmax,0);
        for(int b=0;b<Bmax;b++) if(slot[b].req>=0){ toks[b]=slot[b].last; pos[b]=slot[b].pos; }
        be.decode_step(st, toks.data(), pos.data(), Bmax);
        be.logits_to_host(hlog.data(), Bmax);

        // 3) 슬롯별 argmax → 스트림 → greedy 피드백 → 완료(슬롯 격리: 각 슬롯 자기 logit row만).
        for(int b=0;b<Bmax;b++){ Slot&s=slot[b]; if(s.req<0) continue;
            int next=argmax_row(b);
            printf("%ld %d\n", s.req, next); fflush(stdout);
            s.last=next; s.pos++; s.gen++;
            if(s.gen>=s.max_tokens || s.pos>=maxlen-1){ printf("%ld END\n", s.req); fflush(stdout); s=Slot(); }
        }
    }
    shutdown=true; cv.notify_all();
    if(reader.joinable()) reader.detach();      // stdin blocking getline — detach(프로세스 종료 시 정리)
    be.free_state(st);
}

#endif // DGX_SERVING_ONLINE_H
