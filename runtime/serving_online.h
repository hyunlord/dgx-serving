#ifndef DGX_SERVING_ONLINE_H
#define DGX_SERVING_ONLINE_H
// 모델-중립 온라인 연속 배칭 daemon 루프 (실시간 스트림-구동 greedy). serving_core.h(trace-구동)의 실시간 짝.
//   ★헌법(모델-중립 코어): 이 루프는 OnlineBackend 인터페이스로만 모델 접촉 — 모델 분기/커널/device 버퍼 무지.
//   backend가 device 전송·rope·weight·상태·logit D2H 소유 → 루프는 host 토큰/argmax/스케줄만(gemma4·qwen 공유).
//   신규(bench 대비): 실시간 비동기 주입(stdin reader 스레드+큐) + 실토큰 prefill + greedy 피드백(argmax) + 태그드 스트림 I/O.
//   ★정확성 불변: 온라인 배칭이 각 요청 계산 안 바꿈 — 슬롯 격리(ragged decode 독립). 동시 N == 단독 bit-id(test_serve_online).
//   프로토콜(태그드): stdin "<req_id> <max_tokens> [SMPL <temp> <topp> <repp> <seed>] <id0> ..." / "SHUTDOWN".
//     stdout "<req_id> <tok>" / "<req_id> END"(single-writer). SMPL 없음(기본) = greedy(기존 프로토콜·경로 그대로).
//   ★샘플링은 서빙 옵션(호스트 로직·D2H는 원래 하던 것): 검증/게이트는 greedy 경로(SMPL 미주입)로 불변.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <algorithm>
#include <random>
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
    // 샘플링 파라미터(옵션 — temp<=0 = greedy·기존 경로 그대로). 프로토콜: "<id> <max_tokens> [SMPL t p r seed] <ids...>"
    struct Req { long id; int max_tokens; std::vector<int> ids;
                 float temp=0.f, topp=1.f, repp=1.f; unsigned long long seed=0; };
    struct Slot { long req=-1; int pos=0; int gen=0; int max_tokens=0; int last=0;   // req<0 = free
                  float temp=0.f, topp=1.f, repp=1.f;                                // 샘플러(요청별)
                  std::mt19937_64 rng; std::vector<int> hist; };                     // seed-결정 RNG·생성 이력(반복 페널티)
}

// 모델-중립 온라인 daemon 루프. be=구성된 backend(weight 로드 완료), Bmax=동시 활성 슬롯 상한.
//   반환 시 daemon 종료(stdin EOF/SHUTDOWN). 호출자가 backend 정리.
// STOP_IDS 환경변수 파서(config 주도 — serve_api가 generation_config.eos_token_id를 콤마 목록으로 주입).
//   예: STOP_IDS="1,106,50"(gemma) / "248046,248044"(qwen). 미설정 시 빈 집합(종료 감지 없음 = 기존 동작).
inline std::vector<int> parse_stop_ids_env(){
    std::vector<int> v; const char* e = getenv("STOP_IDS");
    if(e){ std::stringstream ss(e); std::string t; while(std::getline(ss,t,',')) if(!t.empty()) v.push_back(atoi(t.c_str())); }
    return v;
}

//   stop_ids: EOS/종료 토큰 집합(모델 generation_config 권위 — hardcode 금지). 슬롯 argmax가 이 집합이면 종료
//     (그 토큰은 출력 안 함 = EOS는 사용자 텍스트 아님). 종료 사유는 프로토콜 "END <reason>"(stop/length).
//     ★계산 불변: argmax 값은 그대로 판정 — stop_id 만나기 전 토큰들은 stop 없을 때와 bit-identical(종료 조건만 추가).
inline void run_online_daemon(OnlineBackend& be, int Bmax, const std::vector<int>& stop_ids = {}){
    using namespace serving_online_detail;
    const char* log_input_env = getenv("DGX_LOG_INPUT_IDS");
    const bool log_input_ids = log_input_env && std::string(log_input_env) == "1";
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
            std::string nx;                                  // 옵션 SMPL 필드(없으면 기존 프로토콜 그대로 = greedy)
            if(is>>nx){
                if(nx=="SMPL"){ if(!(is>>r.temp>>r.topp>>r.repp>>r.seed)) continue; }
                else r.ids.push_back(atoi(nx.c_str()));
                int id; while(is>>id) r.ids.push_back(id);
            }
            if(r.ids.empty()) continue;
            if(log_input_ids){
                std::ostringstream trace;
                trace << "INPUT_IDS req=" << r.id << " count=" << r.ids.size() << " ids=";
                for(size_t i=0;i<r.ids.size();++i){ if(i) trace << ','; trace << r.ids[i]; }
                fprintf(stderr,"%s\n",trace.str().c_str());
            }
            { std::lock_guard<std::mutex> lk(mtx); queue.push_back(std::move(r)); }
            cv.notify_all();
        }
        shutdown=true; cv.notify_all();                   // stdin EOF
    });

    // 토큰 선택(호스트·슬롯별): temp<=0 → argmax(기존 greedy 경로 바이트 동일 — 게이트 불변).
    //   temp>0 → ★샘플링(서빙 옵션·검증과 분리): 반복 페널티(HF 관례: 생성 이력 토큰 로짓 양수/repp·음수×repp)
    //   → top-64 preselect(호스트 262k 전체 정렬 회피 — 암묵 top-k 64, 문서 명시) → temperature softmax →
    //   top-p 누적 컷 → seed-결정 RNG 추첨. 순수 호스트(D2H는 루프가 원래 하던 것 — 추가 비용 ~0.3ms/tok).
    enum { SK = 64 };
    int   s_idx[SK]; float s_val[SK]; double s_prob[SK]; int s_ord[SK];
    auto pick_row = [&](int b, Slot& s)->int{
        float* p=&hlog[(size_t)b*V];
        if(s.temp<=0.f){ int bi=0; float bv=p[0]; for(int j=1;j<V;j++) if(p[j]>bv){bv=p[j];bi=j;} return bi; }
        // 반복 페널티: ★최근 윈도우 + dedup(전체 이력 복리 누적 repp^N 제거 — 긴 생성 드리프트 원인이었음).
        //   REP_WINDOW(env·기본 128): 최근 W토큰만 대상. REP_WINDOW=0 → 전체 이력이되 dedup(복리 없음).
        //   각 고유 토큰 1회만 v/repp(양수)·v*repp(음수) = HF 관례(복리 아님). 측정 근거: 스칼라 rep↑=축퇴↑ 반증.
        if(s.repp!=1.f){
            static const int REP_W = [](){ const char* e=getenv("REP_WINDOW"); return e ? std::max(0,atoi(e)) : 128; }();
            const int hn=(int)s.hist.size(), lo=(REP_W>0 && hn>REP_W) ? hn-REP_W : 0;
            for(int i=lo;i<hn;++i){ int t=s.hist[i];
                bool seen=false; for(int j=lo;j<i;++j) if(s.hist[j]==t){ seen=true; break; }
                if(seen) continue;                                   // 윈도우 내 1회만(dedup — 복리 방지)
                float v=p[t]; p[t] = v>0.f ? v/s.repp : v*s.repp; }
        }
        const int K = V<SK ? V : SK;
        for(int j=0;j<K;j++){ s_idx[j]=j; s_val[j]=p[j]; }
        int mn=0; for(int j=1;j<K;j++) if(s_val[j]<s_val[mn]) mn=j;
        for(int j=K;j<V;j++) if(p[j]>s_val[mn]){
            s_val[mn]=p[j]; s_idx[mn]=j;
            mn=0; for(int q=1;q<K;q++) if(s_val[q]<s_val[mn]) mn=q;
        }
        float mx=s_val[0]; for(int j=1;j<K;j++) if(s_val[j]>mx) mx=s_val[j];
        double sum=0; for(int j=0;j<K;j++){ s_prob[j]=std::exp(((double)s_val[j]-mx)/s.temp); sum+=s_prob[j]; }
        for(int j=0;j<K;j++) s_ord[j]=j;
        std::sort(s_ord, s_ord+K, [&](int a,int c){ return s_prob[a]>s_prob[c]; });
        double cut=sum*(double)s.topp, cum=0; int keep=0;
        for(;keep<K;){ cum+=s_prob[s_ord[keep]]; keep++; if(cum>=cut) break; }
        std::uniform_real_distribution<double> U(0.0, cum);
        double u=U(s.rng), acc=0; int pick=s_ord[0];
        for(int j=0;j<keep;j++){ acc+=s_prob[s_ord[j]]; if(u<=acc){ pick=s_ord[j]; break; } }
        return s_idx[pick];
    };
    auto n_active = [&](){ int a=0; for(int b=0;b<Bmax;b++) if(slot[b].req>=0)a++; return a; };
    auto is_stop = [&](int t){ for(int s:stop_ids) if(t==s) return true; return false; };  // EOS 집합(config 주도)

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
                slot[b] = Slot{}; Slot& sl=slot[b];
                sl.req=r.id; sl.pos=P; sl.max_tokens=r.max_tokens;
                sl.temp=r.temp; sl.topp=r.topp; sl.repp=r.repp;
                sl.rng.seed(r.seed ? r.seed : 0x9E3779B97F4A7C15ull);   // seed-결정(재현 가능 — serve_api가 기본 랜덤 주입)
                int first = pick_row(0, sl);                       // prefill 로짓 = row 0(greedy면 기존 argmax 그대로)
                sl.last=first;
                if(sl.temp>0.f) sl.hist.push_back(first);
                if(r.max_tokens>0 && is_stop(first)){        // 첫 토큰이 EOS = 빈 응답(즉시 종료·토큰 미출력)
                    printf("%ld END stop\n", r.id); fflush(stdout); slot[b]=Slot();
                } else {
                    if(r.max_tokens>0){ printf("%ld %d\n", r.id, first); fflush(stdout); slot[b].gen=1; }
                    if(slot[b].req>=0 && slot[b].gen>=slot[b].max_tokens){ printf("%ld END length\n", r.id); fflush(stdout); slot[b]=Slot(); }
                }
                lk.lock();
            }
        }
        int A=n_active();
        if(A==0){ if(shutdown){ std::lock_guard<std::mutex> lk(mtx); if(queue.empty()) break; } continue; }

        // 2) decode 스텝: ★active-width(최대 active 슬롯+1까지만 forward — single-stream의 Bmax 전폭 낭비 제거).
        //    현재 admit이 b=0부터 채우므로 active는 앞쪽에 몰림. W=최대 active 인덱스+1.
        //    ★정확성 불변(bit-id): 각 슬롯 KV/state/logit은 독립 → active 슬롯(<W) 결과는 W와 무관하게 동일.
        //    free 슬롯(req<0, W 미만)은 더미 계산이나 출력 무시. single(slot0만)=W1=raw 근접(측정: B8 18→B1 43 t/s).
        //    batch(다 active)면 W=Bmax(무영향). 흩어진 free(중간 완료)는 약간 더미 잔존하나 정확성 무관.
        int W=0; for(int b=0;b<Bmax;b++) if(slot[b].req>=0) W=b+1;
        std::vector<int> toks(W,0), pos(W,0);
        for(int b=0;b<W;b++) if(slot[b].req>=0){ toks[b]=slot[b].last; pos[b]=slot[b].pos; }
        be.decode_step(st, toks.data(), pos.data(), W);
        be.logits_to_host(hlog.data(), W);

        // 3) 슬롯별 토큰 선택(greedy=argmax 그대로·샘플링=옵션) → 스트림 → 피드백 → 완료(슬롯 격리). active는 0..W-1.
        for(int b=0;b<W;b++){ Slot&s=slot[b]; if(s.req<0) continue;
            int next=pick_row(b, s);
            if(is_stop(next)){ printf("%ld END stop\n", s.req); fflush(stdout); s=Slot(); continue; }  // ★EOS: 토큰 미출력·종료(계산 불변 — 값 그대로 판정)
            printf("%ld %d\n", s.req, next); fflush(stdout);
            s.last=next; s.pos++; s.gen++;
            if(s.temp>0.f) s.hist.push_back(next);           // 반복 페널티 이력(샘플링 시에만 — greedy 무비용)
            if(s.gen>=s.max_tokens || s.pos>=maxlen-1){ printf("%ld END length\n", s.req); fflush(stdout); s=Slot(); }
        }
    }
    shutdown=true; cv.notify_all();
    if(reader.joinable()) reader.detach();      // stdin blocking getline — detach(프로세스 종료 시 정리)
    be.free_state(st);
}

#endif // DGX_SERVING_ONLINE_H
