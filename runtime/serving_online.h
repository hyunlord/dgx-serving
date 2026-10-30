#ifndef DGX_SERVING_ONLINE_H
#define DGX_SERVING_ONLINE_H
// 모델-중립 온라인 연속 배칭 daemon 루프 (실시간 스트림-구동 greedy). serving_core.h(trace-구동)의 실시간 짝.
//   ★헌법(모델-중립 코어): 이 루프는 OnlineBackend 인터페이스로만 모델 접촉 — 모델 분기/커널/device 버퍼 무지.
//   backend가 device 전송·rope·weight·상태·logit D2H 소유 → 루프는 host 토큰/argmax/스케줄만(gemma4·qwen 공유).
//   신규(bench 대비): 실시간 비동기 주입(stdin reader 스레드+큐) + 실토큰 prefill + greedy 피드백(argmax) + 태그드 스트림 I/O.
//   ★정확성 불변: 온라인 배칭이 각 요청 계산 안 바꿈 — 슬롯 격리(ragged decode 독립). 동시 N == 단독 bit-id(test_serve_online).
//   프로토콜(태그드): stdin "<req_id> <max_tokens> [SMPL <temp> <topp> <repp> <seed>] <id0> ..." / "CANCEL <req_id>" / "SHUTDOWN".
//     stdout "<req_id> <tok>" / "<req_id> END <stop|length|cancel>"(single-writer). SMPL 없음(기본) = greedy(기존 프로토콜·경로 그대로).
//     CANCEL: 다음 스텝 경계에서 대기/활성 요청을 끝낸다(END cancel). 이미 끝난 req_id는 무시.
//   단일 스트림 fast-path(G4_SERVE_FASTPATH, 기본 1): 활성 요청 1개·greedy·대기 큐 빔이면 backend의
//     generate_single로 S=1 생성 세션을 돌리고, 새 요청/CANCEL이 오면 그 스텝 뒤 배치 루프로 돌아온다.
//     ★backend가 보장: fast-path 토큰·KV == 배치 W=1 경로(R14 게이트 2b·2c) → 전환 시점과 무관한 출력.
//   ★샘플링은 서빙 옵션(호스트 로직·D2H는 원래 하던 것): 검증/게이트는 greedy 경로(SMPL 미주입)로 불변.
#include "runtime/knobs.h"
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
#include <functional>

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
    // 선택: 슬롯 b 단일 스트림 greedy 생성(입력 first, 위치 pos부터 최대 max_steps). 채택된 스텝마다
    //   on_step(token, opaque) 호출, 0이 아니면 그 스텝 뒤 종료(미리 제출된 뒤 스텝은 backend가 drain).
    //   반환 = 채택 스텝 수. 채택 스텝의 토큰·슬롯 KV는 decode_step(W=1) 경로와 비트 동일해야 한다.
    struct SingleStats { unsigned long long executed=0, accepted=0, speculative=0; };
    virtual bool  single_stream() const { return false; }
    virtual int   generate_single(void* st, int b, int pos, int first, int max_steps,
                                  int (*on_step)(int, void*), void* opaque, SingleStats* stats) { return 0; }
    // 최근 logit rows개(V씩)를 host로 D2H 복사(backend가 cuda 소유 — 루프는 순수 host). prefill 후 rows=1, decode 후 rows=Bmax.
    virtual void  logits_to_host(float* h_dst, int rows) = 0;
    // 선택(R17): 슬롯 b 단일 greedy 요청의 speculative 사이클. KV 길이 pos, 아직 KV에 없는 최근 토큰 cur에서
    //   drafter K토큰 → verify(K+1행) → 수락 접두사 m + 교정 1 = tokens[0..n-1](n=m+1) 확정. KV는 pos..pos+m 행이
    //   확정(나머지 verify 행은 다음 사이클이 덮음). K=0 또는 drafter 상태(h)가 없으면 verify 1행 = plain 1스텝.
    //   확정 토큰·KV는 decode_step(W=1) 경로와 비트 동일해야 한다(spec==greedy). spec_invalidate(b)는 슬롯 b가
    //   다른 경로로 진행했을 때 drafter 상태를 버린다.
    struct SpecCycle { int tokens[16]; int n=0, drafted=0, accepted=0, k=0; };
    virtual bool  spec_ready() const { return false; }
    virtual void  spec_invalidate(int b) {}
    virtual void  spec_cycle(void* st, int b, int pos, int cur, int K, SpecCycle* out) {}
};

namespace serving_online_detail {
    // 샘플링 파라미터(옵션 — temp<=0 = greedy·기존 경로 그대로). 프로토콜: "<id> <max_tokens> [SMPL t p r seed] <ids...>"
    struct Req { long id; int max_tokens; std::vector<int> ids;
                 float temp=0.f, topp=1.f, repp=1.f; unsigned long long seed=0; };
    struct Slot { long req=-1; int pos=0; int gen=0; int max_tokens=0; int last=0;   // req<0 = free
                  float temp=0.f, topp=1.f, repp=1.f;                                // 샘플러(요청별)
                  std::mt19937_64 rng; std::vector<int> hist;                        // seed-결정 RNG·생성 이력(반복 페널티)
                  int fast_steps=0, batch_steps=0, fast_entries=0;                   // 경로 카운터(로그)
                  unsigned long long fast_tail=0;
                  int spec_cycles=0, spec_entries=0, spec_tokens=0; long spec_drafted=0, spec_accepted=0;
};
}

// 모델-중립 온라인 daemon 루프. be=구성된 backend(weight 로드 완료), Bmax=동시 활성 슬롯 상한.
//   반환 시 daemon 종료(stdin EOF/SHUTDOWN). 호출자가 backend 정리.
// STOP_IDS 환경변수 파서(config 주도 — serve_api가 generation_config.eos_token_id를 콤마 목록으로 주입).
//   예: STOP_IDS="1,106,50"(gemma) / "248046,248044"(qwen). 미설정 시 빈 집합(종료 감지 없음 = 기존 동작).
inline std::vector<int> parse_stop_ids_env(){
    std::vector<int> v; const char* e = knob_raw("STOP_IDS");
    if(e){ std::stringstream ss(e); std::string t; while(std::getline(ss,t,',')) if(!t.empty()) v.push_back(atoi(t.c_str())); }
    return v;
}

//   stop_ids: EOS/종료 토큰 집합(모델 generation_config 권위 — hardcode 금지). 슬롯 argmax가 이 집합이면 종료
//     (그 토큰은 출력 안 함 = EOS는 사용자 텍스트 아님). 종료 사유는 프로토콜 "END <reason>"(stop/length).
//     ★계산 불변: argmax 값은 그대로 판정 — stop_id 만나기 전 토큰들은 stop 없을 때와 bit-identical(종료 조건만 추가).
inline void run_online_daemon(OnlineBackend& be, int Bmax, const std::vector<int>& stop_ids = {}){
    using namespace serving_online_detail;
    const char* log_input_env = knob_raw("DGX_LOG_INPUT_IDS");
    const bool log_input_ids = log_input_env && std::string(log_input_env) == "1";
    const int V = be.vocab(), maxlen = be.max_len();
    void* st = be.alloc_state(Bmax);
    std::vector<float> hlog((size_t)Bmax * V);
    std::vector<Slot> slot(Bmax);

    // 비동기 stdin reader(스레드 안전 큐). 메인 배치 루프가 스텝 경계에서 drain.
    std::deque<Req> queue; std::mutex mtx; std::condition_variable cv; std::atomic<bool> shutdown{false};
    std::vector<long> cancels;                                                      // mtx 보호(reader→루프)
    const char* fast_env = knob_raw("G4_SERVE_FASTPATH");
    const bool fastpath = be.single_stream() && !(fast_env && std::string(fast_env) == "0");
    unsigned long long n_fast_req=0, n_batch_req=0, n_switch=0, n_tail=0;
    // R17 spec 경로(drafter 적재 backend만, 기본 ON — G4_SPEC_SERVE=0이면 끔): 단일 greedy 요청을 speculative 사이클로.
    const char* spec_env = knob_raw("G4_SPEC_SERVE");
    const bool spec_serve = be.spec_ready() && !(spec_env && std::string(spec_env) == "0");   // default on with a drafter
    const char* speck_env = knob_raw("G4_SPEC_K");
    const int spec_k = speck_env ? atoi(speck_env) : 3;
    unsigned long long n_spec_switch=0;
    if(spec_serve) fprintf(stderr,"[online] spec serving: K=%d (single greedy requests)\n",spec_k);
    std::thread reader([&]{
        std::string line;
        while(std::getline(std::cin, line)){
            std::istringstream is(line); std::string first;
            if(!(is>>first)) continue;
            if(first=="SHUTDOWN"){ shutdown=true; cv.notify_all(); return; }
            if(first=="CANCEL"){ long id; if(is>>id){ std::lock_guard<std::mutex> lk(mtx); cancels.push_back(id); } cv.notify_all(); continue; }
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
            static const int REP_W = [](){ const char* e=knob_raw("REP_WINDOW"); return e ? std::max(0,atoi(e)) : 128; }();
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
    // 요청 종료(프로토콜 END 그대로) + 경로 카운터 로그(stderr — 프로토콜 무관).
    auto finish = [&](Slot& s, const char* reason){
        printf("%ld END %s\n", s.req, reason); fflush(stdout);
        if(s.fast_steps) ++n_fast_req; else ++n_batch_req;
        fprintf(stderr,"[online] PATH req=%ld reason=%s fast_steps=%d batch_steps=%d fast_entries=%d fast_tail=%llu sampling=%d totals_fast_req=%llu totals_batch_req=%llu totals_switch=%llu totals_tail=%llu\n",
                s.req,reason,s.fast_steps,s.batch_steps,s.fast_entries,s.fast_tail,s.temp>0.f?1:0,n_fast_req,n_batch_req,n_switch,n_tail);
        if(s.spec_cycles)
            fprintf(stderr,"[online] SPEC req=%ld cycles=%d entries=%d tokens=%d drafted=%ld accepted=%ld discarded=%ld totals_spec_switch=%llu\n",
                    s.req,s.spec_cycles,s.spec_entries,s.spec_tokens,s.spec_drafted,s.spec_accepted,s.spec_drafted-s.spec_accepted,n_spec_switch);
        s=Slot();
    };
    // CANCEL 적용(스텝 경계): 대기 큐에서 제거하거나 활성 슬롯 종료. 알 수 없는 id는 무시.
    auto apply_cancels = [&](){
        std::vector<long> ids;
        { std::lock_guard<std::mutex> lk(mtx); ids.swap(cancels);
          for(long id:ids) for(auto it=queue.begin(); it!=queue.end(); ++it) if(it->id==id){
              printf("%ld END cancel\n", id); fflush(stdout);
              fprintf(stderr,"[online] PATH req=%ld reason=cancel queued=1\n", id); queue.erase(it); break; } }
        for(long id:ids) for(int b=0;b<Bmax;b++) if(slot[b].req==id) finish(slot[b],"cancel");
    };
    auto pending = [&](long req){ std::lock_guard<std::mutex> lk(mtx);
        if(!queue.empty()) return true; for(long id:cancels) if(id==req) return true; return false; };
    struct FastCtx { std::function<int(int)> fn; static int call(int t, void* o){ return ((FastCtx*)o)->fn(t); } };

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
                be.spec_invalidate(b);                             // drafter 상태(h)는 첫 spec 사이클이 만든다
                be.logits_to_host(hlog.data(), 1);
                slot[b] = Slot{}; Slot& sl=slot[b];
                sl.req=r.id; sl.pos=P; sl.max_tokens=r.max_tokens;
                sl.temp=r.temp; sl.topp=r.topp; sl.repp=r.repp;
                sl.rng.seed(r.seed ? r.seed : 0x9E3779B97F4A7C15ull);   // seed-결정(재현 가능 — serve_api가 기본 랜덤 주입)
                int first = pick_row(0, sl);                       // prefill 로짓 = row 0(greedy면 기존 argmax 그대로)
                sl.last=first;
                if(sl.temp>0.f) sl.hist.push_back(first);
                if(r.max_tokens>0 && is_stop(first)){        // 첫 토큰이 EOS = 빈 응답(즉시 종료·토큰 미출력)
                    finish(slot[b],"stop");
                } else {
                    if(r.max_tokens>0){ printf("%ld %d\n", r.id, first); fflush(stdout); slot[b].gen=1; }
                    if(slot[b].req>=0 && slot[b].gen>=slot[b].max_tokens) finish(slot[b],"length");
                }
                lk.lock();
            }
        }
        apply_cancels();
        int A=n_active();
        if(A==0){ if(shutdown){ std::lock_guard<std::mutex> lk(mtx); if(queue.empty()) break; } continue; }

        // 2s) R17 spec 경로: 활성 1개·greedy·대기 없음·G4_SPEC_SERVE=1. 사이클마다 m+1토큰 확정, 한 토큰씩
        //     기존 규칙(EOS 미출력 종료·max_tokens·maxlen)으로 방출하고 경계를 넘은 토큰은 버린다. 새 요청/CANCEL은
        //     사이클 경계에서 감지해 배치 루프로 복귀(슬롯 pos/last가 이어받음 — 미수락 draft KV 행은 이후 덮임).
        if(spec_serve && A==1){
            int b=0; while(slot[b].req<0) ++b;
            Slot& s=slot[b];
            if(s.temp<=0.f && s.gen<s.max_tokens && s.pos<maxlen-1 && !pending(s.req)){
                ++s.spec_entries;
                const long req=s.req; bool done=false;
                while(!done){
                    const int kroom=std::max(0,(maxlen-2)-s.pos);          // verify 행 pos..pos+K < maxlen-1
                    OnlineBackend::SpecCycle cy;
                    be.spec_cycle(st, b, s.pos, s.last, std::min(spec_k,kroom), &cy);
                    ++s.spec_cycles; s.spec_drafted+=cy.drafted; s.spec_accepted+=cy.accepted;
                    for(int i=0;i<cy.n;i++){
                        const int next=cy.tokens[i];
                        if(is_stop(next)){ finish(s,"stop"); done=true; break; }
                        printf("%ld %d\n", s.req, next); fflush(stdout);
                        s.last=next; s.pos++; s.gen++; s.spec_tokens++;
                        if(s.gen>=s.max_tokens || s.pos>=maxlen-1){ finish(s,"length"); done=true; break; }
                    }
                    if(!done && pending(req)){ ++n_spec_switch; break; }
                }
                continue;
            }
        }

        // 2a) 단일 스트림 fast-path: 활성 1개·greedy·대기 없음. 스텝 처리는 아래 3)과 같은 규칙(EOS·길이 종료).
        //     새 요청/CANCEL이 생기면 그 스텝 뒤 멈추고 배치 루프로 복귀(슬롯 pos/last가 이어받음).
        if(fastpath && A==1){
            int b=0; while(slot[b].req<0) ++b;
            Slot& s=slot[b];
            const int room=std::min(s.max_tokens-s.gen, (maxlen-1)-s.pos);
            if(s.temp<=0.f && room>0 && !pending(s.req)){
                const char* why="limit";
                FastCtx ctx{[&](int next)->int{
                    ++s.fast_steps;
                    if(is_stop(next)){ why="stop"; finish(s,"stop"); return 1; }
                    printf("%ld %d\n", s.req, next); fflush(stdout);
                    s.last=next; s.pos++; s.gen++;
                    if(s.gen>=s.max_tokens || s.pos>=maxlen-1){ why="length"; finish(s,"length"); return 1; }
                    if(pending(s.req)){ why="switch"; return 1; }
                    return 0; }};
                const long req=s.req; const int start=s.pos; ++s.fast_entries;
                OnlineBackend::SingleStats ss;
                be.generate_single(st, b, s.pos, s.last, room, &FastCtx::call, &ctx, &ss);
                be.spec_invalidate(b);
                n_tail+=ss.speculative; if(s.req==req) s.fast_tail+=ss.speculative;
                if(std::string(why)=="switch") ++n_switch;
                fprintf(stderr,"[online] FASTPATH req=%ld start_pos=%d accepted=%llu executed=%llu tail=%llu exit=%s\n",
                        req,start,ss.accepted,ss.executed,ss.speculative,why);
                continue;
            }
        }

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
            be.spec_invalidate(b);
            int next=pick_row(b, s); ++s.batch_steps;
            if(is_stop(next)){ finish(s,"stop"); continue; }  // ★EOS: 토큰 미출력·종료(계산 불변 — 값 그대로 판정)
            printf("%ld %d\n", s.req, next); fflush(stdout);
            s.last=next; s.pos++; s.gen++;
            if(s.temp>0.f) s.hist.push_back(next);           // 반복 페널티 이력(샘플링 시에만 — greedy 무비용)
            if(s.gen>=s.max_tokens || s.pos>=maxlen-1) finish(s,"length");
        }
    }
    shutdown=true; cv.notify_all();
    if(reader.joinable()) reader.detach();      // stdin blocking getline — detach(프로세스 종료 시 정리)
    be.free_state(st);
}

#endif // DGX_SERVING_ONLINE_H
