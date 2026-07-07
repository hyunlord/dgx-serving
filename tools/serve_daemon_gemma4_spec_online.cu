// draft-spec 온라인 daemon — Gemma4 12B target + 0.4B assistant(EAGLE류) speculative 서빙 (4관문).
//   사이클(3관문 test_gemma4_spec_e2e와 동일 수학): 슬롯별 anchor → assistant K토큰 draft(고정 rope행=pos·
//   target KV[0..pos) 직독·재귀 h) → 슬롯-뷰 gemma4_decode(Sq=K+1, commit=0, amort=1) verify → accept 접두사
//   +bonus 방출 → h=verify prenorm 행 m → 반복. ★spec==greedy 불변(verify가 기준 — 게이트 test_gemma4_spec_serving).
//   프로토콜 = serving_online.h와 동일(태그드): stdin "<rid> <max_tokens> <ids...>"/"SHUTDOWN",
//   stdout "<rid> <tok>"/"<rid> END <stop|length>", stderr "[online] ready". serve_api OnlineDaemon 그대로 붙음.
//   동시성 = 슬롯별 라운드로빈 사이클(per-slot 순차 = 단독과 동일 계산 = 격리 구성적 — 배칭 이득은 후속 perf).
//   rope = 온디바이스 생성(plain daemon 동일 — target/assistant 같은 테이블 = 내적 일관·spec 불변식 성립).
//   사용법: serve_daemon_gemma4_spec_online <tgt_weights> <ast_weights> [uq=2] [maxlen=384] [Bmax=4] [K=4] [ast_uq=2]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <sstream>
#include <iostream>
#include <condition_variable>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/serving_online.h"     // parse_stop_ids_env(프로토콜/stop 의미 동일 유지)
#include "../models/gemma4_assistant.h"
#include "../kernels/rope.cuh"
#include "../kernels/embedding.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/argmax.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"[spec-online] CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

static double mem_avail_gb(){
    FILE*f=fopen("/proc/meminfo","r"); if(!f) return -1;
    char k[64]; long v; char u[16]; double r=-1;
    while(fscanf(f,"%63s %ld %15s",k,&v,u)==3){ if(!strcmp(k,"MemAvailable:")){ r=v/1e6; break; } }
    fclose(f); return r;
}

struct SpecSlot { long req=-1; int pos=0; int gen=0; int max_tokens=0; int cur=0; };  // pos=KV 확정 길이(=anchor 위치)

int main(int argc,char**argv){
    std::string TW = argc>1?argv[1]:"artifacts/gemma4-12b/weights";
    std::string AW = argc>2?argv[2]:"artifacts/gemma4-12b-assistant/weights";
    const int uq     = argc>3?atoi(argv[3]):2;
    const int maxlen = argc>4?atoi(argv[4]):384;
    const int Bmax   = argc>5?atoi(argv[5]):4;
    const int K      = argc>6?atoi(argv[6]):4;         // 3관문 최적점(code 1.54×·chat 1.34×)
    const int ast_uq = argc>7?atoi(argv[7]):2;         // drafter 정밀도(2=W4A8 기본[Lever1 게이트 후]·0=fp32)
    double ma = mem_avail_gb();
    fprintf(stderr,"[spec-online] MemAvailable=%.1fGB (필요 ~12GB: 12B q4 7.5+assistant fp32 1.7+KV)\n", ma);
    if(ma>0 && ma<14){ fprintf(stderr,"[spec-online] 메모리 여유 부족(<14GB) — 크래시 가드 중단.\n"); return 1; }

    ModelConfig c = load_config_gemma4((TW+"/manifest.json").c_str());
    AstConfig  ca = load_config_assistant((AW+"/manifest.json").c_str());
    const int V=c.vocab_size, H=c.hidden_size;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    if(ca.vocab_size!=V || ca.backbone!=H){ fprintf(stderr,"[spec-online] target/assistant 형상 불일치 V %d/%d H %d/%d\n",V,ca.vocab_size,H,ca.backbone); return 1; }
    int lf=-1, ls=-1;                                   // shared KV 소스 = 마지막 full/sliding 층
    for(int l=0;l<c.num_layers;l++){ if(c.layer_is_full&&c.layer_is_full[l]) lf=l; else ls=l; }

    // rope 온디바이스(plain daemon 동일 — target·assistant 공용 테이블 = 내적 일관)
    float *d_cf,*d_sf,*d_cs,*d_ss;
    CK(cudaMalloc(&d_cf,(size_t)maxlen*hd_f*4)); CK(cudaMalloc(&d_sf,(size_t)maxlen*hd_f*4));
    CK(cudaMalloc(&d_cs,(size_t)maxlen*hd_s*4)); CK(cudaMalloc(&d_ss,(size_t)maxlen*hd_s*4));
    launch_gen_rope(d_cf,d_sf,maxlen,hd_f,1e4f); launch_gen_rope(d_cs,d_ss,maxlen,hd_s,1e4f); CK(cudaDeviceSynchronize());

    G4Weights  w  = load_weights_gemma4(TW.c_str(), c, uq!=0?1:0, uq==0?1:0);
    AstWeights wa = load_weights_assistant(AW.c_str(), ca, ast_uq);
    G4BatchState bs = alloc_g4_batch_state(c, Bmax, maxlen);

    const int MCAT=Bmax*(K+1);                                          // verify concat 최대 행수(전 슬롯 활성)
    int   *d_tok;  CK(cudaMalloc(&d_tok,(size_t)(maxlen>MCAT?maxlen:MCAT)*4));
    float *d_lg;   CK(cudaMalloc(&d_lg,(size_t)MCAT*V*4));
    float *d_pn;   CK(cudaMalloc(&d_pn,(size_t)maxlen*H*4));           // prefill/verify prenorm rows(maxlen≥MCAT)
    float *d_h;    CK(cudaMalloc(&d_h,(size_t)Bmax*H*4));              // 슬롯별 h_prev(EAGLE 재귀)
    float *d_in;   CK(cudaMalloc(&d_in,(size_t)2*H*4));
    float *d_alg;  CK(cudaMalloc(&d_alg,(size_t)V*4));
    float *d_ho;   CK(cudaMalloc(&d_ho,(size_t)H*4));
    int   *d_am;   CK(cudaMalloc(&d_am,(size_t)MCAT*4));
    // ★draft concat(배치) 버퍼 — A>=2일 때만 사용(단일은 기존 경로)
    float *d_inA;  CK(cudaMalloc(&d_inA,(size_t)Bmax*2*H*4));
    float *d_algA; CK(cudaMalloc(&d_algA,(size_t)Bmax*V*4));
    float *d_hoA;  CK(cudaMalloc(&d_hoA,(size_t)Bmax*H*4));
    float *d_embA; CK(cudaMalloc(&d_embA,(size_t)Bmax*H*4));
    int   *d_tokA; CK(cudaMalloc(&d_tokA,(size_t)Bmax*4));
    int   *d_posA; CK(cudaMalloc(&d_posA,(size_t)Bmax*4));

    // 슬롯-b 뷰(gemma4_verify_batch/prefill_batch_slot과 동일 트릭)
    std::vector<std::vector<float*>> vK(Bmax), vV(Bmax);
    for(int b=0;b<Bmax;b++){
        vK[b].resize(c.num_layers); vV[b].resize(c.num_layers);
        for(int l=0;l<c.num_layers;l++){
            int full=c.layer_is_full?c.layer_is_full[l]:1;
            int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
            size_t stride=(size_t)maxlen*nkv*hd;
            vK[b][l]=bs.Kc[l]+(size_t)b*stride; vV[b][l]=bs.Vc[l]+(size_t)b*stride;
        }
    }
    auto view_of=[&](int b, int pos)->G4DecodeState{
        G4DecodeState v{}; v.n_past=pos; v.max_len=maxlen; v.L=c.num_layers; v.kv_cap=bs.kv_cap;
        v.Kc=vK[b].data(); v.Vc=vV[b].data(); return v;
    };

    const std::vector<int> stop_ids = parse_stop_ids_env();
    auto is_stop=[&](int t){ for(int s:stop_ids) if(t==s) return true; return false; };
    fprintf(stderr,"[online] ready: gemma4-spec 12B+0.4B uq=%d maxlen=%d Bmax=%d K=%d drafter=%s V=%d (슬롯별 spec 사이클·spec==greedy)\n",
            uq,maxlen,Bmax,K,ast_uq==2?"W4A8":"fp32",V); fflush(stderr);

    // reader 스레드(serving_online.h 프로토콜 동일)
    using serving_online_detail::Req;
    std::deque<Req> queue; std::mutex mtx; std::condition_variable cv; std::atomic<bool> shutdown{false};
    std::thread reader([&]{
        std::string line;
        while(std::getline(std::cin,line)){
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
        shutdown=true; cv.notify_all();
    });

    std::vector<SpecSlot> slot(Bmax);
    std::vector<int> h_cat((size_t)Bmax*(K+1)), am_cat((size_t)Bmax*(K+1));  // 활성 a → 행 off=a*(K+1)
    std::vector<int> Nb(Bmax), pos_a(Bmax);
    auto n_active=[&](){ int a=0; for(int b=0;b<Bmax;b++) if(slot[b].req>=0)a++; return a; };
    auto end_slot=[&](int b, const char* why){ printf("%ld END %s\n", slot[b].req, why); fflush(stdout); slot[b]=SpecSlot(); };

    while(true){
        // 1) admit(스텝 경계·serving_online 의미 동일): prefill + 첫 토큰(prefill argmax) 방출.
        {
            std::unique_lock<std::mutex> lk(mtx);
            if(queue.empty() && n_active()==0){
                if(shutdown) break;
                cv.wait(lk,[&]{ return !queue.empty() || shutdown.load(); });
                if(queue.empty() && shutdown) break;
            }
            for(int b=0;b<Bmax && !queue.empty();b++){
                if(slot[b].req>=0) continue;
                Req r=std::move(queue.front()); queue.pop_front();
                lk.unlock();
                int P=(int)r.ids.size();
                if(P>=maxlen-(K+2)){ P=maxlen-(K+2); r.ids.resize(P); }   // verify 청크(K+1)가 항상 maxlen 내
                int budget=maxlen-P-(K+2); if(r.max_tokens>budget) r.max_tokens=budget;
                CK(cudaMemcpy(d_tok,r.ids.data(),(size_t)P*4,cudaMemcpyHostToDevice));
                G4DecodeState pv=view_of(b,0);
                gemma4_decode(c,w,pv,d_tok,P,d_cf,d_sf,d_cs,d_ss,d_lg,0,uq,1,1,d_pn);   // last_only·amort(prefill_batch_slot 동일)·prenorm
                launch_argmax(d_lg,V,d_am); int first; CK(cudaMemcpy(&first,d_am,4,cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(d_h+(size_t)b*H, d_pn+(size_t)(P-1)*H, (size_t)H*4, cudaMemcpyDeviceToDevice));  // h0
                slot[b]=SpecSlot{r.id,P,0,r.max_tokens,first};
                if(r.max_tokens>0 && is_stop(first)){ end_slot(b,"stop"); }
                else {
                    if(r.max_tokens>0){ printf("%ld %d\n", r.id, first); fflush(stdout); slot[b].gen=1; }
                    if(slot[b].req>=0 && slot[b].gen>=slot[b].max_tokens) end_slot(b,"length");
                }
                lk.lock();
            }
        }
        if(n_active()==0){ if(shutdown){ std::lock_guard<std::mutex> lk(mtx); if(queue.empty()) break; } continue; }

        // 2) 활성 슬롯 spec 사이클 1회 — draft(슬롯별) → ★verify concat(활성 전 슬롯 한 SB 호출 = weight 적재
        //    분할상환, 다중 슬롯 −36% measured) → accept/방출(슬롯별). 계산 의미는 per-slot 순차와 동일:
        //    draft는 자기 슬롯 KV[0..pos)만 읽고(이전 사이클 verify가 쓴 접두사), concat==per-slot bit-id는
        //    test_gemma4_verify_concat 게이트(M 경계 스윕·Δ=0)가 잠금 → spec==greedy 유지.
        std::vector<int> act; act.reserve(Bmax);
        for(int b=0;b<Bmax;b++){
            if(slot[b].req<0) continue;
            if(slot[b].pos+K+1>=maxlen){ end_slot(b,"length"); continue; }   // KV 예산 가드
            act.push_back(b);
        }
        const int A=(int)act.size();
        if(A==0) continue;
        // --- draft phase: A==1 기존 슬롯별 경로(단일 불변) / A>=2 ★draft concat(배치 스텝 ×K —
        //     tied lm_head 등 weight 1적재 = verify concat 대칭·게이트 test_gemma4_draft_concat Δ=0) ---
        if(A==1){
            const int b=act[0]; SpecSlot &s=slot[b];
            int dt=s.cur;
            for(int j=0;j<K;j++){
                CK(cudaMemcpy(d_tok,&dt,4,cudaMemcpyHostToDevice));
                launch_embedding(d_tok, w.embed, d_in, 1, H);
                if(c.embed_scale>0.f) launch_scale(d_in,d_in,c.embed_scale,(size_t)H);
                CK(cudaMemcpy(d_in+H, j==0? d_h+(size_t)b*H : d_ho, (size_t)H*4, cudaMemcpyDeviceToDevice));
                gemma4_assistant_step(ca,wa,d_in, vK[b][lf],vV[b][lf],s.pos, vK[b][ls],vV[b][ls],s.pos,
                                      d_cf+(size_t)s.pos*hd_f, d_sf+(size_t)s.pos*hd_f,
                                      d_cs+(size_t)s.pos*hd_s, d_ss+(size_t)s.pos*hd_s, d_alg, d_ho, ast_uq);
                launch_argmax(d_alg,V,d_am); CK(cudaMemcpy(&dt,d_am,4,cudaMemcpyDeviceToHost));
                h_cat[j+1]=dt;
            }
            h_cat[0]=s.cur;
        } else {
            std::vector<const float*> kfA(A),vfA(A),ksA(A),vsA(A); std::vector<int> lenA(A),dtA(A),amA(A);
            for(int a=0;a<A;a++){ const int b=act[a]; SpecSlot &s=slot[b];
                kfA[a]=vK[b][lf]; vfA[a]=vV[b][lf]; ksA[a]=vK[b][ls]; vsA[a]=vV[b][ls];
                lenA[a]=s.pos; dtA[a]=s.cur; h_cat[a*(K+1)]=s.cur; }
            CK(cudaMemcpy(d_posA,lenA.data(),(size_t)A*4,cudaMemcpyHostToDevice));
            for(int j=0;j<K;j++){
                CK(cudaMemcpy(d_tokA,dtA.data(),(size_t)A*4,cudaMemcpyHostToDevice));
                launch_embedding(d_tokA, w.embed, d_embA, A, H);
                if(c.embed_scale>0.f) launch_scale(d_embA,d_embA,c.embed_scale,(size_t)A*H);
                for(int a=0;a<A;a++){                              // 행 a = [emb | h_prev(슬롯 독립 재귀)]
                    CK(cudaMemcpy(d_inA+(size_t)a*2*H, d_embA+(size_t)a*H, (size_t)H*4, cudaMemcpyDeviceToDevice));
                    CK(cudaMemcpy(d_inA+(size_t)a*2*H+H, j==0? d_h+(size_t)act[a]*H : d_hoA+(size_t)a*H,
                                  (size_t)H*4, cudaMemcpyDeviceToDevice));
                }
                gemma4_assistant_step_batch(ca,wa,A,d_inA,kfA.data(),vfA.data(),ksA.data(),vsA.data(),
                                            lenA.data(),d_cf,d_sf,d_cs,d_ss,d_posA,d_algA,d_hoA,ast_uq);
                for(int a=0;a<A;a++) launch_argmax(d_algA+(size_t)a*V,V,d_am+a);
                CK(cudaMemcpy(amA.data(),d_am,(size_t)A*4,cudaMemcpyDeviceToHost));
                for(int a=0;a<A;a++){ dtA[a]=amA[a]; h_cat[a*(K+1)+j+1]=amA[a]; }
            }
        }
        // --- verify phase: concat 1호출(A=1이어도 동일 분해=기존 amort 경로와 같은 커널·게이트 확증) ---
        for(int a=0;a<A;a++){ Nb[a]=K+1; pos_a[a]=slot[act[a]].pos; }
        CK(cudaMemcpy(d_tok,h_cat.data(),(size_t)A*(K+1)*4,cudaMemcpyHostToDevice));
        gemma4_verify_concat(c,w,bs,A,act.data(),Nb.data(),pos_a.data(),d_tok,
                             d_cf,d_sf,d_cs,d_ss,d_lg,uq,d_pn);
        for(int i=0;i<A*(K+1);i++) launch_argmax(d_lg+(size_t)i*V,V,d_am+i);
        CK(cudaMemcpy(am_cat.data(),d_am,(size_t)A*(K+1)*4,cudaMemcpyDeviceToHost));
        // --- accept/방출 phase: 슬롯별(accept d_1..d_m + bonus/교정 — stop/max_tokens 의미 기존 동일) ---
        for(int a=0;a<A;a++){
            const int b=act[a]; SpecSlot &s=slot[b]; if(s.req<0) continue;
            const int off=a*(K+1);
            int m=0; for(int i=1;i<=K;i++){ if(am_cat[off+i-1]==h_cat[off+i]) m++; else break; }
            for(int i=1;i<=m+1 && s.req>=0;i++){
                int tok = (i<=m)? h_cat[off+i] : am_cat[off+m];
                if(is_stop(tok)){ end_slot(b,"stop"); break; }            // EOS 토큰 미출력·종료
                printf("%ld %d\n", s.req, tok); fflush(stdout);
                s.gen++;
                if(s.gen>=s.max_tokens){ end_slot(b,"length"); break; }
            }
            if(s.req>=0){                                                 // 사이클 북키핑(h·cur·pos)
                CK(cudaMemcpy(d_h+(size_t)b*H, d_pn+(size_t)(off+m)*H, (size_t)H*4, cudaMemcpyDeviceToDevice));
                s.cur=am_cat[off+m]; s.pos+=m+1;
            }
        }
    }
    shutdown=true; cv.notify_all();
    if(reader.joinable()) reader.detach();
    free_g4_batch_state(bs); free_weights_assistant(wa,ca); free_weights_gemma4(w); free_config_gemma4(c);
    return 0;
}
