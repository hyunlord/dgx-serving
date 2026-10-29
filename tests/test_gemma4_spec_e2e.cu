// draft-spec 3관문: 사이클 e2e — ①verify(bit-id)+②assistant(HF 골든)를 연결, ★처음으로 곱 measured.
//   사이클(HF SinglePositionMultiToken 의미와 동일): anchor cur(KV 미기재) → assistant K토큰 draft
//   (고정 rope 행=L·target KV[0..L) 직독·EAGLE 재귀 h) → gemma4_decode(Sq=K+1, commit=0) verify →
//   accept 접두사 m + bonus/교정 → h=verify prenorm 행 m·L+=m+1 → 반복.
// 게이트: [A] ★spec==greedy — 사이클 출력 시퀀스 == plain greedy 시퀀스 토큰-동일(전 워크로드·전 K).
//        [B] accept·순 곱 실측 — plain t/s vs spec t/s(warm 2회 중 2번째), draft/verify 시간 버킷.
// 사용: ./tests/test_gemma4_spec_e2e [tgt_w] [tgt_golden] [ast_w] [wl_dir] [uq=2] [N=96]
//   wl_dir/wl_{chat,code,prose}_ids.bin: tools/probe_assistant_spec.py WORKLOADS 프롬프트를
//   apply_chat_template(add_generation_prompt)로 토큰화해 int32 덤프(HF accept와 대조 공정성).
//   rope: tgt_golden/gen_rope_*(전 위치 테이블 — target·assistant 공용. ★bf16 정밀 캡처본 = HF bf16 운영과 동일 정밀.
//   ②의 fp32 rope 행과 ~2e-3 차이는 draft 품질에만 영향 가능 — accept 실측이 판정, spec 불변식과 무관[verify가 기준]).
#include "runtime/knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../models/gemma4_assistant.h"
#include "../kernels/embedding.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/argmax.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
using clk=std::chrono::steady_clock;
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }

static std::vector<float> rff(const std::string &p, size_t n){
    FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"없음: %s\n",p.c_str());exit(1);}
    std::vector<float>v(n); if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v;
}
static long fsz(const std::string &p){ FILE*f=fopen(p.c_str(),"rb"); if(!f)return -1; fseek(f,0,SEEK_END); long b=ftell(f); fclose(f); return b; }
static std::vector<int> rii(const std::string &p){
    long b=fsz(p); if(b<0){fprintf(stderr,"없음: %s\n",p.c_str());exit(1);}
    std::vector<int>v(b/4); FILE*f=fopen(p.c_str(),"rb"); if(fread(v.data(),4,v.size(),f)!=v.size())exit(1); fclose(f); return v;
}
static float* up(const std::vector<float>&h){ float*d; CK(cudaMalloc(&d,h.size()*4)); CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
struct CRow{ int j; float mg, pb; };                           // draft별 확신 진단 행(위치·margin·top-prob)

int main(int argc,char**argv){
    std::string TW = argc>1?argv[1]:"artifacts/gemma4-12b/weights";
    std::string TG = std::string(argc>2?argv[2]:"artifacts/gemma4-12b/golden")+"/";
    std::string AW = argc>3?argv[3]:"artifacts/gemma4-12b-assistant/weights";
    std::string WL = std::string(argc>4?argv[4]:"artifacts/gemma4-12b-assistant/golden")+"/";
    const int uq = argc>5?atoi(argv[5]):2;
    int N = argc>6?atoi(argv[6]):96;
    const int ast_uq = argc>7?atoi(argv[7]):0;               // drafter 정밀도(0=fp32 / 2=W4A8 — Lever1)

    ModelConfig c = load_config_gemma4((TW+"/manifest.json").c_str());
    AstConfig  ca = load_config_assistant((AW+"/manifest.json").c_str());
    const int V=c.vocab_size, H=c.hidden_size, hd_f=c.global_head_dim, hd_s=c.head_dim;
    const int total=(int)(fsz(TG+"gen_rope_full_cos.bin")/4/hd_f);
    int lf=-1, ls=-1;                                          // target 마지막 full/sliding 층(shared KV 소스)
    for(int l=0;l<c.num_layers;l++){ if(c.layer_is_full[l]) lf=l; else ls=l; }
    printf("[spec_e2e] uq=%d N=%d rope_total=%d | KV소스: full L%d·sliding L%d | assistant L%d %s\n",
           uq,N,total,lf,ls,ca.num_layers, ast_uq==2?"W4A8":"fp32");

    float *d_cf=up(rff(TG+"gen_rope_full_cos.bin",(size_t)total*hd_f)), *d_sf=up(rff(TG+"gen_rope_full_sin.bin",(size_t)total*hd_f));
    float *d_cs=up(rff(TG+"gen_rope_sliding_cos.bin",(size_t)total*hd_s)), *d_ss=up(rff(TG+"gen_rope_sliding_sin.bin",(size_t)total*hd_s));
    G4Weights w = load_weights_gemma4(TW.c_str(), c, uq!=0?1:0, uq==0?1:0);
    AstWeights wa = load_weights_assistant(AW.c_str(), ca, ast_uq);

    const int KMAX=8;
    int *d_tok; CK(cudaMalloc(&d_tok,(size_t)64*4));
    float *d_lg; CK(cudaMalloc(&d_lg,(size_t)(KMAX+1)*V*4));
    float *d_pn; CK(cudaMalloc(&d_pn,(size_t)64*H*4));          // verify/prefill prenorm rows(최대 P)
    float *d_in; CK(cudaMalloc(&d_in,(size_t)2*ca.backbone*4));
    float *d_alg; CK(cudaMalloc(&d_alg,(size_t)ca.vocab_size*4));
    float *d_ho; CK(cudaMalloc(&d_ho,(size_t)ca.backbone*4));
    int *d_am; CK(cudaMalloc(&d_am,(size_t)(KMAX+1)*4));

    // ★적응 draft 길이(커널 0 — 호스트 정책): ADAPT=1 TAUS="2,4" [KMAX=8 KMIN=1] — draft마다 확신
    //   margin(top1-top2 로짓, d_alg D2H+호스트 스캔)을 재고 margin<τ면 그 사이클 draft 조기 중단.
    //   spec==greedy 불변(verify가 결정권자 — K는 추측량일 뿐), 판정 비용은 conf ms로 분리 계측.
    // CONF_LOG=path: 고정 K 스윕에서 draft별 (margin, softmax top-prob, 사이클 accept m) CSV — STEP 0
    //   상관 진단 전용(측정 run에 D2H+호스트 softmax 추가 = spec t/s 오염 — 순 곱은 CONF_LOG 없이).
    const int adapt = knob_raw("ADAPT") ? atoi(knob_raw("ADAPT")) : 0;
    int KMIN  = knob_raw("KMIN") ? atoi(knob_raw("KMIN")) : 1;
    int KMAXe = knob_raw("KMAX") ? atoi(knob_raw("KMAX")) : 8;
    if(KMAXe>KMAX) KMAXe=KMAX;
    if(KMIN<1) KMIN=1;
    if(KMIN>KMAXe) KMIN=KMAXe;
    std::vector<float> taus;
    if(const char* ts=knob_raw("TAUS")){
        std::string s(ts); size_t p=0;
        while(p<=s.size()){ size_t q=s.find(',',p); if(q==std::string::npos)q=s.size();
            if(q>p) taus.push_back((float)atof(s.substr(p,q-p).c_str())); p=q+1; }
    }
    if(adapt && taus.empty()) taus={4.f};
    FILE* conf_f=nullptr;
    if(const char* cp=knob_raw("CONF_LOG")){
        conf_f=fopen(cp,"w");
        if(conf_f) fprintf(conf_f,"wl,K,cycle,j,margin,prob,m\n");
        printf("[spec_e2e] CONF_LOG=%s — 진단 run(spec t/s 오염 — accept/상관 전용)\n",cp);
    }
    if(adapt) printf("[spec_e2e] ADAPT: KMIN=%d KMAX=%d taus=%zu개 (margin<τ → 조기 중단·커널 0)\n",KMIN,KMAXe,taus.size());
    const char* cur_wl="";
    std::vector<float> h_alg((size_t)ca.vocab_size);

    auto prefill=[&](G4DecodeState &st, const std::vector<int> &ids, int &anchor){
        CK(cudaMemcpy(d_tok,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
        gemma4_decode(c,w,st,d_tok,(int)ids.size(),d_cf,d_sf,d_cs,d_ss,d_lg,1,uq,1,0,d_pn);   // last_only·prenorm
        launch_argmax(d_lg,V,d_am); CK(cudaMemcpy(&anchor,d_am,4,cudaMemcpyDeviceToHost));
    };

    // plain greedy: N토큰 + 시간(ms). toks[0]=anchor 다음부터 N개 아님 — anchor 이후 생성 N개를 담음.
    auto plain_run=[&](const std::vector<int> &ids, std::vector<int> &toks, double &msec){
        G4DecodeState st = alloc_g4_decode_state(c,total);
        int cur; prefill(st,ids,cur);
        toks.clear(); toks.push_back(cur);
        auto t0=clk::now();
        for(int t=0;t<N;t++){
            CK(cudaMemcpy(d_tok,&cur,4,cudaMemcpyHostToDevice));
            gemma4_decode(c,w,st,d_tok,1,d_cf,d_sf,d_cs,d_ss,d_lg,1,uq,0,0);
            launch_argmax(d_lg,V,d_am); CK(cudaMemcpy(&cur,d_am,4,cudaMemcpyDeviceToHost));
            toks.push_back(cur);
        }
        msec=ms_since(t0);
        free_g4_decode_state(st);
    };

    // spec 사이클: N토큰 생성 + 시간/버킷. 출력 시퀀스는 anchor 포함(plain과 동일 규약).
    //   tau<0 = 고정 K(기존 경로 그대로). tau>=0 = ★적응 draft 길이: draft마다 margin(top1-top2 로짓)을
    //   호스트에서 재고 j+1>=kmin && margin<tau면 조기 중단 → verify Sq=kact+1. 몇 개를 추측하든 verify가
    //   결정권자라 출력 불변([A] 게이트가 매 행 재확인). 판정 비용(D2H 1MB+스캔)은 ms_draft에 포함(순 곱
    //   정직) + ms_conf로 분리 계측. diag=CSV 기록(+호스트 softmax — 그 run 타이밍 오염, STEP 0 전용).
    //   hfdyn>0 = ★사이클-간 동적 K(HF heuristic·판정 비용 0): margin 아닌 verify의 실제 m으로 다음 K 결정
    //   — m==kact(전부 수락)면 K+2, 아니면 K-1 (clamp [1..K]). 초기값 hfdyn.
    auto spec_run=[&](const std::vector<int> &ids, int K, int kmin, float tau, int hfdyn, bool diag,
                      std::vector<int> &toks, double &msec, int &cycles,
                      double &ms_draft, double &ms_verify, double &ms_conf, double &avg_k){
        G4DecodeState st = alloc_g4_decode_state(c,total);
        const int P=(int)ids.size();
        int cur; prefill(st,ids,cur);
        int L=P;                                                // KV 확정 길이 = anchor 위치
        CK(cudaMemcpy(d_in+ca.backbone, d_pn+(size_t)(P-1)*H, (size_t)H*4, cudaMemcpyDeviceToDevice));  // h0 = prenorm 마지막 행
        toks.clear(); toks.push_back(cur);
        cycles=0; ms_draft=0; ms_verify=0; ms_conf=0; long ndraft=0;
        int kdyn = hfdyn>0 ? hfdyn : K;                         // 사이클-간 동적 K 상태
        std::vector<int> chunk(K+1); std::vector<int> am(K+1); std::vector<CRow> crows;
        auto t0=clk::now();
        while((int)toks.size()<N+1 && L+K+1<total){
            // --- draft ≤K (고정 rope 행 L·KV len L·EAGLE 재귀·margin<τ 조기 중단) ---
            auto td=clk::now();
            const int Kc = hfdyn>0 ? kdyn : K;                  // 이번 사이클 draft 예산
            int dt=cur, kact=Kc;
            crows.clear();
            for(int j=0;j<Kc;j++){
                CK(cudaMemcpy(d_tok,&dt,4,cudaMemcpyHostToDevice));
                launch_embedding(d_tok, w.embed, d_in, 1, ca.backbone);
                if(c.embed_scale>0.f) launch_scale(d_in,d_in,c.embed_scale,(size_t)ca.backbone);
                gemma4_assistant_step(ca,wa,d_in, st.Kc[lf],st.Vc[lf],L, st.Kc[ls],st.Vc[ls],L,
                                      d_cf+(size_t)L*hd_f, d_sf+(size_t)L*hd_f,
                                      d_cs+(size_t)L*hd_s, d_ss+(size_t)L*hd_s, d_alg, d_ho, ast_uq);
                launch_argmax(d_alg,ca.vocab_size,d_am); CK(cudaMemcpy(&dt,d_am,4,cudaMemcpyDeviceToHost));
                chunk[j+1]=dt;
                CK(cudaMemcpy(d_in+ca.backbone, d_ho, (size_t)ca.backbone*4, cudaMemcpyDeviceToDevice));  // h 재귀
                if((tau>=0.f && j+1>=kmin) || diag){        // margin은 정책이 발동 가능한 위치에서만(판정 비용 절약)
                    auto tc=clk::now();
                    CK(cudaMemcpy(h_alg.data(), d_alg, (size_t)ca.vocab_size*4, cudaMemcpyDeviceToHost));
                    float t1=h_alg[dt], t2=-1e30f;
                    for(int v=0;v<ca.vocab_size;v++){ float x=h_alg[v]; if(x>t2 && v!=dt) t2=x; }
                    float margin=t1-t2;
                    ms_conf+=ms_since(tc);
                    if(diag){                                   // softmax top-prob(호스트 exp 262k — 진단 전용)
                        double se=0; for(int v=0;v<ca.vocab_size;v++) se+=exp((double)(h_alg[v]-t1));
                        crows.push_back({j+1,margin,(float)(1.0/se)});
                    }
                    if(tau>=0.f && j+1>=kmin && j+1<Kc && margin<tau){ kact=j+1; break; }
                }
            }
            ms_draft+=ms_since(td);
            // --- verify kact+1 (commit=0·전 위치 로짓+prenorm) ---
            auto tv=clk::now();
            chunk[0]=cur;
            CK(cudaMemcpy(d_tok,chunk.data(),(size_t)(kact+1)*4,cudaMemcpyHostToDevice));
            st.n_past=L;
            // amort_mm=1: verify kact+1행을 SB 커널로 weight 1회 적재(비분할상환 gemv는 행마다 재적재 = 첫 실측서
            // verify가 0.7×(K+1)×T_dec로 지배). ★bit-id 유지 근거: SB per-row == plain gemv per-row bit-identical
            // (test_gemma4_prefill_slot 게이트 확증) — 아래 [A] spec==greedy 게이트가 직접 재검증.
            gemma4_decode(c,w,st,d_tok,kact+1,d_cf,d_sf,d_cs,d_ss,d_lg,0,uq,0,1,d_pn);
            for(int i=0;i<=kact;i++) launch_argmax(d_lg+(size_t)i*V,V,d_am+i);
            CK(cudaMemcpy(am.data(),d_am,(size_t)(kact+1)*4,cudaMemcpyDeviceToHost));
            ms_verify+=ms_since(tv);
            int m=0; for(int i=1;i<=kact;i++){ if(am[i-1]==chunk[i]) m++; else break; }
            if(conf_f && diag) for(auto &r:crows)
                fprintf(conf_f,"%s,%d,%d,%d,%.4f,%.6g,%d\n",cur_wl,K,cycles,r.j,r.mg,r.pb,m);
            for(int i=1;i<=m && (int)toks.size()<N+1;i++) toks.push_back(chunk[i]);
            if((int)toks.size()<N+1) toks.push_back(am[m]);
            CK(cudaMemcpy(d_in+ca.backbone, d_pn+(size_t)m*H, (size_t)H*4, cudaMemcpyDeviceToDevice));   // 다음 h = prenorm 행 m
            cur=am[m]; L+=m+1; cycles++; ndraft+=kact;
            if(hfdyn>0) kdyn = (m==kact) ? (kact+2>K?K:kact+2) : (kact-1<1?1:kact-1);   // verify 실측 m 기반
        }
        msec=ms_since(t0);
        avg_k = cycles ? (double)ndraft/cycles : 0.0;
        free_g4_decode_state(st);
    };

    const char* wls[3]={"chat","code","prose"};
    int fails=0;
    printf("\n%-6s %-12s %9s %9s %6s %7s %5s %5s %8s %7s %9s\n",
           "wl","mode","plain t/s","spec t/s","곱","acc","cyc","K~","draft ms","conf ms","verify ms");
    for(int wi=0;wi<3;wi++){
        cur_wl=wls[wi];
        std::vector<int> ids = rii(WL+"wl_"+wls[wi]+"_ids.bin");
        std::vector<int> ptoks, stoks; double pms, sms; int cyc; double md,mv,mc,ak;
        plain_run(ids,ptoks,pms);                                // warm-up
        plain_run(ids,ptoks,pms);                                // 측정
        if(const char* dp=knob_raw("DUMP_PLAIN")){                 // 진단용: plain 궤적 덤프(HF 궤적-난이도 대조 실험)
            char pp[512]; snprintf(pp,sizeof(pp),"%s/traj_%s.bin",dp,wls[wi]);
            FILE*f=fopen(pp,"wb"); if(f){ fwrite(ptoks.data(),4,ptoks.size(),f); fclose(f); }
        }
        if(knob_raw("SKIP_SPEC")){ printf("%-6s plain %.2f t/s (SKIP_SPEC — 궤적 덤프 전용)\n",wls[wi],N/(pms/1e3)); continue; }  // 진단용
        double ptps=N/(pms/1e3);
        auto row=[&](const char* mode, int K, int kmin, float tau, int hfdyn, bool diag){
            spec_run(ids,K,kmin,tau,hfdyn,false,stoks,sms,cyc,md,mv,mc,ak);   // warm-up
            spec_run(ids,K,kmin,tau,hfdyn,diag,stoks,sms,cyc,md,mv,mc,ak);    // 측정(diag=CSV — 그 run 타이밍 오염)
            if(stoks.size()!=ptoks.size()){ fails++; printf("    ★spec 출력 %zu != plain %zu (rope 한계/조기종료 — 잘린 비교 금지)\n",stoks.size(),ptoks.size()); }
            int nc=(int)(ptoks.size()<stoks.size()?ptoks.size():stoks.size());
            int mismatch=-1; for(int i=0;i<nc;i++) if(ptoks[i]!=stoks[i]){ mismatch=i; break; }
            double stps=N/(sms/1e3), acc=(double)N/cyc-1.0;
            printf("%-6s %-12s %9.2f %9.2f %5.2fx %7.2f %5d %5.1f %8.0f %7.0f %9.0f  %s\n",
                   wls[wi],mode,ptps,stps,stps/ptps,acc,cyc,ak,md,mc,mv,
                   mismatch<0?"[A]==plain":"[A]★MISMATCH");
            if(mismatch>=0){ fails++; printf("    ★첫 불일치 idx=%d: plain=%d spec=%d\n",mismatch,ptoks[mismatch],stoks[mismatch]); }
        };
        if(!adapt){
            for(int K : {2,4,8}){ char mb[16]; snprintf(mb,sizeof(mb),"K=%d",K); row(mb,K,1,-1.f,0,conf_f!=nullptr); }
        }else if(adapt==2){                                      // ★HF heuristic: verify 실측 m 기반 사이클-간 K(판정 비용 0)
            row("K=4(ref)",4,1,-1.f,0,false);
            char mb[24]; snprintf(mb,sizeof(mb),"hfdyn4..%d",KMAXe); row(mb,KMAXe,1,-1.f,4,false);
        }else{
            row("K=4(ref)",4,1,-1.f,0,false);                    // 같은 로드·같은 warm 조건의 고정 기준행
            for(float tau : taus){ char mb[24]; snprintf(mb,sizeof(mb),"a%d-%d t%.1f",KMIN,KMAXe,tau); row(mb,KMAXe,KMIN,tau,0,false); }
        }
    }
    printf("\n[test_gemma4_spec_e2e] %s\n", fails?"FAIL(spec!=greedy)":"PASS (전 워크로드·전 K spec==greedy + 곱 measured)");
    return fails?1:0;
}
