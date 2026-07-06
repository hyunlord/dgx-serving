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
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    auto spec_run=[&](const std::vector<int> &ids, int K, std::vector<int> &toks, double &msec,
                      int &cycles, double &ms_draft, double &ms_verify){
        G4DecodeState st = alloc_g4_decode_state(c,total);
        const int P=(int)ids.size();
        int cur; prefill(st,ids,cur);
        int L=P;                                                // KV 확정 길이 = anchor 위치
        CK(cudaMemcpy(d_in+ca.backbone, d_pn+(size_t)(P-1)*H, (size_t)H*4, cudaMemcpyDeviceToDevice));  // h0 = prenorm 마지막 행
        toks.clear(); toks.push_back(cur);
        cycles=0; ms_draft=0; ms_verify=0;
        std::vector<int> chunk(K+1); std::vector<int> am(K+1);
        auto t0=clk::now();
        while((int)toks.size()<N+1 && L+K+1<total){
            // --- draft K (고정 rope 행 L·KV len L·EAGLE 재귀) ---
            auto td=clk::now();
            int dt=cur;
            for(int j=0;j<K;j++){
                CK(cudaMemcpy(d_tok,&dt,4,cudaMemcpyHostToDevice));
                launch_embedding(d_tok, w.embed, d_in, 1, ca.backbone);
                if(c.embed_scale>0.f) launch_scale(d_in,d_in,c.embed_scale,(size_t)ca.backbone);
                gemma4_assistant_step(ca,wa,d_in, st.Kc[lf],st.Vc[lf],L, st.Kc[ls],st.Vc[ls],L,
                                      d_cf+(size_t)L*hd_f, d_sf+(size_t)L*hd_f,
                                      d_cs+(size_t)L*hd_s, d_ss+(size_t)L*hd_s, d_alg, d_ho, ast_uq);
                launch_argmax(d_alg,ca.vocab_size,d_am); CK(cudaMemcpy(&dt,d_am,4,cudaMemcpyDeviceToHost));
                chunk[j+1]=dt;
                CK(cudaMemcpy(d_in+ca.backbone, d_ho, (size_t)ca.backbone*4, cudaMemcpyDeviceToDevice));  // h 재귀
            }
            ms_draft+=ms_since(td);
            // --- verify K+1 (commit=0·전 위치 로짓+prenorm) ---
            auto tv=clk::now();
            chunk[0]=cur;
            CK(cudaMemcpy(d_tok,chunk.data(),(size_t)(K+1)*4,cudaMemcpyHostToDevice));
            st.n_past=L;
            // amort_mm=1: verify K+1행을 SB 커널로 weight 1회 적재(비분할상환 gemv는 행마다 재적재 = 첫 실측서
            // verify가 0.7×(K+1)×T_dec로 지배). ★bit-id 유지 근거: SB per-row == plain gemv per-row bit-identical
            // (test_gemma4_prefill_slot 게이트 확증) — 아래 [A] spec==greedy 게이트가 직접 재검증.
            gemma4_decode(c,w,st,d_tok,K+1,d_cf,d_sf,d_cs,d_ss,d_lg,0,uq,0,1,d_pn);
            for(int i=0;i<=K;i++) launch_argmax(d_lg+(size_t)i*V,V,d_am+i);
            CK(cudaMemcpy(am.data(),d_am,(size_t)(K+1)*4,cudaMemcpyDeviceToHost));
            ms_verify+=ms_since(tv);
            int m=0; for(int i=1;i<=K;i++){ if(am[i-1]==chunk[i]) m++; else break; }
            for(int i=1;i<=m && (int)toks.size()<N+1;i++) toks.push_back(chunk[i]);
            if((int)toks.size()<N+1) toks.push_back(am[m]);
            CK(cudaMemcpy(d_in+ca.backbone, d_pn+(size_t)m*H, (size_t)H*4, cudaMemcpyDeviceToDevice));   // 다음 h = prenorm 행 m
            cur=am[m]; L+=m+1; cycles++;
        }
        msec=ms_since(t0);
        free_g4_decode_state(st);
    };

    const char* wls[3]={"chat","code","prose"};
    int fails=0;
    printf("\n%-6s %-4s %9s %9s %6s %7s %7s %9s %9s\n","wl","K","plain t/s","spec t/s","곱","acc","cyc","draft ms","verify ms");
    for(int wi=0;wi<3;wi++){
        std::vector<int> ids = rii(WL+"wl_"+wls[wi]+"_ids.bin");
        std::vector<int> ptoks, stoks; double pms, sms; int cyc; double md,mv;
        plain_run(ids,ptoks,pms);                                // warm-up
        plain_run(ids,ptoks,pms);                                // 측정
        double ptps=N/(pms/1e3);
        for(int K : {2,4,8}){
            spec_run(ids,K,stoks,sms,cyc,md,mv);                 // warm-up
            spec_run(ids,K,stoks,sms,cyc,md,mv);                 // 측정
            if(stoks.size()!=ptoks.size()){ fails++; printf("    ★spec 출력 %zu != plain %zu (rope 한계/조기종료 — 잘린 비교 금지)\n",stoks.size(),ptoks.size()); }
            int nc=(int)(ptoks.size()<stoks.size()?ptoks.size():stoks.size());
            int mismatch=-1; for(int i=0;i<nc;i++) if(ptoks[i]!=stoks[i]){ mismatch=i; break; }
            double stps=N/(sms/1e3), acc=(double)N/cyc-1.0;
            printf("%-6s K=%-2d %9.2f %9.2f %5.2fx %7.2f %7d %9.0f %9.0f  %s\n",
                   wls[wi],K,ptps,stps,stps/ptps,acc,cyc,md,mv,
                   mismatch<0?"[A]==plain":"[A]★MISMATCH");
            if(mismatch>=0){ fails++; printf("    ★첫 불일치 idx=%d: plain=%d spec=%d\n",mismatch,ptoks[mismatch],stoks[mismatch]); }
        }
    }
    printf("\n[test_gemma4_spec_e2e] %s\n", fails?"FAIL(spec!=greedy)":"PASS (전 워크로드·전 K spec==greedy + 곱 measured)");
    return fails?1:0;
}
