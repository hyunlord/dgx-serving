// M5 온라인 연속 배칭 daemon — Gemma4 plain(greedy). 모델-중립 루프(serving_online.h) + Gemma4OnlineBackend.
//   ★헌법: 배치 루프(reader 스레드·slot·argmax·태그드 I/O)는 serving_online.h에 모델-중립으로 추출됨.
//   이 파일 = Gemma4OnlineBackend(gemma4 primitive 배선) + setup + run_online_daemon 호출. qwen daemon과 루프 공유.
//   dummy rope(gemma4 서빙 bit-id 게이트는 계산 일관성 — API==단독). 사용법: serve_daemon_gemma4_online <w> [uq=2] [maxlen=384] [Bmax=8] [assistant_dir]
//   assistant_dir(선택, R17): 0.4B drafter를 함께 적재 — G4_SPEC_SERVE=1이면 단일 greedy 요청을 speculative 사이클로
//   (serve_daemon_gemma4_spec_online의 A==1 사이클과 같은 수학). 미지정이면 기존과 같다.
#include "runtime/knobs.h"
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/serving_online.h"
#include "../runtime/rope_gemma4.h"    // ★rope 테이블 host 생성(manifest 권위·partial proportional — 절대 게이트 참조)
#include "../models/gemma4_moe.h"      // 26B-A4B MoE 분기(gemma4_moe_decode_batch·prefill_batch_slot)
#include "../kernels/rope.cuh"
#include "../kernels/kv_store.cuh"
#include "../models/gemma4_assistant.h"
#include "../kernels/embedding.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/argmax.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"[online] CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

struct Gemma4OnlineBackend : OnlineBackend {
    ModelConfig c; G4Weights w;
    float *d_cf=nullptr,*d_sf=nullptr,*d_cs=nullptr,*d_ss=nullptr,*d_logit=nullptr;
    int *d_tok=nullptr,*d_pos=nullptr,*d_pf=nullptr;
    int V=0, maxlen_=0, uq=2, moe=0;   // moe=1: 26B-A4B(dense∥MoE) → gemma4_moe_* 분기(c.num_experts>0 자동)
    int vocab() const override { return V; }
    int max_len() const override { return maxlen_; }
    void* alloc_state(int Bmax) override { return new G4BatchState(alloc_g4_batch_state(c,Bmax,maxlen_)); }
    void  free_state(void* st) override { G4BatchState* s=(G4BatchState*)st; free_g4_batch_state(*s); delete s; }
    void  prefill_ids(void* st, int b, const int* h_ids, int P) override {
        G4BatchState* s=(G4BatchState*)st;
        CK(cudaMemcpy(d_pf, h_ids, (size_t)P*4, cudaMemcpyHostToDevice));
        if(moe) gemma4_moe_prefill_batch_slot(c,w,*s,b,d_pf,P,d_cf,d_sf,d_cs,d_ss,d_logit,uq);   // 26B dense∥MoE
        else    gemma4_prefill_batch_slot(c,w,*s,b,d_pf,P,d_cf,d_sf,d_cs,d_ss,d_logit,uq);        // 31B dense
    }
    void  decode_step(void* st, const int* h_toks, const int* h_pos, int Bmax) override {
        G4BatchState* s=(G4BatchState*)st;
        s->B = Bmax;   // ★실행-B를 인자(active-width W)로 설정 — decode_batch가 st.B 슬롯만 forward(qwen 대칭).
                       //   미설정 시 alloc Bmax(8) 고정 → serving_online active-width(W<8)서 stale 슬롯 illegal access.
        CK(cudaMemcpy(d_tok, h_toks, Bmax*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos, h_pos, Bmax*4, cudaMemcpyHostToDevice));
        std::vector<int> hp(h_pos, h_pos+Bmax);
        if(moe) gemma4_moe_decode_batch(c,w,*s,d_tok,d_cf,d_sf,d_cs,d_ss,d_logit,0,uq,d_pos,hp.data());  // 26B(dense mmB + MoE fused)
        else    gemma4_decode_batch(c,w,*s,d_tok,d_cf,d_sf,d_cs,d_ss,d_logit,0,uq,d_pos,hp.data());       // 31B(ragged 슬롯 독립)
    }
    void  logits_to_host(float* h_dst, int rows) override { CK(cudaMemcpy(h_dst, d_logit, (size_t)rows*V*4, cudaMemcpyDeviceToHost)); }

    // 단일 스트림 fast-path: 슬롯 b KV 서브영역(=단일 시퀀스 KV, prefill_batch_slot과 같은 뷰)에서
    //   gemma4_generate(S=1, G4_CUDA_GRAPH 0/1/2). dense 전용(MoE는 배치 경로 유지).
    bool single_stream() const override { return !moe; }
    struct Relay { int (*on_step)(int, void*); void* opaque; };
    static int relay(const float*, int token, int, void* o){ Relay* r=(Relay*)o; return r->on_step(token, r->opaque); }
    int generate_single(void* st, int b, int pos, int first, int max_steps,
                        int (*on_step)(int, void*), void* opaque, SingleStats* stats) override {
        G4BatchState* s=(G4BatchState*)st;
        std::vector<float*> kv(2*(size_t)s->L);
        G4DecodeState view{}; view.n_past=pos; view.max_len=s->max_len; view.L=s->L; view.kv_cap=s->kv_cap;
        view.Kc=kv.data(); view.Vc=kv.data()+s->L;
        const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
        const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
        for(int l=0;l<s->L;l++){
            const int full=c.layer_is_full?c.layer_is_full[l]:1;
            const size_t stride=(size_t)s->max_len*(full?nkv_f*hd_f:nkv_s*hd_s);   // gemma4_batch.cu kv_stride
            view.Kc[l]=kv_at(s->Kc[l],(size_t)b*stride); view.Vc[l]=kv_at(s->Vc[l],(size_t)b*stride);
        }
        Relay r{on_step, opaque}; G4GenerationStats gs;
        const int accepted=gemma4_generate(c,w,view,first,max_steps,-1,d_cf,d_sf,d_cs,d_ss,uq,&relay,&r,&gs);
        if(stats){ stats->executed=gs.executed_steps; stats->accepted=gs.accepted_steps; stats->speculative=gs.speculative_steps; }
        return accepted;
    }

    // ---- R17 speculative cycle (dense only; drafter loaded when an assistant dir is given) ----
    bool has_ast=false; AstConfig ca{}; AstWeights wa{}; int ast_uq=2, lf=-1, ls=-1;
    float *d_h=nullptr,*d_in=nullptr,*d_alg=nullptr,*d_ho=nullptr,*d_lgv=nullptr,*d_pnv=nullptr;
    int *d_tv=nullptr,*d_am=nullptr; std::vector<char> h_valid;
    static constexpr int SPEC_KMAX=8;
    bool spec_ready() const override { return has_ast && !moe; }
    void spec_invalidate(int b) override { if(b<(int)h_valid.size()) h_valid[b]=0; }
    float* slot_k(G4BatchState* s, int l, int b) const { return kv_at(s->Kc[l],(size_t)b*slot_stride(l,s->max_len)); }
    float* slot_v(G4BatchState* s, int l, int b) const { return kv_at(s->Vc[l],(size_t)b*slot_stride(l,s->max_len)); }
    size_t slot_stride(int l, int ml) const {
        const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
        const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
        const int full=c.layer_is_full?c.layer_is_full[l]:1;
        return (size_t)ml*(full?nkv_f*hd_f:nkv_s*hd_s);
    }
    void spec_cycle(void* st, int b, int pos, int cur, int K, SpecCycle* out) override {
        G4BatchState* s=(G4BatchState*)st;
        const int H=c.hidden_size, hd_s=c.head_dim, hd_f=c.global_head_dim?c.global_head_dim:hd_s;
        if(K>SPEC_KMAX) K=SPEC_KMAX;
        if(!h_valid[b]) K=0;                                  // no drafter state yet: one verify row = plain step
        int chunk[SPEC_KMAX+1], am[SPEC_KMAX+1];
        chunk[0]=cur;
        // draft: serve_daemon_gemma4_spec_online A==1 path (shared target KV of the last full/sliding layer, h recursion)
        int dt=cur;
        for(int j=0;j<K;j++){
            CK(cudaMemcpy(d_tv,&dt,4,cudaMemcpyHostToDevice));
            launch_embedding(d_tv, w.embed, d_in, 1, H);
            if(c.embed_scale>0.f) launch_scale(d_in,d_in,c.embed_scale,(size_t)H);
            CK(cudaMemcpy(d_in+H, j==0? d_h+(size_t)b*H : d_ho, (size_t)H*4, cudaMemcpyDeviceToDevice));
            gemma4_assistant_step(ca,wa,d_in, slot_k(s,lf,b),slot_v(s,lf,b),pos, slot_k(s,ls,b),slot_v(s,ls,b),pos,
                                  d_cf+(size_t)pos*hd_f, d_sf+(size_t)pos*hd_f,
                                  d_cs+(size_t)pos*hd_s, d_ss+(size_t)pos*hd_s, d_alg, d_ho, ast_uq);
            launch_argmax(d_alg,V,d_am); CK(cudaMemcpy(&dt,d_am,4,cudaMemcpyDeviceToHost));
            chunk[j+1]=dt;
        }
        // verify: one-slot concat (same call as the spec daemon), all K+1 rows' logits + prenorm
        int Nb=K+1, pa=pos, sb=b;
        CK(cudaMemcpy(d_tv,chunk,(size_t)(K+1)*4,cudaMemcpyHostToDevice));
        gemma4_verify_concat(c,w,*s,1,&sb,&Nb,&pa,d_tv,d_cf,d_sf,d_cs,d_ss,d_lgv,uq,d_pnv);
        for(int i=0;i<=K;i++) launch_argmax(d_lgv+(size_t)i*V,V,d_am+i);
        CK(cudaMemcpy(am,d_am,(size_t)(K+1)*4,cudaMemcpyDeviceToHost));
        int m=0; for(int i=1;i<=K;i++){ if(am[i-1]==chunk[i]) m++; else break; }
        out->n=0; for(int i=1;i<=m;i++) out->tokens[out->n++]=chunk[i];
        out->tokens[out->n++]=am[m];
        out->drafted=K; out->accepted=m; out->k=K;
        CK(cudaMemcpy(d_h+(size_t)b*H, d_pnv+(size_t)m*H, (size_t)H*4, cudaMemcpyDeviceToDevice));   // next h = prenorm row m
        h_valid[b]=1;
    }
    void load_assistant(const std::string& AW, int Bmax){
        ca=load_config_assistant((AW+"/manifest.json").c_str());
        if(ca.vocab_size!=V || ca.backbone!=c.hidden_size){ fprintf(stderr,"[online] target/assistant shape mismatch\n"); exit(1); }
        if(const char* e=knob_raw("G4_SPEC_DRAFT_UQ")) ast_uq=atoi(e);
        for(int l=0;l<c.num_layers;l++){ if(c.layer_is_full&&c.layer_is_full[l]) lf=l; else ls=l; }
        wa=load_weights_assistant(AW.c_str(),ca,ast_uq);
        const int H=c.hidden_size;
        CK(cudaMalloc(&d_h,(size_t)Bmax*H*4)); CK(cudaMalloc(&d_in,(size_t)2*H*4)); CK(cudaMalloc(&d_alg,(size_t)V*4));
        CK(cudaMalloc(&d_ho,(size_t)H*4)); CK(cudaMalloc(&d_lgv,(size_t)(SPEC_KMAX+1)*V*4)); CK(cudaMalloc(&d_pnv,(size_t)(SPEC_KMAX+1)*H*4));
        CK(cudaMalloc(&d_tv,(SPEC_KMAX+1)*4)); CK(cudaMalloc(&d_am,(SPEC_KMAX+1)*4));
        h_valid.assign(Bmax,0); has_ast=true;
    }
};

int main(int argc,char**argv){
    knobs_init("serve_daemon_gemma4_online");
    std::string WD = argc>1?argv[1]:"artifacts/gemma4-31b/weights";
    int uq = argc>2?atoi(argv[2]):2, maxlen = argc>3?atoi(argv[3]):384, Bmax = argc>4?atoi(argv[4]):8;
    const std::string AWD = argc>5?argv[5]:"";
    Gemma4OnlineBackend be; be.uq=uq; be.maxlen_=maxlen;
    be.c = load_config_gemma4((WD+"/manifest.json").c_str());
    be.V = be.c.vocab_size;
    be.moe = be.c.num_experts>0 ? 1 : 0;   // 26B-A4B(E=128) → dense∥MoE 경로 자동 분기
    const int hd_s=be.c.head_dim, hd_f=be.c.global_head_dim?be.c.global_head_dim:be.c.head_dim;
    CK(cudaMalloc(&be.d_cf,(size_t)maxlen*hd_f*4)); CK(cudaMalloc(&be.d_sf,(size_t)maxlen*hd_f*4));
    CK(cudaMalloc(&be.d_cs,(size_t)maxlen*hd_s*4)); CK(cudaMalloc(&be.d_ss,(size_t)maxlen*hd_s*4));
    // ★rope 테이블 = manifest 권위 + partial proportional(full 0.25 = 64쌍만 회전 — runtime/rope_gemma4.h,
    //   절대 게이트 tests/test_gemma4_rope_table.py). 하드코딩 전례 2건(theta 1e4·partial 누락)이 긴 생성
    //   붕괴 원인이었음 — spec==plain 상대 게이트는 공유 오류를 못 잡음.
    if(gemma4_gen_rope_tables((WD+"/manifest.json").c_str(), maxlen, hd_f,be.d_cf,be.d_sf, hd_s,be.d_cs,be.d_ss)!=0){
        fprintf(stderr,"[online] manifest 없음: %s\n",(WD+"/manifest.json").c_str()); return 1; }
    CK(cudaDeviceSynchronize());
    be.w = load_weights_gemma4(WD.c_str(), be.c, uq!=0?1:0, uq==0?1:0);
    CK(cudaMalloc(&be.d_tok,Bmax*4)); CK(cudaMalloc(&be.d_pos,Bmax*4)); CK(cudaMalloc(&be.d_pf,(size_t)maxlen*4));
    CK(cudaMalloc(&be.d_logit,(size_t)Bmax*be.V*4));
    if(!AWD.empty() && !be.moe){ be.load_assistant(AWD,Bmax);
        fprintf(stderr,"[online] drafter loaded: %s uq=%d (spec path needs G4_SPEC_SERVE=1)\n",AWD.c_str(),be.ast_uq); }
    fprintf(stderr,"[online] ready: gemma4%s uq=%d maxlen=%d Bmax=%d V=%d (연속 배칭 greedy, 태그드 ID-in/out)\n", be.moe?"-MoE(26B)":"", uq,maxlen,Bmax,be.V); fflush(stderr);

    run_online_daemon(be, Bmax, parse_stop_ids_env());      // 모델-중립 루프(serving_online.h) + EOS 종료(STOP_IDS env)

    cudaFree(be.d_tok);cudaFree(be.d_pos);cudaFree(be.d_pf);cudaFree(be.d_logit);
    cudaFree(be.d_cf);cudaFree(be.d_sf);cudaFree(be.d_cs);cudaFree(be.d_ss);
    free_weights_gemma4(be.w); free_config_gemma4(be.c);
    return 0;
}
