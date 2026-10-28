#include "../runtime/decode_state_gemma4.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemm_batch.cuh"
#include "../kernels/attention_decode_split.cuh"
#include "../kernels/gemv_w4a8.cuh"   // W4A8 batch: B≤8 SB 커널(27b 축1 이식)
#include "../kernels/gemm_mmq.cuh"    // W4A8 batch: BM=16 split-K(B≥14 opt-in)
#include "../kernels/quantize_x.cuh"  // W4A8: 활성화 int8 양자화
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/gemv_f32.cuh"      // 26B MoE: router scores(fp32 gemv)
#include "../kernels/router_topk.cuh"   // 26B MoE: softmax→top8→renorm + per_expert_scale
#include "../kernels/moe_fused.cuh"     // 26B MoE: grouped gemv(batch experts — M_e<1도 fused, 35b 교훈)

#define BK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
static float *bdm(size_t n){ float*p; BK(cudaMalloc(&p,n*4)); return p; }

G4BatchState alloc_g4_batch_state(const ModelConfig &c, int B, int max_len){
    G4BatchState s{}; s.B=B; s.max_len=max_len; s.n_past=0; s.L=c.num_layers;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    s.kv_cap=(int*)calloc(c.num_layers,sizeof(int));
    s.Kc=(float**)calloc(c.num_layers,sizeof(float*)); s.Vc=(float**)calloc(c.num_layers,sizeof(float*));
    for(int l=0;l<c.num_layers;l++){
        int full=c.layer_is_full?c.layer_is_full[l]:1;
        int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        s.kv_cap[l]=max_len;   // 선형 캡(A-2 안전판). 링버퍼는 별도.
        s.Kc[l]=bdm((size_t)B*max_len*nkv*hd); s.Vc[l]=bdm((size_t)B*max_len*nkv*hd);
    }
    return s;
}
void free_g4_batch_state(G4BatchState &s){
    for(int l=0;l<s.L;l++){ if(s.Kc[l])cudaFree(s.Kc[l]); if(s.Vc[l])cudaFree(s.Vc[l]); }
    free(s.Kc); free(s.Vc); free(s.kv_cap); s.Kc=s.Vc=nullptr; s.kv_cap=nullptr;
}

// batch matmul: X[B,IN] × W[OUT,IN] → Y[B,OUT].
//   use_q4==2 (W4A8): 활성화 int8 양자화 + B-gating(27b qwen35_batch 축1 결정표 이식, qwen과 동일 커널).
//     B≤8 → gemv_w4a8_sb(SMAX dispatch, 1적재) / B≥14 opt-in split-K(BM16, borderline flip=스킴 변경) / else gemm_batch_w4a8.
//     ★bit-identical은 같은 SB family 내(SMAX 버킷 동일)서 성립 — 버킷/커널 경계 넘으면 W4A8 스킴 변경(flip 가능).
//   use_q4!=2 (q4): gemm_batch_q4(B-무관 canonical, 기존 경로 무변경).
// G4_ATTN_SPLIT=1: dense decode-type attention (S=1, verify, concat, batch) uses the split-KV kernel.
static bool g4_attn_split(){ static const bool on=[](){ const char* v=getenv("G4_ATTN_SPLIT"); return v && strcmp(v,"1")==0; }(); return on; }
static int gb_sdtype, gb_use_q4;
static int8_t *gb_xq=nullptr; static float *gb_xs=nullptr; static int32_t *gb_xsum=nullptr; static size_t gb_xcap=0;
static inline void mmB(const QMat &m, const float *X, float *Y, int IN, int OUT, int B){
    if(gb_use_q4==2){
        size_t n=(size_t)B*IN;
        if(gb_xcap<n){ if(gb_xq)cudaFree(gb_xq); if(gb_xs)cudaFree(gb_xs); if(gb_xsum)cudaFree(gb_xsum);
            BK(cudaMalloc(&gb_xq,n)); BK(cudaMalloc(&gb_xs,(n/QBLK)*4)); BK(cudaMalloc(&gb_xsum,(n/QBLK)*4)); gb_xcap=n; }
        launch_quantize_x_q8(X,gb_xq,gb_xs,gb_xsum,IN,B);          // B행 X 양자화(1 launch)
        static int USE_BM16 = getenv("BM16") ? atoi(getenv("BM16")) : 0;
        if(B<=8)                     launch_gemv_w4a8_sb(gb_xq,gb_xs,m.wq,m.sc,Y,IN,OUT,B,gb_sdtype);
        else if(USE_BM16 && B>=14)   launch_gemm_bm16_splitk(gb_xq,gb_xs,m.wq,m.sc,Y,B,OUT,IN,gb_sdtype);  // opt-in(스킴 변경)
        else if(B<=15)               launch_gemv_w4a8_sb(gb_xq,gb_xs,m.wq,m.sc,Y,IN,OUT,B,gb_sdtype);
        else                         launch_gemm_batch_w4a8(gb_xq,gb_xs,m.wq,m.sc,Y,IN,OUT,B,gb_sdtype);   // B=16 특수화
    } else {
        launch_gemm_batch_q4(X, m.wq, m.sc, Y, IN, OUT, B, gb_sdtype);
    }
}

// 26B MoE batch scratch (persistent static, grow-on-demand — gemma4_moe.cu g4m_* 패턴, 매 호출 덮어써짐=수학 불변).
static float *gbm_scores=nullptr,*gbm_dw=nullptr,*gbm_fgu=nullptr,*gbm_fhid=nullptr,*gbm_fsout=nullptr; static int *gbm_didx=nullptr;
static size_t cb_scores=0,cb_dw=0,cb_fgu=0,cb_fhid=0,cb_fsout=0,cb_didx=0;
static inline float* gbm_grow(float*&p,size_t&cap,size_t n){ if(cap<n){ if(p)cudaFree(p); BK(cudaMalloc(&p,n*4)); cap=n; } return p; }
static inline int*   gbm_growi(int*&p,size_t&cap,size_t n){ if(cap<n){ if(p)cudaFree(p); BK(cudaMalloc(&p,n*4)); cap=n; } return p; }

void gemma4_decode_batch(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                         const int *d_tokens, const float *d_cos_f, const float *d_sin_f,
                         const float *d_cos_s, const float *d_sin_s,
                         float *d_logits, int commit, int use_q4,
                         const int *d_pos_b, const int *h_pos_b) {
    gb_sdtype=c.scale_dtype; gb_use_q4=use_q4;
    const int B=st.B, H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, NQ=c.num_q_heads;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const float eps=c.rms_norm_eps, ascale=c.attn_scaling;
    const bool ragged = (d_pos_b != nullptr);
    const int base = st.n_past;                       // same-length 공통 위치
    int base_max = base;                              // ragged: attn shmem 크기용 max(pos_b)+...
    if(ragged && h_pos_b) for(int b=0;b<B;b++) base_max = h_pos_b[b]>base_max ? h_pos_b[b] : base_max;
    const size_t BH=(size_t)B*H;
    const int hd_max=hd_f>hd_s?hd_f:hd_s;
    const size_t qmax=(size_t)B*NQ*hd_max;
    if(g4_attn_split()) attn_split_reserve(B,NQ,st.max_len,hd_max);
    const size_t kvmax=(size_t)B*(nkv_s*hd_s>nkv_f*hd_f?nkv_s*hd_s:nkv_f*hd_f);

    // Like the existing matmul globals: serialized, single-device, non-reentrant calls.
    // Each buffer grows independently; CUDA context/process teardown reclaims storage.
    static float *scratch[16]={};
    static size_t capacity[16]={};
    float *x=gbm_grow(scratch[0],capacity[0],BH);
    float *norm=gbm_grow(scratch[1],capacity[1],BH);
    float *tmp=gbm_grow(scratch[2],capacity[2],BH);
    float *after=gbm_grow(scratch[3],capacity[3],BH);
    float *qb=gbm_grow(scratch[4],capacity[4],qmax);
    float *qn=gbm_grow(scratch[5],capacity[5],qmax);
    float *kb=gbm_grow(scratch[6],capacity[6],kvmax);
    float *kn=gbm_grow(scratch[7],capacity[7],kvmax);
    float *vb=gbm_grow(scratch[8],capacity[8],kvmax);
    float *vn=gbm_grow(scratch[9],capacity[9],kvmax);
    float *attn=gbm_grow(scratch[10],capacity[10],qmax);
    float *ob=gbm_grow(scratch[11],capacity[11],BH);
    float *g=gbm_grow(scratch[12],capacity[12],(size_t)B*FFN);
    float *u=gbm_grow(scratch[13],capacity[13],(size_t)B*FFN);
    float *act=gbm_grow(scratch[14],capacity[14],(size_t)B*FFN);
    float *dn=gbm_grow(scratch[15],capacity[15],BH);

    launch_embedding(d_tokens, w.embed, x, B, H);
    if(c.embed_scale>0.f) launch_scale(x,x,c.embed_scale,BH);

    for(int l=0;l<c.num_layers;l++){
        const G4Layer &L=w.layers[l];
        const int full=L.is_full;
        const int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        const int qdim=NQ*hd, kvdim=nkv*hd;
        const float *cosT=full?d_cos_f:d_cos_s, *sinT=full?d_sin_f:d_sin_s;
        const int window=full?0:c.sliding_window;
        const long kv_stride=(long)st.max_len*kvdim;

        launch_rmsnorm(x, L.input_ln, norm, B, H, eps);
        mmB(L.q, norm, qb, H, qdim, B);
        launch_rmsnorm(qb, L.q_norm, qn, B*NQ, hd, eps);
        // rope: ragged=슬롯별 위치(rope_partial_pos, 전체 테이블), same-length=공통 base(rope_partial_batch). rot=hd(full).
        if(ragged) launch_rope_partial_pos(qn, cosT, sinT, d_pos_b, B, NQ, hd, hd);
        else       launch_rope_partial_batch(qn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, B, NQ, hd, hd);
        mmB(L.k, norm, kb, H, kvdim, B);
        if(full && c.attention_k_eq_v) launch_rmsnorm_noscale(kb, vn, B*nkv, hd, eps);   // V=v_norm(k_proj)
        else { mmB(L.v, norm, vb, H, kvdim, B); launch_rmsnorm_noscale(vb, vn, B*nkv, hd, eps); }
        launch_rmsnorm(kb, L.k_norm, kn, B*nkv, hd, eps);
        if(ragged) launch_rope_partial_pos(kn, cosT, sinT, d_pos_b, B, nkv, hd, hd);
        else       launch_rope_partial_batch(kn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, B, nkv, hd, hd);
        // KV append: 슬롯 b의 위치(ragged=h_pos_b[b], same=base). Kc[l] 레이아웃 [B, max_len, NKV, hd].
        for(int b=0;b<B;b++){
            size_t pos = ragged ? (size_t)h_pos_b[b] : (size_t)base;
            BK(cudaMemcpy(st.Kc[l]+(size_t)b*kv_stride+pos*kvdim, kn+(size_t)b*kvdim, (size_t)kvdim*4, cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy(st.Vc[l]+(size_t)b*kv_stride+pos*kvdim, vn+(size_t)b*kvdim, (size_t)kvdim*4, cudaMemcpyDeviceToDevice));
        }
        if(g4_attn_split())
            launch_attn_decode_split(qn, st.Kc[l], st.Vc[l], attn, B, NQ, nkv, hd, base, 0, ragged?d_pos_b:nullptr, base_max,
                                     kv_stride, attn_split_ns_cap(st.max_len), 0, ascale, window);
        else
            launch_attn_decode_batch(qn, st.Kc[l], st.Vc[l], attn, B, NQ, nkv, hd, base_max, kv_stride, 0, ascale, window, ragged?d_pos_b:nullptr);
        mmB(L.o, attn, ob, qdim, H, B);
        launch_rmsnorm(ob, L.post_attn_ln, tmp, B, H, eps);
        launch_add(after, x, tmp, (int)BH);

        launch_rmsnorm(after, L.pre_ffn_ln, norm, B, H, eps);
        mmB(L.gate, norm, g, H, FFN, B);
        mmB(L.up,   norm, u, H, FFN, B);
        launch_geglu(g, u, act, (size_t)B*FFN);
        mmB(L.down, act, dn, FFN, H, B);
        launch_rmsnorm(dn, L.post_ffn_ln, tmp, B, H, eps);
        launch_add(x, after, tmp, (int)BH);
        if(L.layer_scalar!=1.0f) launch_scale(x, x, L.layer_scalar, BH);
    }

    launch_rmsnorm(x, w.final_norm, norm, B, H, eps);
    mmB(w.lm_head, norm, d_logits, H, V, B);
    if(c.final_logit_softcapping>0.f) launch_softcap(d_logits, d_logits, c.final_logit_softcapping, (size_t)B*V);
    if(commit && !ragged) st.n_past += 1;   // ragged면 per-slot 위치는 호출자(scheduler) 관리
    BK(cudaDeviceSynchronize());
}

// 26B-A4B MoE batch decode — gemma4_decode_batch(31B batch KV attn + dense mmB) + dense∥MoE MLP(B tokens).
//   ★batch 커널 measured(35b 교훈): dense=mmB(gemm_batch 분할상환), MoE experts=moe_fused(grouped gemv — M_e<1도 fused).
//   신규 커널 0(attn_decode_batch·router_topk·per_expert_scale·moe_fused·geglu 재사용). ragged/same-length. batch==solo bit-id(스킴 내).
void gemma4_moe_decode_batch(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                             const int *d_tokens, const float *d_cos_f, const float *d_sin_f,
                             const float *d_cos_s, const float *d_sin_s,
                             float *d_logits, int commit, int use_q4,
                             const int *d_pos_b, const int *h_pos_b) {
    gb_sdtype=c.scale_dtype; gb_use_q4=use_q4;
    const int B=st.B, H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, NQ=c.num_q_heads;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const int E=c.num_experts, K=c.num_experts_per_tok, MI=c.moe_intermediate_size;
    const float eps=c.rms_norm_eps, ascale=c.attn_scaling, root=powf((float)H,-0.5f);
    const long gu_w=(long)2*MI*H, dn_w=(long)H*MI;
    const bool ragged = (d_pos_b != nullptr);
    const int base = st.n_past;
    int base_max = base;
    if(ragged && h_pos_b) for(int b=0;b<B;b++) base_max = h_pos_b[b]>base_max ? h_pos_b[b] : base_max;
    const size_t BH=(size_t)B*H;
    const int hd_max=hd_f>hd_s?hd_f:hd_s;
    const size_t qmax=(size_t)B*NQ*hd_max;
    const size_t kvmax=(size_t)B*(nkv_s*hd_s>nkv_f*hd_f?nkv_s*hd_s:nkv_f*hd_f);

    // Like the existing matmul globals: serialized, single-device, non-reentrant calls.
    // Each buffer grows independently; CUDA context/process teardown reclaims storage.
    static float *scratch[22]={};
    static size_t capacity[22]={};
    float *x=gbm_grow(scratch[0],capacity[0],BH);
    float *norm=gbm_grow(scratch[1],capacity[1],BH);
    float *tmp=gbm_grow(scratch[2],capacity[2],BH);
    float *after=gbm_grow(scratch[3],capacity[3],BH);
    float *qb=gbm_grow(scratch[4],capacity[4],qmax);
    float *qn=gbm_grow(scratch[5],capacity[5],qmax);
    float *kb=gbm_grow(scratch[6],capacity[6],kvmax);
    float *kn=gbm_grow(scratch[7],capacity[7],kvmax);
    float *vb=gbm_grow(scratch[8],capacity[8],kvmax);
    float *vn=gbm_grow(scratch[9],capacity[9],kvmax);
    float *attn=gbm_grow(scratch[10],capacity[10],qmax);
    float *ob=gbm_grow(scratch[11],capacity[11],BH);
    float *g=gbm_grow(scratch[12],capacity[12],(size_t)B*FFN);
    float *u=gbm_grow(scratch[13],capacity[13],(size_t)B*FFN);
    float *act=gbm_grow(scratch[14],capacity[14],(size_t)B*FFN);
    float *dn=gbm_grow(scratch[15],capacity[15],BH);
    float *h1=gbm_grow(scratch[16],capacity[16],BH);
    float *h2=gbm_grow(scratch[17],capacity[17],BH);
    float *moe_in=gbm_grow(scratch[18],capacity[18],BH);
    float *moe_out=gbm_grow(scratch[19],capacity[19],BH);
    float *comb=gbm_grow(scratch[20],capacity[20],BH);
    float *rnorm=gbm_grow(scratch[21],capacity[21],BH);

    launch_embedding(d_tokens, w.embed, x, B, H);
    if(c.embed_scale>0.f) launch_scale(x,x,c.embed_scale,BH);

    for(int l=0;l<c.num_layers;l++){
        const G4Layer &L=w.layers[l];
        const int full=L.is_full;
        const int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        const int qdim=NQ*hd, kvdim=nkv*hd;
        const float *cosT=full?d_cos_f:d_cos_s, *sinT=full?d_sin_f:d_sin_s;
        const int window=full?0:c.sliding_window;
        const long kv_stride=(long)st.max_len*kvdim;

        // --- attention block (batch KV-cached — gemma4_decode_batch와 동일) ---
        launch_rmsnorm(x, L.input_ln, norm, B, H, eps);
        mmB(L.q, norm, qb, H, qdim, B);
        launch_rmsnorm(qb, L.q_norm, qn, B*NQ, hd, eps);
        if(ragged) launch_rope_partial_pos(qn, cosT, sinT, d_pos_b, B, NQ, hd, hd);
        else       launch_rope_partial_batch(qn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, B, NQ, hd, hd);
        mmB(L.k, norm, kb, H, kvdim, B);
        if(full && c.attention_k_eq_v) launch_rmsnorm_noscale(kb, vn, B*nkv, hd, eps);
        else { mmB(L.v, norm, vb, H, kvdim, B); launch_rmsnorm_noscale(vb, vn, B*nkv, hd, eps); }
        launch_rmsnorm(kb, L.k_norm, kn, B*nkv, hd, eps);
        if(ragged) launch_rope_partial_pos(kn, cosT, sinT, d_pos_b, B, nkv, hd, hd);
        else       launch_rope_partial_batch(kn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, B, nkv, hd, hd);
        for(int b=0;b<B;b++){
            size_t pos = ragged ? (size_t)h_pos_b[b] : (size_t)base;
            BK(cudaMemcpy(st.Kc[l]+(size_t)b*kv_stride+pos*kvdim, kn+(size_t)b*kvdim, (size_t)kvdim*4, cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy(st.Vc[l]+(size_t)b*kv_stride+pos*kvdim, vn+(size_t)b*kvdim, (size_t)kvdim*4, cudaMemcpyDeviceToDevice));
        }
        launch_attn_decode_batch(qn, st.Kc[l], st.Vc[l], attn, B, NQ, nkv, hd, base_max, kv_stride, 0, ascale, window, ragged?d_pos_b:nullptr);
        mmB(L.o, attn, ob, qdim, H, B);
        launch_rmsnorm(ob, L.post_attn_ln, tmp, B, H, eps);
        launch_add(after, x, tmp, (int)BH);                          // after = residual (attn 후)

        // --- dense GeGLU MLP (mmB — 분할상환) ---
        launch_rmsnorm(after, L.pre_ffn_ln, norm, B, H, eps);
        mmB(L.gate, norm, g, H, FFN, B);
        mmB(L.up,   norm, u, H, FFN, B);
        launch_geglu(g, u, act, (size_t)B*FFN);
        mmB(L.down, act, dn, FFN, H, B);

        if(L.is_moe){
            launch_rmsnorm(dn, L.post_ff_ln_1, h1, B, H, eps);        // h1 = post_ff_ln_1(dense)
            // router(rin=after=residual)
            launch_rmsnorm(after, L.router_scale, rnorm, B, H, eps);
            launch_scale(rnorm, rnorm, root, BH);
            float *scores=gbm_grow(gbm_scores,cb_scores,(size_t)B*E);
            float *dw    =gbm_grow(gbm_dw,cb_dw,(size_t)B*K);
            int   *didx  =gbm_growi(gbm_didx,cb_didx,(size_t)B*K);
            launch_gemv_f32(rnorm, L.router_proj, scores, H, E, B);
            launch_router_topk(scores, didx, dw, B, E, K);
            launch_apply_per_expert_scale(dw, L.per_expert_scale, didx, B, K);
            // experts(pre_ff_ln_2(rin)) — moe_fused grouped(nslot=B*K, M_e<1도 fused)
            launch_rmsnorm(after, L.pre_ff_ln_2, moe_in, B, H, eps);
            long nslot=(long)B*K;
            float *fgu  =gbm_grow(gbm_fgu,cb_fgu,nslot*(size_t)2*MI);
            float *fhid =gbm_grow(gbm_fhid,cb_fhid,nslot*(size_t)MI);
            float *fsout=gbm_grow(gbm_fsout,cb_fsout,nslot*(size_t)H);
            launch_moe_fused_gu(moe_in, L.experts_gate_up.wq, L.experts_gate_up.sc, didx, fgu, H, 2*MI, K, nslot, gu_w, gb_sdtype);
            for(long slot=0;slot<nslot;slot++)
                launch_geglu(fgu+slot*(2L*MI), fgu+slot*(2L*MI)+MI, fhid+slot*MI, MI);
            launch_moe_fused_down(fhid, L.experts_down.wq, L.experts_down.sc, didx, dw, fsout, MI, H, K, nslot, dn_w, gb_sdtype);
            launch_moe_fused_reduce(fsout, moe_out, H, K, B);
            launch_rmsnorm(moe_out, L.post_ff_ln_2, h2, B, H, eps);   // h2 = post_ff_ln_2(moe)
            launch_add(comb, h1, h2, (int)BH);
            launch_rmsnorm(comb, L.post_ffn_ln, tmp, B, H, eps);
        } else {
            launch_rmsnorm(dn, L.post_ffn_ln, tmp, B, H, eps);
        }
        launch_add(x, after, tmp, (int)BH);
        if(L.layer_scalar!=1.0f) launch_scale(x, x, L.layer_scalar, BH);
    }

    launch_rmsnorm(x, w.final_norm, norm, B, H, eps);
    mmB(w.lm_head, norm, d_logits, H, V, B);
    if(c.final_logit_softcapping>0.f) launch_softcap(d_logits, d_logits, c.final_logit_softcapping, (size_t)B*V);
    if(commit && !ragged) st.n_past += 1;    // ragged면 per-slot 위치는 호출자 관리
    BK(cudaDeviceSynchronize());
}

// ---- chunked prefill (슬롯-b 뷰): 검증필 gemma4_decode(Sq=P) 재사용 ----
//   슬롯 b의 KV 서브영역(Kc[l]+b*stride_l)을 단일-시퀀스 뷰로 노출 → gemma4_decode가 위치 0..P-1을 그 영역에 append.
//   레이아웃 [B,max_len,nkv,hd]에서 슬롯 b의 [max_len,nkv,hd]이 곧 G4DecodeState.Kc[l] → 오프셋 정합(위 append와 동일).
void gemma4_prefill_batch_slot(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                               int b, const int *d_tokens, int P,
                               const float *d_cos_f, const float *d_sin_f,
                               const float *d_cos_s, const float *d_sin_s,
                               float *d_logits, int use_q4){
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    G4DecodeState view{}; view.n_past=0; view.max_len=st.max_len; view.L=st.L; view.kv_cap=st.kv_cap;
    std::vector<float*> Kc(st.L), Vc(st.L);
    for(int l=0;l<st.L;l++){
        int full=c.layer_is_full?c.layer_is_full[l]:1;
        int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        size_t stride=(size_t)st.max_len*nkv*hd;
        Kc[l]=st.Kc[l]+(size_t)b*stride; Vc[l]=st.Vc[l]+(size_t)b*stride;   // 슬롯 b 서브영역
    }
    view.Kc=Kc.data(); view.Vc=Vc.data();
    // commit=0(뷰의 n_past 불필요 — 스케줄러가 위치 관리), last_only=1(마지막 로짓만; 서빙은 discard),
    // amort_mm=1(프리필 SB 커널로 weight amortize — 비-amortized plain gemv의 net-loss 해소, bit-id 유지)
    gemma4_decode(c,w,view,d_tokens,P,d_cos_f,d_sin_f,d_cos_s,d_sin_s,d_logits,0,use_q4,1,1);
}

// ---- spec chain verify (슬롯-b 뷰): 검증필 gemma4_decode(Sq=Nb, commit=0, 전 위치 로짓) 재사용 ----
//   prefill_batch_slot과 동일한 뷰 트릭. amort=0 = plain 1토큰 decode와 동일 커널 선택(전 S 행-독립)
//   → verify 로짓 == plain 로짓 bit-id가 구성적 목표(게이트 test_gemma4_verify가 Δ=0 실측).
//   per-slot 순차 호출(정확성 canonical) — concat 분할상환은 후속 perf lever(27b 축1 패턴).
void gemma4_verify_batch(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                         int B_act, const int *slot_ids, const int *Nb, const int *pos_b,
                         const int *d_tokens_cat,
                         const float *d_cos_f, const float *d_sin_f,
                         const float *d_cos_s, const float *d_sin_s,
                         float *d_logits_cat, int use_q4){
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const int V=c.vocab_size;
    long off=0;
    for(int a=0;a<B_act;a++){
        const int b=slot_ids[a], Sq=Nb[a];
        if(pos_b[a]+Sq>st.max_len){ fprintf(stderr,"[gemma4_verify_batch] slot %d pos %d+%d > max_len %d — KV 예산 초과, 중단.\n",b,pos_b[a],Sq,st.max_len); exit(1); }
        G4DecodeState view{}; view.n_past=pos_b[a]; view.max_len=st.max_len; view.L=st.L; view.kv_cap=st.kv_cap;
        std::vector<float*> Kc(st.L), Vc(st.L);
        for(int l=0;l<st.L;l++){
            int full=c.layer_is_full?c.layer_is_full[l]:1;
            int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
            size_t stride=(size_t)st.max_len*nkv*hd;
            Kc[l]=st.Kc[l]+(size_t)b*stride; Vc[l]=st.Vc[l]+(size_t)b*stride;   // 슬롯 b 서브영역
        }
        view.Kc=Kc.data(); view.Vc=Vc.data();
        gemma4_decode(c,w,view,d_tokens_cat+off,Sq,d_cos_f,d_sin_f,d_cos_s,d_sin_s,
                      d_logits_cat+off*V,0,use_q4,0,0);   // commit=0·last_only=0(전 위치)·amort=0(plain 동일 경로)
        off+=Sq;
    }
}

// ---- spec verify concat 분할상환 (27b 축1 패턴 — 헤더 주석 참조) ----
//   큰 matmul만 M=ΣNb 행 concat(SB ≤16행 그룹 = gemma4_decode amort_mm 동일 분해·스킴 경계 회피),
//   rope=행별 위치(rope_partial_pos)·attention/KV append=슬롯별 기존 커널(단독 경로와 동일 수학).
static inline void mmM(const QMat &m, const float *X, float *Y, int IN, int OUT, int M){
    if(gb_use_q4==2){
        size_t n=(size_t)M*IN;
        if(gb_xcap<n){ if(gb_xq)cudaFree(gb_xq); if(gb_xs)cudaFree(gb_xs); if(gb_xsum)cudaFree(gb_xsum);
            BK(cudaMalloc(&gb_xq,n)); BK(cudaMalloc(&gb_xs,(n/QBLK)*4)); BK(cudaMalloc(&gb_xsum,(n/QBLK)*4)); gb_xcap=n; }
        launch_quantize_x_q8(X,gb_xq,gb_xs,gb_xsum,IN,M);
        for(int s0=0;s0<M;s0+=16){ int sq=(M-s0<16)?(M-s0):16;   // ★항상 SB family(≤16행 그룹) — mmB의 B16 gemm_batch 전환 안 탐
            launch_gemv_w4a8_sb(gb_xq+(size_t)s0*IN, gb_xs+(size_t)s0*(IN/QBLK), m.wq, m.sc, Y+(size_t)s0*OUT, IN, OUT, sq, gb_sdtype); }
    } else if(gb_use_q4){
        fprintf(stderr,"[gemma4_verify_concat] uq=1(tiled)은 M-종속 타일=스킴 위험 — per-slot verify_batch 사용. 중단.\n"); exit(1);
    } else launch_gemv_f32(X,m.wf,Y,IN,OUT,M);
}

void gemma4_verify_concat(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                          int B_act, const int *slot_ids, const int *Nb, const int *pos_b,
                          const int *d_tokens_cat,
                          const float *d_cos_f, const float *d_sin_f,
                          const float *d_cos_s, const float *d_sin_s,
                          float *d_logits_cat, int use_q4, float *d_prenorm_cat){
    gb_sdtype=c.scale_dtype; gb_use_q4=use_q4;
    const int H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, NQ=c.num_q_heads;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const float eps=c.rms_norm_eps, ascale=c.attn_scaling;

    int M=0; for(int a=0;a<B_act;a++) M+=Nb[a];
    std::vector<int> off(B_act); std::vector<int> pos_row(M);
    for(int a=0,o=0;a<B_act;a++){ off[a]=o;
        if(pos_b[a]+Nb[a]>st.max_len){ fprintf(stderr,"[gemma4_verify_concat] slot %d pos %d+%d > max_len %d — KV 예산 초과, 중단.\n",slot_ids[a],pos_b[a],Nb[a],st.max_len); exit(1); }
        for(int j=0;j<Nb[a];j++) pos_row[o+j]=pos_b[a]+j;
        o+=Nb[a]; }
    // Same serialized, single-device lifetime as the float scratch below.
    static int *d_pos_row=nullptr;
    static size_t pos_capacity=0;
    gbm_growi(d_pos_row,pos_capacity,(size_t)M);
    BK(cudaMemcpy(d_pos_row,pos_row.data(),(size_t)M*4,cudaMemcpyHostToDevice));

    const size_t MH=(size_t)M*H;
    const int hd_max=hd_f>hd_s?hd_f:hd_s;
    const size_t qmax=(size_t)M*NQ*hd_max;
    if(g4_attn_split()) attn_split_reserve(M,NQ,st.max_len,hd_max);
    const size_t kvmax=(size_t)M*(nkv_s*hd_s>nkv_f*hd_f?nkv_s*hd_s:nkv_f*hd_f);
    // Like the existing matmul globals: serialized, single-device, non-reentrant calls.
    // Each buffer grows independently; CUDA context/process teardown reclaims storage.
    static float *scratch[16]={};
    static size_t capacity[16]={};
    float *x=gbm_grow(scratch[0],capacity[0],MH);
    float *norm=gbm_grow(scratch[1],capacity[1],MH);
    float *tmp=gbm_grow(scratch[2],capacity[2],MH);
    float *after=gbm_grow(scratch[3],capacity[3],MH);
    float *qb=gbm_grow(scratch[4],capacity[4],qmax);
    float *qn=gbm_grow(scratch[5],capacity[5],qmax);
    float *kb=gbm_grow(scratch[6],capacity[6],kvmax);
    float *kn=gbm_grow(scratch[7],capacity[7],kvmax);
    float *vb=gbm_grow(scratch[8],capacity[8],kvmax);
    float *vn=gbm_grow(scratch[9],capacity[9],kvmax);
    float *attn=gbm_grow(scratch[10],capacity[10],qmax);
    float *ob=gbm_grow(scratch[11],capacity[11],MH);
    float *g=gbm_grow(scratch[12],capacity[12],(size_t)M*FFN);
    float *u=gbm_grow(scratch[13],capacity[13],(size_t)M*FFN);
    float *act=gbm_grow(scratch[14],capacity[14],(size_t)M*FFN);
    float *dn=gbm_grow(scratch[15],capacity[15],MH);

    launch_embedding(d_tokens_cat, w.embed, x, M, H);
    if(c.embed_scale>0.f) launch_scale(x,x,c.embed_scale,MH);

    for(int l=0;l<c.num_layers;l++){
        const G4Layer &L=w.layers[l];
        const int full=L.is_full;
        const int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        const int qdim=NQ*hd, kvdim=nkv*hd;
        const float *cosT=full?d_cos_f:d_cos_s, *sinT=full?d_sin_f:d_sin_s;
        const int window=full?0:c.sliding_window;
        const long kv_stride=(long)st.max_len*kvdim;

        launch_rmsnorm(x, L.input_ln, norm, M, H, eps);
        mmM(L.q, norm, qb, H, qdim, M);
        launch_rmsnorm(qb, L.q_norm, qn, M*NQ, hd, eps);
        launch_rope_partial_pos(qn, cosT, sinT, d_pos_row, M, NQ, hd, hd);   // 행별 절대 위치(ragged 기게이트)
        mmM(L.k, norm, kb, H, kvdim, M);
        if(full && c.attention_k_eq_v) launch_rmsnorm_noscale(kb, vn, M*nkv, hd, eps);   // V=v_norm(k_proj), rope 전
        else { mmM(L.v, norm, vb, H, kvdim, M); launch_rmsnorm_noscale(vb, vn, M*nkv, hd, eps); }
        launch_rmsnorm(kb, L.k_norm, kn, M*nkv, hd, eps);
        launch_rope_partial_pos(kn, cosT, sinT, d_pos_row, M, nkv, hd, hd);
        // KV append + attention: 슬롯별(각자 KV 서브영역·위치) — 단독 verify와 문자 그대로 동일 커널/수학
        for(int a=0;a<B_act;a++){
            const int b=slot_ids[a];
            BK(cudaMemcpy(st.Kc[l]+(size_t)b*kv_stride+(size_t)pos_b[a]*kvdim, kn+(size_t)off[a]*kvdim, (size_t)Nb[a]*kvdim*4, cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy(st.Vc[l]+(size_t)b*kv_stride+(size_t)pos_b[a]*kvdim, vn+(size_t)off[a]*kvdim, (size_t)Nb[a]*kvdim*4, cudaMemcpyDeviceToDevice));
        }
        for(int a=0;a<B_act;a++){
            const int b=slot_ids[a];
            if(g4_attn_split())
                launch_attn_decode_split(qn+(size_t)off[a]*qdim, st.Kc[l]+(size_t)b*kv_stride, st.Vc[l]+(size_t)b*kv_stride,
                                         attn+(size_t)off[a]*qdim, Nb[a], NQ, nkv, hd, pos_b[a], 1, nullptr, pos_b[a]+Nb[a]-1, 0,
                                         attn_split_ns_cap(st.max_len), 0, ascale, window);
            else
                launch_attn_decode(qn+(size_t)off[a]*qdim, st.Kc[l]+(size_t)b*kv_stride, st.Vc[l]+(size_t)b*kv_stride,
                                   attn+(size_t)off[a]*qdim, Nb[a], NQ, nkv, hd, pos_b[a], 0, ascale, window);
        }
        mmM(L.o, attn, ob, qdim, H, M);
        launch_rmsnorm(ob, L.post_attn_ln, tmp, M, H, eps);
        launch_add(after, x, tmp, (int)MH);

        launch_rmsnorm(after, L.pre_ffn_ln, norm, M, H, eps);
        mmM(L.gate, norm, g, H, FFN, M);
        mmM(L.up,   norm, u, H, FFN, M);
        launch_geglu(g, u, act, (size_t)M*FFN);
        mmM(L.down, act, dn, FFN, H, M);
        launch_rmsnorm(dn, L.post_ffn_ln, tmp, M, H, eps);
        launch_add(x, after, tmp, (int)MH);
        if(L.layer_scalar!=1.0f) launch_scale(x, x, L.layer_scalar, MH);
    }

    if(d_prenorm_cat) BK(cudaMemcpy(d_prenorm_cat, x, MH*4, cudaMemcpyDeviceToDevice));   // EAGLE h 재귀용(행 off[a]+m)
    launch_rmsnorm(x, w.final_norm, norm, M, H, eps);
    mmM(w.lm_head, norm, d_logits_cat, H, V, M);
    if(c.final_logit_softcapping>0.f) launch_softcap(d_logits_cat, d_logits_cat, c.final_logit_softcapping, (size_t)M*V);
    BK(cudaDeviceSynchronize());
}
