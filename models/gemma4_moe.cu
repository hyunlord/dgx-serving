#include "gemma4_moe.h"
#include "../runtime/decode_state_gemma4.h"   // G4DecodeState (26B decode = 31B KV 구조 + dense∥MoE MLP)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_tiled.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/router_topk.cuh"
#include "../kernels/moe_fused.cuh"

#define GK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float *dmalloc(size_t n) { float *p; GK(cudaMalloc(&p, n * 4)); return p; }

// 행렬곱 dispatch (gemma4.cu / qwen35.cu 와 동일 규약: 0=fp32 / 1=q4 tiled / 2=W4A8) — 공용 커널만 호출.
static int8_t *g4m_xq=nullptr; static float *g4m_xs=nullptr; static int32_t *g4m_xsum=nullptr; static size_t g4m_xqcap=0;
static int g4m_sdtype;
static inline void mm(const QMat &m, const float *X, float *Y, int IN, int OUT, int S, int use_q4) {
    if (use_q4==2) {
        size_t n=(size_t)S*IN;
        if (g4m_xqcap < n) { if(g4m_xq)cudaFree(g4m_xq); if(g4m_xs)cudaFree(g4m_xs); if(g4m_xsum)cudaFree(g4m_xsum);
            GK(cudaMalloc(&g4m_xq,n)); GK(cudaMalloc(&g4m_xs,(n/QBLK)*4)); GK(cudaMalloc(&g4m_xsum,(n/QBLK)*4)); g4m_xqcap=n; }
        launch_quantize_x_q8(X, g4m_xq, g4m_xs, g4m_xsum, IN, S);
        launch_gemv_w4a8(g4m_xq, g4m_xs, g4m_xsum, m.wq, m.sc, Y, IN, OUT, S, g4m_sdtype);
    } else if (use_q4) launch_gemm_tiled_q4(X, m.wq, m.sc, Y, S, OUT, IN, g4m_sdtype);
    else               launch_gemv_f32(X, m.wf, Y, IN, OUT, S);
}

// MoE fused scratch (persistent static, grow-on-demand — moe_block.cuh 패턴: cudaFree API 오버헤드 회피).
//   각 버퍼는 매 호출 커널 출력으로 완전 덮어써진 뒤 읽힘 → 재사용해도 수학 불변.
static float *g4m_scores=nullptr,*g4m_dw=nullptr,*g4m_fgu=nullptr,*g4m_fhid=nullptr,*g4m_fsout=nullptr; static int *g4m_didx=nullptr;
static size_t c_scores=0,c_dw=0,c_fgu=0,c_fhid=0,c_fsout=0,c_didx=0;
static inline float* g4m_grow(float*&p,size_t&cap,size_t n){ if(cap<n){ if(p)cudaFree(p); GK(cudaMalloc(&p,n*4)); cap=n; } return p; }
static inline int*   g4m_growi(int*&p,size_t&cap,size_t n){ if(cap<n){ if(p)cudaFree(p); GK(cudaMalloc(&p,n*4)); cap=n; } return p; }

// Gemma4 26B-A4B MoE 프리필 forward: 31B dense graph(sandwich attn·dual-rope·k_eq_v·softcap) + dense∥MoE MLP.
//   MLP 위상(modeling Gemma4TextDecoderLayer.forward verbatim):
//     residual=after(attn 후); pre_ff_ln(residual)→dense GeGLU MLP→dn
//     h1=post_ff_ln_1(dn); rin=residual; router(rin); h2in=pre_ff_ln_2(rin); experts(h2in); h2=post_ff_ln_2(experts)
//     h=h1+h2; post_ff_ln(h); x=residual+h; *=layer_scalar
//   ★Gemma4RMSNorm=plain-w(*(weight), NOT (1+w)) → 전부 launch_rmsnorm. router.scale도 plain-w rmsnorm weight로 흡수.
//   ★experts act=gelu_pytorch_tanh(GeGLU) → moe_fused_swiglu(silu)不사용, 기존 launch_geglu를 slot 루프로(신규 커널 0).
void gemma4_moe_forward(const ModelConfig &c, const G4Weights &w,
                        const int *d_ids, int S,
                        const float *d_cos_f, const float *d_sin_f,
                        const float *d_cos_s, const float *d_sin_s,
                        float *d_logits, float *hidden_dump, int use_q4,
                        int *router_didx_dump, float *router_dw_dump) {
    g4m_sdtype = c.scale_dtype;
    const int H = c.hidden_size, FFN = c.intermediate_size, V = c.vocab_size;
    const int NQ = c.num_q_heads;
    const int hd_s = c.head_dim, nkv_s = c.num_kv_heads;
    const int hd_f = c.global_head_dim ? c.global_head_dim : hd_s;
    const int nkv_f = c.num_global_kv_heads ? c.num_global_kv_heads : nkv_s;
    const int E = c.num_experts, K = c.num_experts_per_tok, MI = c.moe_intermediate_size;
    const float eps = c.rms_norm_eps;
    const float root = powf((float)H, -0.5f);        // router scalar_root_size = H^-0.5
    const size_t SH = (size_t)S * H;
    const int hd_max = hd_f > hd_s ? hd_f : hd_s;
    const size_t qmax = (size_t)S * NQ * hd_max;
    const size_t kvmax = (size_t)S * (nkv_s * hd_s > nkv_f * hd_f ? nkv_s * hd_s : nkv_f * hd_f);
    const long gu_w = (long)2 * MI * H, dn_w = (long)H * MI;   // expert당 weight 원소 수

    float *x = dmalloc(SH), *norm = dmalloc(SH), *tmp = dmalloc(SH), *after = dmalloc(SH);
    float *qb = dmalloc(qmax), *qn = dmalloc(qmax);
    float *kb = dmalloc(kvmax), *kn = dmalloc(kvmax), *vb = dmalloc(kvmax), *vn = dmalloc(kvmax);
    float *attn = dmalloc(qmax), *ob = dmalloc(SH);
    float *g = dmalloc((size_t)S * FFN), *u = dmalloc((size_t)S * FFN), *act = dmalloc((size_t)S * FFN), *dn = dmalloc(SH);
    // MoE 경로 SH 버퍼
    float *h1 = dmalloc(SH), *h2 = dmalloc(SH), *moe_in = dmalloc(SH), *moe_out = dmalloc(SH), *comb = dmalloc(SH), *rnorm = dmalloc(SH);

    launch_embedding(d_ids, w.embed, x, S, H);
    if (c.embed_scale > 0.f) launch_scale(x, x, c.embed_scale, SH);

    for (int l = 0; l < c.num_layers; l++) {
        const G4Layer &L = w.layers[l];
        if (hidden_dump) GK(cudaMemcpy(hidden_dump + (size_t)l * SH, x, SH * 4, cudaMemcpyDeviceToDevice));
        const int full = L.is_full;
        const int hd = full ? hd_f : hd_s, nkv = full ? nkv_f : nkv_s;
        const int qdim = NQ * hd, kvdim = nkv * hd;
        const float *cosT = full ? d_cos_f : d_cos_s, *sinT = full ? d_sin_f : d_sin_s;
        const int window = full ? 0 : c.sliding_window;
        const float sc = c.attn_scaling;

        // --- attention block (input_ln → attn → post_attn_ln → residual add) — 31B와 동일 ---
        launch_rmsnorm(x, L.input_ln, norm, S, H, eps);
        mm(L.q, norm, qb, H, qdim, S, use_q4);
        launch_rmsnorm(qb, L.q_norm, qn, S * NQ, hd, eps);
        launch_rope(qn, cosT, sinT, S, NQ, hd);
        mm(L.k, norm, kb, H, kvdim, S, use_q4);
        if (full && c.attention_k_eq_v)
            launch_rmsnorm_noscale(kb, vn, S * nkv, hd, eps);
        else {
            mm(L.v, norm, vb, H, kvdim, S, use_q4);
            launch_rmsnorm_noscale(vb, vn, S * nkv, hd, eps);
        }
        launch_rmsnorm(kb, L.k_norm, kn, S * nkv, hd, eps);
        launch_rope(kn, cosT, sinT, S, nkv, hd);
        launch_attn(qn, kn, vn, attn, S, NQ, nkv, hd, 0, sc, window);
        mm(L.o, attn, ob, qdim, H, S, use_q4);
        launch_rmsnorm(ob, L.post_attn_ln, tmp, S, H, eps);
        launch_add(after, x, tmp, (int)SH);                        // after = residual (attn 후)

        // --- dense GeGLU MLP (pre_ff_ln(residual) → gate/up → GeGLU → down) ---
        launch_rmsnorm(after, L.pre_ffn_ln, norm, S, H, eps);
        mm(L.gate, norm, g, H, FFN, S, use_q4);
        mm(L.up,   norm, u, H, FFN, S, use_q4);
        launch_geglu(g, u, act, (size_t)S * FFN);
        mm(L.down, act, dn, FFN, H, S, use_q4);                    // dn = dense mlp 출력

        if (L.is_moe) {
            // h1 = post_ff_ln_1(dense)
            launch_rmsnorm(dn, L.post_ff_ln_1, h1, S, H, eps);

            // --- router(rin = after = residual) — Gemma4TextRouter verbatim ---
            //   norm_noscale(rin) * scale[H] * H^-0.5  ==  launch_rmsnorm(rin, router.scale)[=plain-w] * root
            launch_rmsnorm(after, L.router_scale, rnorm, S, H, eps);
            launch_scale(rnorm, rnorm, root, SH);
            float *scores = g4m_grow(g4m_scores, c_scores, (size_t)S * E);
            float *dw     = g4m_grow(g4m_dw, c_dw, (size_t)S * K);
            int   *didx   = g4m_growi(g4m_didx, c_didx, (size_t)S * K);
            launch_gemv_f32(rnorm, L.router_proj, scores, H, E, S);   // scores[S,E], router.proj fp32 [E,H]
            launch_router_topk(scores, didx, dw, S, E, K);            // softmax→top8→renorm
            launch_apply_per_expert_scale(dw, L.per_expert_scale, didx, S, K);  // dw *= per_expert_scale[idx]
            // 골든 대조 캡처(선택): MoE 층별 top-K expert idx + 최종 routing weight(renorm×per_expert_scale).
            if (router_didx_dump) GK(cudaMemcpy(router_didx_dump + (size_t)l * S * K, didx, (size_t)S * K * 4, cudaMemcpyDeviceToDevice));
            if (router_dw_dump)   GK(cudaMemcpy(router_dw_dump   + (size_t)l * S * K, dw,   (size_t)S * K * 4, cudaMemcpyDeviceToDevice));

            // --- experts(input = pre_ff_ln_2(rin)) — 128e top8 GeGLU (moe_fused grouped gemv) ---
            launch_rmsnorm(after, L.pre_ff_ln_2, moe_in, S, H, eps);
            long nslot = (long)S * K;
            float *fgu   = g4m_grow(g4m_fgu, c_fgu, nslot * (size_t)2 * MI);
            float *fhid  = g4m_grow(g4m_fhid, c_fhid, nslot * (size_t)MI);
            float *fsout = g4m_grow(g4m_fsout, c_fsout, nslot * (size_t)H);
            launch_moe_fused_gu(moe_in, L.experts_gate_up.wq, L.experts_gate_up.sc, didx, fgu, H, 2 * MI, K, nslot, gu_w, g4m_sdtype);
            // GeGLU(gemma) = gelu_pytorch_tanh(gate)*up. moe_fused_swiglu(silu) 대신 기존 launch_geglu slot 루프 (신규 커널 0).
            for (long slot = 0; slot < nslot; slot++)
                launch_geglu(fgu + slot * (2L * MI), fgu + slot * (2L * MI) + MI, fhid + slot * MI, MI);
            launch_moe_fused_down(fhid, L.experts_down.wq, L.experts_down.sc, didx, dw, fsout, MI, H, K, nslot, dn_w, g4m_sdtype);
            launch_moe_fused_reduce(fsout, moe_out, H, K, S);         // moe_out[S,H]
            launch_rmsnorm(moe_out, L.post_ff_ln_2, h2, S, H, eps);   // h2 = post_ff_ln_2(moe)

            launch_add(comb, h1, h2, (int)SH);                       // h = h1 + h2 (dense ∥ MoE)
            launch_rmsnorm(comb, L.post_ffn_ln, tmp, S, H, eps);     // post_ff_ln(h)
        } else {
            launch_rmsnorm(dn, L.post_ffn_ln, tmp, S, H, eps);       // dense-only (31B 위상 폴백)
        }
        launch_add(x, after, tmp, (int)SH);                          // x = residual + post_ff_ln
        if (L.layer_scalar != 1.0f) launch_scale(x, x, L.layer_scalar, SH);
    }

    launch_rmsnorm(x, w.final_norm, norm, S, H, eps);
    if (hidden_dump) GK(cudaMemcpy(hidden_dump + (size_t)c.num_layers * SH, norm, SH * 4, cudaMemcpyDeviceToDevice));
    mm(w.lm_head, norm, d_logits, H, V, S, use_q4);
    if (c.final_logit_softcapping > 0.f)
        launch_softcap(d_logits, d_logits, c.final_logit_softcapping, (size_t)S * V);
    GK(cudaDeviceSynchronize());

    float *all[] = {x, norm, tmp, after, qb, qn, kb, kn, vb, vn, attn, ob, g, u, act, dn,
                    h1, h2, moe_in, moe_out, comb, rnorm};
    for (float *p : all) cudaFree(p);
    // MoE fused scratch(g4m_*)는 persistent static — free 안 함(재호출 재사용).
}

// Gemma4 26B-A4B MoE decode graph — gemma4_decode(31B KV-cached attn 구조) + dense∥MoE MLP(gemma4_moe_forward 위상, S→Sq).
//   ★신규 커널 0(전부 기존: attn_decode·router_topk·apply_per_expert_scale·moe_fused·geglu). mm/g4m_* scratch 재사용.
//   서빙 decode: KV 이어받기(base=n_past)·rope+base·attn_decode(window). MoE MLP는 위치-독립(prefill forward와 동형 수학).
//   프리필-모드(Sq=S, base=0) → gemma4_moe_forward와 동일 logit(엔진 bit-id 게이트). last_only=1: 마지막 행 logit만.
void gemma4_moe_decode(const ModelConfig &c, const G4Weights &w, G4DecodeState &st,
                       const int *d_tokens, int Sq,
                       const float *d_cos_f, const float *d_sin_f,
                       const float *d_cos_s, const float *d_sin_s,
                       float *d_logits, int commit, int use_q4, int last_only) {
    g4m_sdtype = c.scale_dtype;
    const int H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, NQ=c.num_q_heads;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const int E=c.num_experts, K=c.num_experts_per_tok, MI=c.moe_intermediate_size;
    const float eps=c.rms_norm_eps, ascale=c.attn_scaling;
    const float root=powf((float)H,-0.5f);
    const int base=st.n_past;
    const size_t SH=(size_t)Sq*H;
    const int hd_max=hd_f>hd_s?hd_f:hd_s;
    const size_t qmax=(size_t)Sq*NQ*hd_max;
    const size_t kvmax=(size_t)Sq*(nkv_s*hd_s>nkv_f*hd_f?nkv_s*hd_s:nkv_f*hd_f);
    const long gu_w=(long)2*MI*H, dn_w=(long)H*MI;

    float *x=dmalloc(SH),*norm=dmalloc(SH),*tmp=dmalloc(SH),*after=dmalloc(SH);
    float *qb=dmalloc(qmax),*qn=dmalloc(qmax);
    float *kb=dmalloc(kvmax),*kn=dmalloc(kvmax),*vb=dmalloc(kvmax),*vn=dmalloc(kvmax);
    float *attn=dmalloc(qmax),*ob=dmalloc(SH);
    float *g=dmalloc((size_t)Sq*FFN),*u=dmalloc((size_t)Sq*FFN),*act=dmalloc((size_t)Sq*FFN),*dn=dmalloc(SH);
    float *h1=dmalloc(SH),*h2=dmalloc(SH),*moe_in=dmalloc(SH),*moe_out=dmalloc(SH),*comb=dmalloc(SH),*rnorm=dmalloc(SH);

    launch_embedding(d_tokens, w.embed, x, Sq, H);
    if(c.embed_scale>0.f) launch_scale(x,x,c.embed_scale,SH);

    for(int l=0;l<c.num_layers;l++){
        const G4Layer &L=w.layers[l];
        const int full=L.is_full;
        const int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        const int qdim=NQ*hd, kvdim=nkv*hd;
        const float *cosT=full?d_cos_f:d_cos_s, *sinT=full?d_sin_f:d_sin_s;
        const int window=full?0:c.sliding_window;

        // --- attention block (KV-cached — gemma4_decode와 동일 구조) ---
        launch_rmsnorm(x, L.input_ln, norm, Sq, H, eps);
        mm(L.q, norm, qb, H, qdim, Sq, use_q4);
        launch_rmsnorm(qb, L.q_norm, qn, Sq*NQ, hd, eps);
        launch_rope(qn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, Sq, NQ, hd);
        mm(L.k, norm, kb, H, kvdim, Sq, use_q4);
        if(full && c.attention_k_eq_v) launch_rmsnorm_noscale(kb, vn, Sq*nkv, hd, eps);
        else { mm(L.v, norm, vb, H, kvdim, Sq, use_q4); launch_rmsnorm_noscale(vb, vn, Sq*nkv, hd, eps); }
        launch_rmsnorm(kb, L.k_norm, kn, Sq*nkv, hd, eps);
        launch_rope(kn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, Sq, nkv, hd);
        GK(cudaMemcpy(st.Kc[l]+(size_t)base*kvdim, kn, (size_t)Sq*kvdim*4, cudaMemcpyDeviceToDevice));
        GK(cudaMemcpy(st.Vc[l]+(size_t)base*kvdim, vn, (size_t)Sq*kvdim*4, cudaMemcpyDeviceToDevice));
        launch_attn_decode(qn, st.Kc[l], st.Vc[l], attn, Sq, NQ, nkv, hd, base, 0, ascale, window);
        mm(L.o, attn, ob, qdim, H, Sq, use_q4);
        launch_rmsnorm(ob, L.post_attn_ln, tmp, Sq, H, eps);
        launch_add(after, x, tmp, (int)SH);                          // after = residual (attn 후)

        // --- dense GeGLU MLP (pre_ff_ln(residual)) ---
        launch_rmsnorm(after, L.pre_ffn_ln, norm, Sq, H, eps);
        mm(L.gate, norm, g, H, FFN, Sq, use_q4);
        mm(L.up,   norm, u, H, FFN, Sq, use_q4);
        launch_geglu(g, u, act, (size_t)Sq*FFN);
        mm(L.down, act, dn, FFN, H, Sq, use_q4);

        if(L.is_moe){
            launch_rmsnorm(dn, L.post_ff_ln_1, h1, Sq, H, eps);      // h1 = post_ff_ln_1(dense)
            // router(rin=after=residual) — rmsnorm(router.scale)[plain-w] * H^-0.5 → proj → top8 → per_expert_scale
            launch_rmsnorm(after, L.router_scale, rnorm, Sq, H, eps);
            launch_scale(rnorm, rnorm, root, SH);
            float *scores=g4m_grow(g4m_scores,c_scores,(size_t)Sq*E);
            float *dw    =g4m_grow(g4m_dw,c_dw,(size_t)Sq*K);
            int   *didx  =g4m_growi(g4m_didx,c_didx,(size_t)Sq*K);
            launch_gemv_f32(rnorm, L.router_proj, scores, H, E, Sq);
            launch_router_topk(scores, didx, dw, Sq, E, K);
            launch_apply_per_expert_scale(dw, L.per_expert_scale, didx, Sq, K);
            // experts(pre_ff_ln_2(rin)) — 128e top8 GeGLU
            launch_rmsnorm(after, L.pre_ff_ln_2, moe_in, Sq, H, eps);
            long nslot=(long)Sq*K;
            float *fgu  =g4m_grow(g4m_fgu,c_fgu,nslot*(size_t)2*MI);
            float *fhid =g4m_grow(g4m_fhid,c_fhid,nslot*(size_t)MI);
            float *fsout=g4m_grow(g4m_fsout,c_fsout,nslot*(size_t)H);
            launch_moe_fused_gu(moe_in, L.experts_gate_up.wq, L.experts_gate_up.sc, didx, fgu, H, 2*MI, K, nslot, gu_w, g4m_sdtype);
            for(long slot=0;slot<nslot;slot++)
                launch_geglu(fgu+slot*(2L*MI), fgu+slot*(2L*MI)+MI, fhid+slot*MI, MI);
            launch_moe_fused_down(fhid, L.experts_down.wq, L.experts_down.sc, didx, dw, fsout, MI, H, K, nslot, dn_w, g4m_sdtype);
            launch_moe_fused_reduce(fsout, moe_out, H, K, Sq);
            launch_rmsnorm(moe_out, L.post_ff_ln_2, h2, Sq, H, eps);  // h2 = post_ff_ln_2(moe)
            launch_add(comb, h1, h2, (int)SH);                       // h = h1 + h2 (dense ∥ MoE)
            launch_rmsnorm(comb, L.post_ffn_ln, tmp, Sq, H, eps);
        } else {
            launch_rmsnorm(dn, L.post_ffn_ln, tmp, Sq, H, eps);      // dense-only 폴백
        }
        launch_add(x, after, tmp, (int)SH);                          // x = residual + post_ff_ln
        if(L.layer_scalar!=1.0f) launch_scale(x, x, L.layer_scalar, SH);
    }

    launch_rmsnorm(x, w.final_norm, norm, Sq, H, eps);
    const float *hn = last_only ? norm+(size_t)(Sq-1)*H : norm;
    int rows = last_only ? 1 : Sq;
    mm(w.lm_head, hn, d_logits, H, V, rows, use_q4);
    if(c.final_logit_softcapping>0.f) launch_softcap(d_logits, d_logits, c.final_logit_softcapping, (size_t)rows*V);
    if(commit) st.n_past += Sq;
    GK(cudaDeviceSynchronize());

    float *all[]={x,norm,tmp,after,qb,qn,kb,kn,vb,vn,attn,ob,g,u,act,dn,h1,h2,moe_in,moe_out,comb,rnorm};
    for(float *p:all) cudaFree(p);
}
