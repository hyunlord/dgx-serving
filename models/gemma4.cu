#include "gemma4.h"
#include <cstdio>
#include <cstdlib>
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

#define GK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float *dmalloc(size_t n) { float *p; GK(cudaMalloc(&p, n * 4)); return p; }

// 행렬곱 dispatch (qwen35.cu 와 동일 규약: 0=fp32 / 1=q4 tiled / 2=W4A8) — 공용 커널만 호출.
static int8_t *g4_xq=nullptr; static float *g4_xs=nullptr; static int32_t *g4_xsum=nullptr; static size_t g4_xqcap=0;
static int g4_sdtype;
static inline void mm(const QMat &m, const float *X, float *Y, int IN, int OUT, int S, int use_q4) {
    if (use_q4==2) {
        size_t n=(size_t)S*IN;
        if (g4_xqcap < n) { if(g4_xq)cudaFree(g4_xq); if(g4_xs)cudaFree(g4_xs); if(g4_xsum)cudaFree(g4_xsum);
            GK(cudaMalloc(&g4_xq,n)); GK(cudaMalloc(&g4_xs,(n/QBLK)*4)); GK(cudaMalloc(&g4_xsum,(n/QBLK)*4)); g4_xqcap=n; }
        launch_quantize_x_q8(X, g4_xq, g4_xs, g4_xsum, IN, S);
        launch_gemv_w4a8(g4_xq, g4_xs, g4_xsum, m.wq, m.sc, Y, IN, OUT, S, g4_sdtype);
    } else if (use_q4) launch_gemm_tiled_q4(X, m.wq, m.sc, Y, S, OUT, IN, g4_sdtype);
    else               launch_gemv_f32(X, m.wf, Y, IN, OUT, S);
}

// Gemma4 dense forward: sandwich 4-norm(plain-w RMSNorm), 층 타입별 attention
// (sliding: window=c.sliding_window, hd/nkv 기본 / full(global): window=0, global hd·nkv,
//  k_eq_v — V = rmsnorm_noscale(k_proj 출력, RoPE 전)), GeGLU, embed×√H, layer_scalar,
// final logit softcap. 전부 config/graph 흡수 — 커널에 모델 분기 없음.
void gemma4_forward(const ModelConfig &c, const G4Weights &w,
                    const int *d_ids, int S,
                    const float *d_cos_f, const float *d_sin_f,
                    const float *d_cos_s, const float *d_sin_s,
                    float *d_logits, float *hidden_dump, int use_q4) {
    g4_sdtype = c.scale_dtype;
    const int H = c.hidden_size, FFN = c.intermediate_size, V = c.vocab_size;
    const int NQ = c.num_q_heads;
    const int hd_s = c.head_dim, nkv_s = c.num_kv_heads;
    const int hd_f = c.global_head_dim ? c.global_head_dim : hd_s;
    const int nkv_f = c.num_global_kv_heads ? c.num_global_kv_heads : nkv_s;
    const float eps = c.rms_norm_eps;
    const size_t SH = (size_t)S * H;
    // 버퍼는 층 타입 최대 치수로
    const int hd_max = hd_f > hd_s ? hd_f : hd_s;
    const size_t qmax = (size_t)S * NQ * hd_max;
    const size_t kvmax = (size_t)S * (nkv_s * hd_s > nkv_f * hd_f ? nkv_s * hd_s : nkv_f * hd_f);

    float *x = dmalloc(SH), *norm = dmalloc(SH), *tmp = dmalloc(SH), *after = dmalloc(SH);
    float *qb = dmalloc(qmax), *qn = dmalloc(qmax);
    float *kb = dmalloc(kvmax), *kn = dmalloc(kvmax), *vb = dmalloc(kvmax), *vn = dmalloc(kvmax);
    float *attn = dmalloc(qmax), *ob = dmalloc(SH);
    float *g = dmalloc((size_t)S * FFN), *u = dmalloc((size_t)S * FFN), *act = dmalloc((size_t)S * FFN), *dn = dmalloc(SH);

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
        const float sc = c.attn_scaling;   // Gemma4: 1.0 (0이면 launch_attn이 1/sqrt(hd))

        // --- attention block (input_ln → attn → post_attn_ln → residual add) ---
        launch_rmsnorm(x, L.input_ln, norm, S, H, eps);
        mm(L.q, norm, qb, H, qdim, S, use_q4);
        launch_rmsnorm(qb, L.q_norm, qn, S * NQ, hd, eps);          // q_norm(per-head) → RoPE
        launch_rope(qn, cosT, sinT, S, NQ, hd);
        mm(L.k, norm, kb, H, kvdim, S, use_q4);
        if (full && c.attention_k_eq_v)
            launch_rmsnorm_noscale(kb, vn, S * nkv, hd, eps);       // V = v_norm(k_proj 출력, RoPE 전)
        else {
            mm(L.v, norm, vb, H, kvdim, S, use_q4);
            launch_rmsnorm_noscale(vb, vn, S * nkv, hd, eps);       // v_norm(scale 없음)은 모든 층 적용
        }
        launch_rmsnorm(kb, L.k_norm, kn, S * nkv, hd, eps);
        launch_rope(kn, cosT, sinT, S, nkv, hd);
        launch_attn(qn, kn, vn, attn, S, NQ, nkv, hd, 0, sc, window);
        mm(L.o, attn, ob, qdim, H, S, use_q4);
        launch_rmsnorm(ob, L.post_attn_ln, tmp, S, H, eps);         // post norm (residual add 전)
        launch_add(after, x, tmp, (int)SH);

        // --- mlp block (pre_ffn_ln → GeGLU → post_ffn_ln → residual add) ---
        launch_rmsnorm(after, L.pre_ffn_ln, norm, S, H, eps);
        mm(L.gate, norm, g, H, FFN, S, use_q4);
        mm(L.up,   norm, u, H, FFN, S, use_q4);
        launch_geglu(g, u, act, (size_t)S * FFN);
        mm(L.down, act, dn, FFN, H, S, use_q4);
        launch_rmsnorm(dn, L.post_ffn_ln, tmp, S, H, eps);
        launch_add(x, after, tmp, (int)SH);
        if (L.layer_scalar != 1.0f) launch_scale(x, x, L.layer_scalar, SH);
    }

    launch_rmsnorm(x, w.final_norm, norm, S, H, eps);
    if (hidden_dump) GK(cudaMemcpy(hidden_dump + (size_t)c.num_layers * SH, norm, SH * 4, cudaMemcpyDeviceToDevice));
    mm(w.lm_head, norm, d_logits, H, V, S, use_q4);
    if (c.final_logit_softcapping > 0.f)
        launch_softcap(d_logits, d_logits, c.final_logit_softcapping, (size_t)S * V);
    GK(cudaDeviceSynchronize());

    float *all[] = {x, norm, tmp, after, qb, qn, kb, kn, vb, vn, attn, ob, g, u, act, dn};
    for (float *p : all) cudaFree(p);
}
