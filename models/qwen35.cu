#include "qwen35.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_q4.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/gdn.cuh"

#define QK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float *dmalloc(size_t n) { float *p; QK(cudaMalloc(&p, n * 4)); return p; }

// docs/02a-qwen35-blueprint.md 의 forward graph. config 로 dims/레이어타입 받음.
void qwen35_forward(const ModelConfig &c, const QwenWeights &w,
                    const int *d_ids, int S, const float *d_cos, const float *d_sin,
                    float *d_logits, float *hidden_dump, float *prenorm_out) {
    const int H = c.hidden_size, FFN = c.intermediate_size, HD = c.head_dim, V = c.vocab_size;
    const int NQ = c.num_q_heads, NKV = c.num_kv_heads, rot = c.rotary_dim;
    const float eps = c.rms_norm_eps;
    const int qdim2 = NQ * HD * (c.attn_output_gate ? 2 : 1);
    const int qdim = NQ * HD, kvdim = NKV * HD;
    const int conv_dim = cfg_gdn_conv_dim(&c), vdim = cfg_gdn_value_dim(&c), kdim = cfg_gdn_key_dim(&c);
    const int nvh = c.gdn_num_v_heads, hv = c.gdn_head_v_dim;
    const size_t SH = (size_t)S * H;

    float *x = dmalloc(SH), *norm = dmalloc(SH), *after = dmalloc(SH), *n2 = dmalloc(SH);
    // attention-side
    float *qg = dmalloc((size_t)S * qdim2), *qbuf = dmalloc((size_t)S * qdim), *gate = dmalloc((size_t)S * qdim);
    float *qn = dmalloc((size_t)S * qdim), *kbuf = dmalloc((size_t)S * kvdim), *kn = dmalloc((size_t)S * kvdim);
    float *vbuf = dmalloc((size_t)S * kvdim), *attn = dmalloc((size_t)S * qdim), *obuf = dmalloc(SH);
    // gdn-side
    float *mixed = dmalloc((size_t)S * conv_dim), *mconv = dmalloc((size_t)S * conv_dim);
    float *qg2 = dmalloc((size_t)S * kdim), *kg2 = dmalloc((size_t)S * kdim), *vg2 = dmalloc((size_t)S * vdim);
    float *zbuf = dmalloc((size_t)S * vdim), *abuf = dmalloc((size_t)S * nvh), *bbuf = dmalloc((size_t)S * nvh);
    float *gg = dmalloc((size_t)S * nvh), *beta = dmalloc((size_t)S * nvh);
    float *core = dmalloc((size_t)S * vdim), *normed = dmalloc((size_t)S * vdim), *gdnout = dmalloc(SH);
    // mlp
    float *g_mlp = dmalloc((size_t)S * FFN), *u_mlp = dmalloc((size_t)S * FFN), *act = dmalloc((size_t)S * FFN), *dbuf = dmalloc(SH);

    launch_embedding(d_ids, w.embed, x, S, H);

    for (int l = 0; l < c.num_layers; l++) {
        const QLayer &L = w.layers[l];
        if (hidden_dump) QK(cudaMemcpy(hidden_dump + (size_t)l * SH, x, SH * 4, cudaMemcpyDeviceToDevice));
        launch_rmsnorm_zc(x, L.input_ln, norm, S, H, eps);

        if (L.is_full) {
            // q_proj → [S,NQ,2HD] → split Q|gate (per head)
            launch_gemv_q4(norm, L.q, L.q_s, qg, H, qdim2, S);
            QK(cudaMemcpy2D(qbuf, HD * 4, qg,      2 * HD * 4, HD * 4, (size_t)S * NQ, cudaMemcpyDeviceToDevice));
            QK(cudaMemcpy2D(gate, HD * 4, qg + HD, 2 * HD * 4, HD * 4, (size_t)S * NQ, cudaMemcpyDeviceToDevice));
            launch_rmsnorm_zc(qbuf, L.q_norm, qn, S * NQ, HD, eps);     // q_norm (RoPE 전)
            launch_gemv_q4(norm, L.k, L.k_s, kbuf, H, kvdim, S);
            launch_rmsnorm_zc(kbuf, L.k_norm, kn, S * NKV, HD, eps);    // k_norm
            launch_gemv_q4(norm, L.v, L.v_s, vbuf, H, kvdim, S);
            launch_rope_partial(qn, d_cos, d_sin, S, NQ, HD, rot);
            launch_rope_partial(kn, d_cos, d_sin, S, NKV, HD, rot);
            launch_attn(qn, kn, vbuf, attn, S, NQ, NKV, HD);           // scale=1/sqrt(HD)
            launch_gate_sigmoid(attn, gate, (int)((size_t)S * qdim));  // attn *= sigmoid(gate)
            launch_gemv_q4(attn, L.o, L.o_s, obuf, qdim, H, S);
            launch_add(after, x, obuf, (int)SH);
        } else {
            // GDN
            launch_gemv_q4(norm, L.in_qkv, L.in_qkv_s, mixed, H, conv_dim, S);
            launch_gdn_conv1d_silu(mixed, L.conv1d, mconv, S, conv_dim, c.gdn_conv_kernel);
            QK(cudaMemcpy2D(qg2, kdim * 4, mconv,             conv_dim * 4, kdim * 4, S, cudaMemcpyDeviceToDevice));
            QK(cudaMemcpy2D(kg2, kdim * 4, mconv + kdim,      conv_dim * 4, kdim * 4, S, cudaMemcpyDeviceToDevice));
            QK(cudaMemcpy2D(vg2, vdim * 4, mconv + 2 * kdim,  conv_dim * 4, vdim * 4, S, cudaMemcpyDeviceToDevice));
            launch_gemv_q4(norm, L.in_z, L.in_z_s, zbuf, H, vdim, S);
            launch_gemv_f32(norm, L.in_a, abuf, H, nvh, S);
            launch_gemv_f32(norm, L.in_b, bbuf, H, nvh, S);
            launch_gdn_decay_beta(abuf, bbuf, L.A_log, L.dt_bias, gg, beta, S, nvh);
            launch_gdn_delta_rule(qg2, kg2, vg2, gg, beta, core, S, nvh, hv);
            launch_rms_norm_gated(core, zbuf, L.gdn_norm, normed, S * nvh, hv, eps);
            launch_gemv_q4(normed, L.out_p, L.out_s, gdnout, vdim, H, S);
            launch_add(after, x, gdnout, (int)SH);
        }

        // MLP (공통, SwiGLU)
        launch_rmsnorm_zc(after, L.post_ln, n2, S, H, eps);
        launch_gemv_q4(n2, L.gate, L.gate_s, g_mlp, H, FFN, S);
        launch_gemv_q4(n2, L.up,   L.up_s,   u_mlp, H, FFN, S);
        launch_swiglu(g_mlp, u_mlp, act, (size_t)S * FFN);
        launch_gemv_q4(act, L.down, L.down_s, dbuf, FFN, H, S);
        launch_add(x, after, dbuf, (int)SH);
    }

    if (prenorm_out) QK(cudaMemcpy(prenorm_out, x, SH * 4, cudaMemcpyDeviceToDevice));  // MTP 입력 (pre-norm)
    launch_rmsnorm_zc(x, w.final_norm, norm, S, H, eps);                                 // final norm = (1+w)
    if (hidden_dump) QK(cudaMemcpy(hidden_dump + (size_t)c.num_layers * SH, norm, SH * 4, cudaMemcpyDeviceToDevice));
    launch_gemv_q4(norm, w.lm_head, w.lm_head_s, d_logits, H, V, S);
    QK(cudaDeviceSynchronize());

    float *all[] = {x, norm, after, n2, qg, qbuf, gate, qn, kbuf, kn, vbuf, attn, obuf,
                    mixed, mconv, qg2, kg2, vg2, zbuf, abuf, bbuf, gg, beta, core, normed, gdnout,
                    g_mlp, u_mlp, act, dbuf};
    for (float *p : all) cudaFree(p);
}
