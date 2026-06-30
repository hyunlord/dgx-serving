#include "qwen35.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_q4.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_batch.cuh"
#include "../kernels/gemm_tiled.cuh"   // 챕터7 Phase A: software tiled q4 GEMM (prefill)
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/gdn.cuh"
#include "moe_block.cuh"   // moe_mlp_block (prefill·decode 공유)

#define QK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float *dmalloc(size_t n) { float *p; QK(cudaMalloc(&p, n * 4)); return p; }

// 디버그: device 버퍼를 $DBG0/<name>.bin 으로 (env DBG0 설정 시, layer 0 GDN 단계 격리용)
static void dbg_dump(const char *name, const float *d, size_t n) {
    const char *od = getenv("DBG0"); if (!od) return;
    std::vector<float> h(n); QK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
    char path[512]; snprintf(path, sizeof(path), "%s/%s.bin", od, name);
    FILE *f = fopen(path, "wb"); if (f) { fwrite(h.data(), 4, n, f); fclose(f); }
}

// 행렬곱 dispatch: q4(int4 dequant) / fp32 / W4A8(int8 act × int4 weight) 경로.
static int8_t *gf_xq=nullptr; static float *gf_xs=nullptr; static int32_t *gf_xsum=nullptr; static size_t gf_xqcap=0;
static int gf_sdtype;  // scale_dtype cached from ModelConfig
static inline void mm(const QMat &m, const float *X, float *Y, int IN, int OUT, int S, int use_q4) {
    if (use_q4==2) {
        size_t n=(size_t)S*IN;
        if (gf_xqcap < n) { if(gf_xq)cudaFree(gf_xq); if(gf_xs)cudaFree(gf_xs); if(gf_xsum)cudaFree(gf_xsum);
            QK(cudaMalloc(&gf_xq,n)); QK(cudaMalloc(&gf_xs,(n/QBLK)*4)); QK(cudaMalloc(&gf_xsum,(n/QBLK)*4)); gf_xqcap=n; }
        launch_quantize_x_q8(X, gf_xq, gf_xs, gf_xsum, IN, S);
        launch_gemv_w4a8(gf_xq, gf_xs, gf_xsum, m.wq, m.sc, Y, IN, OUT, S, gf_sdtype);
    } else if (use_q4) launch_gemm_tiled_q4(X, m.wq, m.sc, Y, S, OUT, IN, gf_sdtype);   // prefill tiled GEMM(Phase A); S=1은 gemv
    else               launch_gemv_f32(X, m.wf, Y, IN, OUT, S);
}

// docs/02a-qwen35-blueprint.md 의 forward graph. config 로 dims/레이어타입 받음.
void qwen35_forward(const ModelConfig &c, const QwenWeights &w,
                    const int *d_ids, int S, const float *d_cos, const float *d_sin,
                    float *d_logits, float *hidden_dump, float *prenorm_out, int use_q4) {
    gf_sdtype = c.scale_dtype;
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
            mm(L.q, norm, qg, H, qdim2, S, use_q4);
            QK(cudaMemcpy2D(qbuf, HD * 4, qg,      2 * HD * 4, HD * 4, (size_t)S * NQ, cudaMemcpyDeviceToDevice));
            QK(cudaMemcpy2D(gate, HD * 4, qg + HD, 2 * HD * 4, HD * 4, (size_t)S * NQ, cudaMemcpyDeviceToDevice));
            launch_rmsnorm_zc(qbuf, L.q_norm, qn, S * NQ, HD, eps);     // q_norm (RoPE 전)
            mm(L.k, norm, kbuf, H, kvdim, S, use_q4);
            launch_rmsnorm_zc(kbuf, L.k_norm, kn, S * NKV, HD, eps);    // k_norm
            mm(L.v, norm, vbuf, H, kvdim, S, use_q4);
            launch_rope_partial(qn, d_cos, d_sin, S, NQ, HD, rot);
            launch_rope_partial(kn, d_cos, d_sin, S, NKV, HD, rot);
            launch_attn(qn, kn, vbuf, attn, S, NQ, NKV, HD);           // scale=1/sqrt(HD)
            launch_gate_sigmoid(attn, gate, (int)((size_t)S * qdim));  // attn *= sigmoid(gate)
            mm(L.o, attn, obuf, qdim, H, S, use_q4);
            launch_add(after, x, obuf, (int)SH);
        } else {
            // GDN
            mm(L.in_qkv, norm, mixed, H, conv_dim, S, use_q4);
            if (l == 0) dbg_dump("eng_in_proj_qkv", mixed, (size_t)S * conv_dim);
            launch_gdn_conv1d_silu(mixed, L.conv1d, mconv, S, conv_dim, c.gdn_conv_kernel);
            if (l == 0) dbg_dump("eng_post_conv", mconv, (size_t)S * conv_dim);
            QK(cudaMemcpy2D(qg2, kdim * 4, mconv,             conv_dim * 4, kdim * 4, S, cudaMemcpyDeviceToDevice));
            QK(cudaMemcpy2D(kg2, kdim * 4, mconv + kdim,      conv_dim * 4, kdim * 4, S, cudaMemcpyDeviceToDevice));
            QK(cudaMemcpy2D(vg2, vdim * 4, mconv + 2 * kdim,  conv_dim * 4, vdim * 4, S, cudaMemcpyDeviceToDevice));
            mm(L.in_z, norm, zbuf, H, vdim, S, use_q4);
            launch_gemv_f32(norm, L.in_a, abuf, H, nvh, S);            // in_proj_a/b 는 항상 fp32
            launch_gemv_f32(norm, L.in_b, bbuf, H, nvh, S);
            launch_gdn_decay_beta(abuf, bbuf, L.A_log, L.dt_bias, gg, beta, S, nvh);
            launch_gdn_delta_rule(qg2, kg2, vg2, gg, beta, core, S, nvh, hv, c.gdn_num_k_heads);
            if (l == 0) { dbg_dump("eng_dr_core", core, (size_t)S * vdim); dbg_dump("eng_gg", gg, (size_t)S * nvh); dbg_dump("eng_beta", beta, (size_t)S * nvh); }
            launch_rms_norm_gated(core, zbuf, L.gdn_norm, normed, S * nvh, hv, eps);
            if (l == 0) dbg_dump("eng_norm_out", normed, (size_t)S * vdim);
            mm(L.out_p, normed, gdnout, vdim, H, S, use_q4);
            if (l == 0) dbg_dump("eng_out_proj", gdnout, (size_t)S * H);
            launch_add(after, x, gdnout, (int)SH);
        }

        // MLP: dense SwiGLU 또는 MoE 블록 (config num_experts 로 분기 — 백본은 위까지 전부 공유)
        launch_rmsnorm_zc(after, L.post_ln, n2, S, H, eps);
        if (c.num_experts > 0) {
            if (l == 0) dbg_dump("eng_moe_in", n2, (size_t)S * H);
            moe_mlp_block(c, L, n2, dbuf, S);
            if (l == 0) dbg_dump("eng_moe_out", dbuf, (size_t)S * H);
        } else {
            mm(L.gate, n2, g_mlp, H, FFN, S, use_q4);
            mm(L.up,   n2, u_mlp, H, FFN, S, use_q4);
            launch_swiglu(g_mlp, u_mlp, act, (size_t)S * FFN);
            mm(L.down, act, dbuf, FFN, H, S, use_q4);
        }
        launch_add(x, after, dbuf, (int)SH);
    }

    if (prenorm_out) QK(cudaMemcpy(prenorm_out, x, SH * 4, cudaMemcpyDeviceToDevice));  // MTP 입력 (pre-norm)
    launch_rmsnorm_zc(x, w.final_norm, norm, S, H, eps);                                 // final norm = (1+w)
    if (hidden_dump) QK(cudaMemcpy(hidden_dump + (size_t)c.num_layers * SH, norm, SH * 4, cudaMemcpyDeviceToDevice));
    mm(w.lm_head, norm, d_logits, H, V, S, use_q4);
    QK(cudaDeviceSynchronize());

    float *all[] = {x, norm, after, n2, qg, qbuf, gate, qn, kbuf, kn, vbuf, attn, obuf,
                    mixed, mconv, qg2, kg2, vg2, zbuf, abuf, bbuf, gg, beta, core, normed, gdnout,
                    g_mlp, u_mlp, act, dbuf};
    for (float *p : all) cudaFree(p);
}
