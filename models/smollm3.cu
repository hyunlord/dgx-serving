#include "smollm3.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_q4.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"

#define SCK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// docs/01-stage1-blueprint.md 의 forward graph 그대로. config로 H/L/FFN/heads 받음.
void smollm3_forward(const ModelConfig &c, const EngineWeights &w,
                     const int *d_ids, int S,
                     const float *d_cos, const float *d_sin,
                     float *d_logits, float *hidden_dump) {
    const int H = c.hidden_size, FFN = c.intermediate_size, HD = c.head_dim;
    const int NQ = c.num_q_heads, NKV = c.num_kv_heads;
    const int qdim = NQ * HD, kvdim = NKV * HD, V = c.vocab_size;
    const float eps = c.rms_norm_eps;
    const size_t SH = (size_t)S * H;

    float *x, *norm, *q, *k, *v, *attn, *o, *after, *n2, *g, *u, *act, *down, *fn;
    SCK(cudaMalloc(&x,    SH * 4));
    SCK(cudaMalloc(&norm, SH * 4));
    SCK(cudaMalloc(&q,    (size_t)S * qdim * 4));
    SCK(cudaMalloc(&k,    (size_t)S * kvdim * 4));
    SCK(cudaMalloc(&v,    (size_t)S * kvdim * 4));
    SCK(cudaMalloc(&attn, (size_t)S * qdim * 4));
    SCK(cudaMalloc(&o,    SH * 4));
    SCK(cudaMalloc(&after,SH * 4));
    SCK(cudaMalloc(&n2,   SH * 4));
    SCK(cudaMalloc(&g,    (size_t)S * FFN * 4));
    SCK(cudaMalloc(&u,    (size_t)S * FFN * 4));
    SCK(cudaMalloc(&act,  (size_t)S * FFN * 4));
    SCK(cudaMalloc(&down, SH * 4));
    SCK(cudaMalloc(&fn,   SH * 4));

    launch_embedding(d_ids, w.embed, x, S, H);

    for (int l = 0; l < c.num_layers; l++) {
        const LayerW &L = w.layers[l];
        if (hidden_dump) SCK(cudaMemcpy(hidden_dump + (size_t)l * SH, x, SH * 4, cudaMemcpyDeviceToDevice));

        // --- attention block ---
        launch_rmsnorm(x, L.input_ln, norm, S, H, eps);
        launch_gemv_q4(norm, L.q, L.q_s, q, H, qdim, S);
        launch_gemv_q4(norm, L.k, L.k_s, k, H, kvdim, S);
        launch_gemv_q4(norm, L.v, L.v_s, v, H, kvdim, S);
        if (!c.use_rope || c.use_rope[l]) {   // NoPE: use_rope[l]==0 이면 skip
            launch_rope(q, d_cos, d_sin, S, NQ, HD);
            launch_rope(k, d_cos, d_sin, S, NKV, HD);
        }
        launch_attn(q, k, v, attn, S, NQ, NKV, HD);
        launch_gemv_q4(attn, L.o, L.o_s, o, qdim, H, S);
        launch_add(after, x, o, (int)SH);

        // --- mlp block ---
        launch_rmsnorm(after, L.post_ln, n2, S, H, eps);
        launch_gemv_q4(n2, L.gate, L.gate_s, g, H, FFN, S);
        launch_gemv_q4(n2, L.up,   L.up_s,   u, H, FFN, S);
        launch_swiglu(g, u, act, (size_t)S * FFN);
        launch_gemv_q4(act, L.down, L.down_s, down, FFN, H, S);
        launch_add(x, after, down, (int)SH);
    }

    launch_rmsnorm(x, w.final_norm, fn, S, H, eps);
    // HF output_hidden_states 의 마지막 원소는 final-norm 적용 후 값 → fn 을 덤프 (pre-norm x 아님).
    if (hidden_dump) SCK(cudaMemcpy(hidden_dump + (size_t)c.num_layers * SH, fn, SH * 4, cudaMemcpyDeviceToDevice));
    launch_gemv_q4(fn, w.lm_head, w.lm_head_s, d_logits, H, V, S);
    SCK(cudaDeviceSynchronize());

    cudaFree(x); cudaFree(norm); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(attn);
    cudaFree(o); cudaFree(after); cudaFree(n2); cudaFree(g); cudaFree(u); cudaFree(act);
    cudaFree(down); cudaFree(fn);
}
