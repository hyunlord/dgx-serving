#ifndef DGX_QUANTIZE_X_CUH
#define DGX_QUANTIZE_X_CUH
#include <cstdint>
#include "../runtime/config.h"   // QBLK

// activation 동적 int8 양자화 (per-block-QBLK, symmetric). W4A8용. (챕터2b)
//   X[S, IN] fp32 → xq[S, IN] int8 + xs[S, IN/QBLK] scale(fp32) + xsum[S, IN/QBLK] Σxq(int32).
//   블록당(QBLK=32): d = amax/127, xq_i = round(x_i/d) clamp[-127,127], Σxq = Σ xq_i.
//   xsum은 int4(zero-point 8) dot 보정용: dot = sx·sw·(Σ xq·wq − 8·Σxq).
//   llama.cpp q8_1과 동일 스킴(QK8_1=32, amax/127) — docs/ch2b-w4a8-blueprint.md.
//   1 warp = 1 블록(QBLK=32 lane). IN은 QBLK 배수.

__inline__ __device__ float qx_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__inline__ __device__ int qx_warp_sum_i(int v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

static __global__ void quantize_x_q8_kernel(const float *X, int8_t *xq, float *xs, int32_t *xsum,
                                            int IN, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 글로벌 warp = 글로벌 블록 id
    int lane = threadIdx.x & 31;
    long nblk_total = (long)S * (IN / QBLK);
    if (gw >= nblk_total) return;
    int nb = IN / QBLK;
    int s = (int)(gw / nb), b = (int)(gw % nb);
    long idx = (long)s * IN + (long)b * QBLK + lane;      // QBLK==32==warp width
    float x = X[idx];
    float amax = qx_warp_max(fabsf(x));
    float d = amax / 127.0f;
    int q = (amax == 0.0f) ? 0 : (int)rintf(x / d);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    int ssum = qx_warp_sum_i(q);
    xq[idx] = (int8_t)q;
    if (lane == 0) { xs[gw] = d; xsum[gw] = ssum; }
}

static inline void launch_quantize_x_q8(const float *X, int8_t *xq, float *xs, int32_t *xsum,
                                        int IN, int S, cudaStream_t st = 0) {
    int block = 256;
    long warps = (long)S * (IN / QBLK);
    long grid = (warps * 32 + block - 1) / block;
    quantize_x_q8_kernel<<<(unsigned)grid, block, 0, st>>>(X, xq, xs, xsum, IN, S);
}

#endif // DGX_QUANTIZE_X_CUH
