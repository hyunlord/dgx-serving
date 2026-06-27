#ifndef DGX_GEMV_W4A8_CUH
#define DGX_GEMV_W4A8_CUH
#include <cstdint>
#include "reduce.cuh"
#include "../runtime/config.h"   // QBLK

__inline__ __device__ int warp_reduce_sum_i(int v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// W4A8 행렬-벡터곱: int8 activation(per-block scale+Σxq) × int4 weight(per-block scale, zero-point 8).
//   Y[s,o] = Σ_blocks  xs[s,b]·sw[o,b] · ( Σ_i xq·wq − 8·Σxq )      (docs/ch2b-w4a8-blueprint.md)
//   weight: gemv_q4와 동일 레이아웃 (Wq[o,IN/2] packed 2/byte, sc[o,IN/QBLK] fp32).
//   activation: xq[s,IN] int8, xs[s,IN/QBLK], xsum[s,IN/QBLK] (quantize_x_q8 출력).
//   STEP1 정확성 우선: 1 warp/output, 블록마다 warp_reduce(int dot) 후 scale. DP4A 최적화는 STEP2.
// IN은 QBLK(=32) 배수. warp 폭(32)==QBLK 가정.
static __global__ void gemv_w4a8_kernel(const int8_t *xq, const float *xs, const int32_t *xsum,
                                        const uint8_t *Wq, const float *sw,
                                        float *Y, int IN, int OUT, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long)S * OUT) return;
    int s = (int)(gw / OUT), o = (int)(gw % OUT);
    const int8_t  *xqr  = xq   + (size_t)s * IN;
    const float   *xsr  = xs   + (size_t)s * (IN / QBLK);
    const int32_t *xsmr = xsum + (size_t)s * (IN / QBLK);
    const uint8_t *w    = Wq   + (size_t)o * (IN / 2);
    const float   *swr  = sw   + (size_t)o * (IN / QBLK);
    int nb = IN / QBLK;
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
        int n = b * QBLK + lane;                       // QBLK==32==warp width
        uint8_t by = w[n >> 1];
        int wqv = (n & 1) ? (by >> 4) : (by & 0x0F);   // raw nibble 0..15
        int prod = (int)xqr[n] * wqv;                  // int8 × int4
        int dot = warp_reduce_sum_i(prod);             // Σ_i xq·wq (블록)
        if (lane == 0) acc += xsr[b] * swr[b] * ((float)dot - 8.0f * (float)xsmr[b]);
    }
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}

static inline void launch_gemv_w4a8(const int8_t *xq, const float *xs, const int32_t *xsum,
                                    const uint8_t *Wq, const float *sw,
                                    float *Y, int IN, int OUT, int S, cudaStream_t st = 0) {
    int block = 256;
    long warps = (long)S * OUT;
    long grid = (warps * 32 + block - 1) / block;
    gemv_w4a8_kernel<<<(unsigned)grid, block, 0, st>>>(xq, xs, xsum, Wq, sw, Y, IN, OUT, S);
}

#endif // DGX_GEMV_W4A8_CUH
