#ifndef DGX_GEMV_Q4_CUH
#define DGX_GEMV_Q4_CUH
#include <cstdint>
#include "reduce.cuh"
#include "../runtime/config.h"   // QBLK

// int4 양자화 행렬-벡터곱 (모델 독립). Y[s,:] = Wq · X[s,:]
//   weight [OUT, IN] row-major, q4 packed 2/byte (low=even idx, high=odd), scale fp32 per QBLK block.
//   dequant: (q-8)*scale.  (docs/01-stage1-blueprint.md)
//   warp당 1 (s,o) 출력. lane이 IN을 32 stride로 누적. 1단계는 정확성 우선(m0 방식).
// X: [S, IN], Y: [S, OUT]. IN은 QBLK 배수.
__global__ void gemv_q4_kernel(const float *X, const uint8_t *Wq, const float *sc,
                               float *Y, int IN, int OUT, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long)S * OUT) return;
    int s = (int)(gw / OUT), o = (int)(gw % OUT);
    const float   *x = X + (size_t)s * IN;
    const uint8_t *w = Wq + (size_t)o * (IN / 2);
    const float   *sca = sc + (size_t)o * (IN / QBLK);
    float acc = 0.f;
    for (int i = lane; i < IN; i += 32) {
        uint8_t b = w[i >> 1];
        int q = (i & 1) ? (b >> 4) : (b & 0x0F);
        acc += x[i] * (((float)q - 8.0f) * sca[i / QBLK]);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}

static inline void launch_gemv_q4(const float *X, const uint8_t *Wq, const float *sc,
                                  float *Y, int IN, int OUT, int S, cudaStream_t st = 0) {
    int block = 256;
    long warps = (long)S * OUT;
    long grid = (warps * 32 + block - 1) / block;
    gemv_q4_kernel<<<(unsigned)grid, block, 0, st>>>(X, Wq, sc, Y, IN, OUT, S);
}

#endif // DGX_GEMV_Q4_CUH
