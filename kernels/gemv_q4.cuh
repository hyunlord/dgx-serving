#ifndef DGX_GEMV_Q4_CUH
#define DGX_GEMV_Q4_CUH
#include <cstdint>
#include "reduce.cuh"
#include "qscale.cuh"
#include "../runtime/config.h"   // QBLK

// int4 양자화 행렬-벡터곱 (모델 독립). Y[s,:] = Wq · X[s,:]
//   weight [OUT, IN] row-major, q4 packed 2/byte (low=even idx, high=odd), scale fp32 per QBLK block.
//   dequant: (q-8)*scale.  (docs/01-stage1-blueprint.md)
//   warp당 1 (s,o) 출력. lane이 IN을 32 stride로 누적. 1단계는 정확성 우선(m0 방식).
// X: [S, IN], Y: [S, OUT]. IN은 QBLK 배수.
// ch2 STEP1 시도/기각: uint4 contiguous-per-lane 벡터 로드는 weight를 coalesce하나 X를 uncoalesce
//   (sectors/req 2→30, in-engine 176→104 회귀). weight wide-load과 X coalescing 양립불가 →
//   X-shared 필요(별도 설계). docs/ch2-q4kernel-blueprint.md STEP1 결과 참고. 현 baseline 유지.
template<typename ST>
static __global__ void gemv_q4_kernel(const float *X, const uint8_t *Wq, const ST *sc,
                               float *Y, int IN, int OUT, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long)S * OUT) return;
    int s = (int)(gw / OUT), o = (int)(gw % OUT);
    const float   *x = X + (size_t)s * IN;
    const uint8_t *w = Wq + (size_t)o * (IN / 2);
    const ST      *sca = sc + (size_t)o * (IN / QBLK);
    float acc = 0.f;
    for (int i = lane; i < IN; i += 32) {
        uint8_t b = w[i >> 1];
        int q = (i & 1) ? (b >> 4) : (b & 0x0F);
        acc += x[i] * (((float)q - 8.0f) * sval(sca[i / QBLK]));
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}

static inline void launch_gemv_q4(const float *X, const uint8_t *Wq, const void *sc,
                                  float *Y, int IN, int OUT, int S, int scale_dtype = 0, cudaStream_t st = 0) {
    int block = 256;
    long warps = (long)S * OUT;
    long grid = (warps * 32 + block - 1) / block;
    if (scale_dtype == 1) gemv_q4_kernel<__half><<<(unsigned)grid, block, 0, st>>>(X, Wq, (const __half*)sc, Y, IN, OUT, S);
    else                  gemv_q4_kernel<float ><<<(unsigned)grid, block, 0, st>>>(X, Wq, (const float*)sc, Y, IN, OUT, S);
}

#endif // DGX_GEMV_Q4_CUH
