#ifndef DGX_GEMV_Q3_CUH
#define DGX_GEMV_Q3_CUH
#include <cstdint>
#include "reduce.cuh"
#include "qscale.cuh"
#include "../runtime/config.h"   // QBLK

// q3(3bit) 양자화 행렬-벡터곱 — decode bytes↓ lever(q4 5bit/w → q3 4bit/w = 0.8×, 지형도 153318a의 q3 축).
//   ★packing = bit-plane 분해(풀기 싼 형태 — 시프트 2회+OR, 바이트 정렬 깔끔):
//     그룹(QBLK=32값) = low 2bit plane 8B(u32×2, 값 j의 2bit = low[j>>4] >> (2*(j&15)))
//                      + hi 1bit plane 4B(u32×1, 값 j의 1bit = hi >> j)   = 12B/그룹 + scale 4B.
//     q = low2 | (hi1<<2) ∈ 0..7 (8레벨 대칭), dequant = (q - 3.5f) * scale  (tools/q2_quality.py와 동일 스킴 — 품질 measured 연속).
//   레이아웃: W3q [OUT][IN/QBLK][3×u32] (low0, low1, hi 순), scale [OUT][IN/QBLK].
//   커널 구조 = gemv_q4/gemv_f32와 동일(warp당 1 (s,o)·lane 32-stride·warp_reduce) → lane j = 그룹의 j번째 값
//     = 그룹당 lane은 u32 워드 2개(warp 내 broadcast)만 접근. 누적 순서가 gemv_f32와 동일 → CPU-dequant ref와 bit-id 골든 가능.
template<typename ST>
static __global__ void gemv_q3_kernel(const float *X, const uint32_t *W3q, const ST *sc,
                                      float *Y, int IN, int OUT, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long)S * OUT) return;
    int s = (int)(gw / OUT), o = (int)(gw % OUT);
    int ngrp = IN / QBLK;
    const float    *x = X + (size_t)s * IN;
    const uint32_t *w = W3q + (size_t)o * ngrp * 3;
    const ST       *sca = sc + (size_t)o * ngrp;
    float acc = 0.f;
    for (int g = 0; g < ngrp; g++) {                   // unroll 4 시도 = 악화(1.18→1.24×, 레지스터 압력 — MLP_U 스윕과 동일 패턴) → 기본 유지
        uint32_t lo = w[g*3 + (lane >> 4)];            // low 2bit plane (lane 0..15 → word0, 16..31 → word1)
        uint32_t hi = w[g*3 + 2];                      // hi 1bit plane (전 lane broadcast)
        int q = (int)((lo >> ((lane & 15) * 2)) & 3u) | ((int)((hi >> lane) & 1u) << 2);
        acc += x[g * QBLK + lane] * (((float)q - 3.5f) * sval(sca[g]));
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}

static inline void launch_gemv_q3(const float *X, const uint32_t *W3q, const void *sc,
                                  float *Y, int IN, int OUT, int S, int scale_dtype = 0, cudaStream_t st = 0) {
    int block = 256;
    long warps = (long)S * OUT;
    long grid = (warps * 32 + block - 1) / block;
    if (scale_dtype == 1) gemv_q3_kernel<__half><<<(unsigned)grid, block, 0, st>>>(X, W3q, (const __half*)sc, Y, IN, OUT, S);
    else                  gemv_q3_kernel<float ><<<(unsigned)grid, block, 0, st>>>(X, W3q, (const float*)sc, Y, IN, OUT, S);
}

#endif // DGX_GEMV_Q3_CUH
