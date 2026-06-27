#ifndef DGX_GEMM_BATCH_CUH
#define DGX_GEMM_BATCH_CUH
#include "reduce.cuh"

// batch gemm (fp32): Y[B,OUT] = X[B,IN] · Wᵀ, W[OUT,IN] row-major.
//   챕터3 STEP1 thesis: weight row를 1회 로드 → B행에 재사용(누적기 B개) = weight read OUT×IN(현 gemv는 OUT×B×IN).
//   = mmvq multiple-output의 일반화. weight 1로드당 B× compute로 DRAM latency 숨김 + weight bytes/token B×↓.
//   1 warp = 1 output o. lane이 IN을 32 stride. acc[B] 레지스터(B=register pressure, occupancy 감시 — 2d 교훈).
// B는 template(컴파일타임 누적기 수) → occupancy 스윕을 B별 정확히 측정.
template<int B>
static __global__ void gemm_batch_f32_kernel(const float *X, const float *W, float *Y, int IN, int OUT) {
    long o = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 1 warp/output
    int lane = threadIdx.x & 31;
    if (o >= OUT) return;
    const float *w = W + (size_t)o * IN;
    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; b++) acc[b] = 0.f;
    for (int i = lane; i < IN; i += 32) {
        float wv = w[i];                                   // weight 1회 로드 → B행 공유
        #pragma unroll
        for (int b = 0; b < B; b++) acc[b] += X[(size_t)b * IN + i] * wv;
    }
    #pragma unroll
    for (int b = 0; b < B; b++) {
        float r = warp_reduce_sum(acc[b]);
        if (lane == 0) Y[(size_t)b * OUT + o] = r;
    }
}

template<int B>
static inline void launch_gemm_batch_f32_B(const float *X, const float *W, float *Y, int IN, int OUT, cudaStream_t st) {
    int block = 256; long grid = ((long)OUT * 32 + block - 1) / block;
    gemm_batch_f32_kernel<B><<<(unsigned)grid, block, 0, st>>>(X, W, Y, IN, OUT);
}

// 런타임 B 디스패치 (지원 B: 1,2,4,8,16; 그 외는 호출측에서 분할).
static inline void launch_gemm_batch_f32(const float *X, const float *W, float *Y, int IN, int OUT, int B, cudaStream_t st = 0) {
    switch (B) {
        case 1:  launch_gemm_batch_f32_B<1>(X, W, Y, IN, OUT, st); break;
        case 2:  launch_gemm_batch_f32_B<2>(X, W, Y, IN, OUT, st); break;
        case 4:  launch_gemm_batch_f32_B<4>(X, W, Y, IN, OUT, st); break;
        case 8:  launch_gemm_batch_f32_B<8>(X, W, Y, IN, OUT, st); break;
        case 16: launch_gemm_batch_f32_B<16>(X, W, Y, IN, OUT, st); break;
        default: { for (int b = 0; b < B; b++) launch_gemm_batch_f32_B<1>(X + (size_t)b*IN, W, Y + (size_t)b*OUT, IN, OUT, st); }
    }
}

#endif // DGX_GEMM_BATCH_CUH
