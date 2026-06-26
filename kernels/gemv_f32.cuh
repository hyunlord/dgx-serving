#ifndef DGX_GEMV_F32_CUH
#define DGX_GEMV_F32_CUH
#include "reduce.cuh"

// fp32 행렬-벡터곱 (양자화 안 한 작은 weight 용, 예: GDN in_proj_a/b).
//   X: [S, IN], W: [OUT, IN] row-major fp32, Y: [S, OUT].  warp당 1 (s,o).
static __global__ void gemv_f32_kernel(const float *X, const float *W, float *Y, int IN, int OUT, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long)S * OUT) return;
    int s = (int)(gw / OUT), o = (int)(gw % OUT);
    const float *x = X + (size_t)s * IN;
    const float *w = W + (size_t)o * IN;
    float acc = 0.f;
    for (int i = lane; i < IN; i += 32) acc += x[i] * w[i];
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}
static inline void launch_gemv_f32(const float *X, const float *W, float *Y,
                                   int IN, int OUT, int S, cudaStream_t st = 0) {
    int block = 256; long warps = (long)S * OUT; long grid = (warps * 32 + block - 1) / block;
    gemv_f32_kernel<<<(unsigned)grid, block, 0, st>>>(X, W, Y, IN, OUT, S);
}

#endif // DGX_GEMV_F32_CUH
