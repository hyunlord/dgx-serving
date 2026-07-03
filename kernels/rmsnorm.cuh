#ifndef DGX_RMSNORM_CUH
#define DGX_RMSNORM_CUH
#include "reduce.cuh"

// RMSNorm (모델 독립). 블록당 1 row. y = x * rsqrt(mean(x^2)+eps) * W.
// X,Y: [rows, hh]. W: [hh]. grid = rows, block = THREADS(권장 256).
static __global__ void rmsnorm_kernel(const float *X, const float *W, float *Y, int hh, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float *xr = X + (size_t)row * hh;
    float *yr = Y + (size_t)row * hh;
    __shared__ float sh[32];
    __shared__ float inv;
    float ss = 0.f;
    for (int i = tid; i < hh; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    ss = block_reduce_sum(ss, sh);
    if (tid == 0) inv = rsqrtf(ss / hh + eps);
    __syncthreads();
    for (int i = tid; i < hh; i += blockDim.x) yr[i] = xr[i] * inv * W[i];
}

static inline void launch_rmsnorm(const float *X, const float *W, float *Y,
                                  int rows, int hh, float eps, cudaStream_t s = 0) {
    rmsnorm_kernel<<<rows, 256, 0, s>>>(X, W, Y, hh, eps);
}

// Zero-centered RMSNorm (Qwen3.5 컨벤션): y = x * rsqrt(mean(x^2)+eps) * (1 + W).
// HF Qwen3_5RMSNorm: output = _norm(x) * (1.0 + weight.float()). input/post/final + q/k_norm 에 사용.
// q/k_norm 으로 쓸 때: rows = S*num_heads, hh = head_dim (per-head 정규화).
static __global__ void rmsnorm_zc_kernel(const float *X, const float *W, float *Y, int hh, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float *xr = X + (size_t)row * hh;
    float *yr = Y + (size_t)row * hh;
    __shared__ float sh[32];
    __shared__ float inv;
    float ss = 0.f;
    for (int i = tid; i < hh; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    ss = block_reduce_sum(ss, sh);
    if (tid == 0) inv = rsqrtf(ss / hh + eps);
    __syncthreads();
    for (int i = tid; i < hh; i += blockDim.x) yr[i] = xr[i] * inv * (1.0f + W[i]);
}

static inline void launch_rmsnorm_zc(const float *X, const float *W, float *Y,
                                     int rows, int hh, float eps, cudaStream_t s = 0) {
    rmsnorm_zc_kernel<<<rows, 256, 0, s>>>(X, W, Y, hh, eps);
}

// Scale 없는 RMSNorm: y = x * rsqrt(mean(x^2)+eps). (with_scale=False norm — 예: Gemma4 v_norm)
static __global__ void rmsnorm_ns_kernel(const float *X, float *Y, int hh, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float *xr = X + (size_t)row * hh;
    float *yr = Y + (size_t)row * hh;
    __shared__ float sh[32];
    __shared__ float inv;
    float ss = 0.f;
    for (int i = tid; i < hh; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    ss = block_reduce_sum(ss, sh);
    if (tid == 0) inv = rsqrtf(ss / hh + eps);
    __syncthreads();
    for (int i = tid; i < hh; i += blockDim.x) yr[i] = xr[i] * inv;
}

static inline void launch_rmsnorm_noscale(const float *X, float *Y,
                                          int rows, int hh, float eps, cudaStream_t s = 0) {
    rmsnorm_ns_kernel<<<rows, 256, 0, s>>>(X, Y, hh, eps);
}

#endif // DGX_RMSNORM_CUH
