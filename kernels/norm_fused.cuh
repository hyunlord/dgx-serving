#ifndef DGX_NORM_FUSED_CUH
#define DGX_NORM_FUSED_CUH
// Parallel RMSNorm row function and the norm fusions built on it (model independent).
//
// Row function R(hh): a row of hh floats (hh % 4 == 0) is read as float4 by T(hh) threads, T = hh/4
// rounded up to a warp, capped at 1024; thread t owns float4 t, t+T, ... (at most 4, kept in
// registers). Each thread sums its squares in that order, then block_reduce_sum (reduce.cuh) adds
// the per-thread partials. The order depends only on hh, never on the number of rows, so S=1 decode,
// verify, batch, concat and prefill rows of equal content give bit-identical results.
//
// Fusions repeat the unfused element math exactly:
//   norm:     y = (x * inv) * W                    (rmsnorm_kernel: xr[i] * inv * W[i])
//   rope:     t1' = t1 * c - t2 * sn, t2' = t2 * c + t1 * sn   (rope.cuh, same expressions)
//   residual: out = res + y                        (add_kernel: a[i] + b[i], a = residual)
//   scale:    out = out * s                        (scale_kernel)
//   KV store: kv_cvt (kv_store.cuh, round to nearest)
// Products and sums that the unfused kernels round separately use __fmul_rn/__fadd_rn so the
// compiler cannot contract them into an FMA. tests/test_norm_fused.cu checks fused == unfused
// (with this row function) bit for bit, and the row function against fp64.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "reduce.cuh"

#define NORM_PAR_MAXV 4   // float4 per thread kept in registers

static inline int norm_par_threads(int hh) {
    int t = ((hh / 4 + 31) / 32) * 32;
    return t > 1024 ? 1024 : t;
}
static inline bool norm_par_supported(int hh) {
    return hh % 4 == 0 && hh / 4 <= NORM_PAR_MAXV * norm_par_threads(hh);
}

// Loads the row into v[] and returns the row's sum of squares (all threads get the same value).
__device__ __forceinline__ float norm_par_sumsq(const float *row, int hh, float4 (&v)[NORM_PAR_MAXV], float *sh) {
    const int n4 = hh >> 2;
    const float4 *r4 = reinterpret_cast<const float4 *>(row);
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < NORM_PAR_MAXV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) {
            v[k] = r4[i];
            ss += v[k].x * v[k].x; ss += v[k].y * v[k].y; ss += v[k].z * v[k].z; ss += v[k].w * v[k].w;
        }
    }
    return block_reduce_sum(ss, sh);
}
__device__ __forceinline__ float norm_par_mul(float x, float inv, const float *W, int i) {
    return W ? __fmul_rn(__fmul_rn(x, inv), W[i]) : __fmul_rn(x, inv);
}

// Plain RMSNorm with the row function (W == nullptr: no scale, Gemma 4 v_norm).
static __global__ void rmsnorm_par_kernel(const float *X, const float *W, float *Y, int hh, float eps) {
    __shared__ float sh[32];
    const float *xr = X + (size_t)blockIdx.x * hh;
    float *yr = Y + (size_t)blockIdx.x * hh;
    float4 v[NORM_PAR_MAXV];
    const float inv = rsqrtf(norm_par_sumsq(xr, hh, v, sh) / hh + eps);
#pragma unroll
    for (int k = 0; k < NORM_PAR_MAXV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < (hh >> 2)) {
            const int e = i * 4;
            float4 o;
            o.x = norm_par_mul(v[k].x, inv, W, e); o.y = norm_par_mul(v[k].y, inv, W, e + 1);
            o.z = norm_par_mul(v[k].z, inv, W, e + 2); o.w = norm_par_mul(v[k].w, inv, W, e + 3);
            reinterpret_cast<float4 *>(yr)[i] = o;
        }
    }
}
static inline void launch_rmsnorm_par(const float *X, const float *W, float *Y, int rows, int hh, float eps, cudaStream_t s = 0) {
    rmsnorm_par_kernel<<<rows, norm_par_threads(hh), 0, s>>>(X, W, Y, hh, eps);
}

__device__ __forceinline__ void norm_store(float v, float *d) { *d = v; }
__device__ __forceinline__ void norm_store(float v, __half *d) { *d = __float2half_rn(v); }
__device__ __forceinline__ void norm_store(float v, __nv_bfloat16 *d) { *d = __float2bfloat16_rn(v); }

// Per-head RMSNorm (+ RoPE) (+ store in the KV dtype). Block = one (token s, head h) row of hd floats.
//   position p = pos_arr ? pos_arr[s] : base + s * pos_step;   cos/sin rows are [p * hd, p * hd + hd/2)
//   output row = Y + s * slot_stride + p * row_stride + h * hd  (q scratch: slot_stride = NH*hd,
//   row_stride = 0; KV cache slice: slot_stride = per-slot capacity, row_stride = NKV*hd)
template <typename TO, bool ROPE>
static __global__ void norm_rope_store_kernel(const float *X, const float *W, TO *Y, int NH, int hd, float eps,
                                              const float *cosT, const float *sinT, int base, int pos_step,
                                              const int *pos_arr, long slot_stride, long row_stride) {
    __shared__ float sh[32];
    __shared__ float ys[512];
    const int r = blockIdx.x, s = r / NH, h = r % NH;
    const float *xr = X + (size_t)r * hd;
    const long p = pos_arr ? pos_arr[s] : (long)base + (long)s * pos_step;
    TO *yr = Y + s * slot_stride + p * row_stride + (long)h * hd;
    float4 v[NORM_PAR_MAXV];
    const float inv = rsqrtf(norm_par_sumsq(xr, hd, v, sh) / hd + eps);
    float y[NORM_PAR_MAXV * 4];
#pragma unroll
    for (int k = 0; k < NORM_PAR_MAXV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < (hd >> 2)) {
            const int e = i * 4;
            y[k * 4] = norm_par_mul(v[k].x, inv, W, e); y[k * 4 + 1] = norm_par_mul(v[k].y, inv, W, e + 1);
            y[k * 4 + 2] = norm_par_mul(v[k].z, inv, W, e + 2); y[k * 4 + 3] = norm_par_mul(v[k].w, inv, W, e + 3);
            if (ROPE) { ys[e] = y[k * 4]; ys[e + 1] = y[k * 4 + 1]; ys[e + 2] = y[k * 4 + 2]; ys[e + 3] = y[k * 4 + 3]; }
        }
    }
    if (ROPE) __syncthreads();
    const int half = hd >> 1;
    const float *cr = cosT + p * hd, *sr = sinT + p * hd;
#pragma unroll
    for (int k = 0; k < NORM_PAR_MAXV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < (hd >> 2)) {
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int e = i * 4 + q;
                float o = y[k * 4 + q];
                if (ROPE) {
                    if (e < half) { const float t1 = ys[e], t2 = ys[e + half], c = cr[e], sn = sr[e]; o = t1 * c - t2 * sn; }
                    else { const int j = e - half; const float t1 = ys[j], t2 = ys[e], c = cr[j], sn = sr[j]; o = t2 * c + t1 * sn; }
                }
                norm_store(o, yr + e);
            }
        }
    }
}
template <typename TO>
static inline void launch_norm_rope_store(const float *X, const float *W, TO *Y, int rows, int NH, int hd, float eps,
                                          const float *cosT, const float *sinT, int base, int pos_step, const int *pos_arr,
                                          long slot_stride, long row_stride, bool rope, cudaStream_t s = 0) {
    if (rope) norm_rope_store_kernel<TO, true><<<rows, norm_par_threads(hd), 0, s>>>(X, W, Y, NH, hd, eps, cosT, sinT, base, pos_step, pos_arr, slot_stride, row_stride);
    else      norm_rope_store_kernel<TO, false><<<rows, norm_par_threads(hd), 0, s>>>(X, W, Y, NH, hd, eps, cosT, sinT, base, pos_step, pos_arr, slot_stride, row_stride);
}

// Post-norm + residual (+ scale) (+ next pre-norm), one row of hh floats per block:
//   out1 = (res + R(in) * W1) [* scale];   out2 = R(out1) * W2   (skipped when W2 == nullptr)
// Replaces rmsnorm(in, W1) -> add(out1, res, tmp) -> [scale] -> rmsnorm(out1, W2).
static __global__ void post_norm_residual_kernel(const float *in, const float *W1, const float *res, float *out1,
                                                 float scale, int apply_scale, const float *W2, float *out2,
                                                 int hh, float eps) {
    __shared__ float sh[32];
    const size_t off = (size_t)blockIdx.x * hh;
    float4 v[NORM_PAR_MAXV];
    const float inv1 = rsqrtf(norm_par_sumsq(in + off, hh, v, sh) / hh + eps);
    const int n4 = hh >> 2;
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < NORM_PAR_MAXV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) {
            const int e = i * 4;
            const float4 r = reinterpret_cast<const float4 *>(res + off)[i];
            float4 o;
            o.x = __fadd_rn(r.x, norm_par_mul(v[k].x, inv1, W1, e));
            o.y = __fadd_rn(r.y, norm_par_mul(v[k].y, inv1, W1, e + 1));
            o.z = __fadd_rn(r.z, norm_par_mul(v[k].z, inv1, W1, e + 2));
            o.w = __fadd_rn(r.w, norm_par_mul(v[k].w, inv1, W1, e + 3));
            if (apply_scale) { o.x = __fmul_rn(o.x, scale); o.y = __fmul_rn(o.y, scale); o.z = __fmul_rn(o.z, scale); o.w = __fmul_rn(o.w, scale); }
            reinterpret_cast<float4 *>(out1 + off)[i] = o;
            v[k] = o;
            ss += o.x * o.x; ss += o.y * o.y; ss += o.z * o.z; ss += o.w * o.w;   // same order as norm_par_sumsq
        }
    }
    if (!W2) return;
    const float inv2 = rsqrtf(block_reduce_sum(ss, sh) / hh + eps);
#pragma unroll
    for (int k = 0; k < NORM_PAR_MAXV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) {
            const int e = i * 4;
            float4 o;
            o.x = norm_par_mul(v[k].x, inv2, W2, e); o.y = norm_par_mul(v[k].y, inv2, W2, e + 1);
            o.z = norm_par_mul(v[k].z, inv2, W2, e + 2); o.w = norm_par_mul(v[k].w, inv2, W2, e + 3);
            reinterpret_cast<float4 *>(out2 + off)[i] = o;
        }
    }
}
static inline void launch_post_norm_residual(const float *in, const float *W1, const float *res, float *out1, float scale,
                                             const float *W2, float *out2, int rows, int hh, float eps, cudaStream_t s = 0) {
    post_norm_residual_kernel<<<rows, norm_par_threads(hh), 0, s>>>(in, W1, res, out1, scale, scale != 1.0f, W2, out2, hh, eps);
}

#endif // DGX_NORM_FUSED_CUH
