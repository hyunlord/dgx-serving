#ifndef DGX_KV_STORE_CUH
#define DGX_KV_STORE_CUH
// KV cache storage dtype (process-wide knob G4_KV_DTYPE = fp32 | f16 | bf16). Default (R15): f16, or fp32
//   when G4_ATTN_SPLIT=0 (the original decode kernels read fp32 only).
//   Caches keep their float* handles; for 16-bit dtypes the buffer holds that dtype and every element
//   offset must go through kv_at() (byte arithmetic). fp32 behaves exactly as before (kv_at == p + n).
//   K/V are produced in fp32 scratch and written with kv_store() (round-to-nearest conversion); readers
//   load with kv_load() / kv_load4() and compute in fp32.
//   G4_KV_STATS=1 counts non-finite stored values and the max |value| of stored K and V (printed at exit).
#include "runtime/knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

enum { KV_FP32 = 0, KV_F16 = 1, KV_BF16 = 2 };

// Process-wide override (shared across translation units): MoE models keep fp32 KV (their batch
// attention path is unchanged), see g4_kv_force_fp32().
inline int g_kv_dtype_override = -1;
inline int g4_kv_dtype() {
    if (g_kv_dtype_override >= 0) return g_kv_dtype_override;
    static const int dtype = []() {
        const char *v = knob_raw("G4_KV_DTYPE");
        if (!v || !*v) {
            const char *split = knob_raw("G4_ATTN_SPLIT");
            return (split && !strcmp(split, "0")) ? (int)KV_FP32 : (int)KV_F16;
        }
        if (!strcmp(v, "fp32")) return (int)KV_FP32;
        if (!strcmp(v, "f16")) return (int)KV_F16;
        if (!strcmp(v, "bf16")) return (int)KV_BF16;
        fprintf(stderr, "Invalid G4_KV_DTYPE=%s (fp32, f16 or bf16)\n", v); exit(1);
    }();
    return dtype;
}
// Called when a MoE model's state is allocated (before any KV access in that process).
inline void g4_kv_force_fp32(const char *why) {
    if (g_kv_dtype_override == 0) return;
    if (knob_raw("G4_KV_DTYPE") && strcmp(knob_raw("G4_KV_DTYPE"), "fp32")) fprintf(stderr, "G4_KV_DTYPE ignored (%s): fp32 KV\n", why);
    g_kv_dtype_override = 0;
}
static inline size_t g4_kv_elem() { return g4_kv_dtype() == KV_FP32 ? 4 : 2; }
template<class P> static inline P *kv_at(P *p, size_t elems) {
    using B = typename std::conditional<std::is_const<P>::value, const char, char>::type;
    return reinterpret_cast<P *>(reinterpret_cast<B *>(p) + elems * g4_kv_elem());
}

__device__ __forceinline__ float kv_load(const float *p, size_t i) { return p[i]; }
__device__ __forceinline__ float kv_load(const __half *p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float kv_load(const __nv_bfloat16 *p, size_t i) { return __bfloat162float(p[i]); }
// Four consecutive elements starting at i (i multiple of 4).
__device__ __forceinline__ float4 kv_load4(const float *p, size_t i) { return *reinterpret_cast<const float4 *>(p + i); }
__device__ __forceinline__ float4 kv_load4(const __half *p, size_t i) {
    const uint2 u = *reinterpret_cast<const uint2 *>(p + i);
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float4 kv_load4(const __nv_bfloat16 *p, size_t i) {
    const uint2 u = *reinterpret_cast<const uint2 *>(p + i);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ void kv_cvt(float v, __half *d) { *d = __float2half_rn(v); }
__device__ __forceinline__ void kv_cvt(float v, __nv_bfloat16 *d) { *d = __float2bfloat16_rn(v); }
__device__ __forceinline__ float kv_back(__half v) { return __half2float(v); }
__device__ __forceinline__ float kv_back(__nv_bfloat16 v) { return __bfloat162float(v); }

// stats[0] = non-finite stored K, [1] = non-finite stored V, [2] = max|K| bits, [3] = max|V| bits.
static __device__ unsigned int g_kv_stats[4];
template<typename T>
static __global__ void kv_store_kernel(const float *src, T *dst, size_t n, int stats, int which) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    T out; kv_cvt(src[i], &out); dst[i] = out;
    if (stats) {
        const float back = kv_back(out);
        if (!isfinite(back)) atomicAdd(&g_kv_stats[which], 1u);
        else atomicMax(&g_kv_stats[2 + which], __float_as_uint(fabsf(back)));
    }
}
static inline bool g4_kv_stats_enabled() {
    static const bool on = []() {
        const char *v = knob_raw("G4_KV_STATS");
        if (!(v && v[0] == '1')) return false;
        atexit([]() {
            unsigned int h[4] = {};
            if (cudaMemcpyFromSymbol(h, g_kv_stats, sizeof(h)) == cudaSuccess) {
                float mk, mv; memcpy(&mk, &h[2], 4); memcpy(&mv, &h[3], 4);
                fprintf(stderr, "G4_KV_STATS dtype=%d nonfinite_k=%u nonfinite_v=%u max_abs_k=%.9g max_abs_v=%.9g\n",
                        g4_kv_dtype(), h[0], h[1], mk, mv);
            }
        });
        return true; }();
    return on;
}
// dst is a cache pointer already offset with kv_at(); which = 0 for K, 1 for V. fp32 copies.
static inline void kv_store(const float *src, float *dst, size_t n, int which, cudaStream_t st = 0) {
    const int dt = g4_kv_dtype();
    if (dt == KV_FP32) {
        cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, st);
        return;
    }
    const int stats = g4_kv_stats_enabled() ? 1 : 0;
    const unsigned blocks = (unsigned)((n + 255) / 256);
    if (dt == KV_F16) kv_store_kernel<__half><<<blocks, 256, 0, st>>>(src, reinterpret_cast<__half *>(dst), n, stats, which);
    else kv_store_kernel<__nv_bfloat16><<<blocks, 256, 0, st>>>(src, reinterpret_cast<__nv_bfloat16 *>(dst), n, stats, which);
}

#endif // DGX_KV_STORE_CUH
