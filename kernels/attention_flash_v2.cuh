#ifndef DGX_ATTENTION_FLASH_V2_CUH
#define DGX_ATTENTION_FLASH_V2_CUH
// Prefill flash attention v2 (R21, opt-in G4_FLASH_V2=1; Gemma 4 dense prefill only, the v1 kernel in
// attention_flash.cuh stays for Qwen and for G4_FLASH_V2=0).
//
// Block = Tq query tokens x all GRP query heads that share one kv head (rows = Tq*GRP = RG*16), so each
// K/V tile is loaded once for every head of the group. NW = RG*WPR warps: warp (rg, wp) owns query rows
// rg*16..rg*16+15 and output dims wp*HD/WPR.., and computes QK^T for keys wp*BC/WPR.. of each tile.
// Softmax runs in registers; only the per-warp row max / row sum partials and the bf16 P tile go through
// shared memory. Fragments are loaded with ldmatrix from padded rows (stride HD+8 / BC+8 halves: no bank
// conflicts). Numerics: bf16 MMA, fp32 accumulation (QK^T, P.V, m, l), same as v1.
//
// Summation-order rule (chunk invariance): key tiles are BC keys at fixed absolute positions (j0 = k*BC),
// visited in increasing order for every row; a tile in which a row attends no key leaves that row's state
// exactly unchanged (alpha = 1, P = 0). The whole row is one pass (no KV split, no pass-2). A query row's
// result therefore depends only on its own q, its position and the keys, never on which rows share the
// block or how the prompt was chunked. Tq, BC, RG, WPR are compile-time constants, independent of Sq,
// the KV length, the batch and the SM count.
#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace fav2 {

__device__ __forceinline__ void ldsm_x4(unsigned &r0, unsigned &r1, unsigned &r2, unsigned &r3, const void *p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned &r0, unsigned &r1, unsigned &r2, unsigned &r3, const void *p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(unsigned &r0, unsigned &r1, const void *p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <bool F16> __device__ __forceinline__ void mma_t(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    if constexpr (F16) mma_f16(c, a0, a1, a2, a3, b0, b1); else mma_bf16(c, a0, a1, a2, a3, b0, b1);
}
// Two floats -> one 32-bit pair of MMA operands (round to nearest).
template <bool F16> __device__ __forceinline__ unsigned pack2(float a, float b) {
    if constexpr (F16) { __half2 h = __floats2half2_rn(a, b); return *reinterpret_cast<unsigned *>(&h); }
    else { __nv_bfloat162 h = __floats2bfloat162_rn(a, b); return *reinterpret_cast<unsigned *>(&h); }
}
// R21 PART C: 8 KV elements as f16 operands. An f16 cache is copied unchanged (no conversion).
__device__ __forceinline__ uint4 kv8_f16(const __half *p) { return __ldg(reinterpret_cast<const uint4 *>(p)); }
__device__ __forceinline__ uint4 kv8_f16(const __nv_bfloat16 *p) {
    const uint4 raw = __ldg(reinterpret_cast<const uint4 *>(p)); const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&raw);
    uint4 out; unsigned *o = reinterpret_cast<unsigned *>(&out);
    #pragma unroll
    for (int i = 0; i < 4; i++) o[i] = pack2<true>(__low2float(h[i]), __high2float(h[i]));
    return out;
}
__device__ __forceinline__ uint4 kv8_f16(const float *p) {
    const float4 a = __ldg(reinterpret_cast<const float4 *>(p)), b = __ldg(reinterpret_cast<const float4 *>(p) + 1);
    return make_uint4(pack2<true>(a.x, a.y), pack2<true>(a.z, a.w), pack2<true>(b.x, b.y), pack2<true>(b.z, b.w));
}
template <bool F16, typename KV> __device__ __forceinline__ uint4 kv8(const KV *p);
// 8 consecutive KV elements -> 8 bf16 (one 16-byte shared store). f16 and bf16 sources are one 16-byte load.
__device__ __forceinline__ uint4 kv8_bf16(const __half *p) {
    const uint4 raw = __ldg(reinterpret_cast<const uint4 *>(p)); const __half2 *h = reinterpret_cast<const __half2 *>(&raw);
    uint4 out; __nv_bfloat162 *o = reinterpret_cast<__nv_bfloat162 *>(&out);
    #pragma unroll
    for (int i = 0; i < 4; i++) o[i] = __floats2bfloat162_rn(__low2float(h[i]), __high2float(h[i]));
    return out;
}
__device__ __forceinline__ uint4 kv8_bf16(const __nv_bfloat16 *p) { return __ldg(reinterpret_cast<const uint4 *>(p)); }
__device__ __forceinline__ uint4 kv8_bf16(const float *p) {
    const float4 a = __ldg(reinterpret_cast<const float4 *>(p)), b = __ldg(reinterpret_cast<const float4 *>(p) + 1);
    uint4 out; __nv_bfloat162 *o = reinterpret_cast<__nv_bfloat162 *>(&out);
    o[0] = __floats2bfloat162_rn(a.x, a.y); o[1] = __floats2bfloat162_rn(a.z, a.w);
    o[2] = __floats2bfloat162_rn(b.x, b.y); o[3] = __floats2bfloat162_rn(b.z, b.w);
    return out;
}

template <bool F16, typename KV> __device__ __forceinline__ uint4 kv8(const KV *p) {
    if constexpr (F16) return kv8_f16(p); else return kv8_bf16(p);
}

template <int HD, int RG, int WPR, int BC, int KCH>
struct Cfg {
    static constexpr int R = RG * 16, NW = RG * WPR, NT = NW * 32;
    static constexpr int SQ = HD + 8, SK = KCH + 8, SV = HD + 8, SP = BC + 8;   // padded row strides (halves)
    static constexpr int ODIM = HD / WPR, ONT = ODIM / 8;                        // output dims / n8 tiles per warp
    static constexpr int KPW = BC / WPR, NPK = KPW / 8;                          // QK^T keys / n8 tiles per warp
    static constexpr size_t SMEM = (size_t)(R * SQ + BC * SK + BC * SV + R * SP) * 2 + (size_t)R * WPR * 2 * 4;
    static_assert(HD % KCH == 0 && KCH % 16 == 0, "K chunk");
    static_assert(KPW % 8 == 0 && NPK >= 1 && NPK <= 2, "keys per warp");
    static_assert(ODIM % 16 == 0, "output dims per warp");
    static_assert(BC % 16 == 0 && 512 % BC == 0, "tile");
};

template <int HD, int RG, int WPR, int BC, int KCH, typename KV, bool F16>
__global__ void __launch_bounds__(RG * WPR * 32, 1)
flash_v2_kernel(const float *__restrict__ Q, const KV *__restrict__ Kc, const KV *__restrict__ Vc, float *__restrict__ O,
                int Sq, int NQ, int NKV, int base, float scale, int window) {
    using C = Cfg<HD, RG, WPR, BC, KCH>;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short *sQ = reinterpret_cast<unsigned short *>(smem);
    unsigned short *sK = sQ + C::R * C::SQ;
    unsigned short *sV = sK + BC * C::SK;
    unsigned short *sP = sV + BC * C::SV;
    float *sMax = reinterpret_cast<float *>(sP + C::R * C::SP);   // [R][WPR]
    float *sSum = sMax + C::R * WPR;                              // [R][WPR]

    const int GRP = NQ / NKV, Tq = C::R / GRP;
    const int kv = blockIdx.y;
    const int tb = gridDim.x - 1 - blockIdx.x;                    // longest (latest) query blocks first
    const int q0 = tb * Tq;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int rg = warp / WPR, wp = warp % WPR;
    const size_t kvrow = (size_t)NKV * HD;

    // Q tile (fp32 -> bf16, round to nearest as v1). Row r = token q0 + r/GRP, head kv*GRP + r%GRP.
    for (int i = tid; i < C::R * (HD / 4); i += C::NT) {
        const int r = i / (HD / 4), c = (i % (HD / 4)) * 4, tok = q0 + r / GRP, h = kv * GRP + r % GRP;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < Sq) v = __ldg(reinterpret_cast<const float4 *>(Q + ((size_t)tok * NQ + h) * HD + c));
        unsigned *d = reinterpret_cast<unsigned *>(sQ + r * C::SQ + c);
        d[0] = pack2<F16>(v.x, v.y); d[1] = pack2<F16>(v.z, v.w);
    }
    // Rows of this thread: lr0 = rg*16+g, lr1 = lr0+8.
    const int lr0 = rg * 16 + g, lr1 = lr0 + 8;
    const int tok0 = q0 + lr0 / GRP, tok1 = q0 + lr1 / GRP;
    const int qp0 = base + tok0, qp1 = base + tok1;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.f, l1 = 0.f;
    float acc[C::ONT][4];
    #pragma unroll
    for (int n = 0; n < C::ONT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;

    // Key range of the block: keys [lo, hi) cover every attended key of every valid row.
    const int qlast = (q0 + Tq < Sq ? q0 + Tq : Sq) - 1;
    const int hi = base + qlast + 1;
    int lo = 0;
    if (window > 0) { lo = base + q0 - window + 1; if (lo < 0) lo = 0; }
    const int j_begin = (lo / BC) * BC;

    for (int j0 = j_begin; j0 < hi; j0 += BC) {
        __syncthreads();   // previous tile's sK/sV/sP readers are done (and the Q tile is written)
        // V tile (all dims) and K chunk 0.
        for (int i = tid; i < BC * (HD / 8); i += C::NT) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8, gk = j0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gk < hi) v = kv8<F16>(Vc + (size_t)gk * kvrow + (size_t)kv * HD + c);
            *reinterpret_cast<uint4 *>(sV + r * C::SV + c) = v;
        }
        float s[C::NPK][4];
        #pragma unroll
        for (int n = 0; n < C::NPK; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
        #pragma unroll 1
        for (int kc = 0; kc < HD; kc += KCH) {
            if (kc) __syncthreads();   // previous K chunk consumed
            for (int i = tid; i < BC * (KCH / 8); i += C::NT) {
                const int r = i / (KCH / 8), c = (i % (KCH / 8)) * 8, gk = j0 + r;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (gk < hi) v = kv8<F16>(Kc + (size_t)gk * kvrow + (size_t)kv * HD + kc + c);
                *reinterpret_cast<uint4 *>(sK + r * C::SK + c) = v;
            }
            __syncthreads();
            // S[rows rg*16.., keys wp*KPW..] += Q[:, kc..kc+KCH) K^T, k-steps in increasing dim order.
            #pragma unroll 4
            for (int ks = 0; ks < KCH; ks += 16) {
                unsigned a0, a1, a2, a3;
                ldsm_x4(a0, a1, a2, a3, sQ + (rg * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * C::SQ + kc + ks + (lane >> 4) * 8);
                if constexpr (C::NPK == 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3, sK + (wp * C::KPW + (lane & 7) + (lane >> 4) * 8) * C::SK + ks + ((lane >> 3) & 1) * 8);
                    mma_t<F16>(s[0], a0, a1, a2, a3, b0, b1);
                    mma_t<F16>(s[C::NPK - 1], a0, a1, a2, a3, b2, b3);
                } else {
                    unsigned b0, b1;
                    ldsm_x2(b0, b1, sK + (wp * C::KPW + (lane & 7)) * C::SK + ks + ((lane >> 3) & 1) * 8);
                    mma_t<F16>(s[0], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        // Scale + mask; per-thread then per-quad row max over this warp's keys.
        float pm0 = -FLT_MAX, pm1 = -FLT_MAX;
        #pragma unroll
        for (int n = 0; n < C::NPK; n++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                const int gk = j0 + wp * C::KPW + n * 8 + 2 * t + e;
                const bool a0k = tok0 < Sq && gk <= qp0 && (window <= 0 || gk >= qp0 - window + 1);
                const bool a1k = tok1 < Sq && gk <= qp1 && (window <= 0 || gk >= qp1 - window + 1);
                s[n][e] = a0k ? s[n][e] * scale : -FLT_MAX;
                s[n][2 + e] = a1k ? s[n][2 + e] * scale : -FLT_MAX;
                pm0 = fmaxf(pm0, s[n][e]); pm1 = fmaxf(pm1, s[n][2 + e]);
            }
        }
        pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffffu, pm0, 1)); pm0 = fmaxf(pm0, __shfl_xor_sync(0xffffffffu, pm0, 2));
        pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffffu, pm1, 1)); pm1 = fmaxf(pm1, __shfl_xor_sync(0xffffffffu, pm1, 2));
        if (t == 0) { sMax[lr0 * WPR + wp] = pm0; sMax[lr1 * WPR + wp] = pm1; }
        __syncthreads();
        float tm0 = -FLT_MAX, tm1 = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < WPR; w++) { tm0 = fmaxf(tm0, sMax[lr0 * WPR + w]); tm1 = fmaxf(tm1, sMax[lr1 * WPR + w]); }
        const bool live0 = tm0 != -FLT_MAX, live1 = tm1 != -FLT_MAX;
        const float mn0 = live0 ? fmaxf(m0, tm0) : m0, mn1 = live1 ? fmaxf(m1, tm1) : m1;
        const float al0 = live0 ? expf(m0 - mn0) : 1.f, al1 = live1 ? expf(m1 - mn1) : 1.f;
        float ps0 = 0.f, ps1 = 0.f;
        #pragma unroll
        for (int n = 0; n < C::NPK; n++) {
            float p[4];
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                p[e] = s[n][e] == -FLT_MAX ? 0.f : expf(s[n][e] - mn0);
                p[2 + e] = s[n][2 + e] == -FLT_MAX ? 0.f : expf(s[n][2 + e] - mn1);
            }
            ps0 += p[0]; ps0 += p[1]; ps1 += p[2]; ps1 += p[3];
            const int kcol = wp * C::KPW + n * 8 + 2 * t;
            *reinterpret_cast<unsigned *>(sP + lr0 * C::SP + kcol) = pack2<F16>(p[0], p[1]);
            *reinterpret_cast<unsigned *>(sP + lr1 * C::SP + kcol) = pack2<F16>(p[2], p[3]);
        }
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1); ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1); ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        if (t == 0) { sSum[lr0 * WPR + wp] = ps0; sSum[lr1 * WPR + wp] = ps1; }
        __syncthreads();
        float ts0 = 0.f, ts1 = 0.f;
        #pragma unroll
        for (int w = 0; w < WPR; w++) { ts0 += sSum[lr0 * WPR + w]; ts1 += sSum[lr1 * WPR + w]; }
        l0 = l0 * al0 + ts0; l1 = l1 * al1 + ts1; m0 = mn0; m1 = mn1;
        // O[rows, wp dims] = O*alpha + P V over the BC keys of the tile (k-steps in increasing key order).
        #pragma unroll
        for (int n = 0; n < C::ONT; n++) { acc[n][0] *= al0; acc[n][1] *= al0; acc[n][2] *= al1; acc[n][3] *= al1; }
        #pragma unroll
        for (int ks = 0; ks < BC; ks += 16) {
            unsigned a0, a1, a2, a3;
            ldsm_x4(a0, a1, a2, a3, sP + (rg * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * C::SP + ks + (lane >> 4) * 8);
            #pragma unroll
            for (int n = 0; n < C::ONT; n += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, sV + (ks + (lane & 7) + ((lane >> 3) & 1) * 8) * C::SV + wp * C::ODIM + n * 8 + (lane >> 4) * 8);
                mma_t<F16>(acc[n], a0, a1, a2, a3, b0, b1);
                mma_t<F16>(acc[n + 1], a0, a1, a2, a3, b2, b3);
            }
        }
    }
    // Normalize and write O[(tok*NQ + head)*HD + d] straight from registers.
    const int h0 = kv * GRP + lr0 % GRP, h1 = kv * GRP + lr1 % GRP;
    const float inv0 = 1.f / l0, inv1 = 1.f / l1;
    #pragma unroll
    for (int n = 0; n < C::ONT; n++) {
        const int d = wp * C::ODIM + n * 8 + 2 * t;
        if (tok0 < Sq) *reinterpret_cast<float2 *>(O + ((size_t)tok0 * NQ + h0) * HD + d) = make_float2(acc[n][0] * inv0, acc[n][1] * inv0);
        if (tok1 < Sq) *reinterpret_cast<float2 *>(O + ((size_t)tok1 * NQ + h1) * HD + d) = make_float2(acc[n][2] * inv1, acc[n][3] * inv1);
    }
}

template <int HD, int RG, int WPR, int BC, int KCH, typename KV, bool F16>
static inline void launch_v2(const float *Q, const KV *Kc, const KV *Vc, float *O, int Sq, int NQ, int NKV, int base,
                             cudaStream_t st, float scale, int window) {
    using C = Cfg<HD, RG, WPR, BC, KCH>;
    static const bool configured = [] {
        cudaError_t e = cudaFuncSetAttribute(flash_v2_kernel<HD, RG, WPR, BC, KCH, KV, F16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C::SMEM);
        if (e != cudaSuccess) { fprintf(stderr, "[flash_v2] HD%d needs %zu shared bytes: %s\n", HD, C::SMEM, cudaGetErrorString(e)); abort(); }
        return true;
    }();
    (void)configured;
    const int GRP = NQ / NKV;
    if (NQ % NKV || C::R % GRP) { fprintf(stderr, "[flash_v2] GQA group %d must divide %d rows\n", GRP, C::R); abort(); }
    const int Tq = C::R / GRP;
    dim3 grid((Sq + Tq - 1) / Tq, NKV);
    flash_v2_kernel<HD, RG, WPR, BC, KCH, KV, F16><<<grid, C::NT, C::SMEM, st>>>(Q, Kc, Vc, O, Sq, NQ, NKV, base, scale == 0.f ? 1.0f / sqrtf((float)HD) : scale, window);
}

}  // namespace fav2

// Tile configuration (compile-time; see the design notes in docs): HD256: 64 rows, 8 warps (4 row groups x 2),
// BC 32. HD512: 32 rows, 8 warps (2 row groups x 4), BC 32, K loaded in two 256-dim chunks (shared-memory budget).
// f16_mma (R21 PART C, G4_FLASH_F16_MMA): Q, P and K/V as f16 MMA operands (an f16 cache is used unchanged),
// fp32 accumulation as before. Otherwise bf16 operands.
template <typename KV>
static inline void launch_flash_attn_v2(const float *Q, const KV *Kc, const KV *Vc, float *O, int Sq, int NQ, int NKV, int HD, int base,
                                        cudaStream_t st = 0, float scale = 0.f, int window = 0, bool f16_mma = false) {
    if (HD == 256 && f16_mma) fav2::launch_v2<256, 4, 2, 32, 256, KV, true>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st, scale, window);
    else if (HD == 256) fav2::launch_v2<256, 4, 2, 32, 256, KV, false>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st, scale, window);
    else if (HD == 512 && f16_mma) fav2::launch_v2<512, 2, 4, 32, 256, KV, true>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st, scale, window);
    else if (HD == 512) fav2::launch_v2<512, 2, 4, 32, 256, KV, false>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st, scale, window);
    else { fprintf(stderr, "[flash_v2] unsupported head_dim=%d (256/512 only)\n", HD); abort(); }
}

#endif  // DGX_ATTENTION_FLASH_V2_CUH
