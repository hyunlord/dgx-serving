#ifndef DGX_GEMV_MC_CUH
#define DGX_GEMV_MC_CUH
// R18 multi-column W4A8 GEMV families (S = 1..8 columns, one kernel family per S range).
//
// Family "mca" (order preserving, integer path). For every output row o and column s the float sum is formed
// exactly as in gemv_w4a8_kernel / gemv_w4a8_sb_kernel (kernels/gemv_w4a8.cuh):
//   lane l visits base = 8l, 8l + 256, 8l + 512, ... in increasing order; per visit
//     dot = dp4a(x[base..base+3], w[base..base+3]) then dp4a(x[base+4..+7], w[base+4..+7], dot)   (int, exact)
//     acc = fma(xs[s][blk] * sw[o][blk], (float)dot, acc)                                         (one rounding each)
//   then the warp sum is the shfl_down tree 16, 8, 4, 2, 1 (lane 0 holds the result).
// So every column of every S equals the current plain GEMV bit for bit; S only changes how many columns share
// one weight load. Changes against SB: R output rows per warp share each activation load, S is a template
// constant (fully unrolled), the k loop is unrolled by U (more loads in flight, same visiting order), and the
// final reductions are done as a reduce-scatter that forms the same pairwise sums (IEEE addition is
// commutative, so lane l + lane l^off == lane l^off + lane l).
#include <cstring>
#include <cuda_fp16.h>
#include "gemv_w4a8.cuh"
#include "qscale.cuh"
#include "../runtime/config.h"

// Reduce-scatter of NV accumulators (NV a power of two, 1..32): after the call, lane l holds in v[0] the sum of
// accumulator (l / (32 / NV)) over all lanes, formed with the same pairs as warp_reduce_sum (shfl_down 16..1).
template <int NV>
__device__ __forceinline__ float mc_reduce_scatter(float (&v)[NV], int lane) {
    // steps with offset >= 32/NV split the accumulator set; the remaining offsets reduce in place.
#pragma unroll
    for (int off = 16, n = NV; off >= 1; off >>= 1) {
        if (n > 1) {
            const bool upper = (lane & off) != 0;
            const int h = n >> 1;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                if (i < h) {
                    const float send = upper ? v[i] : v[i + h];
                    const float keep = upper ? v[i + h] : v[i];
                    const float got = __shfl_xor_sync(0xffffffff, send, off);
                    v[i] = keep + got;
                }
            }
            n = h;
        } else {
            v[0] = v[0] + __shfl_xor_sync(0xffffffff, v[0], off);
        }
    }
    return v[0];
}

template <typename ST, int S, int R, int U>
static __global__ void __launch_bounds__(256) gemv_mca_kernel(const int8_t *__restrict__ xq, const float *__restrict__ xs,
                                                              const uint8_t *__restrict__ Wq, const ST *__restrict__ sw,
                                                              float *__restrict__ Y, int IN, int OUT) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int o0 = warp * R;
    if (o0 >= OUT) return;
    const int nb = IN / QBLK;
    float acc[R * S];
#pragma unroll
    for (int i = 0; i < R * S; i++) acc[i] = 0.f;
    const uint8_t *w[R]; const ST *swr[R];
#pragma unroll
    for (int r = 0; r < R; r++) { const int o = o0 + r < OUT ? o0 + r : OUT - 1; w[r] = Wq + (size_t)o * (IN / 2); swr[r] = sw + (size_t)o * nb; }
    const int STEP = 32 * 8;
#pragma unroll U
    for (int base = lane * 8; base < IN; base += STEP) {
        const int blk = base / QBLK;
        unsigned W[R]; float swb[R];
#pragma unroll
        for (int r = 0; r < R; r++) { W[r] = __ldg(reinterpret_cast<const unsigned *>(w[r] + (base >> 1))); swb[r] = sval(swr[r][blk]); }
        int2 x[S]; float xsb[S];
#pragma unroll
        for (int s = 0; s < S; s++) {
            x[s] = __ldg(reinterpret_cast<const int2 *>(xq + (size_t)s * IN + base));
            xsb[s] = __ldg(xs + (size_t)s * nb + blk);
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int pk0 = w4a8_pk(W[r] & 0xFF, (W[r] >> 8) & 0xFF);
            const int pk1 = w4a8_pk((W[r] >> 16) & 0xFF, (W[r] >> 24) & 0xFF);
#pragma unroll
            for (int s = 0; s < S; s++) {
                int dot = __dp4a(x[s].x, pk0, 0);
                dot = __dp4a(x[s].y, pk1, dot);
                acc[r * S + s] = __fmaf_rn(__fmul_rn(xsb[s], swb[r]), (float)dot, acc[r * S + s]);
            }
        }
    }
    constexpr int NV = R * S;
    if constexpr (NV == 1 || NV == 2 || NV == 4 || NV == 8 || NV == 16 || NV == 32) {
        const float v = mc_reduce_scatter<NV>(acc, lane);
        const int idx = lane / (32 / NV);           // accumulator index held by this lane
        if (lane % (32 / NV) == 0) {
            const int r = idx / S, s = idx % S;
            if (o0 + r < OUT) Y[(size_t)s * OUT + o0 + r] = v;
        }
    } else {
#pragma unroll
        for (int i = 0; i < NV; i++) {
            const float v = warp_reduce_sum(acc[i]);
            if (lane == 0 && o0 + i / S < OUT) Y[(size_t)(i % S) * OUT + o0 + i / S] = v;
        }
    }
}

template <int S, int R, int U>
static inline void launch_gemv_mca_t(const int8_t *xq, const float *xs, const uint8_t *Wq, const void *sw, float *Y,
                                     int IN, int OUT, int sdtype, cudaStream_t st) {
    const int block = 256;
    const long warps = (OUT + R - 1) / R;
    const unsigned grid = (unsigned)((warps * 32 + block - 1) / block);
    if (sdtype == 1) gemv_mca_kernel<__half, S, R, U><<<grid, block, 0, st>>>(xq, xs, Wq, (const __half *)sw, Y, IN, OUT);
    else             gemv_mca_kernel<float, S, R, U><<<grid, block, 0, st>>>(xq, xs, Wq, (const float *)sw, Y, IN, OUT);
}

// S dispatch for one (R, U) configuration.
template <int R, int U>
static inline void launch_gemv_mca_ru(const int8_t *xq, const float *xs, const uint8_t *Wq, const void *sw, float *Y,
                                      int IN, int OUT, int S, int sdtype, cudaStream_t st) {
    switch (S) {
        case 1: launch_gemv_mca_t<1, R, U>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 2: launch_gemv_mca_t<2, R, U>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 3: launch_gemv_mca_t<3, R, U>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 4: launch_gemv_mca_t<4, R, U>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 5: launch_gemv_mca_t<5, R, U>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 6: launch_gemv_mca_t<6, R, U>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 7: launch_gemv_mca_t<7, R, U>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        default: launch_gemv_mca_t<8, R, U>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
    }
}

// Family "mcb" (tensor core, new addition order). mma.sync m16n8k32 s8 x s8 -> s32 computes, for 16 output rows
// x 8 columns, the exact int32 dot of one 32-element quant block (columns >= S are zero). The k order inside the
// mma is irrelevant (integer). A and B use the same permutation of the 32 block elements: thread tig (= lane % 4)
// supplies physical elements 8 tig .. 8 tig + 7 as fragment k = 4 tig .. 4 tig + 3 (a0/a1, b0) and
// 16 + 4 tig .. 16 + 4 tig + 3 (a2/a3, b1), i.e. one 4-byte weight load per row and one 8-byte activation load.
// Float order for output (o, s): KS = 4 warps per 16-row tile; warp w visits block pairs p = w, w + 4, ...
// (blocks 2p then 2p + 1) and accumulates acc = fma(xs[s][b] * sw[o][b], (float)int_dot, acc) in that order;
// then out = ((acc_w0 + acc_w1) + acc_w2) + acc_w3. Nothing depends on S, so every column of every S equals the
// S = 1 result bit for bit. The order differs from the plain GEMV (numeric change against main).
template <typename ST, int S>
static __global__ void __launch_bounds__(128) gemv_mcb_kernel(const int8_t *__restrict__ xq, const float *__restrict__ xs,
                                                              const uint8_t *__restrict__ Wq, const ST *__restrict__ sw,
                                                              float *__restrict__ Y, int IN, int OUT) {
    constexpr int KS = 4;
    __shared__ float red[KS - 1][32][4];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int g = lane >> 2, tig = lane & 3;
    const int o0 = blockIdx.x * 16;
    const int nb = IN / QBLK, npair = nb / 2;
    const int ra = o0 + g < OUT ? o0 + g : OUT - 1, rb = o0 + g + 8 < OUT ? o0 + g + 8 : OUT - 1;
    const uint8_t *wa = Wq + (size_t)ra * (IN / 2) + 4 * tig, *wb = Wq + (size_t)rb * (IN / 2) + 4 * tig;
    const ST *sa = sw + (size_t)ra * nb, *sb = sw + (size_t)rb * nb;
    const bool colv = g < S;                                   // this thread's B column g carries a real column
    const int8_t *xb = xq + (size_t)(colv ? g : 0) * IN + 8 * tig;
    const int c0 = 2 * tig, c1 = 2 * tig + 1;                  // accumulator columns of this thread
    const float *xs0 = xs + (size_t)(c0 < S ? c0 : 0) * nb, *xs1 = xs + (size_t)(c1 < S ? c1 : 0) * nb;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
    for (int p = wid; p < npair; p += KS) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int b = 2 * p + h;
            const unsigned Wa = __ldg(reinterpret_cast<const unsigned *>(wa + (size_t)b * 16));
            const unsigned Wb = __ldg(reinterpret_cast<const unsigned *>(wb + (size_t)b * 16));
            int2 x = make_int2(0, 0);
            if (colv) x = __ldg(reinterpret_cast<const int2 *>(xb + (size_t)b * QBLK));
            const int a0 = w4a8_pk(Wa & 0xFF, (Wa >> 8) & 0xFF), a2 = w4a8_pk((Wa >> 16) & 0xFF, (Wa >> 24) & 0xFF);
            const int a1 = w4a8_pk(Wb & 0xFF, (Wb >> 8) & 0xFF), a3 = w4a8_pk((Wb >> 16) & 0xFF, (Wb >> 24) & 0xFF);
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(x.x), "r"(x.y));
            const float swa = sval(sa[b]), swb = sval(sb[b]);
            const float x0 = c0 < S ? __ldg(xs0 + b) : 0.f, x1 = c1 < S ? __ldg(xs1 + b) : 0.f;
            acc[0] = __fmaf_rn(__fmul_rn(x0, swa), (float)d0, acc[0]);
            acc[1] = __fmaf_rn(__fmul_rn(x1, swa), (float)d1, acc[1]);
            acc[2] = __fmaf_rn(__fmul_rn(x0, swb), (float)d2, acc[2]);
            acc[3] = __fmaf_rn(__fmul_rn(x1, swb), (float)d3, acc[3]);
        }
    }
    if (wid > 0) {
#pragma unroll
        for (int j = 0; j < 4; j++) red[wid - 1][lane][j] = acc[j];
    }
    __syncthreads();
    if (wid == 0) {
#pragma unroll
        for (int k = 0; k < KS - 1; k++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[j] = acc[j] + red[k][lane][j];
        if (c0 < S) { if (o0 + g < OUT) Y[(size_t)c0 * OUT + o0 + g] = acc[0]; if (o0 + g + 8 < OUT) Y[(size_t)c0 * OUT + o0 + g + 8] = acc[2]; }
        if (c1 < S) { if (o0 + g < OUT) Y[(size_t)c1 * OUT + o0 + g] = acc[1]; if (o0 + g + 8 < OUT) Y[(size_t)c1 * OUT + o0 + g + 8] = acc[3]; }
    }
}

template <int S>
static inline void launch_gemv_mcb_t(const int8_t *xq, const float *xs, const uint8_t *Wq, const void *sw, float *Y,
                                     int IN, int OUT, int sdtype, cudaStream_t st) {
    const unsigned grid = (unsigned)((OUT + 15) / 16);
    if (sdtype == 1) gemv_mcb_kernel<__half, S><<<grid, 128, 0, st>>>(xq, xs, Wq, (const __half *)sw, Y, IN, OUT);
    else             gemv_mcb_kernel<float, S><<<grid, 128, 0, st>>>(xq, xs, Wq, (const float *)sw, Y, IN, OUT);
}
// IN must be a multiple of 64 (block pairs); all 12B dense shapes are.
static inline void launch_gemv_mcb(const int8_t *xq, const float *xs, const uint8_t *Wq, const void *sw, float *Y,
                                   int IN, int OUT, int S, int sdtype, cudaStream_t st) {
    switch (S) {
        case 1: launch_gemv_mcb_t<1>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 2: launch_gemv_mcb_t<2>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 3: launch_gemv_mcb_t<3>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 4: launch_gemv_mcb_t<4>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 5: launch_gemv_mcb_t<5>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 6: launch_gemv_mcb_t<6>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        case 7: launch_gemv_mcb_t<7>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
        default: launch_gemv_mcb_t<8>(xq, xs, Wq, sw, Y, IN, OUT, sdtype, st); break;
    }
}

// Harness entry (tools/r18_gemv_bench.cu): family names "mca_r<R>u<U>", "mcb". Returns false for unknown names.
static inline bool gemv_mc_launch(const char *fam, const int8_t *xq, const float *xs, const uint8_t *Wq, const __half *sw,
                                  float *Y, int IN, int OUT, int S, cudaStream_t st = 0) {
#define MCA(RR, UU) if (!strcmp(fam, "mca_r" #RR "u" #UU)) { launch_gemv_mca_ru<RR, UU>(xq, xs, Wq, sw, Y, IN, OUT, S, 1, st); return true; }
    MCA(1, 1) MCA(1, 2) MCA(1, 4) MCA(2, 1) MCA(2, 2) MCA(2, 4) MCA(4, 1) MCA(4, 2)
#undef MCA
    if (!strcmp(fam, "mcb")) { launch_gemv_mcb(xq, xs, Wq, sw, Y, IN, OUT, S, 1, st); return true; }
    return false;
}

#endif // DGX_GEMV_MC_CUH
