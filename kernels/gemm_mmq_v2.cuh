#pragma once
// R20 PART C: prefill W4A8 MMQ (128x128 tile, 8 warps 2x4, mma m16n8k32 s8) with the same outputs as
// gemm_mmq_kernel<KT> (kernels/gemm_mmq.cuh). Every mma receives the same int8 A/B fragments and every
// accumulator update is the same expression acc += c * xr * scl in the same order; only data movement changes:
//   C1  the 4-bit B tile is unpacked to int8 once per stage (sBu[n][halfword]) instead of by every warp at every
//       k-step; the fragment word for (g, t) is halfword ks*8 + t (and + 4), exactly the bits mmq_unpack4v saw.
//   C2  row/column scales travel with the tiles through cp.async (double-buffered, raw ST, converted with sval at
//       use) instead of a synchronous gather before the barrier.
//   C3  16-byte cp.async for the A tile (and the B tile when C1 pads sB rows to 16-byte multiples).
// Barriers per K-tile stay at two (C1 moves the first one before the unpack/scale writes and drops the trailing one:
// the next tile's first barrier already orders the reuse of every shared buffer). Requires K % KT == 0 (all 12B shapes); otherwise
// the launcher falls back to the original kernel.
#include "gemm_mmq.cuh"

__device__ __forceinline__ void mmq_cpa16(void *s, const void *g, bool pred) {
    unsigned a = __cvta_generic_to_shared(s); int ss = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n" ::"r"(a), "l"(g), "r"(ss));
}

template <int KT, typename ST, bool C1, bool C2, bool C3>
static __global__ void __launch_bounds__(256, 2) gemm_mmq_v2_kernel(const int8_t *X, const float *xs, const uint8_t *Wq,
                                                                    const ST *sw, float *Y, int M, int N, int K) {
    constexpr int MS = MMQ_BM / 32, NS = MMQ_BN / 32, KS = KT / QBLK;
    constexpr int BROW = C1 ? (KT / 2 + 16) : (KT / 2 + MMQ_BPAD);   // sB row bytes (16-byte multiple with C1)
    constexpr int UROW = KT / 4 + 4;                                  // sBu row ints (conflict-free fragment reads)
    constexpr bool B16 = C3 && C1;
    __shared__ __align__(16) int8_t sA[2][MMQ_BM][KT + MMQ_AP];
    __shared__ __align__(16) uint8_t sB[2][MMQ_BN][BROW];
    __shared__ int sBu[C1 ? MMQ_BN : 1][C1 ? UROW : 1];
    __shared__ float sXs[C2 ? 2 : 1][MMQ_BM][KS];
    __shared__ ST sSwR[C2 ? 2 : 1][MMQ_BN][KS];
    __shared__ float sSw[C2 ? 1 : MMQ_BN][C2 ? 1 : KS];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, wm = warp >> 2, wn = warp & 3;
    int m0 = blockIdx.y * MMQ_BM, n0 = blockIdx.x * MMQ_BN, nblk = K / QBLK, K2 = K / 2, niter = (nblk + KS - 1) / KS;
    int arow = lane & 15, ahalf = lane >> 4;
    float acc[MS][NS][4];
#pragma unroll
    for (int a = 0; a < MS; a++) for (int b = 0; b < NS; b++) for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

    auto load = [&](int buf, int it) {
        int k0 = it * KT;
        if (C3) {
#pragma unroll
            for (int i = tid; i < MMQ_BM * KT / 16; i += 256) { int m = i / (KT / 16), kw = (i % (KT / 16)) * 16, gm = m0 + m;
                mmq_cpa16(&sA[buf][m][kw], &X[(size_t)(gm < M ? gm : 0) * K + k0 + kw], gm < M); }
        } else {
#pragma unroll
            for (int i = tid; i < MMQ_BM * KT / 4; i += 256) { int m = i / (KT / 4), kw = (i % (KT / 4)) * 4, gm = m0 + m, gk = k0 + kw;
                mmq_cpa4(&sA[buf][m][kw], &X[(size_t)gm * K + gk], gm < M && gk < K); }
        }
        if (B16) {
#pragma unroll
            for (int i = tid; i < MMQ_BN * (KT / 2) / 16; i += 256) { int n = i / (KT / 2 / 16), kw = (i % (KT / 2 / 16)) * 16, gn = n0 + n;
                mmq_cpa16(&sB[buf][n][kw], &Wq[(size_t)(gn < N ? gn : 0) * K2 + k0 / 2 + kw], gn < N); }
        } else {
#pragma unroll
            for (int i = tid; i < MMQ_BN * (KT / 2) / 4; i += 256) { int n = i / (KT / 2 / 4), kw = (i % (KT / 2 / 4)) * 4, gn = n0 + n, gk = k0 + kw * 2;
                mmq_cpa4(&sB[buf][n][kw], &Wq[(size_t)gn * K2 + k0 / 2 + kw], gn < N && gk < K); }
        }
        if (C2) {   // KS consecutive scales per row: KS*4 bytes (xs) and KS*sizeof(ST) bytes (sw), 4-byte chunks
            const int bk0 = it * KS;
#pragma unroll
            for (int i = tid; i < MMQ_BM * KS; i += 256) { int m = i / KS, b = i % KS, gm = m0 + m;
                mmq_cpa4(&sXs[buf][m][b], &xs[(size_t)(gm < M ? gm : 0) * nblk + bk0 + b], gm < M); }
            constexpr int CH = KS * (int)sizeof(ST) / 4;
#pragma unroll
            for (int i = tid; i < MMQ_BN * CH; i += 256) { int n = i / CH, c = i % CH, gn = n0 + n;
                mmq_cpa4(reinterpret_cast<char *>(&sSwR[buf][n][0]) + 4 * c,
                         reinterpret_cast<const char *>(&sw[(size_t)(gn < N ? gn : 0) * nblk + bk0]) + 4 * c, gn < N); }
        }
        asm volatile("cp.async.commit_group;\n");
    };
    load(0, 0);
    for (int it = 0; it < niter; it++) {
        const int cur = it & 1;
        asm volatile("cp.async.wait_group 0;\n");
        if (C1) __syncthreads();   // C1: everything below writes shared buffers the previous tile may still read
        if (!C2) {
#pragma unroll
            for (int i = tid; i < MMQ_BM * KS; i += 256) { int m = i / KS, b = i % KS, bk = it * KS + b, gm = m0 + m;
                sXs[0][m][b] = (gm < M && bk < nblk) ? xs[(size_t)gm * nblk + bk] : 0.f; }
#pragma unroll
            for (int i = tid; i < MMQ_BN * KS; i += 256) { int n = i / KS, b = i % KS, bk = it * KS + b, gn = n0 + n;
                sSw[n][b] = (gn < N && bk < nblk) ? sval(sw[(size_t)gn * nblk + bk]) : 0.f; }
        }
        if (C1) {
#pragma unroll
            for (int i = tid; i < MMQ_BN * KT / 4; i += 256) { int n = i / (KT / 4), j = i % (KT / 4);
                sBu[n][j] = mmq_unpack4v(*reinterpret_cast<const unsigned short *>(&sB[cur][n][j * 2])); }
        }
        __syncthreads();
        if (it + 1 < niter) load((it + 1) & 1, it + 1);
        const int sc = C2 ? cur : 0;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            int af[MS][4];
#pragma unroll
            for (int ms = 0; ms < MS; ms++) {
                unsigned ad = __cvta_generic_to_shared(&sA[cur][wm * 64 + ms * 16 + arow][ks * 32 + ahalf * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];\n"
                             : "=r"(af[ms][0]), "=r"(af[ms][1]), "=r"(af[ms][2]), "=r"(af[ms][3]) : "r"(ad));
            }
            int bf[NS][2];
#pragma unroll
            for (int ns = 0; ns < NS; ns++) { int nb = wn * 32 + ns * 8;
                if (C1) { bf[ns][0] = sBu[nb + g][ks * 8 + t]; bf[ns][1] = sBu[nb + g][ks * 8 + 4 + t]; }
                else { int kb = ks * 16;
                    unsigned w0 = *(const unsigned *)&sB[cur][nb + g][kb + (t >> 1) * 4], w1 = *(const unsigned *)&sB[cur][nb + g][kb + 8 + (t >> 1) * 4];
                    bf[ns][0] = mmq_unpack4v((t & 1) ? (w0 >> 16) : w0); bf[ns][1] = mmq_unpack4v((t & 1) ? (w1 >> 16) : w1); }
            }
            float xr[MS][2], scl[NS][2];
#pragma unroll
            for (int ms = 0; ms < MS; ms++) { int r = wm * 64 + ms * 16 + g; xr[ms][0] = sXs[sc][r][ks]; xr[ms][1] = sXs[sc][r + 8][ks]; }
#pragma unroll
            for (int ns = 0; ns < NS; ns++) { int cc = wn * 32 + ns * 8 + t * 2;
                if (C2) { scl[ns][0] = sval(sSwR[sc][cc][ks]); scl[ns][1] = sval(sSwR[sc][cc + 1][ks]); }
                else    { scl[ns][0] = sSw[cc][ks]; scl[ns][1] = sSw[cc + 1][ks]; } }
#pragma unroll
            for (int ms = 0; ms < MS; ms++) {
#pragma unroll
                for (int ns = 0; ns < NS; ns++) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                 : "r"(af[ms][0]), "r"(af[ms][1]), "r"(af[ms][2]), "r"(af[ms][3]), "r"(bf[ns][0]), "r"(bf[ns][1]));
                    acc[ms][ns][0] += c0 * xr[ms][0] * scl[ns][0]; acc[ms][ns][1] += c1 * xr[ms][0] * scl[ns][1];
                    acc[ms][ns][2] += c2 * xr[ms][1] * scl[ns][0]; acc[ms][ns][3] += c3 * xr[ms][1] * scl[ns][1];
                }
            }
        }
        if (!C1) __syncthreads();
    }
#pragma unroll
    for (int ms = 0; ms < MS; ms++) { int mb = wm * 64 + ms * 16;
#pragma unroll
        for (int ns = 0; ns < NS; ns++) { int nb = wn * 32 + ns * 8;
            int rm0 = m0 + mb + g, rm1 = m0 + mb + g + 8, cn0 = n0 + nb + t * 2, cn1 = n0 + nb + t * 2 + 1;
            if (rm0 < M) { if (cn0 < N) Y[(size_t)rm0 * N + cn0] = acc[ms][ns][0]; if (cn1 < N) Y[(size_t)rm0 * N + cn1] = acc[ms][ns][1]; }
            if (rm1 < M) { if (cn0 < N) Y[(size_t)rm1 * N + cn0] = acc[ms][ns][2]; if (cn1 < N) Y[(size_t)rm1 * N + cn1] = acc[ms][ns][3]; }
        }
    }
}

// Gemma 4 prefill dispatch (models/gemma4_decode.cu). Knobs: G4_MMQ_UNPACK (C1), G4_MMQ_SCALE_ASYNC (C2),
// G4_MMQ_CP16 (C3). All off -> launch_gemm_mmq_w4a8_auto (original path, MMQ256 still honored).
template <typename ST, bool C1, bool C2>
static inline void mmq_v2_l3(bool c3, dim3 grid, cudaStream_t st, const int8_t *X, const float *xs, const uint8_t *Wq, const void *sw,
                             float *Y, int M, int N, int K) {
    if (c3) gemm_mmq_v2_kernel<MMQ_KT, ST, C1, C2, true><<<grid, 256, 0, st>>>(X, xs, Wq, (const ST *)sw, Y, M, N, K);
    else    gemm_mmq_v2_kernel<MMQ_KT, ST, C1, C2, false><<<grid, 256, 0, st>>>(X, xs, Wq, (const ST *)sw, Y, M, N, K);
}
template <typename ST>
static inline void mmq_v2_l(bool c1, bool c2, bool c3, dim3 grid, cudaStream_t st, const int8_t *X, const float *xs, const uint8_t *Wq,
                            const void *sw, float *Y, int M, int N, int K) {
    if (c1) { if (c2) mmq_v2_l3<ST, true, true>(c3, grid, st, X, xs, Wq, sw, Y, M, N, K); else mmq_v2_l3<ST, true, false>(c3, grid, st, X, xs, Wq, sw, Y, M, N, K); }
    else    { if (c2) mmq_v2_l3<ST, false, true>(c3, grid, st, X, xs, Wq, sw, Y, M, N, K); else mmq_v2_l3<ST, false, false>(c3, grid, st, X, xs, Wq, sw, Y, M, N, K); }
}
static inline void launch_gemm_mmq_w4a8_g4(const int8_t *X, const float *xs, const uint8_t *Wq, const void *sw, float *Y,
                                           int M, int N, int K, int scale_dtype = 0, cudaStream_t st = 0) {
    static const int c1 = knob_raw("G4_MMQ_UNPACK") ? atoi(knob_raw("G4_MMQ_UNPACK")) : 0;
    static const int c2 = knob_raw("G4_MMQ_SCALE_ASYNC") ? atoi(knob_raw("G4_MMQ_SCALE_ASYNC")) : 0;
    static const int c3 = knob_raw("G4_MMQ_CP16") ? atoi(knob_raw("G4_MMQ_CP16")) : 0;
    static const int sh = knob_raw("MMQ256") ? atoi(knob_raw("MMQ256")) : 0;
    if ((!c1 && !c2 && !c3) || sh != 0 || K % MMQ_KT != 0) { launch_gemm_mmq_w4a8_auto(X, xs, Wq, sw, Y, M, N, K, scale_dtype, st); return; }
    dim3 grid((N + MMQ_BN - 1) / MMQ_BN, (M + MMQ_BM - 1) / MMQ_BM);
    if (scale_dtype == 1) mmq_v2_l<__half>(c1, c2, c3, grid, st, X, xs, Wq, sw, Y, M, N, K);
    else                  mmq_v2_l<float>(c1, c2, c3, grid, st, X, xs, Wq, sw, Y, M, N, K);
}
