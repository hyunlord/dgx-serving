#ifndef DGX_ATTENTION_DECODE_SPLIT_CUH
#define DGX_ATTENTION_DECODE_SPLIT_CUH
// Split-KV decode attention (flash-decoding) with GQA sharing, for decode / verify / batch rows.
//   Pass 1 grid = (chunk c, kv head, row r), 256 threads. One block reads the K/V rows of one kv head
//   for keys [c*C, (c+1)*C) ∩ [lo, pos] exactly once and serves all GRP query heads of that kv head.
//   It writes the unnormalized partial (m, l, acc[HD]) per (row, q head, chunk) to scratch.
//   Pass 2 grid = (row, q head), block = HD: combines the row's valid chunks c_lo..c_hi in ascending order.
// Invariant: chunk boundaries are absolute key positions (multiples of C) and the valid chunk range is
//   a function of (pos, window, C) only. Grid size (rows, batch width, max position) only decides which
//   blocks exist, never the arithmetic of a block or of the combine, so a query row at position pos is
//   bit-identical whether computed alone (S=1), as a verify row, a concat row or a batch row.
//   No atomics. Shared memory does not depend on the context length.
// Rows: row r has position pos_arr ? pos_arr[r] : base + r*pos_step and KV at Kc + r*kv_stride
//   (kv_stride = 0 when all rows share one sequence's KV, e.g. verify).
// Layout: Q/O [rows, NQ, HD]; K/V cache [len, NKV, HD] per sequence; scratch index (r*NQ + h)*ns_cap + c.
#include "runtime/knobs.h"
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "kv_store.cuh"

#define ATTN_SPLIT_THREADS 256

static inline int attn_split_chunk() {
    static const int chunk = []() {
        int c = 64;
        if (const char *v = knob_raw("G4_ATTN_CHUNK")) {
            c = atoi(v);
            if (c != 32 && c != 64 && c != 128 && c != 256) {
                fprintf(stderr, "Invalid G4_ATTN_CHUNK=%s (32, 64, 128 or 256)\n", v); exit(1);
            }
        }
        return c; }();
    return chunk;
}

__device__ __forceinline__ int attn_split_pos(int r, int base, int pos_step, const int *pos_arr) {
    return pos_arr ? pos_arr[r] : base + r * pos_step;
}

template<int HD, int GRP, typename KT = float>
static __global__ void __launch_bounds__(ATTN_SPLIT_THREADS)
attn_split_p1_kernel(const float *Q, const KT *Kc, const KT *Vc,
                     float *pm, float *pl, float *pacc,
                     int NQ, int NKV, int base, int pos_step, const int *pos_arr, long kv_stride,
                     float scale, int window, int C, int ns_cap) {
    const int c = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
    const int pos = attn_split_pos(r, base, pos_step, pos_arr);
    const int lo = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    const int k0 = c * C, j0 = k0 > lo ? k0 : lo, j1 = (k0 + C < pos + 1) ? k0 + C : pos + 1;
    if (j0 >= j1) return;                                   // chunk outside [lo, pos]: no partial
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    constexpr int NWARP = ATTN_SPLIT_THREADS / 32;
    extern __shared__ __align__(16) float smem[];
    float *qs = smem;                                       // [GRP][HD]
    float *sc = smem + GRP * HD;                            // [GRP][C]
    const KT *K = Kc + (size_t)r * kv_stride, *V = Vc + (size_t)r * kv_stride;
    for (int i = tid; i < GRP * HD; i += ATTN_SPLIT_THREADS) {
        const int g = i / HD, d = i % HD;
        qs[i] = Q[((size_t)r * NQ + kvh * GRP + g) * HD + d];
    }
    __syncthreads();
    // Scores: one warp per key; lane covers d = lane*4 + 128*i (float4), fixed order, xor reduction.
    for (int jj = warp; jj < C; jj += NWARP) {
        const int j = k0 + jj;
        if (j < j0 || j >= j1) {
            if (lane < GRP) sc[lane * C + jj] = -FLT_MAX;
            continue;
        }
        const size_t krow = ((size_t)j * NKV + kvh) * HD;
        float part[GRP];
        #pragma unroll
        for (int g = 0; g < GRP; g++) part[g] = 0.f;
        #pragma unroll
        for (int i = 0; i < HD / 128; i++) {
            const int d = 4 * (lane + 32 * i);
            const float4 k = kv_load4(K, krow + d);
            #pragma unroll
            for (int g = 0; g < GRP; g++) {
                const float *q = qs + g * HD + d;
                part[g] = fmaf(q[0], k.x, part[g]); part[g] = fmaf(q[1], k.y, part[g]);
                part[g] = fmaf(q[2], k.z, part[g]); part[g] = fmaf(q[3], k.w, part[g]);
            }
        }
        #pragma unroll
        for (int g = 0; g < GRP; g++) {
            float v = part[g];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            part[g] = v;
        }
        if (lane == 0) {
            #pragma unroll
            for (int g = 0; g < GRP; g++) sc[g * C + jj] = part[g] * scale;
        }
    }
    __syncthreads();
    // Chunk softmax per head: max, exp, sum (lane-strided partials, xor reduction).
    for (int g = warp; g < GRP; g += NWARP) {
        float m = -FLT_MAX;
        for (int jj = lane; jj < C; jj += 32) m = fmaxf(m, sc[g * C + jj]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float l = 0.f;
        for (int jj = lane; jj < C; jj += 32) {
            const int j = k0 + jj;
            const float p = (j >= j0 && j < j1) ? expf(sc[g * C + jj] - m) : 0.f;
            sc[g * C + jj] = p; l += p;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (lane == 0) {
            const size_t oi = ((size_t)r * NQ + kvh * GRP + g) * ns_cap + c;
            pm[oi] = m; pl[oi] = l;
        }
    }
    __syncthreads();
    // Output partial: thread owns dims d = tid + 256*i; keys ascending inside the chunk.
    constexpr int ND = HD / ATTN_SPLIT_THREADS;
    float acc[GRP][ND];
    #pragma unroll
    for (int g = 0; g < GRP; g++)
        #pragma unroll
        for (int i = 0; i < ND; i++) acc[g][i] = 0.f;
    for (int j = j0; j < j1; j++) {
        const size_t vrow = ((size_t)j * NKV + kvh) * HD;
        float v[ND];
        #pragma unroll
        for (int i = 0; i < ND; i++) v[i] = kv_load(V, vrow + tid + ATTN_SPLIT_THREADS * i);
        #pragma unroll
        for (int g = 0; g < GRP; g++) {
            const float p = sc[g * C + (j - k0)];
            #pragma unroll
            for (int i = 0; i < ND; i++) acc[g][i] = fmaf(p, v[i], acc[g][i]);
        }
    }
    #pragma unroll
    for (int g = 0; g < GRP; g++) {
        const size_t oi = ((size_t)r * NQ + kvh * GRP + g) * ns_cap + c;
        #pragma unroll
        for (int i = 0; i < ND; i++) pacc[oi * HD + tid + ATTN_SPLIT_THREADS * i] = acc[g][i];
    }
}

template<int HD>
static __global__ void attn_split_combine_kernel(const float *pm, const float *pl, const float *pacc, float *O,
                                                 int NQ, int base, int pos_step, const int *pos_arr,
                                                 int window, int C, int ns_cap) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int pos = attn_split_pos(r, base, pos_step, pos_arr);
    const int lo = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    const int c_lo = lo / C, c_hi = pos / C;
    const size_t row = ((size_t)r * NQ + h) * ns_cap;
    float mg = -FLT_MAX;
    for (int c = c_lo; c <= c_hi; c++) mg = fmaxf(mg, pm[row + c]);
    float lg = 0.f, acc = 0.f;
    for (int c = c_lo; c <= c_hi; c++) {
        const float w = expf(pm[row + c] - mg);
        lg = fmaf(pl[row + c], w, lg);
        acc = fmaf(pacc[(row + c) * HD + d], w, acc);
    }
    O[((size_t)r * NQ + h) * HD + d] = acc / lg;
}

// Scratch (unnormalized partials). Reserve before any capture; reservation never shrinks.
static float *g_as_m = nullptr, *g_as_l = nullptr, *g_as_acc = nullptr;
static size_t g_as_cap = 0, g_as_acc_cap = 0;
static inline void attn_split_check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) { fprintf(stderr, "[attn_split] %s: %s\n", what, cudaGetErrorString(e)); abort(); }
}
static inline int attn_split_ns_cap(int max_len) { const int C = attn_split_chunk(); return (max_len + C - 1) / C; }
static inline void attn_split_reserve(int rows, int NQ, int max_len, int hd_max) {
    const size_t need = (size_t)rows * NQ * attn_split_ns_cap(max_len);
    if (g_as_cap < need) {
        if (g_as_m) attn_split_check(cudaFree(g_as_m), "free m");
        if (g_as_l) attn_split_check(cudaFree(g_as_l), "free l");
        attn_split_check(cudaMalloc(&g_as_m, need * sizeof(float)), "alloc m");
        attn_split_check(cudaMalloc(&g_as_l, need * sizeof(float)), "alloc l");
        g_as_cap = need;
    }
    if (g_as_acc_cap < need * hd_max) {
        if (g_as_acc) attn_split_check(cudaFree(g_as_acc), "free acc");
        attn_split_check(cudaMalloc(&g_as_acc, need * hd_max * sizeof(float)), "alloc acc");
        g_as_acc_cap = need * hd_max;
    }
}

template<int HD, int GRP, typename KT>
static inline void attn_split_launch_typed(const float *Q, const KT *Kc, const KT *Vc, float *O, int rows,
                                           int NQ, int NKV, int base, int pos_step, const int *pos_arr, int max_pos,
                                           long kv_stride, float scale, int window, int ns_cap, cudaStream_t st) {
    const int C = attn_split_chunk();
    const size_t shmem = (size_t)(GRP * HD + GRP * C) * sizeof(float);
    dim3 g1((max_pos + C) / C, NKV, rows);   // chunks 0..max_pos/C
    attn_split_p1_kernel<HD, GRP, KT><<<g1, ATTN_SPLIT_THREADS, shmem, st>>>(
        Q, Kc, Vc, g_as_m, g_as_l, g_as_acc, NQ, NKV, base, pos_step, pos_arr, kv_stride, scale, window, C, ns_cap);
    attn_split_combine_kernel<HD><<<dim3(rows, NQ), HD, 0, st>>>(
        g_as_m, g_as_l, g_as_acc, O, NQ, base, pos_step, pos_arr, window, C, ns_cap);
}

// max_pos: largest row position (host), used only for the grid. ns_cap: chunk capacity of the scratch
// reservation (attn_split_ns_cap(max_len)); rows*NQ*ns_cap must not exceed the reservation.
static inline void launch_attn_decode_split(const float *Q, const float *Kc, const float *Vc, float *O, int rows,
                                            int NQ, int NKV, int HD, int base, int pos_step, const int *pos_arr,
                                            int max_pos, long kv_stride, int ns_cap, cudaStream_t st = 0,
                                            float scale = 0.f, int window = 0) {
    const int C = attn_split_chunk();
    if (max_pos / C + 1 > ns_cap || (size_t)rows * NQ * ns_cap > g_as_cap || (size_t)rows * NQ * ns_cap * HD > g_as_acc_cap) {
        fprintf(stderr, "[attn_split] scratch not reserved: rows=%d NQ=%d ns_cap=%d max_pos=%d\n", rows, NQ, ns_cap, max_pos);
        abort();
    }
    const float sc = (scale > 0.f) ? scale : rsqrtf((float)HD);
    const int GRP = NQ / NKV;
    // Kc/Vc point at storage of g4_kv_dtype() (fp32 handles, see kv_store.cuh).
    auto go = [&](auto tag) {
        using KT = decltype(tag);
        const KT *K = reinterpret_cast<const KT *>(Kc), *V = reinterpret_cast<const KT *>(Vc);
        if (HD == 256 && GRP == 2) attn_split_launch_typed<256, 2, KT>(Q, K, V, O, rows, NQ, NKV, base, pos_step, pos_arr, max_pos, kv_stride, sc, window, ns_cap, st);
        else if (HD == 512 && GRP == 16) attn_split_launch_typed<512, 16, KT>(Q, K, V, O, rows, NQ, NKV, base, pos_step, pos_arr, max_pos, kv_stride, sc, window, ns_cap, st);
        else if (HD == 256 && GRP == 1) attn_split_launch_typed<256, 1, KT>(Q, K, V, O, rows, NQ, NKV, base, pos_step, pos_arr, max_pos, kv_stride, sc, window, ns_cap, st);
        else if (HD == 256 && GRP == 4) attn_split_launch_typed<256, 4, KT>(Q, K, V, O, rows, NQ, NKV, base, pos_step, pos_arr, max_pos, kv_stride, sc, window, ns_cap, st);
        else if (HD == 512 && GRP == 2) attn_split_launch_typed<512, 2, KT>(Q, K, V, O, rows, NQ, NKV, base, pos_step, pos_arr, max_pos, kv_stride, sc, window, ns_cap, st);
        else { fprintf(stderr, "[attn_split] unsupported HD=%d GRP=%d\n", HD, GRP); abort(); }
    };
    const int dt = g4_kv_dtype();
    if (dt == KV_FP32) go(float{});
    else if (dt == KV_F16) go(__half{});
    else go(__nv_bfloat16{});
}

#endif // DGX_ATTENTION_DECODE_SPLIT_CUH
