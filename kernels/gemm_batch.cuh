#ifndef DGX_GEMM_BATCH_CUH
#define DGX_GEMM_BATCH_CUH
#include <cstdint>
#include "reduce.cuh"
#include "qscale.cuh"
#include "../runtime/config.h"   // QBLK
#include "gemv_q4.cuh"           // launch_gemv_q4 (S=1 fallback)

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

// ---- q4 batch gemm: weight row 1회 로드 + unpack 1회 → B행 재사용(unpack도 amortize, 챕터2b 통찰) ----
template<int B, typename ST>
static __global__ void gemm_batch_q4_kernel(const float *X, const uint8_t *Wq, const ST *sc, float *Y, int IN, int OUT) {
    long o = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (o >= OUT) return;
    const uint8_t *w = Wq + (size_t)o * (IN / 2);
    const ST *sca = sc + (size_t)o * (IN / QBLK);
    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; b++) acc[b] = 0.f;
    for (int i = lane; i < IN; i += 32) {
        uint8_t by = w[i >> 1];
        int q = (i & 1) ? (by >> 4) : (by & 0x0F);
        float wv = ((float)q - 8.0f) * sval(sca[i / QBLK]);     // unpack 1회 → B행 공유
        #pragma unroll
        for (int b = 0; b < B; b++) acc[b] += X[(size_t)b * IN + i] * wv;
    }
    #pragma unroll
    for (int b = 0; b < B; b++) { float r = warp_reduce_sum(acc[b]); if (lane == 0) Y[(size_t)b * OUT + o] = r; }
}
template<int B, typename ST>
static inline void launch_gemm_batch_q4_B_t(const float *X, const uint8_t *Wq, const ST *sc, float *Y, int IN, int OUT, cudaStream_t st) {
    int block = 256; long grid = ((long)OUT * 32 + block - 1) / block;
    gemm_batch_q4_kernel<B,ST><<<(unsigned)grid, block, 0, st>>>(X, Wq, sc, Y, IN, OUT);
}
template<typename ST>
static inline void launch_gemm_batch_q4_t(const float *X, const uint8_t *Wq, const ST *sc, float *Y, int IN, int OUT, int B, cudaStream_t st) {
    switch (B) {
        case 1:  launch_gemm_batch_q4_B_t<1>(X, Wq, sc, Y, IN, OUT, st); break;
        case 2:  launch_gemm_batch_q4_B_t<2>(X, Wq, sc, Y, IN, OUT, st); break;
        case 4:  launch_gemm_batch_q4_B_t<4>(X, Wq, sc, Y, IN, OUT, st); break;
        case 8:  launch_gemm_batch_q4_B_t<8>(X, Wq, sc, Y, IN, OUT, st); break;
        case 16: launch_gemm_batch_q4_B_t<16>(X, Wq, sc, Y, IN, OUT, st); break;
        default: { for (int b = 0; b < B; b++) launch_gemm_batch_q4_B_t<1>(X + (size_t)b*IN, Wq, sc, Y + (size_t)b*OUT, IN, OUT, st); }
    }
}
static inline void launch_gemm_batch_q4(const float *X, const uint8_t *Wq, const void *sc, float *Y, int IN, int OUT, int B, int scale_dtype = 0, cudaStream_t st = 0) {
    if (scale_dtype == 1) launch_gemm_batch_q4_t(X, Wq, (const __half*)sc, Y, IN, OUT, B, st);
    else                  launch_gemm_batch_q4_t(X, Wq, (const float*)sc, Y, IN, OUT, B, st);
}

// ---- multi-warp q4 gemm (챕터4): NW warp가 한 출력행 IN 분할 협력 + shared 축약. unpack은 per-element ----
template<int B, int NW, int ROWS, typename ST>
static __global__ void gemm_mw_q4_kernel(const float *X, const uint8_t *Wq, const ST *sc, float *Y, int IN, int OUT) {
    int lane = threadIdx.x, wid = threadIdx.y, tid = wid * 32 + lane;
    int row0 = blockIdx.x * ROWS;
    float tmp[B][ROWS];
    #pragma unroll
    for (int cc = 0; cc < B; cc++)
        #pragma unroll
        for (int r = 0; r < ROWS; r++) tmp[cc][r] = 0.f;
    for (int i = tid; i < IN; i += NW * 32) {
        float wv[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (row0 + r < OUT) {
                const uint8_t *w = Wq + (size_t)(row0 + r) * (IN / 2);
                uint8_t by = w[i >> 1]; int q = (i & 1) ? (by >> 4) : (by & 0x0F);
                wv[r] = ((float)q - 8.0f) * sval(sc[(size_t)(row0 + r) * (IN / QBLK) + i / QBLK]);
            } else wv[r] = 0.f;
        }
        #pragma unroll
        for (int cc = 0; cc < B; cc++) {
            float xv = X[(size_t)cc * IN + i];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) tmp[cc][r] += xv * wv[r];
        }
    }
    #pragma unroll
    for (int cc = 0; cc < B; cc++)
        #pragma unroll
        for (int r = 0; r < ROWS; r++) tmp[cc][r] = warp_reduce_sum(tmp[cc][r]);
    __shared__ float sh[NW][B][ROWS];
    if (lane == 0)
        #pragma unroll
        for (int cc = 0; cc < B; cc++)
            #pragma unroll
            for (int r = 0; r < ROWS; r++) sh[wid][cc][r] = tmp[cc][r];
    __syncthreads();
    if (wid == 0 && lane == 0)
        #pragma unroll
        for (int cc = 0; cc < B; cc++)
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float s = 0.f;
                #pragma unroll
                for (int wq2 = 0; wq2 < NW; wq2++) s += sh[wq2][cc][r];
                if (row0 + r < OUT) Y[(size_t)cc * OUT + row0 + r] = s;
            }
}
template<int B, int NW, int ROWS, typename ST>
static inline void launch_gemm_mw_q4_B_t(const float *X, const uint8_t *Wq, const ST *sc, float *Y, int IN, int OUT, cudaStream_t st) {
    dim3 block(32, NW); int grid = (OUT + ROWS - 1) / ROWS;
    gemm_mw_q4_kernel<B, NW, ROWS, ST><<<grid, block, 0, st>>>(X, Wq, sc, Y, IN, OUT);
}
static inline bool launch_gemm_mw_q4(const float *X, const uint8_t *Wq, const void *sc, float *Y, int IN, int OUT, int B, int scale_dtype = 0, cudaStream_t st = 0) {
    if (scale_dtype == 1) {
        switch (B) {
            case 4: launch_gemm_mw_q4_B_t<4,4,2>(X,Wq,(const __half*)sc,Y,IN,OUT,st); return true;
            case 8: launch_gemm_mw_q4_B_t<8,2,2>(X,Wq,(const __half*)sc,Y,IN,OUT,st); return true;
            default: return false;
        }
    } else {
        switch (B) {
            case 4: launch_gemm_mw_q4_B_t<4,4,2>(X,Wq,(const float*)sc,Y,IN,OUT,st); return true;
            case 8: launch_gemm_mw_q4_B_t<8,2,2>(X,Wq,(const float*)sc,Y,IN,OUT,st); return true;
            default: return false;
        }
    }
}

// ---- W4A8 batch gemm: int8 X(B행) × int4 weight. weight uint32 1로드+unpack → B행 재사용, DP4A ----
//   xq:[B,IN], xs:[B,IN/QBLK]. weight (wq−8) signed dp4a. lane-minor 8원소 chunk.
__device__ __forceinline__ int gb_pk(int lo_byte, int hi_byte) {
    return ((( lo_byte       & 0xF) - 8) & 0xFF)
         | ((((lo_byte >> 4)  & 0xF) - 8) & 0xFF) << 8
         | ((( hi_byte       & 0xF) - 8) & 0xFF) << 16
         | ((((hi_byte >> 4)  & 0xF) - 8) & 0xFF) << 24;
}

template<int B, typename ST>
static __global__ void gemm_batch_w4a8_kernel(const int8_t *xq, const float *xs, const uint8_t *Wq, const ST *sw, float *Y, int IN, int OUT) {
    long o = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (o >= OUT) return;
    const uint8_t *w = Wq + (size_t)o * (IN / 2);
    const ST *swr = sw + (size_t)o * (IN / QBLK);
    int nb = IN / QBLK;
    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; b++) acc[b] = 0.f;
    for (int base = lane * 8; base < IN; base += 32 * 8) {
        int blk = base / QBLK; float swb = sval(swr[blk]);
        unsigned W = *reinterpret_cast<const unsigned*>(w + (base >> 1));   // weight 1로드+unpack → B행 공유
        int wA = gb_pk(W & 0xFF, (W >> 8) & 0xFF), wB = gb_pk((W >> 16) & 0xFF, (W >> 24) & 0xFF);
        #pragma unroll
        for (int b = 0; b < B; b++) {
            const int8_t *xqb = xq + (size_t)b * IN;
            int x0 = *reinterpret_cast<const int*>(xqb + base), x1 = *reinterpret_cast<const int*>(xqb + base + 4);
            int dot = __dp4a(x0, wA, 0); dot = __dp4a(x1, wB, dot);
            acc[b] += xs[(size_t)b * nb + blk] * swb * (float)dot;
        }
    }
    #pragma unroll
    for (int b = 0; b < B; b++) { float r = warp_reduce_sum(acc[b]); if (lane == 0) Y[(size_t)b * OUT + o] = r; }
}
template<int B, typename ST>
static inline void launch_gemm_batch_w4a8_B_t(const int8_t *xq, const float *xs, const uint8_t *Wq, const ST *sw, float *Y, int IN, int OUT, cudaStream_t st) {
    int block = 256; long grid = ((long)OUT * 32 + block - 1) / block;
    gemm_batch_w4a8_kernel<B,ST><<<(unsigned)grid, block, 0, st>>>(xq, xs, Wq, sw, Y, IN, OUT);
}
template<typename ST>
static inline void launch_gemm_batch_w4a8_t(const int8_t *xq, const float *xs, const uint8_t *Wq, const ST *sw, float *Y, int IN, int OUT, int B, cudaStream_t st) {
    switch (B) {
        case 1:  launch_gemm_batch_w4a8_B_t<1>(xq, xs, Wq, sw, Y, IN, OUT, st); break;
        case 2:  launch_gemm_batch_w4a8_B_t<2>(xq, xs, Wq, sw, Y, IN, OUT, st); break;
        case 4:  launch_gemm_batch_w4a8_B_t<4>(xq, xs, Wq, sw, Y, IN, OUT, st); break;
        case 8:  launch_gemm_batch_w4a8_B_t<8>(xq, xs, Wq, sw, Y, IN, OUT, st); break;
        case 16: launch_gemm_batch_w4a8_B_t<16>(xq, xs, Wq, sw, Y, IN, OUT, st); break;
        default: { for (int b = 0; b < B; b++) launch_gemm_batch_w4a8_B_t<1>(xq + (size_t)b*IN, xs + (size_t)b*(IN/QBLK), Wq, sw, Y + (size_t)b*OUT, IN, OUT, st); }
    }
}
static inline void launch_gemm_batch_w4a8(const int8_t *xq, const float *xs, const uint8_t *Wq, const void *sw, float *Y, int IN, int OUT, int B, int scale_dtype = 0, cudaStream_t st = 0) {
    if (scale_dtype == 1) launch_gemm_batch_w4a8_t(xq, xs, Wq, (const __half*)sw, Y, IN, OUT, B, st);
    else                  launch_gemm_batch_w4a8_t(xq, xs, Wq, (const float*)sw, Y, IN, OUT, B, st);
}

#endif // DGX_GEMM_BATCH_CUH
