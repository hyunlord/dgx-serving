#ifndef DGX_GEMV_W4A8_CUH
#define DGX_GEMV_W4A8_CUH
#include <cstdint>
#include "reduce.cuh"
#include "../runtime/config.h"   // QBLK

// W4A8 행렬-벡터곱: int8 activation(per-block scale) × int4 weight(per-block scale, zero-point 8).
//   Y[s,o] = Σ_blocks xs[s,b]·sw[o,b] · Σ_i xq_i·(wq_i − 8)     (docs/ch2b-w4a8-blueprint.md)
//   weight를 (wq−8) signed로 unpack해 DP4A하면 −8·Σxq 보정항 불필요(xsum 불요).
//   STEP2 최적화: DP4A int8 dot + lane-minor 파티션(int8 X·weight 둘 다 coalesced — STEP1 uint4 X-uncoalesce 해소).
//   1 warp/output. lane이 연속 4원소(int32 1로드), iteration이 warp폭(128원소) stride. 블록(32)당 scale.
//   per-lane 4원소는 한 블록 내(4|base, 4<32) → lane당 블록 scale 1개. 블록별 partial을 float로 누적 후 warp_reduce.
// IN은 QBLK(=32) 배수(→ 4·8 정렬). xsum 인자는 호환 위해 받되 미사용.
static __global__ void gemv_w4a8_kernel(const int8_t *xq, const float *xs, const int32_t * /*xsum*/,
                                        const uint8_t *Wq, const float *sw,
                                        float *Y, int IN, int OUT, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long)S * OUT) return;
    int s = (int)(gw / OUT), o = (int)(gw % OUT);
    const int8_t  *xqr = xq + (size_t)s * IN;
    const float   *xsr = xs + (size_t)s * (IN / QBLK);
    const uint8_t *w   = Wq + (size_t)o * (IN / 2);
    const float   *swr = sw + (size_t)o * (IN / QBLK);
    float acc = 0.f;
    for (int base = lane * 4; base < IN; base += 32 * 4) {        // lane-minor: 연속 4원소, warp폭 128 stride
        int xqi = *reinterpret_cast<const int*>(xqr + base);     // 4 int8 (coalesced across warp)
        const uint8_t *wp = w + (base >> 1);                     // 2 byte = 4 nibble
        int b0 = wp[0], b1 = wp[1];
        // (wq − 8) signed int8 ×4 packed
        int wpk = ((( b0       & 0xF) - 8) & 0xFF)
                | ((((b0 >> 4)  & 0xF) - 8) & 0xFF) << 8
                | ((( b1       & 0xF) - 8) & 0xFF) << 16
                | ((((b1 >> 4)  & 0xF) - 8) & 0xFF) << 24;
        int dot = __dp4a(xqi, wpk, 0);                           // Σ xq·(wq−8) (4원소)
        int blk = base / QBLK;
        acc += xsr[blk] * swr[blk] * (float)dot;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}

static inline void launch_gemv_w4a8(const int8_t *xq, const float *xs, const int32_t *xsum,
                                    const uint8_t *Wq, const float *sw,
                                    float *Y, int IN, int OUT, int S, cudaStream_t st = 0) {
    int block = 256;
    long warps = (long)S * OUT;
    long grid = (warps * 32 + block - 1) / block;
    gemv_w4a8_kernel<<<(unsigned)grid, block, 0, st>>>(xq, xs, xsum, Wq, sw, Y, IN, OUT, S);
}

#endif // DGX_GEMV_W4A8_CUH
