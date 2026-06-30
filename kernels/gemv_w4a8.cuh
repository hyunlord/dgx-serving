#ifndef DGX_GEMV_W4A8_CUH
#define DGX_GEMV_W4A8_CUH
#include <cstdint>
#include "reduce.cuh"
#include "qscale.cuh"
#include "../runtime/config.h"   // QBLK

// W4A8 행렬-벡터곱: int8 activation(per-block scale) × int4 weight(per-block scale, zero-point 8).
//   Y[s,o] = Σ_blocks xs[s,b]·sw[o,b] · Σ_i xq_i·(wq_i − 8)     (docs/ch2b-w4a8-blueprint.md)
//   weight를 (wq−8) signed로 unpack해 DP4A하면 −8·Σxq 보정항 불필요(xsum 불요).
//   STEP2 최적화: DP4A int8 dot + lane-minor 파티션(int8 X·weight 둘 다 coalesced — STEP1 uint4 X-uncoalesce 해소).
//   1 warp/output. lane이 연속 4원소(int32 1로드), iteration이 warp폭(128원소) stride. 블록(32)당 scale.
//   per-lane 4원소는 한 블록 내(4|base, 4<32) → lane당 블록 scale 1개. 블록별 partial을 float로 누적 후 warp_reduce.
// IN은 QBLK(=32) 배수(→ 4·8 정렬). xsum 인자는 호환 위해 받되 미사용.
template<typename ST>
static __global__ void gemv_w4a8_kernel(const int8_t *xq, const float *xs, const int32_t * /*xsum*/,
                                        const uint8_t *Wq, const ST *sw,
                                        float *Y, int IN, int OUT, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long)S * OUT) return;
    int s = (int)(gw / OUT), o = (int)(gw % OUT);
    const int8_t  *xqr = xq + (size_t)s * IN;
    const float   *xsr = xs + (size_t)s * (IN / QBLK);
    const uint8_t *w   = Wq + (size_t)o * (IN / 2);
    const ST      *swr = sw + (size_t)o * (IN / QBLK);
    // (wq−8) signed int8×4 packing: 2 byte(4 nibble) → int (dp4a 입력)
    auto pk = [](int lo_byte, int hi_byte) -> int {
        return ((( lo_byte       & 0xF) - 8) & 0xFF)
             | ((((lo_byte >> 4)  & 0xF) - 8) & 0xFF) << 8
             | ((( hi_byte       & 0xF) - 8) & 0xFF) << 16
             | ((((hi_byte >> 4)  & 0xF) - 8) & 0xFF) << 24;
    };
    float acc = 0.f;
    for (int base = lane * 8; base < IN; base += 32 * 8) {        // lane-minor 8원소, warp폭 256 stride
        int blk = base / QBLK;                                    // 8|base, 8<32 → chunk가 한 블록 내
        float sc_b = xsr[blk] * sval(swr[blk]);
        unsigned W = *reinterpret_cast<const unsigned*>(w + (base >> 1));  // 4 byte = 8 nibble, 1회 로드(coalesced)
        int x0 = *reinterpret_cast<const int*>(xqr + base);      // xq[base..+3]
        int x1 = *reinterpret_cast<const int*>(xqr + base + 4);  // xq[base+4..+7]
        int dot = __dp4a(x0, pk( W & 0xFF, (W >> 8) & 0xFF), 0);          // weights base..+3
        dot     = __dp4a(x1, pk((W >> 16) & 0xFF, (W >> 24) & 0xFF), dot);// weights base+4..+7
        acc += sc_b * (float)dot;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}

static inline void launch_gemv_w4a8(const int8_t *xq, const float *xs, const int32_t *xsum,
                                    const uint8_t *Wq, const void *sw,
                                    float *Y, int IN, int OUT, int S, int scale_dtype = 0, cudaStream_t st = 0) {
    int block = 256;
    long warps = (long)S * OUT;
    long grid = (warps * 32 + block - 1) / block;
    if (scale_dtype == 1) gemv_w4a8_kernel<__half><<<(unsigned)grid, block, 0, st>>>(xq, xs, xsum, Wq, (const __half*)sw, Y, IN, OUT, S);
    else                  gemv_w4a8_kernel<float ><<<(unsigned)grid, block, 0, st>>>(xq, xs, xsum, Wq, (const float*)sw, Y, IN, OUT, S);
}

#endif // DGX_GEMV_W4A8_CUH
