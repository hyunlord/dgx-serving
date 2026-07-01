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
// (wq−8) signed int8×4 packing: 2 byte(4 nibble) → int (dp4a 입력). 두 커널 공유.
static __device__ __forceinline__ int w4a8_pk(int lo_byte, int hi_byte) {
    return ((( lo_byte       & 0xF) - 8) & 0xFF)
         | ((((lo_byte >> 4)  & 0xF) - 8) & 0xFF) << 8
         | ((( hi_byte       & 0xF) - 8) & 0xFF) << 16
         | ((((hi_byte >> 4)  & 0xF) - 8) & 0xFF) << 24;
}

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
    float acc = 0.f;
    for (int base = lane * 8; base < IN; base += 32 * 8) {        // lane-minor 8원소, warp폭 256 stride
        int blk = base / QBLK;                                    // 8|base, 8<32 → chunk가 한 블록 내
        float sc_b = xsr[blk] * sval(swr[blk]);
        unsigned W = *reinterpret_cast<const unsigned*>(w + (base >> 1));  // 4 byte = 8 nibble, 1회 로드(coalesced)
        int x0 = *reinterpret_cast<const int*>(xqr + base);      // xq[base..+3]
        int x1 = *reinterpret_cast<const int*>(xqr + base + 4);  // xq[base+4..+7]
        int dot = __dp4a(x0, w4a8_pk( W & 0xFF, (W >> 8) & 0xFF), 0);          // weights base..+3
        dot     = __dp4a(x1, w4a8_pk((W >> 16) & 0xFF, (W >> 24) & 0xFF), dot);// weights base+4..+7
        acc += sc_b * (float)dot;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}

// 챕터9 roofline 실험: MLP↑ 변형 — U청크 로드를 먼저 모두 issue 후 compute (long_scoreboard latency를 outstanding
//   load U개로 숨김). acc 누적 순서가 baseline과 동일(같은 base 순열, u-순 누적) → bit-exact. U는 register 압력↑(occupancy
//   trade) → 경험적 측정. baseline gemv_w4a8_kernel과 출력 0-diff 게이트.
template<typename ST, int U>
static __global__ void gemv_w4a8_mlp_kernel(const int8_t *xq, const float *xs, const int32_t *,
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
    float acc = 0.f;
    const int STRIDE = 32*8;
    for (int g = lane * 8; g < IN; g += STRIDE * U) {
        unsigned Wv[U]; int x0v[U], x1v[U]; float scb[U]; bool ok[U];
        #pragma unroll
        for (int u=0; u<U; u++) {                                  // load phase: U개 동시 발사(MLP↑)
            int base = g + u*STRIDE;
            ok[u] = base < IN;
            if (ok[u]) {
                int blk = base / QBLK;
                scb[u] = xsr[blk] * sval(swr[blk]);
                Wv[u]  = *reinterpret_cast<const unsigned*>(w + (base >> 1));
                x0v[u] = *reinterpret_cast<const int*>(xqr + base);
                x1v[u] = *reinterpret_cast<const int*>(xqr + base + 4);
            }
        }
        #pragma unroll
        for (int u=0; u<U; u++) if (ok[u]) {                       // compute phase: u-순 누적 = baseline과 동일 순서(bit-exact)
            int dot = __dp4a(x0v[u], w4a8_pk( Wv[u] & 0xFF, (Wv[u] >> 8) & 0xFF), 0);
            dot     = __dp4a(x1v[u], w4a8_pk((Wv[u] >> 16) & 0xFF, (Wv[u] >> 24) & 0xFF), dot);
            acc += scb[u] * (float)dot;
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}
template<int U>
static inline void launch_gemv_w4a8_mlp(const int8_t *xq, const float *xs, const int32_t *xsum,
                                    const uint8_t *Wq, const void *sw, float *Y,
                                    int IN, int OUT, int S, int scale_dtype = 0, cudaStream_t st = 0) {
    long warps = (long)S * OUT; int block = 256; unsigned grid = (unsigned)((warps*32 + block-1)/block);
    if (scale_dtype == 1) gemv_w4a8_mlp_kernel<__half,U><<<grid,block,0,st>>>(xq,xs,xsum,Wq,(const __half*)sw,Y,IN,OUT,S);
    else                  gemv_w4a8_mlp_kernel<float,U><<<grid,block,0,st>>>(xq,xs,xsum,Wq,(const float*)sw,Y,IN,OUT,S);
}

// ── 소-batch W4A8 GEMV (verify Sq=2~8) ─────────────────────────────────────────
// 챕터8 STEP D: warp 1개 = 출력 o 1개. weight를 1회만 적재(DRAM)하고 S개 활성화 컬럼에
//   대해 S개 accumulator로 DP4A. weight DRAM 트래픽 = Sq=1 gemv와 동일(=1회) → 대역폭 최적.
//   (Sq=1 gemv_w4a8를 S번 부르면 같은 o weight를 s마다 재적재 → S× 트래픽; MMQ 128-tile은
//    소-Sq에 행 대부분 비워 비효율. 이 커널이 verify Sq=K amortize를 1 적재로 회복.)
//   활성화(S×IN int8 ≈ 수십 KB)는 L2 상주 → o마다 재읽기해도 DRAM 아님.
//   결과는 gemv_w4a8_kernel과 누적 순서 동일(같은 lane-minor 블록 순회) → 수치 bit-동일.
template<typename ST, int SMAX>
static __global__ void gemv_w4a8_sb_kernel(const int8_t *xq, const float *xs,
                                           const uint8_t *Wq, const ST *sw,
                                           float *Y, int IN, int OUT, int S) {
    long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= (long)OUT) return;
    int o = (int)warp;
    const uint8_t *w   = Wq + (size_t)o * (IN / 2);
    const ST      *swr = sw + (size_t)o * (IN / QBLK);
    float acc[SMAX];
    #pragma unroll
    for (int s = 0; s < SMAX; s++) acc[s] = 0.f;
    for (int base = lane * 8; base < IN; base += 32 * 8) {
        int blk = base / QBLK;
        unsigned W = *reinterpret_cast<const unsigned*>(w + (base >> 1));  // weight 1회 적재(o당)
        int pk0 = w4a8_pk( W & 0xFF, (W >> 8) & 0xFF);
        int pk1 = w4a8_pk((W >> 16) & 0xFF, (W >> 24) & 0xFF);
        float swb = sval(swr[blk]);
        for (int s = 0; s < S; s++) {                            // S 활성화 컬럼 (L2 상주, weight 재적재 없음)
            const int8_t *xqr = xq + (size_t)s * IN;
            int x0 = *reinterpret_cast<const int*>(xqr + base);
            int x1 = *reinterpret_cast<const int*>(xqr + base + 4);
            int dot = __dp4a(x0, pk0, 0);
            dot     = __dp4a(x1, pk1, dot);
            acc[s] += xs[(size_t)s * (IN / QBLK) + blk] * swb * (float)dot;
        }
    }
    for (int s = 0; s < S; s++) {
        float r = warp_reduce_sum(acc[s]);
        if (lane == 0) Y[(size_t)s * OUT + o] = r;
    }
}

// (챕터9 SB-MLP 실험 제거: SB는 S-accumulator로 이미 ILP 충분 — weight U-unroll이 occupancy만↓, spec 21.3→20.7 역효과.)

static inline void launch_gemv_w4a8(const int8_t *xq, const float *xs, const int32_t *xsum,
                                    const uint8_t *Wq, const void *sw,
                                    float *Y, int IN, int OUT, int S, int scale_dtype = 0, cudaStream_t st = 0) {
    int block = 256;
    long warps = (long)S * OUT;
    long grid = (warps * 32 + block - 1) / block;
    if (scale_dtype == 1) gemv_w4a8_kernel<__half><<<(unsigned)grid, block, 0, st>>>(xq, xs, xsum, Wq, (const __half*)sw, Y, IN, OUT, S);
    else                  gemv_w4a8_kernel<float ><<<(unsigned)grid, block, 0, st>>>(xq, xs, xsum, Wq, (const float*)sw, Y, IN, OUT, S);
}

// 소-batch W4A8 GEMV 런처. grid = OUT warps (S 무관 — weight 1회 적재). S<=16 전제.
//   SMAX dispatch(de-risk tools/probe_sb16): S≤8=SMAX8(regs40, occ100%, 최적) / 9~16=SMAX16(regs48, occ83%, 그래도
//   gemm_batch B×reload 대비 S=12 5.85×). B≤8 회귀 0(구조적). B=16은 호출측이 gemm_batch<16> 특수화로 보냄(더 빠름).
static inline void launch_gemv_w4a8_sb(const int8_t *xq, const float *xs,
                                       const uint8_t *Wq, const void *sw,
                                       float *Y, int IN, int OUT, int S, int scale_dtype = 0, cudaStream_t st = 0) {
    int block = 256;
    long grid = ((long)OUT * 32 + block - 1) / block;
    if (S <= 8) {
        if (scale_dtype == 1) gemv_w4a8_sb_kernel<__half,8><<<(unsigned)grid, block, 0, st>>>(xq, xs, Wq, (const __half*)sw, Y, IN, OUT, S);
        else                  gemv_w4a8_sb_kernel<float, 8><<<(unsigned)grid, block, 0, st>>>(xq, xs, Wq, (const float*)sw, Y, IN, OUT, S);
    } else {  // 9~16: SMAX16(1적재로 gemm_batch 9~15 collapse 제거)
        if (scale_dtype == 1) gemv_w4a8_sb_kernel<__half,16><<<(unsigned)grid, block, 0, st>>>(xq, xs, Wq, (const __half*)sw, Y, IN, OUT, S);
        else                  gemv_w4a8_sb_kernel<float, 16><<<(unsigned)grid, block, 0, st>>>(xq, xs, Wq, (const float*)sw, Y, IN, OUT, S);
    }
}

#endif // DGX_GEMV_W4A8_CUH
