#ifndef DGX_QUANTIZE_X_R10_CUH
#define DGX_QUANTIZE_X_R10_CUH
#include "quantize_x.cuh"
// Isolated candidate: original xq/xs arithmetic and geometry, no sum output.
// No engine routing. Initial engine experiment remains audited fp32-scale decode only.
static __global__ void quantize_x_q8_r10_d4_kernel(const float *X, int8_t *xq, float *xs,
                                            int IN, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 글로벌 warp = 글로벌 블록 id
    int lane = threadIdx.x & 31;
    long nblk_total = (long)S * (IN / QBLK);
    if (gw >= nblk_total) return;
    int nb = IN / QBLK;
    int s = (int)(gw / nb), b = (int)(gw % nb);
    long idx = (long)s * IN + (long)b * QBLK + lane;      // QBLK==32==warp width
    float x = X[idx];
    float amax = qx_warp_max(fabsf(x));
    float d = amax / 127.0f;
    int q = (amax == 0.0f) ? 0 : (int)rintf(x / d);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    xq[idx] = (int8_t)q;
    if (lane == 0) xs[gw] = d;
}

static inline void launch_quantize_x_q8_r10_d4(const float *X, int8_t *xq, float *xs,
                                        int IN, int S, cudaStream_t st = 0) {
    int block = 256;
    long warps = (long)S * (IN / QBLK);
    long grid = (warps * 32 + block - 1) / block;
    quantize_x_q8_r10_d4_kernel<<<(unsigned)grid, block, 0, st>>>(X, xq, xs, IN, S);
}

#endif
