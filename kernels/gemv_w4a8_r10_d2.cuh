#ifndef DGX_GEMV_W4A8_R10_D2_CUH
#define DGX_GEMV_W4A8_R10_D2_CUH
#include "gemv_w4a8.cuh"

// Experimental S=1 pair only. IN is a positive multiple of QBLK=32.
// Gate/up retain separate row-major weights, scales, and non-aliasing outputs.
template<typename ST>
static __global__ void gemv_w4a8_r10_d2_kernel(
    const int8_t *xq, const float *xs, const int32_t * /*xsum*/,
    const uint8_t *gate_wq, const ST *gate_sc,
    const uint8_t *up_wq, const ST *up_sc,
    float *gate, float *up, int IN, int OUT) {
    long o = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (o >= OUT) return;
    const uint8_t *gw = gate_wq + (size_t)o * (IN / 2);
    const uint8_t *uw = up_wq + (size_t)o * (IN / 2);
    const ST *gs = gate_sc + (size_t)o * (IN / QBLK);
    const ST *us = up_sc + (size_t)o * (IN / QBLK);
    float acc_gate = 0.f, acc_up = 0.f;
    for (int base = lane * 8; base < IN; base += 32 * 8) {
        int blk = base / QBLK;
        float xscale = xs[blk];
        float sc_gate = xscale * sval(gs[blk]);
        float sc_up = xscale * sval(us[blk]);
        unsigned G = *reinterpret_cast<const unsigned*>(gw + (base >> 1));
        unsigned U = *reinterpret_cast<const unsigned*>(uw + (base >> 1));
        int x0 = *reinterpret_cast<const int*>(xq + base);
        int x1 = *reinterpret_cast<const int*>(xq + base + 4);
        int dot_gate = __dp4a(x0, w4a8_pk(G & 0xFF, (G >> 8) & 0xFF), 0);
        dot_gate = __dp4a(x1, w4a8_pk((G >> 16) & 0xFF, (G >> 24) & 0xFF), dot_gate);
        int dot_up = __dp4a(x0, w4a8_pk(U & 0xFF, (U >> 8) & 0xFF), 0);
        dot_up = __dp4a(x1, w4a8_pk((U >> 16) & 0xFF, (U >> 24) & 0xFF), dot_up);
        acc_gate += sc_gate * (float)dot_gate;
        acc_up += sc_up * (float)dot_up;
    }
    acc_gate = warp_reduce_sum(acc_gate);
    acc_up = warp_reduce_sum(acc_up);
    if (lane == 0) { gate[o] = acc_gate; up[o] = acc_up; }
}

// Caller validates scale_dtype in {0,1}, positive aligned IN, and OUT > 0.
// No dispatch or quantization changes are part of this isolated candidate.
static inline void launch_gemv_w4a8_r10_d2(
    const int8_t *xq, const float *xs, const int32_t *xsum,
    const uint8_t *gate_wq, const void *gate_sc,
    const uint8_t *up_wq, const void *up_sc,
    float *gate, float *up, int IN, int OUT,
    int scale_dtype = 0, cudaStream_t st = 0) {
    int block = 256;
    unsigned grid = (unsigned)(((long)OUT * 32 + block - 1) / block);
    if (scale_dtype == 1)
        gemv_w4a8_r10_d2_kernel<__half><<<grid,block,0,st>>>(
            xq,xs,xsum,gate_wq,(const __half*)gate_sc,up_wq,(const __half*)up_sc,gate,up,IN,OUT);
    else
        gemv_w4a8_r10_d2_kernel<float><<<grid,block,0,st>>>(
            xq,xs,xsum,gate_wq,(const float*)gate_sc,up_wq,(const float*)up_sc,gate,up,IN,OUT);
}
#endif
