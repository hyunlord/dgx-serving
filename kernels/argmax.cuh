#ifndef DGX_ARGMAX_CUH
#define DGX_ARGMAX_CUH
#include <float.h>

// device argmax over logits[n] → out[0] = index of max. 단일 블록, grid-stride 스캔 후 block-reduce.
// host argmax(V floats D2H + CPU 루프) 대체 — 4바이트만 D2H.
static __global__ void argmax_kernel(const float *x, int n, int *out) {
    int tid = threadIdx.x;
    float bv = -FLT_MAX; int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) { float v = x[i]; if (v > bv) { bv = v; bi = i; } }
    __shared__ float sv[256]; __shared__ int si[256];
    sv[tid] = bv; si[tid] = bi; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { if (sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; } }
        __syncthreads();
    }
    if (tid == 0) out[0] = si[0];
}

static inline void launch_argmax(const float *x, int n, int *out, cudaStream_t st = 0) {
    argmax_kernel<<<1, 256, 0, st>>>(x, n, out);
}

#endif // DGX_ARGMAX_CUH
