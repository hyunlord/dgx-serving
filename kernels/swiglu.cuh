#ifndef DGX_SWIGLU_CUH
#define DGX_SWIGLU_CUH

// SwiGLU (모델 독립): Y = silu(g) * u, silu(x) = x / (1 + e^-x). g,u,Y: [n].
static __global__ void swiglu_kernel(const float *g, const float *u, float *Y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float gg = g[i]; float s = gg / (1.0f + expf(-gg)); Y[i] = s * u[i]; }
}

static inline void launch_swiglu(const float *g, const float *u, float *Y, size_t n,
                                 cudaStream_t st = 0) {
    int block = 256;
    size_t grid = (n + block - 1) / block;
    swiglu_kernel<<<(unsigned)grid, block, 0, st>>>(g, u, Y, n);
}

#endif // DGX_SWIGLU_CUH
