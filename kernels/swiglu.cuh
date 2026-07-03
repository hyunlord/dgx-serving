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

// GeGLU (모델 독립): Y = gelu_tanh(g) * u. gelu_tanh(x) = 0.5x(1+tanh(√(2/π)(x+0.044715x³)))
// — HF ACT2FN["gelu_pytorch_tanh"] 와 동일식 (gated MLP의 activation variant).
static __global__ void geglu_kernel(const float *g, const float *u, float *Y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = g[i];
        float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
        Y[i] = 0.5f * x * (1.0f + t) * u[i];
    }
}

static inline void launch_geglu(const float *g, const float *u, float *Y, size_t n,
                                cudaStream_t st = 0) {
    int block = 256;
    size_t grid = (n + block - 1) / block;
    geglu_kernel<<<(unsigned)grid, block, 0, st>>>(g, u, Y, n);
}

#endif // DGX_SWIGLU_CUH
