#ifndef DGX_ELEMENTWISE_CUH
#define DGX_ELEMENTWISE_CUH

// 모델 독립 elementwise 커널. residual add 등에 사용.
// y[i] += x[i]  (in-place 누적). n은 런타임 인자.

__global__ void add_inplace_kernel(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// out[i] = a[i] + b[i]
__global__ void add_kernel(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

static inline void launch_add_inplace(float *y, const float *x, int n, cudaStream_t s = 0) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_inplace_kernel<<<grid, block, 0, s>>>(y, x, n);
}

// out = a + b
static inline void launch_add(float *out, const float *a, const float *b, int n, cudaStream_t s = 0) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block, 0, s>>>(out, a, b, n);
}

// y[i] *= sigmoid(gate[i])  — Qwen3.5 full-attn output gate (gate는 sigmoid, silu 아님)
__global__ void gate_sigmoid_kernel(float *y, const float *gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] * (1.0f / (1.0f + expf(-gate[i])));
}

static inline void launch_gate_sigmoid(float *y, const float *gate, int n, cudaStream_t s = 0) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gate_sigmoid_kernel<<<grid, block, 0, s>>>(y, gate, n);
}

#endif // DGX_ELEMENTWISE_CUH
