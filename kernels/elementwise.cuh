#ifndef DGX_ELEMENTWISE_CUH
#define DGX_ELEMENTWISE_CUH

// 모델 독립 elementwise 커널. residual add 등에 사용.
// y[i] += x[i]  (in-place 누적). n은 런타임 인자.

static __global__ void add_inplace_kernel(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// out[i] = a[i] + b[i]
static __global__ void add_kernel(float *out, const float *a, const float *b, int n) {
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
static __global__ void gate_sigmoid_kernel(float *y, const float *gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] * (1.0f / (1.0f + expf(-gate[i])));
}

static inline void launch_gate_sigmoid(float *y, const float *gate, int n, cudaStream_t s = 0) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gate_sigmoid_kernel<<<grid, block, 0, s>>>(y, gate, n);
}

// MoE expert 누적: y[i] += alpha * x[i] (alpha = host scalar routing weight)
static __global__ void axpy_acc_kernel(float *y, float alpha, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}
static inline void launch_axpy_acc(float *y, float alpha, const float *x, int n, cudaStream_t s = 0) {
    int block = 256, grid = (n + block - 1) / block;
    axpy_acc_kernel<<<grid, block, 0, s>>>(y, alpha, x, n);
}

// MoE shared expert 누적: out[s,i] += sigmoid(glog[s]) * so[s,i] (토큰별 sigmoid 게이트)
static __global__ void moe_shared_add_kernel(float *out, const float *so, const float *glog, int S, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * H) return;
    float g = 1.0f / (1.0f + expf(-glog[idx / H]));
    out[idx] += g * so[idx];
}
static inline void launch_moe_shared_add(float *out, const float *so, const float *glog, int S, int H, cudaStream_t st = 0) {
    int block = 256, grid = (S * H + block - 1) / block;
    moe_shared_add_kernel<<<grid, block, 0, st>>>(out, so, glog, S, H);
}

#endif // DGX_ELEMENTWISE_CUH
