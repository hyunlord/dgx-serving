#ifndef DGX_ELEMENTWISE_CUH
#define DGX_ELEMENTWISE_CUH
#include <cuda_bf16.h>

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

// y[i] = x[i] * s (스칼라 곱 — embed 스케일·per-layer scalar 등, 모델 독립)
static __global__ void scale_kernel(float *y, const float *x, float s, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}
static inline void launch_scale(float *y, const float *x, float s, size_t n, cudaStream_t st = 0) {
    scale_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, x, s, n);
}

// y[i] = tanh(x[i]/c) * c (logit softcapping — 모델 독립 elementwise)
static __global__ void softcap_kernel(float *y, const float *x, float c, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i] / c) * c;
}
static inline void launch_softcap(float *y, const float *x, float c, size_t n, cudaStream_t st = 0) {
    softcap_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(y, x, c, n);
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

// fp32 → bf16 변환 (KV cache 저장용). out[i] = __float2bfloat16(in[i]).
//   bf16=모델 dtype과 일치 → 정밀도 동일(변환 시점만 저장 쪽으로 이동, 결과 같음).
static __global__ void f32_to_bf16_kernel(const float *in, __nv_bfloat16 *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}
static inline void launch_f32_to_bf16(const float *in, __nv_bfloat16 *out, size_t n, cudaStream_t s = 0) {
    int block = 256;
    size_t grid = (n + block - 1) / block;
    f32_to_bf16_kernel<<<grid, block, 0, s>>>(in, out, n);
}

// 챕터9 bf16 GDN state: fp32 소스(예: spec snapshot) → state dst(fp32면 D2D memcpy, bf16면 변환).
//   ST로 템플릿(GdnStateT 타입 몰라도 dst에서 추론). GDN spec rollback(snap→committed state) 복구용.
template<typename ST>
static inline void state_from_f32(ST *dst, const float *src, size_t n, cudaStream_t s = 0) {
    if (sizeof(ST) == 4) cudaMemcpy(dst, src, n * 4, cudaMemcpyDeviceToDevice);
    else launch_f32_to_bf16(src, (__nv_bfloat16*)dst, n, s);
}

#endif // DGX_ELEMENTWISE_CUH
