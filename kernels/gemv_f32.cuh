#ifndef DGX_GEMV_F32_CUH
#define DGX_GEMV_F32_CUH
#include "reduce.cuh"

// fp32 행렬-벡터곱 (양자화 안 한 작은 weight 용, 예: GDN in_proj_a/b).
//   X: [S, IN], W: [OUT, IN] row-major fp32, Y: [S, OUT].  warp당 1 (s,o).
static __global__ void gemv_f32_kernel(const float *X, const float *W, float *Y, int IN, int OUT, int S) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long)S * OUT) return;
    int s = (int)(gw / OUT), o = (int)(gw % OUT);
    const float *x = X + (size_t)s * IN;
    const float *w = W + (size_t)o * IN;
    float acc = 0.f;
    for (int i = lane; i < IN; i += 32) acc += x[i] * w[i];
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[(size_t)s * OUT + o] = acc;
}
// A1 최적화: warp당 MO output (X float4 재사용 + MO 독립 W 스트림 → MLP↑). X는 global/L1 (smem 안 씀 = bank conflict 없음).
//   가정: IN % 4 == 0 (float4). OUT%MO != 0 이면 마지막 그룹은 런타임 guard로 일부만.
template<int MO>
static __global__ void gemv_f32_mo_kernel(const float *X, const float *W, float *Y, int IN, int OUT, int S) {
    long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int gps = (OUT + MO - 1) / MO;            // groups per token
    if (warp >= (long)S * gps) return;
    int s = (int)(warp / gps), o0 = (int)(warp % gps) * MO;
    const float4 *X4 = (const float4 *)(X + (size_t)s * IN);
    int IN4 = IN >> 2;
    float acc[MO];
    #pragma unroll
    for (int m = 0; m < MO; m++) acc[m] = 0.f;
    for (int i = lane; i < IN4; i += 32) {
        float4 x = X4[i];
        #pragma unroll
        for (int m = 0; m < MO; m++) {
            int o = o0 + m; if (o >= OUT) continue;
            float4 w = ((const float4 *)(W + (size_t)o * IN))[i];
            acc[m] += x.x * w.x + x.y * w.y + x.z * w.z + x.w * w.w;
        }
    }
    #pragma unroll
    for (int m = 0; m < MO; m++) {
        float a = acc[m];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
        if (lane == 0) { int o = o0 + m; if (o < OUT) Y[(size_t)s * OUT + o] = a; }
    }
}

static inline void launch_gemv_f32(const float *X, const float *W, float *Y,
                                   int IN, int OUT, int S, cudaStream_t st = 0) {
    const int MO = 2;
    if ((IN & 3) == 0) {                       // float4 경로 (모든 본체 IN은 4의 배수)
        int block = 256; long gps = (OUT + MO - 1) / MO; long warps = (long)S * gps;
        long grid = (warps * 32 + block - 1) / block;
        gemv_f32_mo_kernel<MO><<<(unsigned)grid, block, 0, st>>>(X, W, Y, IN, OUT, S);
    } else {                                   // fallback (naive)
        int block = 256; long warps = (long)S * OUT; long grid = (warps * 32 + block - 1) / block;
        gemv_f32_kernel<<<(unsigned)grid, block, 0, st>>>(X, W, Y, IN, OUT, S);
    }
}

#endif // DGX_GEMV_F32_CUH
