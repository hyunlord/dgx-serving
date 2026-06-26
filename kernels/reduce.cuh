#ifndef DGX_REDUCE_CUH
#define DGX_REDUCE_CUH

// 모델 독립 warp/block reduce. blockDim.x <= 1024 (warp 최대 32개) 까지 일반 동작.

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// sh: 최소 (blockDim.x/32) floats. 모든 thread가 같은 합을 받음.
__inline__ __device__ float block_reduce_sum(float v, float *sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 31) >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nwarps) ? sh[lane] : 0.0f;
    if (wid == 0) { v = warp_reduce_sum(v); if (lane == 0) sh[0] = v; }
    __syncthreads();
    float r = sh[0];
    __syncthreads();
    return r;
}

#endif // DGX_REDUCE_CUH
