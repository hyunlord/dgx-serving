#ifndef DGX_EMBEDDING_CUH
#define DGX_EMBEDDING_CUH

// 임베딩 룩업 (모델 독립). ids: [S], W: [V, H] fp32, Y: [S, H]. 블록당 1 토큰.
static __global__ void embedding_kernel(const int *ids, const float *W, float *Y, int S, int H) {
    int s = blockIdx.x;
    if (s >= S) return;
    int id = ids[s];
    const float *wr = W + (size_t)id * H;
    float *yr = Y + (size_t)s * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) yr[i] = wr[i];
}

static inline void launch_embedding(const int *ids, const float *W, float *Y,
                                    int S, int H, cudaStream_t st = 0) {
    embedding_kernel<<<S, 256, 0, st>>>(ids, W, Y, S, H);
}

#endif // DGX_EMBEDDING_CUH
