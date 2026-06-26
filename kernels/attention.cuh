#ifndef DGX_ATTENTION_CUH
#define DGX_ATTENTION_CUH
#include <float.h>

// GQA causal attention (prefill, 모델 독립).
//   Q: [S, NQ, HD], K/V: [S, NKV, HD], O: [S, NQ, HD]. scale = 1/sqrt(HD).
//   블록 = (head h, query qi). blockDim.x = HD. threadIdx.x 가 dim(d) 와 key-pos(j) 두 역할.
//   ⇒ 제약: S <= HD (1단계 prefill 용도). 동적 shared: qs[HD] + scores[S] + probs[S].
//   kv head = h / (NQ/NKV) (GQA group).
__global__ void attn_kernel(const float *Q, const float *K, const float *V, float *O,
                            int S, int NQ, int NKV, int HD) {
    int h = blockIdx.x, qi = blockIdx.y, d = threadIdx.x;
    if (qi >= S) return;
    int GRP = NQ / NKV, kv = h / GRP;
    extern __shared__ float sm[];
    float *qs = sm;            // [HD]
    float *scores = sm + HD;   // [S]
    float *probs = scores + S; // [S]

    if (d < HD) qs[d] = Q[((size_t)qi * NQ + h) * HD + d];
    __syncthreads();

    // causal: key pos j = d, j <= qi
    if (d <= qi) {
        const float *kp = K + ((size_t)d * NKV + kv) * HD;
        float dot = 0.f;
        for (int t = 0; t < HD; t++) dot += qs[t] * kp[t];
        scores[d] = dot / sqrtf((float)HD);
    }
    __syncthreads();

    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j <= qi; j++) m = fmaxf(m, scores[j]);
        float sum = 0.f;
        for (int j = 0; j <= qi; j++) { probs[j] = expf(scores[j] - m); sum += probs[j]; }
        for (int j = 0; j <= qi; j++) probs[j] /= sum;
    }
    __syncthreads();

    if (d < HD) {
        float acc = 0.f;
        for (int j = 0; j <= qi; j++) acc += probs[j] * V[((size_t)j * NKV + kv) * HD + d];
        O[((size_t)qi * NQ + h) * HD + d] = acc;
    }
}

static inline void launch_attn(const float *Q, const float *K, const float *V, float *O,
                               int S, int NQ, int NKV, int HD, cudaStream_t st = 0) {
    dim3 grid(NQ, S);
    size_t shmem = (size_t)(HD + 2 * S) * sizeof(float);
    attn_kernel<<<grid, HD, shmem, st>>>(Q, K, V, O, S, NQ, NKV, HD);
}

#endif // DGX_ATTENTION_CUH
