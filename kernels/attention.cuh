#ifndef DGX_ATTENTION_CUH
#define DGX_ATTENTION_CUH
#include <float.h>

// GQA causal attention (prefill, 모델 독립).
//   Q: [S, NQ, HD], K/V: [S, NKV, HD], O: [S, NQ, HD]. scale = 1/sqrt(HD).
//   블록 = (head h, query qi). blockDim.x = HD. threadIdx.x 가 dim(d) 와 key-pos(j) 두 역할.
//   ⇒ 제약: S <= HD (1단계 prefill 용도). 동적 shared: qs[HD] + scores[S] + probs[S].
//   kv head = h / (NQ/NKV) (GQA group).
static __global__ void attn_kernel(const float *Q, const float *K, const float *V, float *O,
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

// Decode/verify attention over a KV cache (위치 독립 cache, 임의 길이). 2a attn_kernel 의 S<=HD 제약 없음.
//   Q: [Sq, NQ, HD] (decode Sq=1, verify Sq=K+1). Kc/Vc: [Lc, NKV, HD] cache (신규토큰 이미 append됨).
//   query i (절대위치 base+i) 는 cache[0 .. base+i] 를 causal attend. scale=1/sqrt(HD), GQA.
//   block=(query i, head h), blockDim=HD. dynamic shared: qs[HD] + scores[base+Sq] (>= 최대 attendlen).
static __global__ void attn_decode_kernel(const float *Q, const float *Kc, const float *Vc, float *O,
                                   int Sq, int NQ, int NKV, int HD, int base) {
    int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int GRP = NQ / NKV, kv = h / GRP;
    int attendlen = base + i + 1;
    extern __shared__ float sm[];
    float *qs = sm;                 // [HD]
    float *scores = sm + HD;        // [attendlen]
    if (d < HD) qs[d] = Q[((size_t)i * NQ + h) * HD + d];
    __syncthreads();
    // scores[j] = (q · Kc[j,kv]) / sqrt(HD), j=0..attendlen-1 (thread가 strided j 담당, 각 full dot)
    for (int j = d; j < attendlen; j += blockDim.x) {
        const float *kp = Kc + ((size_t)j * NKV + kv) * HD;
        float dot = 0.f;
        for (int t = 0; t < HD; t++) dot += qs[t] * kp[t];
        scores[j] = dot / sqrtf((float)HD);
    }
    __syncthreads();
    __shared__ float ssum;
    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < attendlen; j++) m = fmaxf(m, scores[j]);
        float s = 0.f;
        for (int j = 0; j < attendlen; j++) { scores[j] = expf(scores[j] - m); s += scores[j]; }
        ssum = s;
    }
    __syncthreads();
    // out[d] = Σ_j (scores[j]/sum) * Vc[j,kv,d]
    if (d < HD) {
        float acc = 0.f;
        for (int j = 0; j < attendlen; j++) acc += scores[j] * Vc[((size_t)j * NKV + kv) * HD + d];
        O[((size_t)i * NQ + h) * HD + d] = acc / ssum;
    }
}

static inline void launch_attn_decode(const float *Q, const float *Kc, const float *Vc, float *O,
                                      int Sq, int NQ, int NKV, int HD, int base, cudaStream_t st = 0) {
    dim3 grid(Sq, NQ);
    size_t shmem = (size_t)(HD + base + Sq) * sizeof(float);
    attn_decode_kernel<<<grid, HD, shmem, st>>>(Q, Kc, Vc, O, Sq, NQ, NKV, HD, base);
}

// ---- batch decode attention (챕터3): B개 독립 시퀀스, 각 1토큰(위치 base), 각자 KV 캐시 ----
//   Q,O: [B, NQ, HD]. Kc/Vc: [B, max_len, NKV, HD] (시퀀스별), kv_stride = max_len*NKV*HD.
//   각 row b는 자기 KV[b][0..base] attend (attendlen = base+1 공통, same-length).
static __global__ void attn_decode_batch_kernel(const float *Q, const float *Kc, const float *Vc, float *O,
                                   int NQ, int NKV, int HD, int base, long kv_stride) {
    int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int GRP = NQ / NKV, kv = h / GRP;
    int attendlen = base + 1;
    const float *Kcb = Kc + (size_t)b * kv_stride;     // 시퀀스 b의 KV
    const float *Vcb = Vc + (size_t)b * kv_stride;
    extern __shared__ float sm[];
    float *qs = sm, *scores = sm + HD;
    if (d < HD) qs[d] = Q[((size_t)b * NQ + h) * HD + d];
    __syncthreads();
    for (int j = d; j < attendlen; j += blockDim.x) {
        const float *kp = Kcb + ((size_t)j * NKV + kv) * HD;
        float dot = 0.f;
        for (int t = 0; t < HD; t++) dot += qs[t] * kp[t];
        scores[j] = dot / sqrtf((float)HD);
    }
    __syncthreads();
    __shared__ float ssum;
    if (d == 0) {
        float m = -FLT_MAX; for (int j = 0; j < attendlen; j++) m = fmaxf(m, scores[j]);
        float s = 0.f; for (int j = 0; j < attendlen; j++) { scores[j] = expf(scores[j] - m); s += scores[j]; }
        ssum = s;
    }
    __syncthreads();
    if (d < HD) {
        float acc = 0.f;
        for (int j = 0; j < attendlen; j++) acc += scores[j] * Vcb[((size_t)j * NKV + kv) * HD + d];
        O[((size_t)b * NQ + h) * HD + d] = acc / ssum;
    }
}
static inline void launch_attn_decode_batch(const float *Q, const float *Kc, const float *Vc, float *O,
                                            int B, int NQ, int NKV, int HD, int base, long kv_stride, cudaStream_t st = 0) {
    dim3 grid(B, NQ);
    size_t shmem = (size_t)(HD + base + 1) * sizeof(float);
    attn_decode_batch_kernel<<<grid, HD, shmem, st>>>(Q, Kc, Vc, O, NQ, NKV, HD, base, kv_stride);
}

#endif // DGX_ATTENTION_CUH
