#ifndef DGX_GDN_CUH
#define DGX_GDN_CUH
#include "reduce.cuh"

// Gated Delta Net (Qwen3.5 linear attention) 커널. 모델 독립(차원은 인자).
// 사양: docs/02a-qwen35-blueprint.md §7.5 (modeling_qwen3_5.py 직접 정독).
// 1단계 방침대로 정확성 우선 — naive 순차 recurrent (병렬/chunk 최적화는 추후).

// ---- (1) depthwise causal conv1d + silu ----
//   in: [S, C] (row=time), w: [C, K] (conv1d.weight [C,1,K] squeeze), out: [S, C].
//   causal: out[t,c] = silu( Σ_{j=0..K-1} w[c,j] * in[t-(K-1)+j, c] ), in[<0]=0. (HF F.conv1d pad=K-1 후 [:S])
__global__ void gdn_conv1d_silu_kernel(const float *in, const float *w, float *out, int S, int C, int K) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)S * C) return;
    int c = (int)(idx % C), t = (int)(idx / C);
    float acc = 0.f;
    for (int j = 0; j < K; j++) {
        int ti = t - (K - 1) + j;
        if (ti >= 0) acc += w[(size_t)c * K + j] * in[(size_t)ti * C + c];
    }
    out[(size_t)t * C + c] = acc / (1.0f + expf(-acc));  // silu
}
static inline void launch_gdn_conv1d_silu(const float *in, const float *w, float *out,
                                          int S, int C, int K, cudaStream_t st = 0) {
    int block = 256; long n = (long)S * C; int grid = (int)((n + block - 1) / block);
    gdn_conv1d_silu_kernel<<<grid, block, 0, st>>>(in, w, out, S, C, K);
}

// ---- (2) decay g + beta (per head 스칼라) ----
//   a,b: [S,H] (in_proj_a/b 출력). g[s,h] = -exp(A_log[h]) * softplus(a+dt_bias[h]); beta = sigmoid(b).
__device__ __forceinline__ float softplus_stable(float x) { return x > 20.f ? x : log1pf(expf(x)); }
__global__ void gdn_decay_beta_kernel(const float *a, const float *b, const float *A_log,
                                      const float *dt_bias, float *g, float *beta, int S, int H) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)S * H) return;
    int h = (int)(idx % H);
    g[idx]    = -expf(A_log[h]) * softplus_stable(a[idx] + dt_bias[h]);
    beta[idx] = 1.0f / (1.0f + expf(-b[idx]));
}
static inline void launch_gdn_decay_beta(const float *a, const float *b, const float *A_log,
                                         const float *dt_bias, float *g, float *beta,
                                         int S, int H, cudaStream_t st = 0) {
    int block = 256; long n = (long)S * H; int grid = (int)((n + block - 1) / block);
    gdn_decay_beta_kernel<<<grid, block, 0, st>>>(a, b, A_log, dt_bias, g, beta, S, H);
}

// ---- (3) gated delta rule (naive 순차 recurrent) ----
//   Q,K,V: [S, NH, D], G,beta: [S, NH] (beta=sigmoid 적용 후), O: [S, NH, D].
//   block = head, thread = v(0..D-1), state S[k,v] 의 v열(scol[k]) 을 thread 가 소유. D<=128 가정(head_k=head_v).
//   못박은 3개(blueprint §7.5): l2norm(q)&l2norm(k) 둘다(sum, eps1e-6) → q만 *1/sqrt(D);
//   g 스칼라 decay S*=e^g; kv[v]=Σ_k S[k,v]k[k]; Δ=(v-kv)β; S[k,v]+=k[k]Δ; out[v]=Σ_k S[k,v]q[k].
__global__ void gdn_delta_rule_kernel(const float *Q, const float *K, const float *V,
                                      const float *G, const float *Bt, float *O,
                                      int S, int NH, int D) {
    int h = blockIdx.x, v = threadIdx.x;
    if (v >= D) return;
    extern __shared__ float sh[];          // qs[D], ks[D], red[32]
    float *qs = sh, *ks = sh + D, *red = sh + 2 * D;
    float scol[128];                        // S[:, v], k-dim (D<=128)
    for (int k = 0; k < D; k++) scol[k] = 0.f;
    const float invsD = rsqrtf((float)D);   // q scale = 1/sqrt(D)

    for (int t = 0; t < S; t++) {
        size_t base = ((size_t)t * NH + h) * D;
        float qe = Q[base + v], ke = K[base + v];
        // l2norm over D (sum, eps=1e-6)
        float sq = block_reduce_sum(qe * qe, red);
        float sk = block_reduce_sum(ke * ke, red);
        float qn = qe * rsqrtf(sq + 1e-6f) * invsD;   // q: l2norm 후 1/sqrt(D)
        float kn = ke * rsqrtf(sk + 1e-6f);           // k: l2norm 만
        qs[v] = qn; ks[v] = kn;
        __syncthreads();

        float g_t = expf(G[(size_t)t * NH + h]);
        float beta_t = Bt[(size_t)t * NH + h];
        float kv = 0.f;
        #pragma unroll 4
        for (int k = 0; k < D; k++) { scol[k] *= g_t; kv += scol[k] * ks[k]; }
        float delta = (V[base + v] - kv) * beta_t;
        float out = 0.f;
        #pragma unroll 4
        for (int k = 0; k < D; k++) { scol[k] += ks[k] * delta; out += scol[k] * qs[k]; }
        O[base + v] = out;
        __syncthreads();
    }
}
static inline void launch_gdn_delta_rule(const float *Q, const float *K, const float *V,
                                         const float *G, const float *Bt, float *O,
                                         int S, int NH, int D, cudaStream_t st = 0) {
    size_t shmem = (size_t)(2 * D + 32) * sizeof(float);
    gdn_delta_rule_kernel<<<NH, D, shmem, st>>>(Q, K, V, G, Bt, O, S, NH, D);
}

// ---- (4) gated RMSNorm (Qwen3_5RMSNormGated) ----
//   X,gate: [R, D], W: [D]. y = (x*rsqrt(mean(x^2)+eps) * W) * silu(gate). (weight 표준, 1+w 아님)
__global__ void rms_norm_gated_kernel(const float *X, const float *gate, const float *W,
                                      float *Y, int D, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float *xr = X + (size_t)row * D, *gr = gate + (size_t)row * D;
    float *yr = Y + (size_t)row * D;
    __shared__ float sh[32]; __shared__ float inv;
    float ss = 0.f;
    for (int i = tid; i < D; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    ss = block_reduce_sum(ss, sh);
    if (tid == 0) inv = rsqrtf(ss / D + eps);
    __syncthreads();
    for (int i = tid; i < D; i += blockDim.x) {
        float g = gr[i];
        yr[i] = (xr[i] * inv * W[i]) * (g / (1.0f + expf(-g)));
    }
}
static inline void launch_rms_norm_gated(const float *X, const float *gate, const float *W,
                                         float *Y, int rows, int D, float eps, cudaStream_t st = 0) {
    rms_norm_gated_kernel<<<rows, 128, 0, st>>>(X, gate, W, Y, D, eps);
}

#endif // DGX_GDN_CUH
