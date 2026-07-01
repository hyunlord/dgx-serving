#ifndef DGX_GDN_CUH
#define DGX_GDN_CUH
#include "reduce.cuh"
#include "../runtime/decode_state.h"

// Gated Delta Net (Qwen3.5 linear attention) 커널. 모델 독립(차원은 인자).
// 사양: docs/02a-qwen35-blueprint.md §7.5 (modeling_qwen3_5.py 직접 정독).
// 1단계 방침대로 정확성 우선 — naive 순차 recurrent (병렬/chunk 최적화는 추후).

// ---- (1) depthwise causal conv1d + silu ----
//   in: [S, C] (row=time), w: [C, K] (conv1d.weight [C,1,K] squeeze), out: [S, C].
//   causal: out[t,c] = silu( Σ_{j=0..K-1} w[c,j] * in[t-(K-1)+j, c] ), in[<0]=0. (HF F.conv1d pad=K-1 후 [:S])
static __global__ void gdn_conv1d_silu_kernel(const float *in, const float *w, float *out, int S, int C, int K) {
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

// decode/verify용 conv1d+silu: 직전 K-1 토큰 context(prev [K-1, C]) 를 prepend (zero-pad 대신).
//   in: [Sq, C], prev: [K-1, C] (시간순; prev[K-2]=가장 최근), out: [Sq, C].
//   ti = t-(K-1)+j ; ti>=0 → in[ti], ti<0 → prev[K-1+ti]. prev=NULL 이면 zero-pad(=prefill 첫 청크).
static __global__ void gdn_conv1d_silu_decode_kernel(const float *in, const float *prev, const float *w,
                                              float *out, int Sq, int C, int K) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)Sq * C) return;
    int c = (int)(idx % C), t = (int)(idx / C);
    float acc = 0.f;
    for (int j = 0; j < K; j++) {
        int ti = t - (K - 1) + j;
        float xv = 0.f;
        if (ti >= 0) xv = in[(size_t)ti * C + c];
        else if (prev) xv = prev[(size_t)(K - 1 + ti) * C + c];
        acc += w[(size_t)c * K + j] * xv;
    }
    out[(size_t)t * C + c] = acc / (1.0f + expf(-acc));
}
static inline void launch_gdn_conv1d_silu_decode(const float *in, const float *prev, const float *w,
                                                 float *out, int Sq, int C, int K, cudaStream_t st = 0) {
    int block = 256; long n = (long)Sq * C; int grid = (int)((n + block - 1) / block);
    gdn_conv1d_silu_decode_kernel<<<grid, block, 0, st>>>(in, prev, w, out, Sq, C, K);
}

// ---- 챕터9 STEP 2: tree-aware conv1d (노드별 ancestor-path window, parent fork) ----
//   linear conv는 window가 선형 prev(t-1,t-2..); tree는 노드 t의 window = ancestor-path(t→parent→..→root→convctx).
//   parent[t] = tree 부모(<t), root는 -1(convctx로 떨어짐). newest(node t)부터 parent 체인 walk, root 넘으면 convctx.
//   parent[t]=t-1(chain)이면 선형 conv와 bit-identical(아래 검증). conv window 작은 버퍼라 walk K(=3~4)스텝.
static __global__ void gdn_conv1d_silu_decode_tree_kernel(const float *in, const float *prev, const int *parent,
                                              const float *w, float *out, int Sq, int C, int K) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)Sq * C) return;
    int c = (int)(idx % C), t = (int)(idx / C);
    // window 토큰 인덱스 수집(newest-first, parent 체인 walk): widx=인덱스, wsrc 0=in/1=prev/-1=zero. K<=8.
    int widx[8], wsrc[8]; int node = t, p = K - 2;
    for (int jj = 0; jj < K; jj++) {
        if (node >= 0)            { widx[jj]=node; wsrc[jj]=0; node=parent[node]; }   // tree 노드, parent walk
        else if (prev && p >= 0)  { widx[jj]=p;    wsrc[jj]=1; p--; }                 // root 넘음 → convctx(most-recent first)
        else                      { wsrc[jj]=-1; }
    }
    // 합산은 oldest-first(j=0..K-1) — linear conv와 같은 순서 = bit-identical(float reorder 없음). jj=K-1-j가 oldest.
    float acc = 0.f;
    for (int j = 0; j < K; j++) {
        int jj = K - 1 - j; float xv = 0.f;
        if (wsrc[jj]==0) xv = in[(size_t)widx[jj]*C + c];
        else if (wsrc[jj]==1) xv = prev[(size_t)widx[jj]*C + c];
        acc += w[(size_t)c * K + j] * xv;
    }
    out[(size_t)t * C + c] = acc / (1.0f + expf(-acc));
}
static inline void launch_gdn_conv1d_silu_decode_tree(const float *in, const float *prev, const int *parent,
                                                 const float *w, float *out, int Sq, int C, int K, cudaStream_t st = 0) {
    int block = 256; long n = (long)Sq * C; int grid = (int)((n + block - 1) / block);
    gdn_conv1d_silu_decode_tree_kernel<<<grid, block, 0, st>>>(in, prev, parent, w, out, Sq, C, K);
}

// ---- batch decode conv1d (챕터3): B 독립 시퀀스, 각 1토큰 + 자기 prev(convctx[b]) ----
//   in: [B, C] (각 시퀀스 1토큰), prev: [B, K-1, C], out: [B, C]. Sq=1당 ti=j-(K-1): j<K-1→prev[b][j], j=K-1→in[b].
static __global__ void gdn_conv1d_silu_decode_batch_kernel(const float *in, const float *prev, const float *w,
                                              float *out, int B, int C, int K) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)B * C) return;
    int c = (int)(idx % C), b = (int)(idx / C);
    const float *prevb = prev + (size_t)b * (K - 1) * C;
    float acc = 0.f;
    for (int j = 0; j < K - 1; j++) acc += w[(size_t)c * K + j] * prevb[(size_t)j * C + c];
    acc += w[(size_t)c * K + (K - 1)] * in[(size_t)b * C + c];
    out[(size_t)b * C + c] = acc / (1.0f + expf(-acc));
}
static inline void launch_gdn_conv1d_silu_decode_batch(const float *in, const float *prev, const float *w,
                                                       float *out, int B, int C, int K, cudaStream_t st = 0) {
    int block = 256; long n = (long)B * C; int grid = (int)((n + block - 1) / block);
    gdn_conv1d_silu_decode_batch_kernel<<<grid, block, 0, st>>>(in, prev, w, out, B, C, K);
}

// ---- (2) decay g + beta (per head 스칼라) ----
//   a,b: [S,H] (in_proj_a/b 출력). g[s,h] = -exp(A_log[h]) * softplus(a+dt_bias[h]); beta = sigmoid(b).
static __device__ __forceinline__ float softplus_stable(float x) { return x > 20.f ? x : log1pf(expf(x)); }
static __global__ void gdn_decay_beta_kernel(const float *a, const float *b, const float *A_log,
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
static __global__ void gdn_delta_rule_kernel(const float *Q, const float *K, const float *V,
                                      const float *G, const float *Bt, float *O,
                                      int S, int NH, int D, int NHk) {
    // NH=v-head 수(state/V/G/Bt), NHk=k-head 수(Q/K). GQA: v-head h → k-head g=h*NHk/NH. NHk==NH면 동일.
    int h = blockIdx.x, v = threadIdx.x;
    if (v >= D) return;
    int g = h * NHk / NH;                   // 공유 k-head
    extern __shared__ float sh[];          // qs[D], ks[D], red[32]
    float *qs = sh, *ks = sh + D, *red = sh + 2 * D;
    float scol[128];                        // S[:, v], k-dim (D<=128)
    for (int k = 0; k < D; k++) scol[k] = 0.f;
    const float invsD = rsqrtf((float)D);   // q scale = 1/sqrt(D)

    for (int t = 0; t < S; t++) {
        size_t base = ((size_t)t * NH + h) * D;          // v-head (V/O/state)
        size_t base_qk = ((size_t)t * NHk + g) * D;      // k-head (Q/K)
        float qe = Q[base_qk + v], ke = K[base_qk + v];
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
                                         int S, int NH, int D, int NHk, cudaStream_t st = 0) {
    size_t shmem = (size_t)(2 * D + 32) * sizeof(float);
    gdn_delta_rule_kernel<<<NH, D, shmem, st>>>(Q, K, V, G, Bt, O, S, NH, D, NHk);
}

// decode/verify용 gated delta rule: 초기 state Sin 에서 시작, Sq 토큰 순차 처리,
//   각 스텝 후 state 스냅샷을 Ssnap[t] 에 기록(임시 승격용), core_attn_out 출력.
//   state 레이아웃 [NH, D(k), D(v)] = h*D*D + k*D + v. Sin=NULL 이면 0에서 시작.
//   block=head, thread=v. (delta rule 수식은 gdn_delta_rule 과 동일 — docs §7.5)
static __global__ void gdn_step_kernel(const float *Q, const float *K, const float *V,
                                const float *G, const float *Bt,
                                const GdnStateT *Sin, float *Ssnap, float *O,
                                int Sq, int NH, int D, int NHk, int snapshot_all) {
    int h = blockIdx.x, v = threadIdx.x;
    if (v >= D) return;
    int g = h * NHk / NH;                   // GQA: v-head h → 공유 k-head g
    extern __shared__ float sh[];
    float *qs = sh, *ks = sh + D, *red = sh + 2 * D;
    float scol[128];
    for (int k = 0; k < D; k++) scol[k] = Sin ? (float)Sin[((size_t)h * D + k) * D + v] : 0.f;
    const float invsD = rsqrtf((float)D);

    for (int t = 0; t < Sq; t++) {
        size_t base = ((size_t)t * NH + h) * D;          // v-head (V/O/state)
        size_t base_qk = ((size_t)t * NHk + g) * D;      // k-head (Q/K)
        float qe = Q[base_qk + v], ke = K[base_qk + v];
        float sq = block_reduce_sum(qe * qe, red);
        float sk = block_reduce_sum(ke * ke, red);
        qs[v] = qe * rsqrtf(sq + 1e-6f) * invsD;
        ks[v] = ke * rsqrtf(sk + 1e-6f);
        __syncthreads();
        float g_t = expf(G[(size_t)t * NH + h]), beta_t = Bt[(size_t)t * NH + h];
        float kv = 0.f;
        #pragma unroll 4
        for (int k = 0; k < D; k++) { scol[k] *= g_t; kv += scol[k] * ks[k]; }
        float delta = (V[base + v] - kv) * beta_t, out = 0.f;
        #pragma unroll 4
        for (int k = 0; k < D; k++) { scol[k] += ks[k] * delta; out += scol[k] * qs[k]; }
        O[base + v] = out;
        // 스냅샷: snapshot_all 이면 매 스텝 Ssnap[t](verify 승격용), 아니면 최종 state만 Ssnap[0]
        //   (긴 prefill에서 Ssnap=O(1) — Sq×NH×D² 폭발 방지, 32K 크래시 근본 원인 제거).
        if (snapshot_all || t == Sq - 1) {
            size_t slot = snapshot_all ? (size_t)t : 0;
            float *snap = Ssnap + slot * NH * D * D + (size_t)h * D * D;
            for (int k = 0; k < D; k++) snap[(size_t)k * D + v] = scol[k];
        }
        __syncthreads();
    }
}

//   snapshot_all=1(기본): 매 스텝 state 스냅샷(verify 승격용, Ssnap=[Sq,NH,D,D]).
//   snapshot_all=0: 최종 state만 Ssnap[0](plain prefill/decode — Ssnap=[NH,D,D] 1개면 충분, 긴 Sq 안전).
static inline void launch_gdn_step(const float *Q, const float *K, const float *V,
                                   const float *G, const float *Bt, const GdnStateT *Sin,
                                   float *Ssnap, float *O, int Sq, int NH, int D, int NHk,
                                   int snapshot_all = 1, cudaStream_t st = 0) {
    size_t shmem = (size_t)(2 * D + 32) * sizeof(float);
    gdn_step_kernel<<<NH, D, shmem, st>>>(Q, K, V, G, Bt, Sin, Ssnap, O, Sq, NH, D, NHk, snapshot_all);
}

// ---- STEP 1 de-risk: q/k l2norm 사전계산(state-무관 → recurrence 밖 병렬, in-loop block_reduce 제거) ----
//   Q/K: [rows, D] (rows=S*NHk). Qn=q·rsqrt(Σq²+eps)·1/√D, Kn=k·rsqrt(Σk²+eps). block=D, grid=rows.
static __global__ void gdn_qk_norm_kernel(const float *Q, const float *K, float *Qn, float *Kn, int D) {
    int row = blockIdx.x, k = threadIdx.x; if (k >= D) return;
    extern __shared__ float r[];
    float qe = Q[(size_t)row * D + k], ke = K[(size_t)row * D + k];
    float sq = block_reduce_sum(qe * qe, r);
    float sk = block_reduce_sum(ke * ke, r);
    Qn[(size_t)row * D + k] = qe * rsqrtf(sq + 1e-6f) * rsqrtf((float)D);
    Kn[(size_t)row * D + k] = ke * rsqrtf(sk + 1e-6f);
}
static inline void launch_gdn_qk_norm(const float *Q, const float *K, float *Qn, float *Kn,
                                      int rows, int D, cudaStream_t st = 0) {
    gdn_qk_norm_kernel<<<rows, D, 32 * sizeof(float), st>>>(Q, K, Qn, Kn, D);
}
// prenorm 버전 gdn_step: Qn/Kn(사전정규화) 입력 → in-loop block_reduce 제거. recurrence(t순차) 동일 → Sq일관·오차0 보존.
static __global__ void gdn_step_prenorm_kernel(const float *Qn, const float *Kn, const float *V,
                                const float *G, const float *Bt, const GdnStateT *Sin, float *Ssnap, float *O,
                                int Sq, int NH, int D, int NHk, int snapshot_all) {
    int h = blockIdx.x, v = threadIdx.x; if (v >= D) return;
    int g = h * NHk / NH;
    extern __shared__ float sh[]; float *qs = sh, *ks = sh + D;
    float scol[128];
    for (int k = 0; k < D; k++) scol[k] = Sin ? (float)Sin[((size_t)h * D + k) * D + v] : 0.f;
    for (int t = 0; t < Sq; t++) {
        size_t base = ((size_t)t * NH + h) * D, base_qk = ((size_t)t * NHk + g) * D;
        qs[v] = Qn[base_qk + v]; ks[v] = Kn[base_qk + v];
        __syncthreads();
        float g_t = expf(G[(size_t)t * NH + h]), beta_t = Bt[(size_t)t * NH + h];
        float kv = 0.f;
        #pragma unroll 4
        for (int k = 0; k < D; k++) { scol[k] *= g_t; kv += scol[k] * ks[k]; }
        float delta = (V[base + v] - kv) * beta_t, out = 0.f;
        #pragma unroll 4
        for (int k = 0; k < D; k++) { scol[k] += ks[k] * delta; out += scol[k] * qs[k]; }
        O[base + v] = out;
        if (snapshot_all || t == Sq - 1) {
            size_t slot = snapshot_all ? (size_t)t : 0;
            float *snap = Ssnap + slot * NH * D * D + (size_t)h * D * D;
            for (int k = 0; k < D; k++) snap[(size_t)k * D + v] = scol[k];
        }
        __syncthreads();
    }
}
static inline void launch_gdn_step_prenorm(const float *Qn, const float *Kn, const float *V,
                                   const float *G, const float *Bt, const GdnStateT *Sin,
                                   float *Ssnap, float *O, int Sq, int NH, int D, int NHk,
                                   int snapshot_all = 1, cudaStream_t st = 0) {
    gdn_step_prenorm_kernel<<<NH, D, (size_t)2 * D * sizeof(float), st>>>(Qn, Kn, V, G, Bt, Sin, Ssnap, O, Sq, NH, D, NHk, snapshot_all);
}

// k-split: block=D*KSPLIT, k-차원을 KSPLIT warp-그룹이 분할(thread=(v,grp), scol[D/KSPLIT]). occupancy↑(8%→KSPLIT×)로
//   k-loop shared-read stall 은닉. recurrence(t순차) 동일·부분합 결합 → 오차0·Sq일관. D는 2의 거듭제곱(=128). Qn/Kn 사전정규화.
template<int D, int KSPLIT>
static __global__ void __launch_bounds__(D*KSPLIT, 1) gdn_step_ksplit_kernel(
        const float *Qn, const float *Kn, const float *V, const float *G, const float *Bt,
        const GdnStateT *Sin, float *Ssnap, float *O, int Sq, int NH, int NHk, int snapshot_all) {
    constexpr int KL = D / KSPLIT;
    int h = blockIdx.x, tid = threadIdx.x, v = tid & (D-1), grp = tid / D, kb = grp*KL, gq = h*NHk/NH;
    extern __shared__ float sh[]; float *qs = sh, *ks = sh + D, *part = sh + 2*D;  // qs[D],ks[D],part[KSPLIT*D]
    float scol[KL];
    #pragma unroll
    for (int j=0;j<KL;j++) scol[j] = Sin ? (float)Sin[((size_t)h*D + (kb+j))*D + v] : 0.f;
    for (int t=0;t<Sq;t++) {
        size_t base = ((size_t)t*NH+h)*D, base_qk = ((size_t)t*NHk+gq)*D;
        if (grp==0) { qs[v]=Qn[base_qk+v]; ks[v]=Kn[base_qk+v]; }
        __syncthreads();
        float g_t=expf(G[(size_t)t*NH+h]), beta_t=Bt[(size_t)t*NH+h];
        float kvp=0.f;
        #pragma unroll
        for (int j=0;j<KL;j++){ scol[j]*=g_t; kvp+=scol[j]*ks[kb+j]; }
        part[grp*D+v]=kvp; __syncthreads();
        float kv=0.f;
        #pragma unroll
        for (int gg=0;gg<KSPLIT;gg++) kv+=part[gg*D+v];
        float delta=(V[base+v]-kv)*beta_t, outp=0.f;
        #pragma unroll
        for (int j=0;j<KL;j++){ scol[j]+=ks[kb+j]*delta; outp+=scol[j]*qs[kb+j]; }
        __syncthreads();                       // part 재사용 전 (kv read 완료)
        part[grp*D+v]=outp; __syncthreads();
        if (grp==0){ float o=0.f; for(int gg=0;gg<KSPLIT;gg++) o+=part[gg*D+v]; O[base+v]=o; }
        if (snapshot_all || t==Sq-1){ size_t slot=snapshot_all?(size_t)t:0; float*snap=Ssnap+slot*NH*D*D+(size_t)h*D*D;
            #pragma unroll
            for (int j=0;j<KL;j++) snap[(size_t)(kb+j)*D+v]=scol[j]; }
        __syncthreads();
    }
}
template<int D, int KSPLIT>
static inline void launch_gdn_step_ksplit(const float *Qn, const float *Kn, const float *V,
                                   const float *G, const float *Bt, const GdnStateT *Sin,
                                   float *Ssnap, float *O, int Sq, int NH, int NHk,
                                   int snapshot_all = 1, cudaStream_t st = 0) {
    size_t shmem = (size_t)(2*D + KSPLIT*D) * sizeof(float);
    gdn_step_ksplit_kernel<D,KSPLIT><<<NH, D*KSPLIT, shmem, st>>>(Qn,Kn,V,G,Bt,Sin,Ssnap,O,Sq,NH,NHk,snapshot_all);
}

// ---- 챕터9 tree-spec: tree-aware gdn_step (1-forward, 노드별 parent fork) ----
//   선형 carry 대신 각 노드 t가 parent[t]의 snapshot서 state reload → tree 분기 1 forward 처리.
//   parent[t] = tree 부모 노드 인덱스(<t, topo순), 또는 <0이면 root(Sin=committed state서).
//   parent[t]==t-1(선형 chain)이면 Ssnap[t-1] reload = 기존 carry와 bit-identical(snapshot 검증, ch9 de-risk).
//   Ssnap 항상 매-노드 기록(자식이 부모 읽음). matmul/attn amortize는 별도(SB 커널/tree mask); 여기선 GDN recurrence 정확성.
template<int D, int KSPLIT>
static __global__ void __launch_bounds__(D*KSPLIT, 1) gdn_step_ksplit_tree_kernel(
        const float *Qn, const float *Kn, const float *V, const float *G, const float *Bt,
        const GdnStateT *Sin, const int *parent, float *Ssnap, float *O, int Sq, int NH, int NHk) {
    constexpr int KL = D / KSPLIT;
    int h = blockIdx.x, tid = threadIdx.x, v = tid & (D-1), grp = tid / D, kb = grp*KL, gq = h*NHk/NH;
    extern __shared__ float sh[]; float *qs = sh, *ks = sh + D, *part = sh + 2*D;
    float scol[KL];
    for (int t=0;t<Sq;t++) {
        int p = parent[t];                              // 부모 노드(또는 <0=root)
        #pragma unroll
        for (int j=0;j<KL;j++)                          // carry 대신 parent snapshot서 reload (root는 Sin)
            scol[j] = (p>=0) ? Ssnap[((size_t)p*NH+h)*D*D + (size_t)(kb+j)*D + v]
                             : (Sin ? (float)Sin[((size_t)h*D + (kb+j))*D + v] : 0.f);
        size_t base = ((size_t)t*NH+h)*D, base_qk = ((size_t)t*NHk+gq)*D;
        if (grp==0) { qs[v]=Qn[base_qk+v]; ks[v]=Kn[base_qk+v]; }
        __syncthreads();
        float g_t=expf(G[(size_t)t*NH+h]), beta_t=Bt[(size_t)t*NH+h];
        float kvp=0.f;
        #pragma unroll
        for (int j=0;j<KL;j++){ scol[j]*=g_t; kvp+=scol[j]*ks[kb+j]; }
        part[grp*D+v]=kvp; __syncthreads();
        float kv=0.f;
        #pragma unroll
        for (int gg=0;gg<KSPLIT;gg++) kv+=part[gg*D+v];
        float delta=(V[base+v]-kv)*beta_t, outp=0.f;
        #pragma unroll
        for (int j=0;j<KL;j++){ scol[j]+=ks[kb+j]*delta; outp+=scol[j]*qs[kb+j]; }
        __syncthreads();
        part[grp*D+v]=outp; __syncthreads();
        if (grp==0){ float o=0.f; for(int gg=0;gg<KSPLIT;gg++) o+=part[gg*D+v]; O[base+v]=o; }
        { float*snap=Ssnap+(size_t)t*NH*D*D+(size_t)h*D*D;   // 매 노드 snapshot(자식이 읽음)
          #pragma unroll
          for (int j=0;j<KL;j++) snap[(size_t)(kb+j)*D+v]=scol[j]; }
        __syncthreads();
    }
}
template<int D, int KSPLIT>
static inline void launch_gdn_step_ksplit_tree(const float *Qn, const float *Kn, const float *V,
                                   const float *G, const float *Bt, const GdnStateT *Sin, const int *parent,
                                   float *Ssnap, float *O, int Sq, int NH, int NHk, cudaStream_t st = 0) {
    size_t shmem = (size_t)(2*D + KSPLIT*D) * sizeof(float);
    gdn_step_ksplit_tree_kernel<D,KSPLIT><<<NH, D*KSPLIT, shmem, st>>>(Qn,Kn,V,G,Bt,Sin,parent,Ssnap,O,Sq,NH,NHk);
}

// ---- batch decode GDN step (챕터3): B 독립 시퀀스, 각 1스텝(행 간 recurrence 없음 — 병렬) ----
//   Q,K,V,O: [B, NH, D]. G,Bt: [B, NH]. Sin/Sout: [B, NH, D, D] (시퀀스별 committed state). 1 step씩.
//   grid(NH, B): 블록(h,b)가 시퀀스 b·head h의 1스텝. Sout = 갱신된 state(commit). (snapshot 불요 — plain decode.)
static __global__ void gdn_step_batch_kernel(const float *Q, const float *K, const float *V,
                                const float *G, const float *Bt,
                                const float *Sin, float *Sout, float *O,
                                int NH, int D, int NHk) {
    int h = blockIdx.x, b = blockIdx.y, v = threadIdx.x;
    if (v >= D) return;
    int g = h * NHk / NH;                                  // GQA: v-head h → 공유 k-head g
    extern __shared__ float sh[];
    float *qs = sh, *ks = sh + D, *red = sh + 2 * D;
    float scol[128];
    size_t stb = ((size_t)b * NH + h) * D * D;            // state[b][h] 시작 (v-head)
    for (int k = 0; k < D; k++) scol[k] = Sin ? Sin[stb + (size_t)k * D + v] : 0.f;
    const float invsD = rsqrtf((float)D);
    size_t base = ((size_t)b * NH + h) * D;               // V/O row (b, v-head h)
    size_t base_qk = ((size_t)b * NHk + g) * D;           // Q/K row (b, k-head g)
    float qe = Q[base_qk + v], ke = K[base_qk + v];
    float sq = block_reduce_sum(qe * qe, red);
    float sk = block_reduce_sum(ke * ke, red);
    qs[v] = qe * rsqrtf(sq + 1e-6f) * invsD;
    ks[v] = ke * rsqrtf(sk + 1e-6f);
    __syncthreads();
    float g_t = expf(G[(size_t)b * NH + h]), beta_t = Bt[(size_t)b * NH + h];
    float kv = 0.f;
    #pragma unroll 4
    for (int k = 0; k < D; k++) { scol[k] *= g_t; kv += scol[k] * ks[k]; }
    float delta = (V[base + v] - kv) * beta_t, out = 0.f;
    #pragma unroll 4
    for (int k = 0; k < D; k++) { scol[k] += ks[k] * delta; out += scol[k] * qs[k]; }
    O[base + v] = out;
    for (int k = 0; k < D; k++) Sout[stb + (size_t)k * D + v] = scol[k];   // 갱신 state commit
}
// 챕터9 GDN de-risk: D를 template(compile-time)화 → k-루프 unroll → scol[D]가 local mem(stack) 대신 레지스터.
//   런타임 D 버전은 for(k<D) unroll 불가 → scol[128] local mem → scol[k] 512회/thread = long_scoreboard 601(ncu).
//   연산 동일(bit-exact), D만 compile-time. gdn_step_ksplit_tree_kernel<D>와 동일 기법.
template<int D, typename ST=float>
static __global__ void gdn_step_batch_kernel_t(const float *Q, const float *K, const float *V,
                                const float *G, const float *Bt,
                                const ST *Sin, ST *Sout, float *O, int NH, int NHk) {
    int h = blockIdx.x, b = blockIdx.y, v = threadIdx.x;
    int g = h * NHk / NH;
    extern __shared__ float sh[];
    float *qs = sh, *ks = sh + D, *red = sh + 2 * D;
    float scol[D];
    size_t stb = ((size_t)b * NH + h) * D * D;
    #pragma unroll
    for (int k = 0; k < D; k++) scol[k] = Sin ? (float)Sin[stb + (size_t)k * D + v] : 0.f;  // ST=bf16면 mixed(계산 fp32)
    const float invsD = rsqrtf((float)D);
    size_t base = ((size_t)b * NH + h) * D;
    size_t base_qk = ((size_t)b * NHk + g) * D;
    float qe = Q[base_qk + v], ke = K[base_qk + v];
    float sq = block_reduce_sum(qe * qe, red);
    float sk = block_reduce_sum(ke * ke, red);
    qs[v] = qe * rsqrtf(sq + 1e-6f) * invsD;
    ks[v] = ke * rsqrtf(sk + 1e-6f);
    __syncthreads();
    float g_t = expf(G[(size_t)b * NH + h]), beta_t = Bt[(size_t)b * NH + h];
    float kv = 0.f;
    #pragma unroll
    for (int k = 0; k < D; k++) { scol[k] *= g_t; kv += scol[k] * ks[k]; }
    float delta = (V[base + v] - kv) * beta_t, out = 0.f;
    #pragma unroll
    for (int k = 0; k < D; k++) { scol[k] += ks[k] * delta; out += scol[k] * qs[k]; }
    O[base + v] = out;
    #pragma unroll
    for (int k = 0; k < D; k++) Sout[stb + (size_t)k * D + v] = (ST)scol[k];
}
static inline void launch_gdn_step_batch(const float *Q, const float *K, const float *V,
                                         const float *G, const float *Bt, const GdnStateT *Sin,
                                         GdnStateT *Sout, float *O, int B, int NH, int D, int NHk, cudaStream_t st = 0) {
    dim3 grid(NH, B);
    size_t shmem = (size_t)(2 * D + 32) * sizeof(float);
    // D compile-time 특수화(scol 레지스터화) + ST=GdnStateT(state I/O 대역폭). 흔한 head_dim만; 그 외 fp32 런타임 fallback.
#if defined(GDN_FORCE_RUNTIME) && !defined(GDN_BF16_STATE)   // A/B: 런타임(local-mem) 강제 — fp32 전용
    gdn_step_batch_kernel<<<grid, D, shmem, st>>>(Q, K, V, G, Bt, Sin, Sout, O, NH, D, NHk);
#else
    if      (D==128) gdn_step_batch_kernel_t<128,GdnStateT><<<grid, D, shmem, st>>>(Q, K, V, G, Bt, Sin, Sout, O, NH, NHk);
    else if (D==256) gdn_step_batch_kernel_t<256,GdnStateT><<<grid, D, shmem, st>>>(Q, K, V, G, Bt, Sin, Sout, O, NH, NHk);
  #ifndef GDN_BF16_STATE
    else             gdn_step_batch_kernel<<<grid, D, shmem, st>>>(Q, K, V, G, Bt, Sin, Sout, O, NH, D, NHk);  // fp32 fallback(D≠128/256)
  #endif
#endif
}

// ---- (4) gated RMSNorm (Qwen3_5RMSNormGated) ----
//   X,gate: [R, D], W: [D]. y = (x*rsqrt(mean(x^2)+eps) * W) * silu(gate). (weight 표준, 1+w 아님)
static __global__ void rms_norm_gated_kernel(const float *X, const float *gate, const float *W,
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
