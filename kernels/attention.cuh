#ifndef DGX_ATTENTION_CUH
#define DGX_ATTENTION_CUH
#include "runtime/knobs.h"
#include <float.h>
#include <cstdio>
#include <cstdlib>

// ── 임시 가드 (챕터7 FA STEP 0) ───────────────────────────────────────────────
//   현 attention 커널은 full scores[attendlen]를 dynamic shared에 올림 → context가 길면 shmem이
//   기본 48KB를 넘어 launch가 조용히 실패(cudaDeviceSynchronize "no error"라 CK가 못 잡음) → 틀린 출력.
//   가드: 48KB 초과 시 opt-in(최대 ~99KB, base~25K까지) / opt-in 한계 초과 시 명시적 에러+중단.
//   ※ 근본해결은 FlashAttention(tiled KV 상수 shared). 이 가드는 그 전까지 "조용한 오답→명시적 중단".
static inline void attn_smem_guard(const void *kernel, size_t shmem, const char *who, int ctxlen) {
    static int optin = -1;
    if (optin < 0) cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (shmem > (size_t)optin) {
        fprintf(stderr, "[attn] %s: 필요 shared %zuKB > opt-in 한계 %dKB (context~%d토큰). "
            "현 attention 커널은 이 길이 미지원 — FlashAttention 필요. 조용한 오답 대신 중단합니다.\n",
            who, shmem / 1024, optin / 1024, ctxlen);
        abort();
    }
    if (shmem > 48 * 1024)
        cudaFuncSetAttribute(const_cast<void *>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
}

// GQA causal attention (prefill, 모델 독립).
//   Q: [S, NQ, HD], K/V: [S, NKV, HD], O: [S, NQ, HD]. scale = 1/sqrt(HD).
//   블록 = (head h, query qi). blockDim.x = HD. threadIdx.x 가 dim(d) 와 key-pos(j) 두 역할.
//   ⇒ 제약: S <= HD (1단계 prefill 용도). 동적 shared: qs[HD] + scores[S] + probs[S].
//   kv head = h / (NQ/NKV) (GQA group).
//   scale: softmax 전 스코어 스케일(0 → 1/sqrt(HD) 기본). window: sliding window 폭
//   (0 → 무제한 causal; w>0 → key pos j ∈ [qi-w+1, qi]). 마스크 파라미터화 — 모델 분기 아님.
static __global__ void attn_kernel(const float *Q, const float *K, const float *V, float *O,
                            int S, int NQ, int NKV, int HD, float scale, int window) {
    int h = blockIdx.x, qi = blockIdx.y, d = threadIdx.x;
    if (qi >= S) return;
    int GRP = NQ / NKV, kv = h / GRP;
    int lo = (window > 0 && qi - window + 1 > 0) ? qi - window + 1 : 0;
    extern __shared__ float sm[];
    float *qs = sm;            // [HD]
    float *scores = sm + HD;   // [S]
    float *probs = scores + S; // [S]

    if (d < HD) qs[d] = Q[((size_t)qi * NQ + h) * HD + d];
    __syncthreads();

    // causal (+sliding window): key pos j = d, lo <= j <= qi
    if (d >= lo && d <= qi) {
        const float *kp = K + ((size_t)d * NKV + kv) * HD;
        float dot = 0.f;
        for (int t = 0; t < HD; t++) dot += qs[t] * kp[t];
        scores[d] = dot * scale;
    }
    __syncthreads();

    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = lo; j <= qi; j++) m = fmaxf(m, scores[j]);
        float sum = 0.f;
        for (int j = lo; j <= qi; j++) { probs[j] = expf(scores[j] - m); sum += probs[j]; }
        for (int j = lo; j <= qi; j++) probs[j] /= sum;
    }
    __syncthreads();

    if (d < HD) {
        float acc = 0.f;
        for (int j = lo; j <= qi; j++) acc += probs[j] * V[((size_t)j * NKV + kv) * HD + d];
        O[((size_t)qi * NQ + h) * HD + d] = acc;
    }
}

static inline void launch_attn(const float *Q, const float *K, const float *V, float *O,
                               int S, int NQ, int NKV, int HD, cudaStream_t st = 0,
                               float scale = 0.f, int window = 0) {
    dim3 grid(NQ, S);
    size_t shmem = (size_t)(HD + 2 * S) * sizeof(float);
    float sc = (scale > 0.f) ? scale : rsqrtf((float)HD);
    attn_kernel<<<grid, HD, shmem, st>>>(Q, K, V, O, S, NQ, NKV, HD, sc, window);
}

// Decode/verify attention over a KV cache (위치 독립 cache, 임의 길이). 2a attn_kernel 의 S<=HD 제약 없음.
//   Q: [Sq, NQ, HD] (decode Sq=1, verify Sq=K+1). Kc/Vc: [Lc, NKV, HD] cache (신규토큰 이미 append됨).
//   query i (절대위치 base+i) 는 cache[0 .. base+i] 를 causal attend. scale=1/sqrt(HD), GQA.
//   block=(query i, head h), blockDim=HD. dynamic shared: qs[HD] + scores[base+Sq] (>= 최대 attendlen).
//   scale: 0 → 1/sqrt(HD) 기본. window: 0 → 무제한 causal; w>0 → sliding(key pos j ∈ [qpos-w+1, qpos]).
//   (scale/window는 인자 — 모델 분기 아님. Gemma4 sliding/global 층에 config로 주입.)
// Diagnostic only (R15 noise baseline): G4_ATTN_DIAG_REVERSE=1 sums the output over keys in
// descending order in both decode kernels. Same math, different addition order. Default off.
static __device__ int g_attn_diag_reverse = 0;
static inline void attn_diag_init() {
    static const bool done = []() {
        const char *v = knob_raw("G4_ATTN_DIAG_REVERSE");
        int on = (v && v[0] == '1') ? 1 : 0;
        if (on) { cudaMemcpyToSymbol(g_attn_diag_reverse, &on, sizeof(int)); fprintf(stderr, "G4_ATTN_DIAG_REVERSE=1\n"); }
        return true; }();
    (void)done;
}
static __global__ void attn_decode_kernel(const float *Q, const float *Kc, const float *Vc, float *O,
                                   int Sq, int NQ, int NKV, int HD, int base, float scale, int window) {
    int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int GRP = NQ / NKV, kv = h / GRP;
    int qpos = base + i;
    int attendlen = qpos + 1;
    int lo = (window > 0 && qpos - window + 1 > 0) ? qpos - window + 1 : 0;   // sliding window 하한
    extern __shared__ float sm[];
    float *qs = sm;                 // [HD]
    float *scores = sm + HD;        // [attendlen]
    if (d < HD) qs[d] = Q[((size_t)i * NQ + h) * HD + d];
    __syncthreads();
    // scores[j] = (q · Kc[j,kv]) * scale, j=lo..attendlen-1 (window 밖은 skip)
    for (int j = d; j < attendlen; j += blockDim.x) {
        if (j < lo) { scores[j] = -FLT_MAX; continue; }
        const float *kp = Kc + ((size_t)j * NKV + kv) * HD;
        float dot = 0.f;
        for (int t = 0; t < HD; t++) dot += qs[t] * kp[t];
        scores[j] = dot * scale;
    }
    __syncthreads();
    __shared__ float ssum;
    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = lo; j < attendlen; j++) m = fmaxf(m, scores[j]);
        float s = 0.f;
        for (int j = lo; j < attendlen; j++) { scores[j] = expf(scores[j] - m); s += scores[j]; }
        ssum = s;
    }
    __syncthreads();
    // out[d] = Σ_j (scores[j]/sum) * Vc[j,kv,d]
    if (d < HD) {
        float acc = 0.f;
        if (g_attn_diag_reverse) for (int j = attendlen - 1; j >= lo; j--) acc += scores[j] * Vc[((size_t)j * NKV + kv) * HD + d];
        else for (int j = lo; j < attendlen; j++) acc += scores[j] * Vc[((size_t)j * NKV + kv) * HD + d];
        O[((size_t)i * NQ + h) * HD + d] = acc / ssum;
    }
}

static inline void launch_attn_decode(const float *Q, const float *Kc, const float *Vc, float *O,
                                      int Sq, int NQ, int NKV, int HD, int base, cudaStream_t st = 0,
                                      float scale = 0.f, int window = 0) {
    dim3 grid(Sq, NQ);
    size_t shmem = (size_t)(HD + base + Sq) * sizeof(float);
    attn_smem_guard((const void *)attn_decode_kernel, shmem, "attn_decode", base + Sq);
    attn_diag_init();
    float sc = (scale > 0.f) ? scale : rsqrtf((float)HD);
    attn_decode_kernel<<<grid, HD, shmem, st>>>(Q, Kc, Vc, O, Sq, NQ, NKV, HD, base, sc, window);
}

// ---- batch decode attention (챕터3): B개 독립 시퀀스, 각 1토큰(위치 base), 각자 KV 캐시 ----
//   Q,O: [B, NQ, HD]. Kc/Vc: [B, max_len, NKV, HD] (시퀀스별), kv_stride = max_len*NKV*HD.
//   각 row b는 자기 KV[b][0..base] attend (attendlen = base+1 공통, same-length).
//   scale: 0 → 1/sqrt(HD). window: 0 → 무제한; w>0 → sliding [qpos-w+1, qpos].
//   base_b: nullptr → 모든 슬롯 위치 base(same-length). 아니면 슬롯 b 위치 = base_b[b] (ragged 연속배칭).
static __global__ void attn_decode_batch_kernel(const float *Q, const float *Kc, const float *Vc, float *O,
                                   int NQ, int NKV, int HD, int base, long kv_stride, float scale, int window,
                                   const int *base_b) {
    int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int GRP = NQ / NKV, kv = h / GRP;
    int qpos = base_b ? base_b[b] : base;
    int attendlen = qpos + 1;
    int lo = (window > 0 && qpos - window + 1 > 0) ? qpos - window + 1 : 0;
    const float *Kcb = Kc + (size_t)b * kv_stride;     // 시퀀스 b의 KV
    const float *Vcb = Vc + (size_t)b * kv_stride;
    extern __shared__ float sm[];
    float *qs = sm, *scores = sm + HD;
    if (d < HD) qs[d] = Q[((size_t)b * NQ + h) * HD + d];
    __syncthreads();
    for (int j = d + lo; j < attendlen; j += blockDim.x) {
        const float *kp = Kcb + ((size_t)j * NKV + kv) * HD;
        float dot = 0.f;
        for (int t = 0; t < HD; t++) dot += qs[t] * kp[t];
        scores[j] = dot * scale;
    }
    __syncthreads();
    __shared__ float ssum;
    if (d == 0) {
        float m = -FLT_MAX; for (int j = lo; j < attendlen; j++) m = fmaxf(m, scores[j]);
        float s = 0.f; for (int j = lo; j < attendlen; j++) { scores[j] = expf(scores[j] - m); s += scores[j]; }
        ssum = s;
    }
    __syncthreads();
    if (d < HD) {
        float acc = 0.f;
        if (g_attn_diag_reverse) for (int j = attendlen - 1; j >= lo; j--) acc += scores[j] * Vcb[((size_t)j * NKV + kv) * HD + d];
        else for (int j = lo; j < attendlen; j++) acc += scores[j] * Vcb[((size_t)j * NKV + kv) * HD + d];
        O[((size_t)b * NQ + h) * HD + d] = acc / ssum;
    }
}
//   base: same-length면 공통 위치. ragged면 base=max(base_b)(shmem 크기용) + base_b[] per-slot.
static inline void launch_attn_decode_batch(const float *Q, const float *Kc, const float *Vc, float *O,
                                            int B, int NQ, int NKV, int HD, int base, long kv_stride, cudaStream_t st = 0,
                                            float scale = 0.f, int window = 0, const int *base_b = nullptr) {
    dim3 grid(B, NQ);
    size_t shmem = (size_t)(HD + base + 1) * sizeof(float);   // base=최대 attendlen 커버
    attn_smem_guard((const void *)attn_decode_batch_kernel, shmem, "attn_decode_batch", base + 1);
    attn_diag_init();
    float sc = (scale > 0.f) ? scale : rsqrtf((float)HD);
    attn_decode_batch_kernel<<<grid, HD, shmem, st>>>(Q, Kc, Vc, O, NQ, NKV, HD, base, kv_stride, sc, window, base_b);
}

#endif // DGX_ATTENTION_CUH
