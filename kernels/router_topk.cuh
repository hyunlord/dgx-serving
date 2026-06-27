#pragma once
#include <cuda_runtime.h>
// MoE 라우터 top-k: 토큰별 router_logits[E] → top-K expert 인덱스 + renormalized 가중치.
//   HF(Qwen3.5-MoE): softmax(E 전체, fp32) → top-K → top_w/=top_w.sum().
//   수학 항등: softmax(E)→topK→renorm == softmax(topK logits) (full-E Z가 renorm서 상쇄,
//   weight 차 ~3e-8). 후자로 구현(단순·정확, full softmax 불필요). 정확성-우선 naive(1스레드/토큰).
//   ★ 가정: renorm(top_w/=sum) 있을 때만 항등 성립. Qwen3.5-MoE는 renorm 함 → OK.
//     다음 MoE(예: Gemma4 26B-A4B)가 renorm 안 하면(top-K를 full-softmax 값 그대로 사용) 이 항등 깨짐
//     → config 분기(renorm 여부) 필요. 지금은 Qwen-MoE 전용 가정.
//   idx[S,K] (int, top-K expert), w[S,K] (float, 합=1). 효율화(병렬 선택)는 나중(ncu 후).

#ifndef ROUTER_MAX_E
#define ROUTER_MAX_E 512   // qwen3.5_moe E=256; 여유
#endif

static __global__ void router_topk_kernel(const float *logits, int *idx, float *w, int E, int K) {
    int row = blockIdx.x;                       // 토큰 1개 / 블록 (blockDim=1, 정확성-우선)
    const float *lg = logits + (size_t)row * E;
    int *ir = idx + (size_t)row * K;
    float *wr = w + (size_t)row * K;
    bool used[ROUTER_MAX_E];
    for (int i = 0; i < E; i++) used[i] = false;
    float maxl = -1e30f;
    for (int k = 0; k < K; k++) {               // K-pass argmax (선택분 mask)
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < E; i++) if (!used[i] && lg[i] > bv) { bv = lg[i]; best = i; }
        used[best] = true; ir[k] = best; wr[k] = bv;
        if (bv > maxl) maxl = bv;
    }
    float s = 0.f;                              // softmax over top-K logits (= HF renorm 결과)
    for (int k = 0; k < K; k++) { float e = expf(wr[k] - maxl); wr[k] = e; s += e; }
    for (int k = 0; k < K; k++) wr[k] /= s;
}

static inline void launch_router_topk(const float *logits, int *idx, float *w,
                                      int S, int E, int K, cudaStream_t st = 0) {
    router_topk_kernel<<<S, 1, 0, st>>>(logits, idx, w, E, K);
}
