#ifndef DGX_ATTENTION_FLASH_CUH
#define DGX_ATTENTION_FLASH_CUH
#include <float.h>
#include <math.h>

// 챕터7 FA.1a: tiled FlashAttention (online softmax). full-attn 16층용 (GQA causal).
//   현 attn_decode의 두 문제 해결: ① full scores[attendlen]를 shared에 → 48KB 벽(context>12K 깨짐).
//   ② softmax를 단일 스레드 직렬. → FA는 KV를 BC씩 타일 순회 + running(m,l,acc) → shared 상수(BC 고정)
//   + 타일별 softmax. shared = q[HD]+Ks[BC][HD]+Vs[BC][HD]+s[BC] = BC 고정(32K/256K든 동일) → 벽 없음.
//
//   매핑: block=(query i, head h), blockDim=HD. thread d = 출력 dim d 소유(acc[d] running).
//   GQA: kv head = h/(NQ/NKV) — 각 query head가 올바른 KV head attend (데인 함정, 명시).
//   causal: query 절대위치 qpos=base+i, attend [0..qpos] (attendlen=qpos+1). 루프 bound가 causality 보장.
//   BC = 타일 크기(템플릿). 정확성 핵심 = online softmax running 갱신(m 바뀔 때 exp(m_old-m_new) rescale).
template<int HD, int BC>
static __global__ void flash_attn_kernel(const float *Q, const float *Kc, const float *Vc, float *O,
                                         int Sq, int NQ, int NKV, int base) {
    int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int GRP = NQ / NKV, kv = h / GRP;
    int qpos = base + i, attendlen = qpos + 1;
    float scale = 1.0f / sqrtf((float)HD);

    __shared__ float q[HD];
    __shared__ float Ks[BC][HD];
    __shared__ float Vs[BC][HD];
    __shared__ float s[BC];

    q[d] = Q[((size_t)i * NQ + h) * HD + d];
    __syncthreads();

    float m = -FLT_MAX, l = 0.f, acc = 0.f;   // running: m=max, l=denom, acc=출력 dim d
    for (int j0 = 0; j0 < attendlen; j0 += BC) {
        int Bcur = (attendlen - j0 < BC) ? (attendlen - j0) : BC;
        // KV 타일 적재 (thread d = 각 행의 dim d, coalesced)
        for (int r = 0; r < Bcur; r++) {
            Ks[r][d] = Kc[((size_t)(j0 + r) * NKV + kv) * HD + d];
            Vs[r][d] = Vc[((size_t)(j0 + r) * NKV + kv) * HD + d];
        }
        __syncthreads();
        // scores: thread d(<Bcur)가 key j=d 의 full dot (q·Ks[d])
        if (d < Bcur) {
            float dot = 0.f;
            for (int t = 0; t < HD; t++) dot += q[t] * Ks[d][t];
            s[d] = dot * scale;
        }
        __syncthreads();
        // online softmax: 타일 max → running rescale (모든 스레드 중복 계산, Bcur<=BC 작음)
        float tile_m = -FLT_MAX;
        for (int j = 0; j < Bcur; j++) tile_m = fmaxf(tile_m, s[j]);
        float m_new = fmaxf(m, tile_m);
        float alpha = expf(m - m_new);             // 이전 누적 보정 (첫 타일 m=-FLT_MAX → 0)
        float tile_l = 0.f, acc_tile = 0.f;
        for (int j = 0; j < Bcur; j++) {
            float p = expf(s[j] - m_new);
            tile_l += p;
            acc_tile += p * Vs[j][d];              // thread d = 출력 dim d
        }
        l = l * alpha + tile_l;
        acc = acc * alpha + acc_tile;
        m = m_new;
        __syncthreads();                           // 다음 타일이 Ks/Vs 덮기 전
    }
    O[((size_t)i * NQ + h) * HD + d] = acc / l;
}

// 런처: HD는 컴파일타임(head_dim 고정 — 헌법 허용 상수). BC=32 → shared ~33KB 상수(벽 없음).
template<int HD, int BC>
static inline void launch_flash_attn_t(const float *Q, const float *Kc, const float *Vc, float *O,
                                       int Sq, int NQ, int NKV, int base, cudaStream_t st = 0) {
    dim3 grid(Sq, NQ);
    flash_attn_kernel<HD, BC><<<grid, HD, 0, st>>>(Q, Kc, Vc, O, Sq, NQ, NKV, base);
}

// 엔진용 기본 진입 (HD=128, BC=32). attn_decode 교체 대상.
static inline void launch_flash_attn(const float *Q, const float *Kc, const float *Vc, float *O,
                                     int Sq, int NQ, int NKV, int HD, int base, cudaStream_t st = 0) {
    // head_dim은 128 고정(검증 모델군). 다른 값 들어오면 단언 실패로 드러냄(조용한 오답 방지).
    if (HD != 128) { /* 미지원 head_dim — 확장 시 인스턴스 추가 */ return; }
    launch_flash_attn_t<128, 32>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st);
}

#endif // DGX_ATTENTION_FLASH_CUH
