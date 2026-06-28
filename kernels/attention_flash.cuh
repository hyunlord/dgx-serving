#ifndef DGX_ATTENTION_FLASH_CUH
#define DGX_ATTENTION_FLASH_CUH
#include <float.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>

// ── attention 커널 landscape (왜 여러 개인가 — mm() 복붙이 조용히 갈라진 교훈, 명시 기록) ──────────
//   1) launch_attn (attention.cuh)            — 레퍼런스 forward(qwen35.cu/smollm3.cu). golden 생성용.
//        S≤HD 제약, 작은 S만(벽 무관). **의도적 독립**: 엔진 FA와 다른 커널 = 교차검증 ground truth.
//   2) launch_flash_attn (이 파일)            — 엔진 prefill+decode(qwen35_decode.cu). FA.1a. 12K 벽 없음.
//        attn_decode를 **교체**. full-attn 16층.
//   3) launch_attn_decode_batch (attention.cuh) — batch decode(qwen35_batch.cu). **아직 12K 벽 있음**
//        (full scores in shared). batch 긴-context는 나중 FA 교체 예정(TODO).
//   4) launch_attn_decode (attention.cuh)     — 옛 엔진 경로. 이제 test_flash_attn의 벽-안 대조 레퍼런스로만.
//   ⚠ FA 개선(FA.1b/FA.2) 시 1)·3)은 자동 안 받음 → 갈라질 잠재. golden 틀어지면 여기부터 의심.

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

// 엔진용 기본 진입. head_dim별 인스턴스(BC는 shared<48KB 유지): HD256→BC16(33KB), HD128→BC32(33KB).
//   ⚠ 미지원 HD는 abort(조용한 return 금지 — 그게 0.8B/27B head_dim=256을 놓친 버그였음).
static inline void launch_flash_attn(const float *Q, const float *Kc, const float *Vc, float *O,
                                     int Sq, int NQ, int NKV, int HD, int base, cudaStream_t st = 0) {
    if      (HD == 256) launch_flash_attn_t<256, 16>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st);
    else if (HD == 128) launch_flash_attn_t<128, 32>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st);
    else {
        fprintf(stderr, "[flash_attn] 미지원 head_dim=%d (128/256만 인스턴스됨). "
            "확장 시 launch_flash_attn에 인스턴스 추가. 조용한 오답 대신 중단합니다.\n", HD);
        abort();
    }
}

#endif // DGX_ATTENTION_FLASH_CUH
