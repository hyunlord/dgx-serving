#ifndef DGX_DECODE_STATE_H
#define DGX_DECODE_STATE_H
#include "config.h"
#include "weights_qwen.h"

// speculative decode 상태 3종 (docs/02c-speculative-blueprint.md).
//   full-attn KV: full 레이어만 [max_len, NKV, HD] (roped K 저장). 위치독립 → 길이 포인터 rollback.
//   GDN committed state S: linear 레이어만 [NH, D, D]. 순차누적 → 임시승격.
//   conv1d ctx: linear 레이어만 [K-1, conv_dim]. decode시 직전 K-1 토큰.
// 인덱스는 전체 레이어 기준 배열; GDN 레이어의 Kc/Vc=NULL, full 레이어의 S/convctx=NULL.
struct DecodeState {
    int   n_past;     // 확정된 시퀀스 길이 (= committed KV 길이 = 다음 토큰 위치)
    int   max_len;
    int   L;          // 레이어 수 (free 시 사용)
    float **Kc, **Vc;       // [L], full 레이어만 (roped K, V)
    float **S;              // [L], linear 레이어만 (committed GDN state)
    float **convctx;        // [L], linear 레이어만 ([K-1, conv_dim])
};

DecodeState alloc_decode_state(const ModelConfig &c, int max_len);
void free_decode_state(DecodeState &s);
void reset_decode_state(const ModelConfig &c, DecodeState &s);  // n_past=0, S/convctx=0

// 본체 incremental decode/verify. Sq 토큰(d_tokens, pos n_past..n_past+Sq-1) 처리.
//   d_logits: [Sq, V] 출력. d_prenorm: nullable [Sq, H] (final-norm 전 = MTP 입력).
//   commit=1: KV append 확정 + S/convctx/n_past 갱신 (일반 decode).
//   commit=0: KV는 임시 슬롯에 append하나 n_past 미전진, S/convctx committed 불변 (verify, STEP2).
//             대신 S_snap(nullable, [Sq, n_linear, NH, D, D])에 스텝별 GDN state 스냅샷 기록.
//   d_cos/d_sin: [max_len, rot] rope 테이블 (위치 p → row p). use_q4: fp32(0) 기준선.
void qwen35_decode(const ModelConfig &c, const QwenWeights &w, DecodeState &st,
                   const int *d_tokens, int Sq, const float *d_cos, const float *d_sin,
                   float *d_logits, float *d_prenorm, int commit, float *S_snap, int use_q4);

#endif // DGX_DECODE_STATE_H
