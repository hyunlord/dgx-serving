#ifndef DGX_SMOLLM3_H
#define DGX_SMOLLM3_H
#include "../runtime/config.h"
#include "../runtime/weights.h"

// SmolLM3 graph (공용 커널 배선). prefill: S 토큰 한 번에 forward.
//   d_ids   : [S] int (device)
//   d_cos/d_sin : [S, head_dim] (device) — rope 테이블 (1단계는 HF 골든 로드)
//   d_logits: [S, V] (device, 출력)
//   hidden_dump : nullable. 비-NULL이면 [(L+1), S, H] (device)에 기록.
//                 인덱스 l(0..L-1)=레이어 l 입력 hidden, 인덱스 L=final-norm 적용 후 hidden
//                 (HF output_hidden_states 의미와 일치).
void smollm3_forward(const ModelConfig &c, const EngineWeights &w,
                     const int *d_ids, int S,
                     const float *d_cos, const float *d_sin,
                     float *d_logits, float *hidden_dump);

#endif // DGX_SMOLLM3_H
