#ifndef DGX_QWEN35_H
#define DGX_QWEN35_H
#include "../runtime/config.h"
#include "../runtime/weights_qwen.h"

// Qwen3.5-0.8B 본체 graph (full attention + GDN linear attention 혼합). prefill S 토큰.
//   d_ids [S] int, d_cos/d_sin [S, rotary_dim], d_logits [S,V] (출력).
//   hidden_dump : nullable [(L+1),S,H] — l<L: 레이어 l 입력 hidden, l=L: final-norm 적용 후.
//   prenorm_out : nullable [S,H] — final-norm '입력' hidden (= MTP head 가 받을 값, 2b).
void qwen35_forward(const ModelConfig &c, const QwenWeights &w,
                    const int *d_ids, int S, const float *d_cos, const float *d_sin,
                    float *d_logits, float *hidden_dump, float *prenorm_out);

#endif // DGX_QWEN35_H
