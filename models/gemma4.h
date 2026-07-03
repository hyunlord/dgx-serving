#ifndef DGX_MODELS_GEMMA4_H
#define DGX_MODELS_GEMMA4_H
#include "../runtime/config.h"
#include "../runtime/weights_gemma4.h"

// Gemma4 dense 프리필 forward graph (골든 게이트용, S<=min(head_dim들) 프리필).
//   d_cos_f/d_sin_f: full(global) 층 cos/sin [S, global_head_dim] (proportional — 미회전 각도 cos=1/sin=0)
//   d_cos_s/d_sin_s: sliding 층 cos/sin [S, head_dim]
//   hidden_dump: [L+1, S, H] (레이어 입력 + final-norm 후). use_q4: 0=fp32, 1=q4(tiled), 2=W4A8.
void gemma4_forward(const ModelConfig &c, const G4Weights &w,
                    const int *d_ids, int S,
                    const float *d_cos_f, const float *d_sin_f,
                    const float *d_cos_s, const float *d_sin_s,
                    float *d_logits, float *hidden_dump, int use_q4);

#endif // DGX_MODELS_GEMMA4_H
