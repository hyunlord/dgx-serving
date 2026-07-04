#ifndef DGX_MODELS_GEMMA4_MOE_H
#define DGX_MODELS_GEMMA4_MOE_H
#include "../runtime/config.h"
#include "../runtime/weights_gemma4.h"

// Gemma4 26B-A4B MoE 프리필 forward graph (dense MLP ∥ MoE 병렬 위상).
//   gemma4.cu(31B dense) graph 복제 + MLP 스텝을 modeling_gemma4.py Gemma4TextDecoderLayer 위상으로:
//     dense GeGLU MLP → post_ff_ln_1 = h1
//     MoE(rin=residual): router(rmsnorm_noscale*scale*H^-0.5 → gemv_f32 → topk → per_expert_scale)
//                        → experts(pre_ff_ln_2 → moe_fused GeGLU 128e top8) → post_ff_ln_2 = h2
//     h = h1 + h2 → post_ff_ln → residual add.
//   전부 config/graph/기존 커널 흡수(신규 커널=per_expert_scale gather-multiply 1개, moe_fused.cuh).
//   d_cos_f/d_sin_f: full 층 cos/sin [S, global_head_dim] (뒤 미회전=cos1/sin0). d_cos_s/d_sin_s: sliding [S, head_dim].
//   hidden_dump: [L+1, S, H] (nullptr 가능). use_q4: 0=fp32, 1=q4(tiled), 2=W4A8.
//   router_didx_dump/router_dw_dump: [L, S, K] device (nullptr 가능) — MoE 층별 top-K expert idx + routing
//     weight(renorm×per_expert_scale) 캡처. 골든 대조(router.scale raw·per_expert_scale 검증)용. dense 층은 미기록.
void gemma4_moe_forward(const ModelConfig &c, const G4Weights &w,
                        const int *d_ids, int S,
                        const float *d_cos_f, const float *d_sin_f,
                        const float *d_cos_s, const float *d_sin_s,
                        float *d_logits, float *hidden_dump, int use_q4,
                        int *router_didx_dump=nullptr, float *router_dw_dump=nullptr);

// 26B-A4B MoE decode graph (KV-cached — gemma4_decode 대칭 + dense∥MoE MLP). G4DecodeState 필요(decode_state_gemma4.h).
//   commit: n_past += Sq. last_only: 마지막 행 logit만. 프리필-모드(Sq=S,base=0)=gemma4_moe_forward와 동일(엔진 bit-id 게이트).
struct G4DecodeState;
void gemma4_moe_decode(const ModelConfig &c, const G4Weights &w, G4DecodeState &st,
                       const int *d_tokens, int Sq,
                       const float *d_cos_f, const float *d_sin_f,
                       const float *d_cos_s, const float *d_sin_s,
                       float *d_logits, int commit, int use_q4, int last_only);

// 26B-A4B MoE batch decode (KV-cached — gemma4_decode_batch 대칭 + dense∥MoE MLP). G4BatchState 필요(decode_state_gemma4.h).
//   dense=mmB(gemm_batch)·experts=moe_fused(grouped). ragged: d_pos_b!=null(슬롯별 위치). batch==solo bit-id(스킴 내).
struct G4BatchState;
void gemma4_moe_decode_batch(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                             const int *d_tokens, const float *d_cos_f, const float *d_sin_f,
                             const float *d_cos_s, const float *d_sin_s,
                             float *d_logits, int commit, int use_q4,
                             const int *d_pos_b, const int *h_pos_b);

#endif // DGX_MODELS_GEMMA4_MOE_H
