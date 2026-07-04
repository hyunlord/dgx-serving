#ifndef DGX_WEIGHTS_GEMMA4_H
#define DGX_WEIGHTS_GEMMA4_H
#include <cstdint>
#include "config.h"
#include "weights_qwen.h"   // QMat 재사용 (q4/fp32 이중 경로 규약 공유)

// Gemma4 레이어: sandwich 4-norm + 층 타입별 attention (sliding: v_proj 있음 / full: k_eq_v로 없음).
struct G4Layer {
    int    is_full;
    float *input_ln, *post_attn_ln, *pre_ffn_ln, *post_ffn_ln;  // [H] 4-norm sandwich
    float  layer_scalar;                                        // 레이어 출력 스칼라 (host)
    QMat   q, k, v, o;                                          // v.wq==NULL if k_eq_v(full)
    float *q_norm, *k_norm;                                     // [hd_l] per-head norm (v_norm은 scale 없음 → 텐서 없음)
    QMat   gate, up, down;                                      // GeGLU dense MLP

    // ---- Gemma4 MoE (26B-A4B, num_experts>0) — dense MLP ∥ MoE 병렬, shared expert 없음 ----
    int    is_moe;                                              // 1이면 MoE 블록 활성 (num_experts>0)
    float *post_ff_ln_1, *pre_ff_ln_2, *post_ff_ln_2;           // [H] MoE 경로 3-norm (dense post / MoE 입력·출력)
    QMat   experts_gate_up, experts_down;                       // [E,2*MI,H] / [E,H,MI] batched q4 (moe_fused)
    float *router_proj;                                         // fp32 [E,H] (launch_gemv_f32)
    float *router_scale;                                        // fp32 [H]  (rmsnorm weight로 흡수: norm_noscale*scale)
    float *per_expert_scale;                                    // fp32 [E]  (top_w 후처리 gather-multiply)
};

struct G4Weights {
    float  *embed, *final_norm;
    QMat    lm_head;      // tied: embed에서 q4 변환 저장본
    G4Layer *layers;  int L;
    int     have_q4, have_fp32;
};

G4Weights load_weights_gemma4(const char *dir, const ModelConfig &c, int load_q4, int load_fp32);
void free_weights_gemma4(G4Weights &w);

#endif // DGX_WEIGHTS_GEMMA4_H
