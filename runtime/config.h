#ifndef DGX_CONFIG_H
#define DGX_CONFIG_H

// ============================================================================
// 컴파일 타임 상수 (헌법 원칙 3의 절충: 거의 안 바뀌는 것만 허용)
// ============================================================================

// int4 양자화 블록 크기. weight q4 패킹/스케일 레이아웃의 전제이므로 컴파일 타임 고정.
#ifndef QBLK
#define QBLK 32
#endif

// ============================================================================
// ModelConfig — 모델 상수는 #define이 아니라 런타임 config로 받는다 (헌법 원칙 3).
//   H, L, FFN, n_heads 등은 절대 커널/graph에 박지 않는다.
// ============================================================================

typedef struct ModelConfig {
    int   hidden_size;        // H        (SmolLM3: 2048)
    int   intermediate_size;  // FFN      (11008)
    int   num_layers;         // L        (36)
    int   num_q_heads;        // NQ       (16)
    int   num_kv_heads;       // NKV      (4)   GQA: NQ % NKV == 0
    int   head_dim;           // HD       (128)
    int   vocab_size;         // V        (128256)
    float rms_norm_eps;       //          (1e-6)
    float rope_theta;         //          (5e6)
    int   tie_word_embeddings;//          (1 = lm_head shares embedding weight)
    int   max_seq_len;        // KV 캐시 상한

    // SmolLM3 NoRoPE: 레이어별 RoPE 적용 여부. use_rope[i]==1 이면 RoPE 적용, 0이면 skip.
    // 길이 = num_layers. NULL이면 모든 레이어 RoPE 적용으로 간주.
    const int *use_rope;

    // ---- Qwen3.5 확장 (SmolLM3는 사용 안 함) ----
    int rotary_dim;          // partial RoPE 회전 차원 (Qwen 64). 0/head_dim 이면 full rope.
    int attn_output_gate;    // 1이면 q_proj가 2*NQ*HD 출력 + attn_out *= sigmoid(gate)
    // 레이어 타입: layer_is_full[i]==1 full attention, 0 linear/GDN. NULL이면 전부 full.
    const int *layer_is_full;
    // GDN(Gated Delta Net) dims
    int gdn_num_k_heads, gdn_num_v_heads, gdn_head_k_dim, gdn_head_v_dim, gdn_conv_kernel;
    // MoE dims (num_experts==0 → dense 모델). qwen3_5_moe: experts/top-k/shared.
    int num_experts, num_experts_per_tok, moe_intermediate_size, shared_expert_intermediate_size, norm_topk_prob;
} ModelConfig;

// GDN 파생값
static inline int cfg_gdn_conv_dim(const ModelConfig *c) {
    return c->gdn_num_k_heads * c->gdn_head_k_dim * 2 + c->gdn_num_v_heads * c->gdn_head_v_dim;
}
static inline int cfg_gdn_value_dim(const ModelConfig *c) { return c->gdn_num_v_heads * c->gdn_head_v_dim; }
static inline int cfg_gdn_key_dim(const ModelConfig *c) { return c->gdn_num_k_heads * c->gdn_head_k_dim; }

// 파생값 헬퍼 (graph/커널에서 사용)
static inline int cfg_q_dim(const ModelConfig *c)  { return c->num_q_heads  * c->head_dim; } // = H (보통)
static inline int cfg_kv_dim(const ModelConfig *c) { return c->num_kv_heads * c->head_dim; }
static inline int cfg_gqa_group(const ModelConfig *c) { return c->num_q_heads / c->num_kv_heads; }

#endif // DGX_CONFIG_H
