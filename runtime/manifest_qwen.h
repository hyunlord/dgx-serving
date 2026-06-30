#ifndef DGX_MANIFEST_QWEN_H
#define DGX_MANIFEST_QWEN_H
#include "manifest.h"   // mf_* 헬퍼, ModelConfig, load 기반

// Qwen3.5 manifest → ModelConfig (확장 필드 포함). layer_types 문자열 배열 파싱.
static inline ModelConfig load_config_qwen(const char *manifest_path) {
    char *s = mf_read_file(manifest_path);
    ModelConfig c{};
    c.hidden_size       = mf_int(s, "hidden_size");
    c.intermediate_size = mf_int(s, "intermediate_size");
    c.num_layers        = mf_int(s, "num_layers");
    c.num_q_heads       = mf_int(s, "num_q_heads");
    c.num_kv_heads      = mf_int(s, "num_kv_heads");
    c.head_dim          = mf_int(s, "head_dim");
    c.vocab_size        = mf_int(s, "vocab_size");
    c.rms_norm_eps      = (float)mf_num(s, "rms_norm_eps");
    c.rope_theta        = (float)mf_num(s, "rope_theta");
    c.tie_word_embeddings = mf_bool(s, "tie_word_embeddings");
    c.rotary_dim        = mf_int(s, "rotary_dim");
    c.attn_output_gate  = mf_bool(s, "attn_output_gate");
    c.use_rope          = nullptr;   // Qwen 본체는 전 레이어 rope
    // GDN dims
    c.gdn_num_k_heads = mf_int(s, "linear_num_key_heads");
    c.gdn_num_v_heads = mf_int(s, "linear_num_value_heads");
    c.gdn_head_k_dim  = mf_int(s, "linear_key_head_dim");
    c.gdn_head_v_dim  = mf_int(s, "linear_value_head_dim");
    c.gdn_conv_kernel = mf_int(s, "linear_conv_kernel");

    // MoE (dense 모델엔 키 없음 → 0 = dense). num_experts>0 이면 MoE graph.
    c.num_experts                     = mf_int_opt(s, "num_experts", 0);
    c.num_experts_per_tok             = mf_int_opt(s, "num_experts_per_tok", 0);
    c.moe_intermediate_size           = mf_int_opt(s, "moe_intermediate_size", 0);
    c.shared_expert_intermediate_size = mf_int_opt(s, "shared_expert_intermediate_size", 0);
    c.norm_topk_prob                  = mf_int_opt(s, "norm_topk_prob", 1);  // Qwen-MoE renorm 함
    c.scale_dtype                     = mf_int_opt(s, "scale_dtype", 0);     // 0=fp32, 1=fp16 (기존 fp32 회귀안전)

    // layer_types: ["linear_attention","full_attention",...] → layer_is_full[i] (1/0)
    int *lf = (int *)malloc(sizeof(int) * c.num_layers);
    const char *p = strchr(mf_find(s, "layer_types"), '[');
    if (!p) { fprintf(stderr, "layer_types '[' 없음\n"); exit(1); }
    p++;
    for (int i = 0; i < c.num_layers; i++) {
        const char *q = strchr(p, '"');
        if (!q) { fprintf(stderr, "layer_types 파싱 부족 (i=%d)\n", i); exit(1); }
        lf[i] = (strncmp(q + 1, "full", 4) == 0) ? 1 : 0;
        p = strchr(q + 1, '"') + 1;   // 닫는 따옴표 다음
    }
    c.layer_is_full = lf;
    free(s);
    return c;
}

static inline void free_config_qwen(ModelConfig &c) {
    free((void *)c.layer_is_full); c.layer_is_full = nullptr;
}

#endif // DGX_MANIFEST_QWEN_H
