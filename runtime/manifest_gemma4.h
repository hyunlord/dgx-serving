#ifndef DGX_MANIFEST_GEMMA4_H
#define DGX_MANIFEST_GEMMA4_H
#include "manifest.h"   // mf_* 헬퍼, ModelConfig

// Gemma4 manifest → ModelConfig. layer_types: "sliding_attention"/"full_attention".
// (M4 유연성 수정 #3 — graph/config 흡수: Gemma4 특수성은 전부 config 필드, 커널 분기 없음)
static inline float mf_num_opt(const char *s, const char *key, double def) {
    char pat[128]; snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(s, pat); if (!p) return (float)def;
    p = strchr(p + strlen(pat), ':'); if (!p) return (float)def;
    return (float)strtod(p + 1, nullptr);
}

static inline ModelConfig load_config_gemma4(const char *manifest_path) {
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
    c.tie_word_embeddings = mf_bool(s, "tie_word_embeddings");
    c.scale_dtype       = mf_int_opt(s, "scale_dtype", 0);
    c.use_rope          = nullptr;

    // Gemma4 확장 필드 (extract가 config 주도로 기록)
    c.sliding_window        = mf_int_opt(s, "sliding_window", 0);
    c.global_head_dim       = mf_int_opt(s, "global_head_dim", 0);
    c.num_global_kv_heads   = mf_int_opt(s, "num_global_kv_heads", 0);
    c.attention_k_eq_v      = mf_int_opt(s, "attention_k_eq_v", 0);
    c.final_logit_softcapping = mf_num_opt(s, "final_logit_softcapping", 0.0);
    c.embed_scale           = mf_num_opt(s, "embed_scale", 0.0);
    c.attn_scaling          = mf_num_opt(s, "attn_scaling", 0.0);
    c.rotary_dim_full       = mf_int_opt(s, "rotary_dim_full", 0);
    c.rotary_dim_sliding    = mf_int_opt(s, "rotary_dim_sliding", 0);

    // layer_types → layer_is_full[i] (full_attention=1, sliding_attention=0) — qwen 파서와 동일 규약
    int *lf = (int *)malloc(sizeof(int) * c.num_layers);
    const char *p = strchr(mf_find(s, "layer_types"), '[');
    if (!p) { fprintf(stderr, "layer_types '[' 없음\n"); exit(1); }
    p++;
    for (int i = 0; i < c.num_layers; i++) {
        const char *q = strchr(p, '"');
        if (!q) { fprintf(stderr, "layer_types 파싱 부족 (i=%d)\n", i); exit(1); }
        lf[i] = (strncmp(q + 1, "full", 4) == 0) ? 1 : 0;
        p = strchr(q + 1, '"') + 1;
    }
    c.layer_is_full = lf;
    free(s);
    return c;
}

static inline void free_config_gemma4(ModelConfig &c) {
    free((void *)c.layer_is_full); c.layer_is_full = nullptr;
}

#endif // DGX_MANIFEST_GEMMA4_H
