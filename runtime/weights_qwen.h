#ifndef DGX_WEIGHTS_QWEN_H
#define DGX_WEIGHTS_QWEN_H
#include <cstdint>
#include "config.h"

// Qwen3.5 디바이스 가중치. full / GDN 레이어 혼합. 미사용 포인터는 NULL.
// 파일명 = extract_weights_qwen35.py 의 상대키 스킴 (L{i}_<relkey>[_q4|_scale].bin).
struct QLayer {
    int      is_full;
    float   *input_ln, *post_ln;
    // MLP (both)
    uint8_t *gate, *up, *down;  float *gate_s, *up_s, *down_s;
    // full attention
    uint8_t *q, *k, *v, *o;     float *q_s, *k_s, *v_s, *o_s;
    float   *q_norm, *k_norm;
    // GDN
    uint8_t *in_qkv, *in_z, *out_p;  float *in_qkv_s, *in_z_s, *out_s;
    float   *in_a, *in_b, *conv1d, *A_log, *dt_bias, *gdn_norm;
};

struct QwenWeights {
    float   *embed, *final_norm;
    uint8_t *lm_head;  float *lm_head_s;
    QLayer  *layers;   int L;
};

QwenWeights load_weights_qwen(const char *dir, const ModelConfig &c);
void free_weights_qwen(QwenWeights &w);

#endif // DGX_WEIGHTS_QWEN_H
