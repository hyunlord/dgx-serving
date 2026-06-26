#ifndef DGX_WEIGHTS_H
#define DGX_WEIGHTS_H
#include <cstdint>
#include "config.h"

// 디바이스 상주 가중치 (config 기반 범용 로딩). 선형은 q4(uint8)+scale(fp32), norm/embed는 fp32.
struct LayerW {
    float   *input_ln, *post_ln;                          // [H]
    uint8_t *q, *k, *v, *o, *gate, *up, *down;            // q4 packed
    float   *q_s, *k_s, *v_s, *o_s, *gate_s, *up_s, *down_s; // scale
};

struct EngineWeights {
    float   *embed;        // [V,H] fp32
    float   *final_norm;   // [H] fp32
    uint8_t *lm_head;      // q4 [V,H]
    float   *lm_head_s;    // scale
    LayerW  *layers;       // [L]
    int      L;
};

// dir 안의 표준 포맷(L{i}_*.{bin,q4.bin,scale.bin}, w_embed/w_final_norm/w_lm_head) 로드.
EngineWeights load_weights(const char *dir, const ModelConfig &c);
void free_weights(EngineWeights &w);

#endif // DGX_WEIGHTS_H
