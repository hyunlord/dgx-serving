#ifndef DGX_WEIGHTS_QWEN_H
#define DGX_WEIGHTS_QWEN_H
#include <cstdint>
#include "config.h"

// 행렬곱 weight (q4 또는 fp32, precision 토글로 선택). 둘 다 로드하면 한 엔진이 두 경로 모두 실행 가능.
//   q4 경로: wq(int4 packed) + sc(scale). fp32 경로: wf.  미로드 포인터는 NULL.
struct QMat { uint8_t *wq = nullptr; void *sc = nullptr; float *wf = nullptr; };

struct QLayer {
    int      is_full;
    float   *input_ln, *post_ln;
    QMat     gate, up, down;                  // dense MLP (num_experts==0)
    // full attention
    QMat     q, k, v, o;  float *q_norm, *k_norm;
    // GDN
    QMat     in_qkv, in_z, out_p;
    float   *in_a, *in_b, *conv1d, *A_log, *dt_bias, *gdn_norm;
    // MoE (num_experts>0): batched expert q4 [E,2*inter,H]/[E,H,inter] + router/shared
    QMat     experts_gate_up, experts_down;  // expert e = offset e*(out*in) 인덱싱
    float   *router;                          // fp32 [E, H]
    QMat     shared_gate, shared_up, shared_down;
    float   *shared_gate_w;                   // fp32 [1, H] (sigmoid 게이트)
};

struct QwenWeights {
    float  *embed, *final_norm;
    QMat    lm_head;
    QLayer *layers;  int L;
    int     have_q4, have_fp32;   // 로드된 경로
};

// load_q4/load_fp32: 각 경로 weight 로드 여부. 둘 다 1이면 두 경로 모두 가능.
QwenWeights load_weights_qwen(const char *dir, const ModelConfig &c, int load_q4, int load_fp32);
void free_weights_qwen(QwenWeights &w);

#endif // DGX_WEIGHTS_QWEN_H
