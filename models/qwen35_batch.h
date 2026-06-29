#ifndef DGX_QWEN35_BATCH_H
#define DGX_QWEN35_BATCH_H
#include <cuda_bf16.h>
#include "../runtime/config.h"
#include "../runtime/weights_qwen.h"

// 챕터3 batch decode: B개 독립 시퀀스(same-length), 각 1토큰을 한 forward로. (multi-sequence batch, plain, fp32.)
//   per-sequence KV/state. gemm_batch(weight 재사용) + batch attn/gdn/conv. == B개 single decode 시퀀스별 오차0.
struct BatchDecodeState {
    int B, n_past, max_len, L;
    __nv_bfloat16 **Kc, **Vc;  // [L], full 레이어: [B, max_len, NKV*HD] (bf16=모델 dtype, 메모리 절반)
    float **S, **convctx; // [L], gdn 레이어: S=[B, NH*Dv*Dv], convctx=[B, (K-1)*conv_dim]
};
BatchDecodeState alloc_batch_decode_state(const ModelConfig &c, int B, int max_len);
void free_batch_decode_state(BatchDecodeState &s);
void reset_batch_decode_state(const ModelConfig &c, BatchDecodeState &s);

// d_tokens: [B] (각 시퀀스의 현재 토큰, 위치 n_past). d_logits: [B, V]. commit(KV append + state + n_past++).
//   use_q4: 0=fp32, 1=q4, 2=W4A8 (matmul 경로; gemm_batch fp32/q4/w4a8).
void qwen35_decode_batch(const ModelConfig &c, const QwenWeights &w, BatchDecodeState &st,
                         const int *d_tokens, const float *d_cos, const float *d_sin, float *d_logits, int use_q4 = 0);

#endif // DGX_QWEN35_BATCH_H
