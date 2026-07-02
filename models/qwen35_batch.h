#ifndef DGX_QWEN35_BATCH_H
#define DGX_QWEN35_BATCH_H
#include <cuda_bf16.h>
#include "../runtime/config.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"   // GdnStateT (fp32/bf16 by GDN_BF16_STATE)

// 챕터3 batch decode: B개 독립 시퀀스(same-length), 각 1토큰을 한 forward로. (multi-sequence batch, plain, fp32.)
//   per-sequence KV/state. gemm_batch(weight 재사용) + batch attn/gdn/conv. == B개 single decode 시퀀스별 오차0.
struct BatchDecodeState {
    int B, n_past, max_len, L;
    __nv_bfloat16 **Kc, **Vc;  // [L], full 레이어: [B, max_len, NKV*HD] (bf16=모델 dtype, 메모리 절반)
    GdnStateT **S; float **convctx; // [L], gdn: S=[B,NH*Dv*Dv](fp32/bf16), convctx=[B,(K-1)*conv_dim]
};
BatchDecodeState alloc_batch_decode_state(const ModelConfig &c, int B, int max_len);
void free_batch_decode_state(BatchDecodeState &s);
void reset_batch_decode_state(const ModelConfig &c, BatchDecodeState &s);
// 챕터9 연속배칭 slot 재사용: 물리 slot b의 GDN state(S/convctx) clear(잔재 제거). KV는 새 요청이 덮음.
void reset_batch_slot(const ModelConfig &c, BatchDecodeState &s, int b);
// 챕터9 chunked prefill: slot b에 프롬프트 P토큰을 청크(≤chunk_max, 기본512)로 병렬 prefill(1토큰/step 대신, ~62×).
//   qwen35_decode(Sq>1 full-MMQ)를 slot b KV/GDN view로 재사용. 후 slot은 pos=P부터 batch decode 합류.
void prefill_batch_slot(const ModelConfig &c, const QwenWeights &w, BatchDecodeState &st, int b,
                        const int *d_tokens, int P, const float *d_cos, const float *d_sin,
                        float *d_logit_out, int use_q4, int chunk_max = 512);

// d_tokens: [B] (각 시퀀스의 현재 토큰). d_logits: [B, V]. commit(KV append + state).
//   use_q4: 0=fp32, 1=q4, 2=W4A8.
//   h_pos_b: nullptr=same-length(모든 seq 위치=st.n_past, st.n_past++). 비-null=ragged(연속배칭): h_pos_b[b]=시퀀스 b의
//     현재 위치(per-seq rope+KV+attn), st.n_past 불변(호출자가 per-seq 위치 관리). 커널 재사용, 위치만 per-seq.
void qwen35_decode_batch(const ModelConfig &c, const QwenWeights &w, BatchDecodeState &st,
                         const int *d_tokens, const float *d_cos, const float *d_sin, float *d_logits, int use_q4 = 0,
                         const int *h_pos_b = nullptr);

// 챕터10 Phase 1~3: B개 슬롯(요청)의 tree를 한 forward로 verify (batched tree verify, ragged).
//   슬롯별 Nnodes_b[b](Phase3 축2 ragged; uniform은 전부 같은 값). 노드는 concat 레이아웃:
//     행 r = off[b] + i  (off[b]=ΣNnodes_b[0..b-1] prefix-offset, 슬롯 b의 i번째 노드). 총 M = ΣNnodes_b.
//   commit 안 함(verify). qwen35_decode(single tree verify, qwen35_decode.cu:103) 본체를 슬롯 차원으로 확장:
//     · rmsnorm/rope(d_pos)/swiglu/residual/matmul(q/k/v/gate/up/down/o/out_p/lm_head): 행-독립 → concat 행에 그대로.
//       fp32는 M 한 커널(타일 꽉 참). W4A8 canonical(축1): 전 슬롯 >8 → concat-mmq(M) / 전 슬롯 2~8 && M≤16 →
//       concat-sb(M) — 둘 다 단독 spec 커널과 bit-identical(row/컬럼-독립) + weight 1적재=곱. 혼합·M>16 소트리는
//       per-slot(각 슬롯 단독 커널 일치, 곱 부분 손실; borderline-flip 회피).
//     · attention: 슬롯별 루프 — 슬롯 b 노드는 sts[b]->Kc/Vc([0..base_b]+자기 tree)만 attend + 슬롯 b tree mask(d_anc_b).
//     · GDN: 슬롯별 루프 — 슬롯 b 노드는 sts[b]->S서 tree-fork(gdn_step_ksplit_tree + conv1d_silu_decode_tree).
//   ★상태는 슬롯별 독립 DecodeState(sts[b]) — single spec의 KV/GDN/convctx + promote_tree_path 자산을 슬롯별 그대로 재사용.
//     verify 중 각 슬롯 노드 i의 roped K/V는 sts[b]->Kc/Vc의 slot (base_b[b]+i)에 기록(단독 spec과 동일 node-index 배치;
//     이후 호스트가 슬롯별 promote_tree_path로 path 노드를 contiguous compaction). n_past는 불변(commit 안 함).
//   입력: Nnodes_b[B](호스트, 슬롯별 트리 크기); d_tokens_b concat[M] 토큰; base_b[B]=슬롯 committed 길이(sts[b]->n_past);
//     d_parent_b/d_pos_b/d_anc_b concat[M]=슬롯-로컬 부모(-1..Nb-1)/위치(base+depth)/조상 비트마스크(슬롯-로컬, off[b]부터 Nb개).
//     va[B]=슬롯별 VerifyArtifacts(accept 승격용 gdn_snap/conv_mixed). null이면 snapshot 생략.
//   출력: d_logits[M, V]; d_hnextn nullable [M, H] = final-norm 적용 후(postnorm=h_nextn, MTP draft 입력).
//     (single spec qwen35_decode의 d_postnorm과 동일 역할 — test_treespec_e2e/mtp_step이 h_post로 쓰는 값.)
void qwen35_verify_batch(const ModelConfig &c, const QwenWeights &w, DecodeState **sts,
                         const int *d_tokens_b, int B, const int *Nnodes_b, const int *base_b,
                         const float *d_cos, const float *d_sin,
                         float *d_logits, float *d_hnextn, VerifyArtifacts *va, int use_q4,
                         const int *d_parent_b, const unsigned *d_anc_b, const int *d_pos_b);

#endif // DGX_QWEN35_BATCH_H
