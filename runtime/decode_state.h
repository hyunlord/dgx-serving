#ifndef DGX_DECODE_STATE_H
#define DGX_DECODE_STATE_H
#include <cuda_bf16.h>
#include "config.h"
#include "weights_qwen.h"

// 챕터9 bf16 GDN state: state I/O 대역폭 반감(de-risk GO: recurrence contractive g_t<1 → 오차 감쇠 <2% plateau, 커널 1.94×).
//   기본 fp32(회귀 0). -DGDN_BF16_STATE 로 bf16(mixed: 저장 bf16, 계산 fp32). snapshot(gdn_snap)은 fp32 유지(spec 정확성).
#ifdef GDN_BF16_STATE
typedef __nv_bfloat16 GdnStateT;
#else
typedef float GdnStateT;
#endif

// speculative decode 상태 3종 (docs/02c-speculative-blueprint.md).
//   full-attn KV: full 레이어만 [max_len, NKV, HD] (roped K 저장). 위치독립 → 길이 포인터 rollback.
//   GDN committed state S: linear 레이어만 [NH, D, D]. 순차누적 → 임시승격.
//   conv1d ctx: linear 레이어만 [K-1, conv_dim]. decode시 직전 K-1 토큰.
// 인덱스는 전체 레이어 기준 배열; GDN 레이어의 Kc/Vc=NULL, full 레이어의 S/convctx=NULL.
struct DecodeState {
    int   n_past;     // 확정된 시퀀스 길이 (= committed KV 길이 = 다음 토큰 위치)
    int   max_len;
    int   L;          // 레이어 수 (free 시 사용)
    __nv_bfloat16 **Kc, **Vc;  // [L], full 레이어만 (roped K, V; bf16=모델 dtype, 메모리 절반)
    GdnStateT **S;          // [L], linear 레이어만 (committed GDN state; fp32 or bf16 by GDN_BF16_STATE)
    float **convctx;        // [L], linear 레이어만 ([K-1, conv_dim])
};

DecodeState alloc_decode_state(const ModelConfig &c, int max_len);
void free_decode_state(DecodeState &s);
void reset_decode_state(const ModelConfig &c, DecodeState &s);  // n_past=0, S/convctx=0

// verify(commit=0)가 임시승격용으로 노출하는 산출물. alloc/free 헬퍼로 관리.
//   gdn_snap : [Sq, n_linear, NH*D*D] — 각 토큰 처리 후 GDN state 스냅샷.
//   conv_mixed: [Sq, n_linear, conv_dim] — conv 입력(in_proj_qkv 출력) per 토큰 (conv ctx 승격용).
struct VerifyArtifacts {
    int    Sq, n_linear, state_sz, conv_dim;
    float *gdn_snap;
    float *conv_mixed;
};
VerifyArtifacts alloc_verify_artifacts(const ModelConfig &c, int max_Sq);
void free_verify_artifacts(VerifyArtifacts &v);

// 본체 incremental decode/verify. Sq 토큰(d_tokens, pos n_past..n_past+Sq-1) 처리.
//   d_logits: [Sq, V] 출력. d_prenorm: nullable [Sq, H] (final-norm 전 = MTP 입력).
//   commit=1: KV append 확정 + S/convctx/n_past 갱신 (일반 decode). va 무시.
//   commit=0: committed S/convctx/n_past 불변 (verify). va!=NULL이면 스텝별 GDN snapshot + conv mixed 기록.
//   d_cos/d_sin: [max_len, rot] rope 테이블 (위치 p → row p). use_q4: fp32(0) 기준선.
//   d_postnorm: nullable [Sq, H] (final-norm 적용 후 = h_nextn, MTP draft 입력 — 2b 확정).
//   last_logits_only: 1=마지막 토큰만(prefill TTFT, d_logits=[1,V]), 2=lm_head 생략(비-마지막 청크).
void qwen35_decode(const ModelConfig &c, const QwenWeights &w, DecodeState &st,
                   const int *d_tokens, int Sq, const float *d_cos, const float *d_sin,
                   float *d_logits, float *d_prenorm, int commit, VerifyArtifacts *va, int use_q4,
                   float *d_postnorm = nullptr, int last_logits_only = 0,
                   // 챕터9 tree-verify(모두 null=기존 linear): d_parent[Sq]=GDN/conv fork 부모, d_anc[Sq]=attn 조상 비트마스크, d_pos[Sq]=rope 위치(base+depth)
                   const int *d_parent = nullptr, const unsigned *d_anc = nullptr, const int *d_pos = nullptr);

// verify 후 앞 m 토큰을 committed 로 승격 (임시승격: GDN snap[m-1] 복사 + conv ctx 갱신 + n_past+=m).
//   KV/cache 의 [n_past..n_past+m-1] 은 verify 중 이미 기록됨 → 길이 포인터만 전진.
void promote_prefix(const ModelConfig &c, DecodeState &st, const VerifyArtifacts &va, int m);

// 챕터9 STEP4: tree verify 후 accept된 path(노드 인덱스 path[0..plen-1], path[0]=root, strictly increasing) 승격.
//   GDN S=snap[path末], conv ctx=path mixed 마지막 K-1, KV(full-attn)=path 노드를 contiguous slot으로 compaction. n_past+=plen.
void promote_tree_path(const ModelConfig &c, DecodeState &st, const VerifyArtifacts &va, const int *path, int plen);

#endif // DGX_DECODE_STATE_H
