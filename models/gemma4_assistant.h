#ifndef DGX_GEMMA4_ASSISTANT_H
#define DGX_GEMMA4_ASSISTANT_H

// Gemma4 assistant(0.4B MTP drafter, EAGLE류) graph — draft-spec 2관문.
//   구조(HF 코드/텐서 실사·docs/gemma-assistant-spec-measured.md):
//   입력 = concat(target 스케일드 embed[3840], h_prev[3840]) → pre_projection(7680→1024) →
//   4층 트렁크(sliding×3+full): q_proj/q_norm/rope(고정 위치)만 자체 — ★K/V는 target의 마지막
//   full층(47)·sliding층(46) KV 캐시 직독(cross-attn·자체 KV 없음·prefill 없음), sandwich 4-norm·
//   layer_scalar·GeGLU(8192) → model.norm → 두 갈래: tied lm_head(=embed, softcap 없음) 로짓 +
//   post_projection(1024→3840) = 다음 draft 스텝의 h_prev(재귀).
//   ★scaling=1.0(QK-norm 계열)·centroid head 비활성(체크포인트에 텐서 자체 없음).
//   게이트: tests/test_gemma4_assistant (HF fp32 골든 — 스텝별 로짓/argmax/hidden + 재귀 + 결정성).

struct AstConfig {
    int   num_layers;         // 4
    int   hidden_size;        // 1024
    int   intermediate_size;  // 8192 (GeGLU)
    int   num_q_heads;        // 16
    int   head_dim;           // 256 (sliding)
    int   global_head_dim;    // 512 (full)
    int   num_kv_heads;       // 8  = target sliding KV 헤드수(cross-attn GQA)
    int   num_global_kv_heads;// 1  = target full KV 헤드수
    int   layer_is_full[64];  // 층 타입(파싱 상한 64 — 실제 4)
    int   sliding_window;     // 1024
    float rms_norm_eps;
    int   vocab_size;         // 262144 (12B 정합)
    int   backbone;           // 3840 (target hidden)
    float attn_scaling;       // 1.0
};

#include "../runtime/weights_qwen.h"   // QMat(wq/sc/wf — 공용 매트 컨테이너)

struct AstLayer {
    int    is_full;
    float  layer_scalar;                                      // host 스칼라(≠1 가능 — 학습값)
    float *input_ln, *q_norm, *post_attn_ln, *pre_ffn_ln, *post_ffn_ln;   // device fp32
    QMat   q, o, gate, up, down;                              // use_q4=0→wf(fp32) / 2→wq+sc(W4A8, Lever1)
};

struct AstWeights {
    QMat     embed;        // [V,1024] tied lm_head(매트멀 전용 — 룩업 아님이라 q4 안전)
    float   *final_norm;   // [1024]
    QMat     pre_proj;     // [1024, 7680]
    QMat     post_proj;    // [3840, 1024]
    AstLayer layers[64];
};

AstConfig  load_config_assistant(const char *manifest_path);
// use_q4: 0=fp32(wf — 골든 기준) / 2=W4A8(wq/sc — draft bytes 8×↓, 스킴은 target blueprint 동일)
AstWeights load_weights_assistant(const char *dir, const AstConfig &c, int use_q4 = 0);
void       free_weights_assistant(AstWeights &w, const AstConfig &c);

// 1 draft 스텝(Sq=1, fp32): d_in7680 = [emb | h_prev] concat(device, 호출자 준비).
//   d_kf/d_vf: target 마지막 full층 KV [len_f, NKV_f, hd_f] / d_ks/d_vs: 마지막 sliding층 [len_s, NKV_s, hd_s]
//     (엔진 KV 캐시 레이아웃 그대로 — 포인터 직독·복사 0).
//   cross-attn = launch_attn_decode(base=len-1, Sq=1): sliding window가 KV 끝 기준 마지막 w개
//     = HF bidirectional-flip 마스크의 q_len=1 의미와 정확히 일치(코드 실사).
//   rope: 고정 위치(pos=현재 길이-1)의 cos/sin 행 [hd] — 호출자가 준비(골든 덤프 행 or gen_rope 테이블 행).
//   출력: d_logits[V](softcap 없음), d_hout[3840](다음 스텝 h_prev — 재귀).
void gemma4_assistant_step(const AstConfig &c, const AstWeights &w,
                           const float *d_in7680,
                           const float *d_kf, const float *d_vf, int len_f,
                           const float *d_ks, const float *d_vs, int len_s,
                           const float *d_cos_f1, const float *d_sin_f1,
                           const float *d_cos_s1, const float *d_sin_s1,
                           float *d_logits, float *d_hout, int use_q4 = 0);

// ★draft concat(배치) 스텝 — verify concat(15dbd3c)의 대칭: 활성 슬롯 B개의 draft 스텝 j를 한 번에.
//   큰 matmul(pre_proj·trunk·post_proj·tied lm_head)만 B행으로 묶어 weight 1적재(gemv_w4a8_sb ≤16행 —
//   SB per-row == plain gemv per-row bit-id 기게이트·B==1은 기존 gemv 그대로=단일 경로 불변),
//   cross-attn은 슬롯별 기존 커널(각자 target KV 포인터·base=len[b]-1 — KV 매핑 명시적·자체 KV 없음),
//   rope는 행별 위치(rope_partial_pos·전체 테이블 + d_pos[b]=len[b]). 신규 커널 0.
//   d_in_B: [B,7680] 행별 [emb|h_prev](호출자 준비 — 재귀·argmax도 호출자=슬롯별 독립 재귀).
//   kf/vf/ks/vs: host 배열[B] = 슬롯별 target full/sliding KV 포인터. len_b: host [B](full==sliding 동일 규약).
//   출력 d_logits_B [B,V]·d_hout_B [B,3840]. 게이트: tests/test_gemma4_draft_concat(배치==순차 bit-id).
void gemma4_assistant_step_batch(const AstConfig &c, const AstWeights &w, int B,
                                 const float *d_in_B,
                                 const float* const* kf, const float* const* vf,
                                 const float* const* ks, const float* const* vs,
                                 const int *len_b,
                                 const float *d_cos_f, const float *d_sin_f,
                                 const float *d_cos_s, const float *d_sin_s,
                                 const int *d_pos,
                                 float *d_logits_B, float *d_hout_B, int use_q4 = 0);

#endif // DGX_GEMMA4_ASSISTANT_H
