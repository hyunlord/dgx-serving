#ifndef DGX_DECODE_STATE_GEMMA4_H
#define DGX_DECODE_STATE_GEMMA4_H
#include "config.h"
#include "weights_gemma4.h"

// Gemma4 decode 상태: KV 캐시만(GDN state 경로 없음 — 27b보다 단순).
//   층 타입별 head 구성: full(global) hd_f·NKV_f, sliding hd_s·NKV_s.
//   KV는 fp32(golden 우선; bf16 최적화는 STEP B). 위치독립 cache → base 포인터 전진.
//   sliding 층 캡(STEP A-2): kv_cap[l] = sliding이면 min(max_len, sliding_window), full이면 max_len.
//   물리 슬롯 = pos % kv_cap[l] (링버퍼). A-1은 캡=max_len(선형)으로 시작, A-2에서 w=1024 캡.
struct G4DecodeState {
    int    n_past;      // 확정 시퀀스 길이(= 다음 토큰 위치)
    int    max_len;
    int    L;
    int   *kv_cap;      // [L] 층별 KV 캡(sliding=min(max_len,w) / full=max_len)
    float **Kc, **Vc;   // [L] fp32 KV 캐시 (roped K, v_norm된 V). 크기 = kv_cap[l]*NKV_l*hd_l
};

G4DecodeState alloc_g4_decode_state(const ModelConfig &c, int max_len);
void free_g4_decode_state(G4DecodeState &s);
void reset_g4_decode_state(G4DecodeState &s);   // n_past=0 (KV 재사용)

// 증분 decode/프리필 통합. Sq 토큰(d_tokens, 위치 n_past..n_past+Sq-1) 처리.
//   base=0 & Sq=S → 프리필. base>0 & Sq=1 → decode 스텝. KV는 슬롯 base..에 append.
//   d_cos_f/d_sin_f: full 층 rope [max_len, hd_f], d_cos_s/d_sin_s: sliding 층 [max_len, hd_s].
//     위치 p의 row = p (엔진 진입 전 전체 위치 테이블 준비 — golden은 HF 캡처본).
//   d_logits: [Sq,V] (softcap 후). commit=1이면 n_past += Sq.
//   d_hidden: nullable [Sq(또는 last), H] 진단용. last_only: 1=마지막 토큰만 logits.
//   d_prenorm(nullable, [Sq,H]): final-norm 직전 residual(x = 마지막 층 출력, layer_scalar 후) 복사 —
//     HF hidden_states[-1]과 동일(실측: final-norm 전). assistant draft의 h_prev 입력(EAGLE hook·3관문).
void gemma4_decode(const ModelConfig &c, const G4Weights &w, G4DecodeState &st,
                   const int *d_tokens, int Sq,
                   const float *d_cos_f, const float *d_sin_f,
                   const float *d_cos_s, const float *d_sin_s,
                   float *d_logits, int commit, int use_q4, int last_only = 0, int amort_mm = 0,
                   float *d_prenorm = nullptr);

// ---- Batch decode (A-3-2): B슬롯 동시. same-length(모든 슬롯 위치 base 공통) — ragged/chunked는 A-3 후속. ----
//   KV: 층별 [B, max_len, NKV_l, hd_l] fp32 (kv_stride = max_len*NKV_l*hd_l). GDN 없어 KV만.
struct G4BatchState {
    int    n_past, max_len, L, B;
    int   *kv_cap;
    float **Kc, **Vc;   // [L], 각 [B, max_len, NKV_l, hd_l]
};
G4BatchState alloc_g4_batch_state(const ModelConfig &c, int B, int max_len);
void free_g4_batch_state(G4BatchState &s);

// B슬롯 1토큰 동시 decode. matmul M=B(gemm_batch), attn 슬롯별.
//   d_tokens: [B]. d_logits: [B,V] (softcap 후).
//   same-length: d_pos_b=h_pos_b=nullptr → 모든 슬롯 위치 st.n_past. commit=1이면 n_past += 1.
//   ragged(연속배칭): d_pos_b[B](device)+h_pos_b[B](host) 슬롯별 위치. per-slot 위치는 호출자(scheduler)가 관리
//     (commit 무관 — n_past 안 건드림). KV append 오프셋은 h_pos_b, rope는 d_pos_b(전체 테이블), attn window는 슬롯별.
void gemma4_decode_batch(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                         const int *d_tokens, const float *d_cos_f, const float *d_sin_f,
                         const float *d_cos_s, const float *d_sin_s,
                         float *d_logits, int commit, int use_q4,
                         const int *d_pos_b = nullptr, const int *h_pos_b = nullptr);

// chunked prefill: 슬롯 b에 P토큰(위치 0..P-1) 프리필. KV 레이아웃 [B,max_len,nkv,hd]에서 슬롯 b 서브영역
//   (Kc[l]+b*kv_stride)이 곧 단일-시퀀스 KV → 검증필 gemma4_decode(Sq=P) 프리필을 슬롯-b 뷰로 재사용(신규 커널 0).
//   d_tokens[P](device), d_logits=[1,V] scratch(last_only). 서빙은 KV만 필요(prefill 로짓 discard).
//   ★정확성: KV 내용(roped K/v_norm V)은 위치별·attn 무관이라 1-tok ingest와 동일 KV여야 함(bit-id 게이트로 실측).
void gemma4_prefill_batch_slot(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                               int b, const int *d_tokens, int P,
                               const float *d_cos_f, const float *d_sin_f,
                               const float *d_cos_s, const float *d_sin_s,
                               float *d_logits, int use_q4);

// spec chain verify (draft-spec 1관문, dense 전용 — 12B assistant 경로): 활성 슬롯 a의 청크
//   [anchor, d1..dK_a](= Nb[a] 토큰, d_tokens_cat의 off[a]..)를 위치 pos_b[a]..pos_b[a]+Nb[a]-1 에서 한 forward로 검증.
//   d_logits_cat: [ΣNb, V] 전 위치 로짓(softcap 후) — row off[a]+i = "토큰 i까지 본 뒤 다음 토큰" 분포.
//   KV는 슬롯 서브영역 pos_b[a]..에 append. ★commit 없음 — 수락 길이만큼 위치 전진은 호출자(호스트) 산술이고,
//   미수락 위치의 KV는 다음 청크 append가 덮어씀 = 롤백은 위치 관리뿐(GDN 없음 — 27b와 달리 state snapshot 불요).
//   ★bit-id canonical: 슬롯별 gemma4_decode(Sq=Nb, commit=0, last_only=0, amort=0) 뷰 호출 = plain 1토큰 decode와
//   문자 그대로 동일 커널 경로(mm/rope/attn/lm_head 전부 행-독립 — 청크 prefill Δ=0 게이트와 같은 성질).
//   게이트: tests/test_gemma4_verify (spec==greedy bit-id·거부/롤백·결정성·batch==solo).
//   concat 분할상환(27b qwen35_verify_batch 축1)은 후속 perf lever — 정확성 관문 먼저.
void gemma4_verify_batch(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                         int B_act, const int *slot_ids, const int *Nb, const int *pos_b,
                         const int *d_tokens_cat,
                         const float *d_cos_f, const float *d_sin_f,
                         const float *d_cos_s, const float *d_sin_s,
                         float *d_logits_cat, int use_q4);

#endif // DGX_DECODE_STATE_GEMMA4_H
