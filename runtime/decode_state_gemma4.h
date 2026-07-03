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
void gemma4_decode(const ModelConfig &c, const G4Weights &w, G4DecodeState &st,
                   const int *d_tokens, int Sq,
                   const float *d_cos_f, const float *d_sin_f,
                   const float *d_cos_s, const float *d_sin_s,
                   float *d_logits, int commit, int use_q4, int last_only = 0);

// ---- Batch decode (A-3-2): B슬롯 동시. same-length(모든 슬롯 위치 base 공통) — ragged/chunked는 A-3 후속. ----
//   KV: 층별 [B, max_len, NKV_l, hd_l] fp32 (kv_stride = max_len*NKV_l*hd_l). GDN 없어 KV만.
struct G4BatchState {
    int    n_past, max_len, L, B;
    int   *kv_cap;
    float **Kc, **Vc;   // [L], 각 [B, max_len, NKV_l, hd_l]
};
G4BatchState alloc_g4_batch_state(const ModelConfig &c, int B, int max_len);
void free_g4_batch_state(G4BatchState &s);

// B슬롯 1토큰 동시 decode (same-length, 위치 = st.n_past). matmul M=B(gemm_batch), attn 슬롯별.
//   d_tokens: [B]. d_logits: [B,V] (softcap 후). commit=1이면 n_past += 1.
void gemma4_decode_batch(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                         const int *d_tokens, const float *d_cos_f, const float *d_sin_f,
                         const float *d_cos_s, const float *d_sin_s,
                         float *d_logits, int commit, int use_q4);

#endif // DGX_DECODE_STATE_GEMMA4_H
