# 2b 설계안 — Qwen3.5-0.8B MTP head (STEP 0, 작성 중)

> 2a 본체(fp32 next-token=11751 HF 일치, q4 dual-path) 위에 MTP head. 과거 Python 0% accept를 깬다.
> **이 문서 합의 후 STEP 1 착수.** 미확정 3개(§4)는 조사(llama.cpp KV / HF MTP 모듈) 결과로 확정.

## 1. 확정 — MTP weight 텐서 (실측, BF16)

embed/lm_head는 본체와 tie(공유). MTP 전용:
| 텐서 | shape | 역할 |
|---|---|---|
| `mtp.pre_fc_norm_embedding` | (1024,) | enorm (e 정규화) |
| `mtp.pre_fc_norm_hidden` | (1024,) | hnorm (h 정규화) |
| `mtp.fc` | (1024, 2048) | eh_proj: concat[e_n,h_n]([2H=2048]) → H=1024 |
| `mtp.layers.0.input_layernorm` | (1024,) | (1+w) zero-centered |
| `mtp.layers.0.self_attn.q_proj` | (4096,1024) | =8*256*2 (Q+gate, output_gate) |
| `mtp.layers.0.self_attn.k/v_proj` | (512,1024) | NKV*HD=2*256 |
| `mtp.layers.0.self_attn.q/k_norm` | (256,) | head_dim RMSNorm |
| `mtp.layers.0.self_attn.o_proj` | (1024,2048) | NQ*HD=2048 → H |
| `mtp.layers.0.post_attention_layernorm` | (1024,) | |
| `mtp.layers.0.mlp.{gate,up}` / `down` | (3584,1024)/(1024,3584) | SwiGLU |
| `mtp.norm` | (1024,) | shared_head_norm (lm_head 전) |

→ MTP attn = **2a full-attn 형과 동일** (q/k_norm + partial rope + output_gate). GDN 아님.
  신규 연산은 **eh_proj 앞단(enorm/hnorm/concat/fc)** 뿐. 나머지는 2a 커널 재사용.

## 2. MTP forward 그래프 (위치 P, 의뢰서 + 2a 정독 기반 — 조사로 KV/pos 확정 예정)

```
e = embed(token[P+1])            # ★ 페어링: 다음 토큰 (token[P] 아님)
h = prenorm_hidden[P]            # 본체 위치 P의 final-norm 전 hidden (2a prenorm_out, 1.4e-5 검증)
e_n = rmsnorm_zc(e, enorm); h_n = rmsnorm_zc(h, hnorm)
x = fc(concat[e_n, h_n])         # ★ [e,h] 순서 (e 먼저)
cur = rmsnorm_zc(x, input_layernorm)
q|gate = q_proj(cur) → split; q=q_norm(q); k=k_norm(k_proj(cur)); v=v_proj(cur)
rope_partial(q,k, pos)           # θ=1e7, 앞 64dim
attn = causal_attend(q, k[prefix], v[prefix])   # ★ prefix attend (self 아님), KV=§4.1
attn *= sigmoid(gate); cur = o_proj(attn) + x
cur = swiglu(rmsnorm_zc(cur, post_attention_layernorm)) + cur
logits = lm_head(rmsnorm_zc(cur, mtp.norm))      # tie
draft = argmax(logits)           # 타깃 = token[P+2]
```
주의: 모든 norm은 2a에서 검증한 `(1+w)` zero-centered (`rmsnorm_zc`). enorm/hnorm도 동일 컨벤션(조사 확인).

## 3. 커널/도구 계획
- **재사용(2a)**: gemv(q4/fp32 dispatch), rmsnorm_zc, rope_partial, attention(full-attn), gate_sigmoid, swiglu, embedding.
- **신규**: eh_proj 앞단 — enorm/hnorm(rmsnorm_zc 재사용) + concat + fc(gemv). concat[e_n,h_n]는 버퍼 배치로 처리.
- **도구**: extract에 mtp.* 추출 추가(fc/enorm/hnorm/layers.0/norm; embed·lm_head 공유). q4+fp32 둘 다(2a 방침).
- **본체 변경**: prenorm_out(이미 있음) + 본체 KV 또는 MTP catch-up용 hidden 노출 (§4.1 결과 따라).

## 4. 조사 결과 (B=llama.cpp 완료) + 미확정 (C 대기)

### 4.1 MTP KV: **자체 캐시 + catch-up decode** (확정, llama.cpp 소스)
- MTP attn은 **자체 K/V** (MTP 레이어 wk/wv로 prefix에서 생성), **별도 KV 캐시**(MTP 레이어 1개만).
  본체 KV 안 씀. (Gemma=공유, Qwen=자체 — `llama-model.cpp:2047-2051,2144`, `qwen35.cpp:540`)
- **catch-up**: `speculative.cpp:1069-1105` — `is_mem_shared==false`라 prefix 전체를 MTP로 1회 decode해
  자체 KV를 채운 뒤 draft. (`build_attn_inp_kv`)
- → **CUDA 구현**: MTP 레이어용 KV 캐시(K,V) 별도 유지 + prefix 1회 훑어 채우기. 본체 KV에서 파생 아님.
- 주의: 의뢰서가 가리킨 `speculative.cpp:642`는 EAGLE3. 실제 Qwen MTP 드라이버는
  `common_speculative_impl_draft_mtp` (line 896~), 그래프는 `qwen35.cpp:graph_mtp` (488-644).

### 4.2 pos: **본체와 동일 절대위치** (확정)
- MTP는 `build_inp_pos()`로 본체와 같은 절대 pos. RoPE는 trunk와 **동일** `ggml_rope_multi`
  (IMRoPE, 같은 sections/n_rot/θ) — `qwen35.cpp:587-592` = trunk `302-312`. draft 토큰 pos=n_past+i+1.
- → text-only면 2a `rope_partial`(앞 64dim, θ=1e7) 그대로 적용.

### 4.3 eh_proj concat 순서: **e 먼저, h 나중** (확정)
- `qwen35.cpp:548`: `ggml_concat(e_norm, h_norm, dim=0)` → e leading. eh_proj `{2H,H}`. enorm/hnorm=RMSNorm.

### 4.4 ⚠️ h 정의 충돌 — pre-norm vs post-norm (★ 보고/합의 필요)
- **의뢰서**: h = "norm 前 hidden = model.norm의 입력" (pre-norm). 2a `prenorm_out`이 이것.
- **llama.cpp (직접 확인 `qwen35.cpp:209-212`)**: `cur = build_norm(cur, model.output_norm)` **후**
  `cb(cur,"h_nextn")` → MTP가 받는 h = **본체 final-norm 적용 *後*** hidden. (post-norm)
- 정면 충돌. 페어링 핵심이라 0%의 진범 후보. **C(HF 공식 MTP 모듈 있으면 그 코드가 확정) 결과로 판정**,
  없으면 사용자 합의로 결정. 2a가 pre(`prenorm_out`)·post(`hidden_dump[L]`) 둘 다 노출하므로 구현은 준비됨.
  → 안전책: 두 h로 각각 accept 측정해 실증.

### 4.5 골든 전략 (확정): HF에 MTP 모듈 **없음** → 부품별 + llama.cpp
- 직접 확인: transformers qwen3_5/qwen3_next에 MTP/NextN 클래스 **전무**, `_keys_to_ignore ^mtp.*`로
  weight만 무시. → **HF로 MTP 골든 불가**.
- 채택: **(B) 부품별 결정적 골든** — eh_proj/enorm/hnorm/concat은 torch로 정확 골든(입력 e,h+weight),
  rmsnorm/attn/swiglu/proj는 2a 검증 커널이 골든. 부품 다 맞고 accept 0%면 구조(h/KV/pos)로 좁혀
  **(A) llama.cpp cb_eval 골든**(이름 `mtp_tok_embd/hnorm/enorm/concat/eh_proj/attn_pregate/h_nextn/result_output`;
  0.8B GGUF 변환 필요). accept는 **fp32 본체 기준선**으로만.

## 4.6 과거 실패 정밀 해부 (조사 C — 같은 실수 금지)
| 스크립트 | e | h | attn | RoPE | concat | 활성 버그 |
|---|---|---|---|---|---|---|
| reproduce | tn[p]=argmax ✗ | pre-norm | self-only(out=V) | 손수, /half 지수버그 | [e,h] | 진범 3개 |
| v4 | 5개 sweep(정답 포함) | pre-norm | self-only | /half 버그 | [e,h] | attn버그가 다 가림 |
| v5(최완성) | ids[P+1] & tn[P] | pre-norm | causal/S, **prefix KV 없음** | transformers util ✓ | [e,h] | **pos off-by-1, prefix KV 부재** |

교훈: (1) attn self-only 금지(prefix attend), (2) RoPE는 2a 검증 `rope_partial` 사용(손수짜기 금지),
(3) **pos**: 시퀀스 위치 P의 MTP 입력은 위치 P+1에 대응 → off-by-1 주의, (4) **prefix KV 필수**.

## 4.7 (확정) 접근 — torch 레퍼런스 먼저 non-zero, 그 다음 CUDA
HF 무모듈이므로 **손수 만든 torch MTP가 곧 골든**이다. 과거 0%를 Python에서 먼저 깬다:
1. `tools/mtp_reference.py`: mtp.* weight 로드 + MTP forward 직접 구현 (2a 본체의 h 사용,
   진범3 + pos/KV/h-norm 정확히). transformers `apply_rotary_pos_emb`만 검증 util로 차용.
2. fp32 본체 기준선에서 **accept 측정** → non-zero 나오면 h/pos/KV 확정 (경험적). 이게 CUDA의 골든.
3. 그 torch 레퍼런스의 부품별 중간값을 덤프 → CUDA 부품과 오차 0 대조 → 조립 → 엔진 accept = torch accept 재현.
→ "Python 0%"를 Python에서 먼저 해결(빠른 반복)하고, CUDA는 검증된 레퍼런스를 재현만. 위험 최소.

## 4.8 ★ STEP 1 결과 — 과거 0% 돌파 (tools/mtp_reference.py)

프롬프트 "The capital of France is Paris, and the capital of Japan is Tokyo." (15토큰, 14 draft 위치).
accept = mean_P(MTP draft[P] == body greedy argmax[P+1]):

| h 시점 | pos | accept |
|---|---|---|
| pre  | P1/P0 | 57.1% |
| **post (h_nextn)** | **P1/P0** | **78.6%** ← best |

- **78.6% accept** — 과거 4번(0%)을 깸. 진범 4개 동시 반영의 결과.
- **h = post-norm(h_nextn) 확정** (78.6 vs pre 57.1). llama.cpp 일치. **4번째 진범=h 시점** 실증.
- pos P1/P0 무차이(짧은 text). 페어링 민감도: token 변경 → draft 4종(정상, e 살아있음 → eh_proj/concat 정확).
- → **이 torch forward(post-norm)가 CUDA 이식의 검증 골든.** 부품별/최종 logits 대조 후 엔진 accept=78.6% 재현.

## 4.9 ★ STEP 2-3 결과 — CUDA MTP가 골든 재현 (tests/test_mtp.cu)

CUDA MTP head (fp32 경로, eh_proj + 2a full-attn 커널 재사용 + ffn + head):
- 부품별 vs torch golden: eh_proj x **1.4e-5**, attn cur2 **1.6e-5**, logits **5.0e-5**.
- **accept = 78.6% (11/14)** — torch 골든과 동일. 과거 0% 완전 돌파를 CUDA에서 입증.
- 신규 커널 없음 — eh_proj는 rmsnorm_zc+concat(memcpy2D)+gemv_f32, 나머지 2a 재사용.

남은 것(STEP 4 speculative 루프)은 **autoregressive decode + KV 캐시 인프라** 필요(2a에서 stage2+로 미룸).
2b 핵심 목표(0% 돌파 + 의미있는 accept)는 **달성**. 생성 루프는 decode 인프라 단계에서.

## 5. (확정) STEP 0 결론 요약
- KV: 자체+catch-up. pos: 본체 동일(IMRoPE=partial rope). concat: e먼저. eh_proj `{2H,H}`. enorm/hnorm RMS.
- h 정의: **llama.cpp=post-norm(h_nextn)** 가 유일 권위 레퍼런스(HF 무모듈). 의뢰서=pre-norm. → **양쪽 실측**.
- 골든: 부품별(torch+2a커널), 필요시 llama.cpp cb_eval. fp32 기준선.
- 신규 구현: eh_proj 앞단(enorm/hnorm/concat/fc) + MTP 자체 KV catch-up. 나머지 2a 커널 재사용.

## 5. 과거 0% 방지 단위 테스트 (의뢰서 교훈)
- **페어링 민감도**: 같은 h, token만 바꿔 draft가 변하는지 (안 변하면 e가 죽음 = eh_proj/concat 오류).
- 부품별 오차 0 누적 후 조립. 끝값(draft)만 보고 추측 금지.

## 6. 작업 순서
STEP1 eh_proj(+추출) → STEP2 MTP attn(prefix) → STEP3 ffn+head 통합 → STEP4 accept 측정 + speculative 루프.
각 단계 끝 검증+커밋.
