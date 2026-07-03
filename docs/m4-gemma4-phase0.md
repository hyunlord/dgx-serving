# M4 Phase 0 — Gemma4 31B dense: 해부 리포트 (유연성 시험)

## STEP 0 — mw1 확정 (M2 잔여 완결)

- warm-start A/B가 발견한 B16 K7W2 **mw1**을 확정 측정(λ4 SHORTP 0.3, nreq60):
  **seed42 53.2 / seed777 53.9 tok/s vs mw0 51.0(같은 trace controlled) = +4.3~5.7%**, tpot95도 개선
  (0.324/0.299 vs 0.340). **full ISO 12/12 PASS.** screening 허상 아님 — 진짜.
- mw0 51.0은 레시피 claim 51.7을 1.4% 이내 재현(측정 파이프 정합 부수 확인).
- 레시피 갱신: regime-table **mw1 갱신판 아티팩트**(원본 provenance 참조물 보존) → `recipe.py export` →
  verify 지문/가드 PASS. λ4 long-heavy 승자 = **B16 K7W2 mw1 53.2**(engine ec4d7a1).

> 원칙: 전부 **config.json + transformers modeling 코드 + safetensors index에서 추출**(추측 0건).
> 프레임 = 성능이 아니라 유연성 시험 — 수정 3분류 기록이 산출물. 커널 오염(③)만 금지 신호.

## 0. 자산 확보

- `google/gemma-4-31B-it` HF 스냅샷 62.5GB(bf16), 텐서 1188개. transformers 5.12.1이 Gemma4 지원(골든 가능).
- **멀티모달 래퍼**(`Gemma4ForConditionalGeneration` = vision_tower 27층 + audio 토큰 + language_model 60층).
  **텍스트 서빙 대상 = `model.language_model.*`만** — vision/audio는 Phase 0~ 스코프 제외(명시).

## 1. 아키텍처 델타 (vs Qwen3.5-27B — 전부 코드/config 근거)

| 항목 | Qwen3.5-27B | Gemma4-31B | 근거 |
|---|---|---|---|
| 형상 | L64 H5120 FFN17408 V248320 | **L60 H5376 FFN21504 V262144** | config |
| 레이어 혼합 | GDN 48 : full 16 (3:1) | **sliding(w=1024) 50 : full 10 (5:1)** — GDN류 없음 | layer_types |
| KV heads | full: NKV4 hd256 | **sliding: NKV16 hd256 / full(global): NKV4 hd512** | config+modeling |
| **k_eq_v** | 없음 | **full 층 v_proj 없음(50/60만 존재)** — V=v_norm(k_proj(x)), K=rope(k_norm(k_proj(x))) | modeling L1219·index |
| QK norm | (엔진 확인 필요) | **q_norm/k_norm(head_dim RMS) + v_norm(scale 없음)**, softmax **scaling=1.0** | modeling L1194·1209~ |
| RoPE | theta 1e7, partial 0.25(분모=rotary_dim) | **이원화**: full=proportional theta 1e6 partial 0.25 **분모=head_dim(512), 64각만 회전** / sliding=default theta 1e4 전체 회전 | rope_utils L187~ |
| Norm | RMSNorm **(1+w)** (MTP 교훈) | **plain w**(fp32 캐스팅) — **(1+w) 아님, 역방향 함정** | modeling L193~ |
| Norm 배치 | pre 2개 | **sandwich 4개**(input/post_attn/pre_ffn/post_ffn — residual add 전에 post norm) | L1411~1445 |
| MLP | SwiGLU | **GeGLU**(gate/up/down, gelu_pytorch_tanh) | L1083 |
| embed | untied (dq4 필수였음) | **tied**(lm_head 텐서 없음) + **embed×√H 스케일** | L1602·index |
| logits | softcap 없음 | **final_logit_softcapping 30**(/30→tanh→×30) | L1889~ |
| 신규 텐서 | — | **layer_scalar**(레이어당 1, 출력 전체 곱 — 체크포인트 값 로드 필수) | L1381·1454 |
| MTP head | 있음(spec 자산) | **없음** — spec은 별도 draft 필요(Phase 0 밖) | index |
| attn softcap / KV공유 / MoE / per-layer input | — | 전부 **비활성**(31B 설정) | config |

## 2. 커널 자산 매핑 표 (재사용/조정/신규)

| 연산 | 판정 | 비고(수정 분류 예상) |
|---|---|---|
| W4A8 matmul(gemm_mmq/sb/gemm_batch/split-K) | **재사용** | 치수는 config — 수정 없음 기대 |
| rmsnorm | **재사용(설정)** | plain w = 우리 커널 기본형. (1+w) 스위치는 이미 config 축. fp32 캐스팅 순서 골든에서 검증 |
| rope | **조정(config)** | 분모=head_dim 파라미터화(현행 rotary_dim 분모와 분리) — ① graph/config 흡수 예상 |
| flash/attn | **조정(config)** | sliding window 마스크(w=1024) + 층별 hd 256/512·NKV 16/4 — 마스크는 인자, 커널 내부 분기 금지 감시 |
| KV 캐시 | **조정(graph)** | sliding 층은 ring buffer(w=1024)로 축소 가능(1차는 full 저장도 OK) — k_eq_v층 V는 norm만 다름(K/V 둘 다 저장) |
| activation | **신규(소형)** | gelu_tanh(GeGLU) — swiglu 커널에 act 함수 축 추가 or gelu 버전 1개 ① |
| softcap | **신규(소형)** | 로짓 /c→tanh→×c elementwise — lm_head 뒤 1회 ① |
| layer_scalar | **신규(사소)** | 스칼라 곱 — 기존 elementwise/graph 배선 ① |
| embed | **조정(graph)** | ×√H 스케일 + tied lm_head(27b lm_head q4 경로 재사용) ① |
| QK/V norm | **재사용(배선)** | head_dim 단위 rmsnorm — 기존 rmsnorm 재사용(scale 없음 옵션 추가 가능성 ②) |
| GDN | **불필요** | state 경로 없음 — decode step이 27b보다 단순 |

## 3. roofline (착수 전 기대 좌표 — 상한≠예측)

- params 29.6B(형상 추정, 명목 31B의 -4.5% — vision 제외 정합). W4A8 bytes/token ≈ 15.7GB.
- **single decode 상한 17.4 t/s**(27b 19.0보다 낮음 — params↑). batch B12 상한 208, spec은 draft 부재로 미정.
- 27b 달성률 전례(single 62.5%) 대입 시 기대 좌표 ~11 t/s 부근 — 예측 아님, 참고 좌표.

## 3.5 메모리 예산 사전 계산 (unified 가드 — 시스템 다운 전례)

- HF bf16 스냅샷: 디스크 62.5GB(확보). 변환은 safetensors **레이어 단위 스트리밍** — RAM 피크 = 최대 단일
  텐서(embed 262144×5376×2B = 2.8GB) + 버퍼 ≈ 6GB 이내(가드 여유).
- W4A8 산출물: ~16GB 디스크(27b 14.3GB 대비 +10%).
- **골든 fp32 full 로드 = 118GB → 가드(0.7×avail ≈ 83GB) 초과, 불가 확정.** 27b 골든 생성 방식(fp32를
  어떻게 다뤘는지) 확인 후 결정: 후보 (a) HF bf16 forward 로짓을 fp32 캐스팅 골든 (b) CPU 레이어 스트리밍
  fp32 (c) fp16s 스킴 전례 재사용. — STEP C 착수 전 해소 항목.

## 4. STEP B — 모델 등록 구조 + 변환 파이프

- **분리 구조 확정**(기존이 이미 config-주도라 대칭 확장): `models/gemma4.{h,cu}`(graph ~110줄, 공용 커널만
  호출) + `runtime/weights_gemma4.{h,cu}`(로더 — QMat 재사용) + `runtime/manifest_gemma4.h`(파서) +
  `runtime/config.h`(Gemma4 필드 추가). qwen35 경로 무변경(회귀 격리).
- **변환**: `extract_weights_qwen35.py` 일반화(prefix 자동 감지가 `model.language_model.` 이미 처리 →
  Gemma4 언랩 공짜). q4-only 추출 24GB(410 q4/420 fp32-norm, mean rel-err 0.10=27b와 동급).
  메모리: bf16 스트리밍 로드 피크 <10GB(가드 여유).
- **골든**: `make_reference_gemma4.py` — dual RoPE(full/sliding) cos/sin 2세트 캡처가 유일한 신규 축.

## 5. STEP C — 첫 golden 게이트 (PASS)

- **결과: argmax 전위치(5/5, vocab 262144) 일치, last-tok eng=gold=7001.** graph 배선 정확.
- 수치: logits maxabs 0.93(softcap ±30), hidden L0 0.097→L60 5.9.
- **게이트 판정 = argmax 일치**(정직): fp32 full 레퍼런스 118GB=가드 초과라 불가 → bf16 로드 골든.
  bf16 레퍼런스에선 fp32-exact err<1e-2 원천 불가. **격리 검증(추측 아님)**: ①L0 격차 0.097 규명 =
  golden이 embed_scale(√H)도 **bf16 캐스팅**(modeling L1469) → 엔진 fp32가 오히려 정밀 ②층별 오차 smooth
  누적, **full 층 경계(5/11/…/59) 국소 점프 없음**(L23 3.4→L24 2.47 복귀) = bf16 서명이지 버그 아님.
  = 27b/0.8b(fp32 골든, err<1e-2)보다 약한 게이트지만 31B에서 achievable한 최선.

## 6. 유연성 수정 기록 (3분류) — ★핵심 산출물

| # | 수정 | 분류 | 내용 |
|---|---|---|---|
| 1 | plan.py 해부기 | ① 도구 | text_config 언랩 + sliding/full 패턴 + HF params 추정기 |
| 2 | extract_weights 일반화 | ① config | 타입별 RoPE·sliding·k_eq_v·softcap·embed_scale·layer_scalar 필드(config 주도) |
| 3 | config.h + manifest_gemma4.h | ① config | Gemma4 필드 9종 + 파서(qwen 규약 재사용) |
| 4 | weights_gemma4 + models/gemma4 | ① graph | 로더(sandwich 4-norm/k_eq_v v_proj 부재/layer_scalar) + graph 110줄(공용 커널만) |
| 5 | swiglu.launch_geglu | 신규(독립) | gelu_tanh gated — SwiGLU 병렬 activation variant, 모델 독립 |
| 6 | elementwise.launch_scale/softcap | 신규(독립) | embed×√H·layer_scalar 스칼라곱 / logit 소프트캡 — 범용 elementwise |
| 7 | rmsnorm.launch_rmsnorm_noscale | 신규(독립) | scale 없는 RMSNorm(v_norm) — 모델 독립 variant |
| 8 | attention.launch_attn | **② 커널 인터페이스** | scale+window 인자 추가(**기본값으로 기존 동작 보존**). 사유: sliding window 마스크 + attn_scaling=1.0. 마스크/스케일은 **인자**이지 커널 내부 모델 분기 아님 |

**★유연성 판정: ③(커널 내부 모델 분기 = 금지 신호) = 0건.** 구조적으로 꽤 다른 2번째 모델
(sliding/full 혼합 vs GDN/full, GeGLU vs SwiGLU, sandwich 4-norm vs 2-norm, tied+softcap+layer_scalar+
embed_scale, dual RoPE, k_eq_v)을 **config/graph/모델독립 신규커널로 전부 흡수** — 커널 오염 0.
②는 1건(attn scale/window, 가산적·기본값 보존). "커널 라이브러리+얇은 graph" 헌법이 2번째 모델에서 유지됨.

**회귀**: smollm3 PASS, qwen35-0.8b PASS(q4 worst 4.98e-5) — 커널 변경 순수 가산적 확증.

## 7. 남은 것(이후 세션 — Phase 0 밖)
- 성능/튜닝/spec(MTP head 부재 → 별도 draft 필요), sliding window KV 캐시 축소(현 프리필은 마스크만),
  fp32 레퍼런스 미확보로 argmax 게이트 — 더 긴 프롬프트/디코드 다토큰으로 argmax 신뢰도 보강.
