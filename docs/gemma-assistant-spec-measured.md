# Gemma 공식 assistant(0.4B) draft-spec 재판정 — GO (measured, 구현 착수 근거)

> E2B 판정([gemma-draft-spec-verdict.md](./gemma-draft-spec-verdict.md), 순 곱 ~1.0× → 접음)은 **그 draft의
> 판정이었지 draft-spec 경로 자체의 판정이 아니었다.** 공식 drafter **google/gemma-4-12B-it-assistant**(0.423B,
> Apache-2.0)로 재측정: HF 좌표 순 곱 **2.1~4.0×**(워크로드별) — E2B와 질적으로 다름 → **구현 정당화 GO**.
> 이번도 구현 0(HF 공식 assisted-generation 경로만·`tools/probe_assistant_spec.py`).

## STEP 0 — 구조 (코드 실사: "작은 모델"이 아니라 target-결합 drafter)

- **EAGLE류 hook**: draft 입력 = concat(target embed[3840], target last hidden[3840]) → pre_projection(7680→1024).
  4층 trunk(sliding×3+full, H1024·FFN8192 GeGLU·tied embed 262144×1024) → 출력 두 갈래: lm_head(logits)와
  post_projection(1024→3840, **다음 draft 스텝의 hidden으로 재귀**).
- **자체 KV 없음**: `use_cache=False`·고정 position — 4층 전부 **cross-attention으로 target의 shared KV를 읽음**.
  12B는 `num_kv_shared_layers=0`이라 shared KV = **target 마지막 full층(47)·마지막 sliding층(46)의 KV** —
  우리 엔진에선 KV 캐시 포인터 2개 전달(복사 0). ★assistant attention 형상이 target KV와 정확히 일치하게 설계
  (full: hd512·KV헤드 1 / sliding: hd256·KV헤드 8) = KV 변환 불요.
- **centroid head는 이 체크포인트에서 OFF**(`use_ordered_embeddings=false` → plain lm_head) — 탑재 스코프 축소.
- vocab 262144 == 12B 정합 ✓. **draft 비용 실측(HF·GB10)**: assistant 1 forward ≈ **5.4ms** vs target 스텝
  143.6ms → **c ≈ 0.038 = E2B(c 0.270)의 1/7**. draft 중 target 추가 read ≈ 0(embed row 1개+KV는 캐시 재사용).

## STEP A — 순 곱 measured (HF 공식 경로·구현 0·GB10 warm·128토큰·greedy)

`tools/probe_assistant_spec.py` — E2B verdict와 **동일 프롬프트·동일 템플릿**(대조 공정). HF plain 12B bf16 = 6.96 t/s.

| 워크로드 | K=2 | K=4 | K=8 | heuristic(기본) | E2B 최고(참고) |
|---|---|---|---|---|---|
| chat  | 2.76× (acc 1.56) | 3.64× (2.66) | **3.99× (3.57)** | 3.04× (3.74) | 0.95×(손해) |
| code  | 2.83× (1.61) | 3.65× (2.66) | **3.72× (3.27)** | 3.72× (3.27) | 1.22× |
| prose | 2.05× (0.89) | **2.15× (1.13)** | 2.10× (1.37) | 2.10× (1.37) | 1.04× |

(acc = 사이클당 평균 수락 토큰 = tokens/cycle − 1. 순 곱 = assisted t/s ÷ plain t/s, warm best of 2.)

- **E2B 접음의 원인 = draft 선택이었음이 measured로 확정**: 같은 프롬프트에서 0.95~1.22× → **2.1~4.0×**.
  B200 참고치 3.1×는 GB10 HF 좌표에서 chat 3.99×로 재현·초과.
- **스펙 불변식(중요)**: chat은 전 K에서 출력 == plain greedy **완전 일치**. code/prose는 K-무관 **동일 위치·동일
  토큰쌍**(code@80, prose@8)에서 분기 — 실측 진단: code@80은 top-2 로짓 **완전 동률**(둘 다 20.3750, gap=0),
  prose@8은 gap 6.25e-2 = 로짓 ~10.5에서 **bf16 1 ULP**. 즉 target 자체의 근소 동률이 batched verify(M>1)의
  부동소수 합산 순서로 뒤집히는 **tie-flip**(재실행 M=1 argmax도 assisted 쪽 선택 = plain조차 동률에선 유일하지
  않음). **acceptance 버그 아님** → accept 수치 신뢰 가능. 우리 엔진 구현은 27b spec처럼 spec==greedy **bit-id
  게이트**(같은 커널 경로 규율)로 이 부류를 원천 차단하는 것이 기준.

## STEP B — 엔진 전이 산술 + 판정

HF 곱은 HF plain(6.96 t/s) 좌표. 우리 엔진 plain은 **23.3 t/s(3.3× 빠름)** → draft/verify 오버헤드 비중이 커져
곱이 줄어든다(정직 전이). 성분: a(K)=tokens/cycle(모델 속성=전이됨), T_dec=42.9ms(23.3 t/s),
T_verify(K+1≤9)=T_dec×1.0~1.2(27b spec 실측 패턴), T_ast=2.5~5.5ms(HF 실측 5.4ms을 상계로, CUDA 포팅·q4 시 하계).

순 곱(엔진) = a·T_dec ÷ (K·T_ast + T_verify):

| 워크로드(최적 K) | 보수(V=52ms·A=5.5ms) | 낙관(V=43ms·A=2.5ms) |
|---|---|---|
| chat (K8, a 4.57) | **2.0×** | 3.1× |
| code (K8, a 4.27) | **1.9×** | 2.9× |
| prose (K4, a 2.13) | 1.2× | 1.7× |

**판정: GO.** chat/code는 보수 하한도 1.9~2.0×로 문턱(1.5×) 확실 상회, prose는 경계(1.2~1.7× — 워크로드 민감
정직, 27b spec의 6.99/4.55/3.5 민감도와 동형). 예상 좌표: 엔진 순 곱 **~2~3×** → 12B plain 23.3 → **~45~70 t/s**
= llama.cpp spec 66.9 t/s와 **같은 리그**(구현 후 실측이 확정 — 여기 숫자는 산술 전이, overclaim 아님).

## 구현 스코프 (정직 산정 — 두 부분 + 완화 요인)

| 부분 | 내용 | 규모 |
|---|---|---|
| ① gemma4_verify_batch | K+1 청크 forward + 전 위치 logits/argmax + KV 롤백(길이 복원) | ~130줄 — gemma4 chunked prefill(슬롯-b 뷰)·mmB·batch argmax 재사용. ★Gemma는 GDN 없음 = 27b보다 쉬움(state snapshot/rollback 불요, KV 길이 복원만) |
| ② assistant graph | pre/post projection(gemv)·4층 cross-attn trunk(★target KV 포인터 직결·형상 일치·자체 KV 없음)·tied lm_head(q4 후보)·dual-rope Q | ~250-350줄 신규 graph + loader/extract 확장 + **HF 골든 게이트**(0.4B라 fp32 골든 가능) |
| target-side | 최종 hidden(3840) 노출 + 층47 full/층46 sliding KV 포인터 전달 | 사소(~20-40줄, 신규 커널 0) |
| 사이클/서빙 | draft K회 → verify → accept/rollback 호스트 사이클·serve_api spec 모드 | 27b spec 사이클 패턴 재사용 |

new-paradigm(26B MoE급) — **단계 게이트**: ① verify POC(spec==greedy **bit-id**) → ② assistant HF-골든(fp32/
bf16 argmax) → ③ 사이클 e2e(accept·곱 실측 vs 위 전이 산술) → ④ 서빙/웹UI(spec 켠 12B 채팅). 예상 2~4 세션.

> **① 통과(a81f686)**: `gemma4_verify_batch` + `test_gemma4_verify` — perfect-draft K∈{2,3,4,8} 전 위치 로짓
> **bit-identical(diff 0)** vs plain greedy·오염 24사이클 거부/교정/롤백 정확·결정성 30회·batch==solo 전부 PASS.
> gemma4 경로는 전 구간 행-독립이라 verify 로짓이 plain과 같은 비트 = **tie-flip 구조적 불가**(HF와 대조).
>
> **② 통과(7182f57)**: `gemma4_assistant.{h,cu}` + HF 골든(`make_reference_assistant.py`) — 스텝별 로짓
> MEAN-rel **3~6e-07**(fp32 동급)·argmax 8/8==HF·**재귀 8스텝 draft 전부 일치**(오차 축적 없음)·결정성 30회.
> 신규 커널 0(cross-attn=attn_decode base=len-1 매핑이 HF flip-마스크 의미와 정확 일치). ★③ de-risk 실측:
> target concat용 hidden = `hidden_states[-1]` = **final-norm 전**(층 출력) — target-side는 pre-norm residual 노출.
>
> **③ 통과(사이클 e2e — 곱 첫 실측)**: draft②→verify①→accept 연결(`test_gemma4_spec_e2e`·target hook =
> `gemma4_decode` d_prenorm). ★**spec==greedy 사이클 유지**: 3워크로드×K{2,4,8} 전부 출력==plain greedy 토큰-동일.
> ★곱 실측(12B W4A8·N96·warm): 비분할상환 verify는 0.24~0.81×(verify가 0.7×(K+1)×T_dec 지배 — measured) →
> `amort_mm=1`(SB==gemv bit-id 기게이트 성질·[A] 재검증) 후 **code K4 1.54×(34.1 t/s)·chat K4 1.34×(29.7)·
> prose ~1.0×**. 남은 lever(측정 명명): assistant fp32 7.9ms/스텝(bf16/q4화 → 산술 ~1.7-1.9×)·**accept 갭**
> (chat K4 1.46 vs HF 2.66 — W4A8 target 조건화 가설, 도달 시 ~2.5×=전이 산술 정합)·serving 통합(④).
>
> **④ 통과(서빙·실사용 — draft-spec 완성)**: `serve_daemon_gemma4_spec_online`(슬롯별 spec 사이클·프로토콜 동일·
> 메모리 가드) + `serve_api --model gemma4-spec --assistant ...`(웹UI spec 배지/메트릭 그대로). 게이트
> `test_gemma4_spec_serving.py` 전부 PASS: **spec daemon 출력 == plain daemon 출력 토큰-동일**(3워크로드 —
> decode_batch vs verify-청크 조합까지 bit-id)·동시 3요청 격리·stop/length 종결·생존. ★캡처 재현(HTTP
> client-observed, 128tok): **chat 34.6 t/s(plain 22.0의 1.57×)·code 30.6(1.40×)**, 응답 텍스트 plain==spec.
> llama 12B spec 66.9 대비 **0.46~0.52×** = 같은 개념끼리의 정직 좌표. 후속(선택): drafter bf16/q4·accept 갭
> 진단(uq 대조)·verify concat 분할상환·spec 배칭.

## Lever 세션 (곱 키우기 — 명명된 2 lever measured)

**Lever1 = drafter W4A8화(적용)**: 큰 matmul 전부(embed=tied lm_head 포함 — 룩업 아님이라 안전) target blueprint
스킴 q4 + 활성화 int8, 커널 전부 재사용(quantize_x_q8·gemv_w4a8 — 신규 0). 골든 재게이트(HF q4-라운드트립
`golden_q4`): 로짓 rel 1.0e-2(<2e-2)·h_out 2.7e-2(<5e-2 — 양자화 2단 경로)·argmax 8/8·재귀 8/8·결정성 PASS.
draft 비용 **7.9 → 2.0ms**(≈4×↓). ★사이클 곱(N96·warm·spec==greedy 전 구성 유지):

| wl(최적 K) | fp32 drafter | **W4A8 drafter** | t/s |
|---|---|---|---|
| code K4 | 1.54× | **2.16×** | **48.2** |
| chat K4 | 1.34× | **1.80×** | **40.1** |
| prose K2 | 1.22× | **1.59×** | 35.5 |

**Lever2 = accept 갭 진단(가설 2개 기각·measured)**: ① uq=1(활성화 fp32) 대조 — accept 불변(chat K4 1.46 동일)
→ **활성화-양자화 조건화 기각**. ② fp32 rope 테이블(rotary 모듈 직접 생성·bf16 캡처와 Δ1.95e-3) 대조 — accept
불변 → **rope 정밀 기각**. 남은 후보(분리엔 bf16 target 필요=대형 후속): target **q4 가중치** 조건화 or 궤적
차이(우리 q4-greedy 텍스트 ≠ HF bf16 텍스트 = 본질적 draft 난이도 차). HF-tier accept(2.66) 도달 시 ~2.5×
산술은 유효하게 남음.

**서빙 재측정(W4A8 drafter 기본화)**: 서빙 게이트 전부 재PASS(spec==plain daemon 토큰-동일·동시 격리·생존).
★캡처 재현(HTTP client-observed·128tok): **chat 50.5 t/s(2.30×)·code 46.7(2.12×)** — llama 12B spec 66.9 대비
**0.70~0.75×**(시작 0.46~0.52×에서 접근). 남은 lever: accept 갭(위 잔여 후보)·verify concat 분할상환·spec 배칭.

## 하지 않은 것 / 다음
- **구현 0(이번)** — 곱의 확정은 구현 후 엔진 실측. num_assistant_tokens>8·tree-draft·batch+spec 결합은 후속.
- 메모리: 12B(24GB) + assistant(0.9GB) 동시 상주 — W4A8 시 ~7.5GB로 여유(128GB 가드 내).
- 다음 세션 = ① verify POC부터(정확성 먼저·bit-id 게이트).
