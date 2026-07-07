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

## accept 갭 규명 (진단 완결 — bf16 target 실험 · 전 축 measured)

메모리 가드 하 fp32(bf16값) target 12B 엔진 실험(예산 50.3GB 사전 산정=실측 51.2GB·가용 117GB) +
HF-스택 궤적-난이도 대조(`tools/probe_traj_difficulty.py` — 위치별 1스텝 일치율 a1, ★a1(HF,HF텍스트)=0.844
→accept4 환산 2.66 = HF 실측 2.66 정확 재현으로 방법론 자체 검증):

| 가설 | 실험 | 판정 |
|---|---|---|
| W4A8 활성화 조건화 | uq=1(활성화 fp32) 대조 | **기각**(accept 불변) |
| rope bf16 정밀 | fp32 테이블 × q4·fp32 target 양쪽 | **기각**(fp32-target 조합서도 소수점 동일) |
| 궤적/텍스트 난이도 | HF-스택 a1 on 우리 q4/fp32 궤적 | **기각**(q4 텍스트 a1 0.885=오히려 쉬움·fp32 chat 0.844=정상) |
| **target q4 가중치 조건화** | fp32-target 엔진 accept | **code 확정**: K4 1.74→**2.56**(≈HF 2.66)·K8 1.91→3.92 |
| assistant graph 결함 | ②게이트(HF 입력→엔진==HF) | 무죄 |
| chat/prose 잔여 | 위 전부 소거 후 | **미확정 미시 축** — chat fp32-target 1.56 vs 같은 텍스트 HF-a1 2.66 |

- **prose 해소**: HF 스택도 prose는 a1 0.60(→1.32) = 본질적 어려운 워크로드(HF 실측 1.13과 정합) — 엔진 0.83과의
  잔여는 chat과 동류.
- **실용 결론(정직)**: 현 q4 서빙 좌표(50.5/46.7 t/s)는 **q4-스택의 상한 근처** — code는 조건화 확정이라 q4 유지
  시 accept 개선 불가(fp32 target은 plain 4.9 t/s=서빙 비실용·진단 전용). chat 잔여 축의 신규 후보(미시험):
  **bf16-반올림 분포 정합**(drafter가 bf16-연산 스택에 학습 — 엔진 h/KV를 bf16 반올림해 투입하는 실험이 다음 분리).
  풀리면 chat ~2.66-tier(산술 ~60 t/s) 여지.

## llama-on-GB10 공정 벤치 (캡처값 검증 — 진짜 기준선)

기존 비교의 llama 66.9는 캡처값(출처 장비 불명)이었음 → MTP fork `llama-server`(gemma4-assistant 네이티브)를
**같은 GB10·같은 조건**(같은 프롬프트·같은 공식 assistant를 f16 GGUF로 변환·K4=`--spec-draft-n-max 4`·greedy·
warm·single-stream·client-observed 동일 회계·전층 GPU)으로 직접 벤치(`tools/bench_llama_gb10.py`):

| K4·128tok | llama-GB10 | dgx | dgx/llama | 곱(llama/dgx) |
|---|---|---|---|---|
| plain | 25.6 | 22.0~22.3 | 0.86× | — |
| spec chat | 56.2 | 50.5 | **0.90×** | 2.20×/**2.30×** |
| spec code | **69.6** | 46.7 | 0.67× | 2.72×/2.12× |
| spec prose | 45.6 | 31.9 | 0.70× | 1.78×/1.43× |

- **66.9는 GB10 실좌표로 확인**(code 69.6서 재현) — "0.70~0.75×" 뭉뚱그림의 진짜 분해 = chat 0.90×·code 0.67×.
- **spec 곱은 chat에서 우리 우세**(2.30 vs 2.20) — chat 갭의 대부분은 plain 갭(0.86×)의 상속.
- ★**q4 조건화의 정밀화**: llama도 4bit(Q4_K_M) target인데 code accept 우위 → 원인은 "4bit이라서"가 아니라
  **양자화 스킴 품질**(K-quant super-block·6bit scale vs 우리 blueprint q4 rel-err 0.1016) = 수정 가능 축 승격.
- 조건 차 명시(정직): target Q4_K_M 6.9GB vs w4a8(동급 4bit·다른 스킴)·drafter f16 vs W4A8(llama에 유리)·
  prose는 우리만 EOS 조기종료(106tok — t/s 회계는 n/dt로 유효).

## K-quant 품질 주입 실험 (구현 0 — 스킴 가설의 정량 판정)

Q4_K_M GGUF를 fp32 dequant해 엔진 uq=0에 주입(`tools/probe_kquant_inject.py` — 정합 골든: 교체 330·
fallback 0·worst cosine 0.996·norm-ratio 가드). **가중치 보존은 실측 32% 우수**(rel-err 0.0695 vs blueprint
0.1016·혼합 정밀 Q6_K[v/embed] 포함). accept(전부 fp32-활성·가중치 값만 대조):

| accept K4 | blueprint q4 | **K-quant 값** | bf16 값 | HF |
|---|---|---|---|---|
| code | 1.74 | **1.91 (+0.17 소폭)** | 2.56 | 2.66 |
| chat | 1.46 | 1.67 | 1.56 | 2.66 |
| prose | 0.71 | **1.13 (HF 동률 — 완전 회복)** | 0.83 | 1.13 |

**판정(스킴 가설 부분 기각)**: 가중치 32% 정확해져도 **code accept는 소폭만 회복 — code의 벽은 스킴 품질이
아니라 4bit 양자화 자체**(bf16값에서만 2.56). prose는 완전 회복(워크로드 선택적). 풀기 비용은 함정 아니었음
(Q4_K 4.5 vs blueprint 5.0 bits/weight로 오히려 가볍고 니블 정렬·llama plain 25.6>22.0 실측 상한) —
그러나 accept 열쇠가 아니므로 **K-quant 커널 구현 보류**(q2 패턴: 구현 전 성패 숫자가 차단). 잔여 이득
후보로만 기록: bytes 10%↓=plain +8~10% 산술·prose accept. ★llama code 우위의 잔여 가설(신규):
**적응적 draft 길이**(llama --spec-draft-n-min~max = 확신 낮으면 짧게 draft·낭비 사이클 절감 vs 우리 고정 K4)
— 호스트 로직만으로 시험 가능(커널 0)·다음 분리 실험.

## 적응적 draft 길이 실험 (커널 0 — 세 정책 삼중 판정 · 접힘)

llama code 우위의 잔여 가설이었던 "적응 draft 길이"를 **호스트 정책 3종으로 전부 measured**(e2e·uq=2·
ast_uq=2·N=96·warm 2번째·전 행 [A] spec==greedy PASS·ref 재현 변동 ±2%). 구현: e2e에 env 게이트
(`ADAPT/TAUS/KMIN/KMAX`=margin 정책·`ADAPT=2`=HF heuristic·`CONF_LOG`=진단 CSV, `tools/probe_adaptive_k.py`
분석), 기본 경로 무변경.

**STEP 0 — 확신 신호는 진짜다**(draft별 top1-top2 로짓 margin, D2H 1MB+호스트 스캔 ~0.6ms/draft):
margin→accept AUC **chat 0.87 / code 0.90 / prose 0.72**, 고정 K4 낭비(뽑고 버림) chat 65%/code 55%/
prose 77%, m=0 사이클(전부 버림) 34~55%. 신호도 낭비도 커서 정책 여지가 충분해 보였다 — 그러나:

| 곱(spec/plain) | 고정 K4(ref) | ①margin 조기중단 a1-7 최적 | ②연장 전용 a4-7 최적 | ③HF 동적 K hfdyn4..7 |
|---|---|---|---|---|
| chat | 1.82~1.83× | 1.89× (τ2, **+3.3%**) | 1.76× (−3%) | 1.74× (−3%) |
| code | 2.15~2.19× | 1.99× (τ3, **−9%**) | 2.21× (τ6, +1.4%=잡음권) | 1.87× (**−13%**) |
| prose | 1.51~1.53× | 1.43× (−7%) | 1.46× (−4%) | 1.43× (−5%) |

**판정(접힘) — 실패 원인은 정책이 아니라 비용 구조(산술·measured)**:
- **절단 손익분기**: draft 1개 = 2.2ms(+판정 0.6ms), 수락 토큰 1개의 시간 가치 ≈ 사이클 49ms/acc 2.7 ≈
  **18ms** → 절단이 이득이려면 P(accept)<~13% 필요. 실측은 최저-margin 구간(<0.5)조차 **accept 20~38%** —
  어떤 문턱에서도 자르면 손해. ①이 acc를 깎는 이유(code 1.74→1.53)·③도 동일(신호가 verify 실측 m이어도
  같은 산술 — code −13%).
- **연장의 상쇄**: ②는 acc를 진짜 올림(code 1.74→**2.10**, +21%) — 그러나 margin은 drafter 자기확신이지
  "접두사 전체 생존(m=4)" 예측이 아니라(AUC 0.90은 per-draft), m<4 사이클의 연장 draft가 그림자 낭비 →
  draft+판정 추가(+91ms)가 verify 절약(−75ms)과 상쇄, 순 +1.4%=잡음권.
- **llama 적응 이득은 동형 함정**: 그쪽은 draft가 상대적으로 비싼 비용 구조라 절단이 pay — 우리 draft는
  2.2ms(사이클의 14%)라 낭비 자체가 싸다. llama code 69.6 우위의 설명은 적응이 아니라 4bit accept 벽(위
  K-quant 판정)+plain 상속으로 이미 완결.
- chat ①의 +3.3%는 유일 양수지만 ref 재현 변동(±2%)에 근접 — 워크로드-조건부 정책 복잡도 대비 미채택.
- **부산물(다음 lever의 산술 근거)**: verify ms/cyc 실측 **Sq3 46.5 / Sq5 49 / Sq9 63** — 행 수 3배에
  +35%뿐(SB 분할상환 확인·Sq9 점프는 버킷 패딩 의심). → **verify concat 분할상환**(여러 슬롯의 verify를
  한 SB 호출로): 2슬롯 Sq5×2=Sq10 ≈ ~63ms vs 각각 98ms = **verify −36% 산술 여지**(spec 배칭의 실질
  형태·스킴 경계[verify M 변경=bit flip 가능] 주의) = 다음 후보 승격.

## verify concat 분할상환 (다중 슬롯 실 WIN — 적응 K 부산물 산술의 실현)

적응 K 접힘의 부산물(verify Sq 둔감 46.5/49/63ms)을 구현으로 전환: **활성 슬롯들의 verify를 한 호출로
concat** — 큰 matmul만 M=Σ(K+1)행으로 묶고(`gemma4_verify_concat`, gemma4_batch.cu), rope는 행별 위치
(rope_partial_pos)·attention/KV append는 슬롯별 기존 커널. **신규 커널 0**.

**스킴 경계 flip 원천 회피(구성적)**: matmul은 어떤 M이든 `gemv_w4a8_sb` ≤16행 그룹으로만 분해
(gemma4_decode amort_mm과 동일 — SB per-row == plain gemv per-row bit-id 기게이트). mmB의 B16
gemm_batch 전환(진짜 커널 패밀리 경계)을 아예 타지 않음. uq=1(tiled)은 M-종속 타일=스킴 위험이라 loud
abort(per-slot 경로 사용).

게이트(test_gemma4_verify_concat) 전부 PASS: **concat==per-slot canonical==amort-slot 3중 로짓 Δ=0**
(M 경계 전수: 5·8·9·10·15·16·17·20·36 — SMAX 8→9·그룹 16→17 경계·ragged 혼합 포함), prenorm Δ=0,
KV 불침범 Δ=0, 결정성 Δ=0. 서빙 게이트(test_gemma4_spec_serving) 재PASS: spec==plain 토큰 동일·동시
3요청==솔로(=concat 경로 실사용 격리)·생존 → **spec==greedy 유지**.

속도(단위, warm·K4 청크): per-slot vs concat = B2 90.9→58.4ms(**−36%** — 예측 −36% 적중)·B3 −38%·B4 −35%.
서빙 A/B(daemon-직결·같은 wl N개 동시·client-observed aggregate·warm·96tok):

| aggregate t/s | N=1 | N=2 | N=4 |
|---|---|---|---|
| code: per-slot→concat | 45.2→44.3(잡음권) | 44.9→**62.5 (+39%)** | 44.6→**59.9 (+34%)** |
| chat: per-slot→concat | 54.0→53.7 | 53.9→**74.8 (+39%)** | 54.1→**72.1 (+33%)** |
| prose: per-slot→concat | 31.3→31.2 | 31.4→**44.2 (+41%)** | 31.2→**42.5 (+36%)** |

- **per-slot(구)는 N에 완전 평평** = verify 직렬화가 동시성을 전부 먹고 있었음(measured) — concat이
  spec 서빙에서 다중 슬롯을 처음으로 유효하게 만듦. TTFT 0.12~0.49s 건강(admission 즉시).
- **단일 스트림 불변(정직)**: N=1 차이 −0.3~2% = 잡음권(bit-id는 게이트가 보증 — A=1도 concat 함수
  단일 경로이나 기존 amort와 동일 분해).
- N=4(M=20)가 N=2(M=10)보다 소폭 낮음 = SB 16행 그룹 2회 적재 산술 그대로(K4 기준 N=3(M=15)이
  단일 적재 스윗스팟). 다중 슬롯의 다음 병목 = **draft 페이즈**(N4 라운드의 ~20% — assistant concat이
  다음 lever 후보).
- 좌표 맥락(정직): 12B plain 서빙은 B8 ~96 t/s(배칭 자체가 강함) — spec의 다중 슬롯 강점은 raw
  aggregate가 아니라 **낮은 동시성(N2~4)에서 per-user t/s(요청별 15~42)와 aggregate의 균형**.
  llama-server 다중 슬롯+spec 동시 벤치는 인프라 확인 필요(후속) — 단일 스트림 좌표(chat 0.90×/code
  0.67×)는 불변.

## draft concat (assistant 배칭 — verify concat의 대칭·사이클 완전 배칭)

verify concat 후 남은 반쪽: draft 페이즈가 슬롯별 순차(N4 라운드의 ~20%)여서 N=4 < N=2였음 →
**`gemma4_assistant_step_batch`**: 활성 슬롯 B개의 draft 스텝을 한 번에 — 큰 matmul(pre_proj·trunk·
post_proj·**tied lm_head=지배항**)만 B행 SB ≤16행 1적재(B==1은 기존 gemv 그대로=단일 경로 불변),
cross-attn은 슬롯별 기존 커널(각자 target KV 포인터·base=len−1 — KV 매핑 명시적·drafter는 자체 KV가
없어 append 위험 0), rope는 행별 위치(rope_partial_pos). 재귀(h)·argmax는 호출자(슬롯별 독립). **신규 커널 0**.

게이트(test_gemma4_draft_concat) 전부 PASS: **배치==순차 bit-id Δ=0** — B∈{2,3,4} × K=6 재귀 전 스텝의
로짓·h_out·draft 토큰(함정 3종: 재귀 연쇄·ragged 위치·KV 매핑 전부 통과). draft 페이즈 단위(K4):
순차 4슬롯 30.2ms → **배치4 13.1ms(−57%)**·배치2 9.3ms. daemon은 A==1 기존 경로/A≥2 배치(3페이즈:
draft concat→verify concat→accept). 서빙 게이트 재PASS(spec==plain·동시==솔로·생존).

서빙 A/B(verify-only concat 대비, daemon-직결·warm·96tok):

| aggregate t/s | N=1 | N=2 | N=3 | N=4 |
|---|---|---|---|---|
| code | 44.8→44.7 | 63.1→67.2 (+6.5%) | 65.2→67.4 | 60.1→**66.0 (+9.8%)** |
| chat | 53.5→54.0 | 75.4→75.6 | 78.0→**80.4** | 72.1→**79.3 (+10.0%)** |
| prose | 31.2→31.3 | 44.3→44.6 | 45.8→47.6 | 42.4→**47.0 (+10.8%)** |

- **N=4가 N=2를 따라잡음**(chat/prose는 추월·code 동률권) — draft 병목 제거가 정확히 N4에서 최대.
  N-스케일 곡선 평탄화(N2~4 어디든 근사 최적). 피크 chat N3 80.4 t/s.
- **누적(per-slot 순차 기점): N4 code 44.6→66.0=+48%·chat 54.1→79.3=+47%·prose 31.2→47.0=+51%**
  — 다중 슬롯 spec 서빙 1.5×(사이클 완전 배칭: draft+verify 둘 다 weight 1적재).
- 단일 스트림 불변(±1% 잡음권)·TTFT 불변(0.12~0.55s). 남은 잔여: verify N4의 SB 16+4 2회 적재
  (K4 기준 N3=M15 단일 적재가 산술 최적 — Bmax=3 운용 선택지)·assistant cross-attn 슬롯 루프(작음).

## 하지 않은 것 / 다음
- **구현 0(이번)** — 곱의 확정은 구현 후 엔진 실측. num_assistant_tokens>8·tree-draft·batch+spec 결합은 후속.
- 메모리: 12B(24GB) + assistant(0.9GB) 동시 상주 — W4A8 시 ~7.5GB로 여유(128GB 가드 내).
- 다음 세션 = ① verify POC부터(정확성 먼저·bit-id 게이트).
- 적응 K 접힘 + verify concat + draft concat 실현(위) = **다중 슬롯 사이클 완전 배칭 완결**. 남은 잔여
  후보: K-quant bytes 10%↓(plain 소폭)·bf16-반올림 정합(chat accept)·verify N4 그룹 2회 적재 미세축.
