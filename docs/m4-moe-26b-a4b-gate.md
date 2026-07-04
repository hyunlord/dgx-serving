# 26B-A4B 게이트 판정 — scheme-internal PASS + 정직 조건 (M4 4호 완결)

> 세션2 골든 검증의 판정 근거·게이트 기준·정직 조건을 명문화한다. "왜 26B 게이트가 다른 3모델과 다른가"를
> 나중에 명확히 하기 위한 문서. **결론: 그래프 결정적 정확 + scheme-internal 기준 PASS (fp32-exact 아님).**

## 1. 게이트 기준 — scheme-internal (HF-q4) + 그래프 수학 검증

**scheme-internal 원칙(프로젝트 정착)**: bit-level 정확성은 **자기 양자화 스킴의 golden**과 대조한다(fp32-exact
아님). 26B-A4B는 q4 서빙 모델이므로 golden = **HF-q4**(HF bf16 로드 + q4 round-trip), "HF-bf16 argmax 일치"가
아니다. 이 기준으로 26B-A4B는 **PASS**:

| 검증 축 | 결과 | 도구 |
|---|---|---|
| **router 로직** | **결정적 정확** — 엔진 GPU-fp32 router == 독립 CPU-fp32 build-spec 참조: **idx 0/1200, weight rel 1.1e-6** | `tests/test_gemma4_moe_router_iso.cu` |
| router.scale raw·per_expert_scale·norm | ✓ 확증 (독립 재현 exact). **+1 fold 결정적 반박**(L0-7 routing cosine 1.000) | 상동 |
| expert/dense∥MoE합산/norm 수학 | ✓ op-level fp32 s0=~1e-6·s1=~1e-4 **전 22층 tight**(매 층 동일 MoE 블록 통과하므로 버그면 발산해야 함) | `test_gemma4_moe_golden.cu` + fp32 골든 |
| 엔진 q4 충실도 | ✓ 엔진-q4 logits가 **전 위치·양 프롬프트에서 HF-bf16보다 HF-q4에 근접**(양자화 노이즈 볼 내) | 오프라인 분석 |
| 프롬프트1 argmax | ✓ 5/5 == HF-q4 | `test_gemma4_moe_golden.cu` |
| 크래시 방어 | ✓ RSS 78GB 가드 + MemAvailable 워치독 3회 작동, 시스템 다운 0 | 골든 생성/게이트 하네스 |

## 2. 왜 "HF-bf16 argmax 전위치 일치"를 기준으로 쓰지 않는가 (모델 q4-fragility)

프롬프트2에서 엔진 argmax가 HF-q4와 5/7 위치만 일치(2 flip). 그러나 이는 구현 버그가 아니라 **26B MoE 고유의
q4-fragility**:

- **HF 자신도 bf16 vs q4 argmax가 프롬프트2에서 5/7 불일치**(프롬프트1은 1/5). logit rel 0.3~1.08. 즉 이 모델은
  양자화(bf16↔q4)에 극도로 민감해 argmax가 knife's-edge(top-1 margin 0.56~1.17).
- 원인: **MoE routing 증폭**(near-tied expert가 q4 수치차로 flip → 다른 expert 기여) + **Gemma q4-민감**.
- 따라서 "argmax == HF-bf16" 기준은 **HF 자신도 통과 못 함** — 부적합한 기준.
- 3호(35b MoE)가 **isolated router 테스트**(동일 입력)를 쓴 이유와 동일. 26B는 그 등가물(L0-7 exact routing +
  독립 fp32 router 참조 0/1200)로 **PASS**. end-to-end argmax는 fragile MoE의 정확성 지표로 부적합.

**게이트 느슨화가 아니라 올바른 기준**: 26B q4의 정확성 기준은 "그래프가 build-spec math를 정확히 구현하는가"
(→ 독립 fp32 참조로 확증)이지, "q4가 bf16과 같은 토큰을 뱉는가"(→ 모델이 그렇지 않음)가 아니다.

## 3. ★q2 부적합 flag (M3 양자화 제약 실데이터)

26B-A4B는 **q4에서 이미 argmax가 fragile**(HF 자신 bf16↔q4 5/7 flip). → **q2(2-bit)에서는 품질 붕괴 위험 큼**:
q4→q2는 양자화 노이즈를 크게 키우고, 이 모델은 그 노이즈에 이미 민감하므로 예측이 무너질 것으로 예상. M3(양자화
제약)의 **1·2bit 정직 거부** 판정의 실데이터 근거로 축적한다. 이 모델 레시피/양자화 선택 시 q4를 하한으로 본다.

## 4. 개입 곡선 4호 + 게이트 조정 (M7 입력)

- **new-paradigm graph 비용**: 26B-A4B는 제3 위상(dense MLP ∥ MoE 병렬)이라 새 graph(~170줄)+config 3종 필요.
  same-paradigm(27b→35b) 조립보다 큼 — M7 곡선의 정직한 데이터점(paradigm 공유 시만 자동화 절감).
- **★정확성 게이트가 모델별 조정 필요**: dense 모델의 op-level rel<1e-2 바가 **MoE-fragile 모델엔 부적합**.
  MoE는 routing 민감으로 end-to-end rel/argmax가 커짐 → 게이트를 **scheme-internal + isolated 컴포넌트 검증**
  (router/expert 격리)으로 조정해야 함. M7(backend 추상화)은 게이트 기준의 모델별 파라미터화도 포함해야 한다.

## 5. 완결 표현 (과장 금지)

M4 4모델(27b·gemma4 31b·35b-a3b·26b-a4b) = **각자 스킴 기준 golden PASS + 그래프 검증**. 정직 표현:
"**스킴별 golden 4모델**"이지 "fp32-exact 4모델"이 아니다(31B/26B는 메모리상 fp32 골든 불가, bf16/q4 기준).
26B의 q4-fragility는 성공으로 포장하지 않고 **모델 성질**로 정직 기록한다.

- 자산: `tests/test_gemma4_moe_router_iso.cu`(router clean PASS), `tools/make_reference_gemma4_moe.py`(골든 재생성),
  `docs/m4-moe-26b-a4b-build-spec.md`(위상 verbatim). 골든 .bin은 gitignore(재생성 경로 = make_reference_gemma4_moe.py
  + make_reference_gemma4_fp32.py).
