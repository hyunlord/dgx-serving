# (C) 동적/활성화 sparsity 진단 — soft+dynamic, 예측비용 함정, ~1.33× 상한 (measured)

> q2 막힘(양자화로 bytes↓ 불가) → sparsity(안 읽기=양자화 없이 bytes↓)가 대역폭 돌파 2번째 시드. ★measured 진단(구현 아님).
> ★결론: **dense FFN 활성 sparsity(SwiGLU)는 soft(31% clean at 0.66%err·ReLU 90% 아님)+100% dynamic(0% static)
> → 매 토큰 predictor 필수(예측비용 함정)·절약 상한 ~1.33×(2× 아님). MoE만 싼 predictor(router) 있어 sparsity 작동.**

## STEP 0 — FFN 활성 sparsity 실재 (Qwen3.5-0.8B, 63 토큰 decode 프록시)

SwiGLU 중간활성 h=silu(gate(x))·up(x)(down_proj 입력). h[j]≈0이면 down col j skippable.
| thr(|h|<thr×max) | skippable 뉴런% | 잃는 L2 에너지% |
|---|---|---|
| 0.001 | 5.1% | 0.005% |
| **0.01** | **31.1%** | **0.66%**(clean) |
| 0.05 | 75.8% | 14.3%(과다=품질 저하) |
| 0.10 | 90.8% | 28.2%(붕괴) |

- **soft sparsity**: clean(저 에너지손) skippable ≈ **31%**(thr 0.01). ReLU 모델(90%+ exact-zero, PowerInfer 대상)과 달리
  SwiGLU는 smooth = soft(가설 measured 확인). → 절약 상한: FFN이 decode weight ~80% × 31% = **~25% bytes → decode ~1.33×**(2× 아님).

## STEP A — 예측가능성 (★성패 = 예측비용)

| | % |
|---|---|
| 항상-skip(static·예측 쉬움) | **0.0%** |
| dynamic(토큰별·predictor 필요) | 99.7% |

- ★**100% dynamic**: 어느 뉴런이 작은지 토큰마다 완전히 다름 → **static pruning 불가**(0%)·**매 토큰 예측 필수**.
- ★**예측비용 함정(q2 혼합정밀 함정과 동형·심각)**: h 예측하려면 gate·x 필요 → **gate read = FFN의 1/3** = 절약(25%)의 상당분 먹음.
  대안 = 별도 trained predictor(PowerInfer식 작은 MLP) = **오프라인 학습 필요(대형 노력)** + 정확도 위험(틀리면 중요 뉴런 skip=품질↓).
- **MoE 대조**: MoE routing = 이미 싼 predictor(tiny router가 expert 선택) = 예측비용 낮음 → **그래서 MoE sparsity 작동**(moe_fused
  이미 잡음). dense FFN엔 그런 싼 router 없음 = 새로 predictor 훈련해야 = 함정.

## STEP B — 판정 (measured)

★**dense FFN sparsity = 깨끗한 대역폭 돌파 아님(measured)**:
- **상한 modest**(~1.33×·soft 31%·2× 아님) + **100% dynamic**(predictor 필수) + **예측비용 함정**(gate read or 훈련된 predictor)
  + **품질 게이트**(skip=0.66%err at 31%, 더 skip하면 급증). = 순 이득 불확실·큰 연구(predictor 훈련).
- **왜 안 되나(measured 답, 도망 아님)**: (1)SwiGLU가 soft(ReLU 아님)라 sparsity 자체 modest·(2)dynamic이라 예측 필수·
  (3)예측비용이 절약 먹음(q2 함정 동형). = 세 겹으로 dense FFN sparsity는 easy lever 아님.

### 다음 (measured가 정함)
- **sparsity가 작동하는 곳 = MoE**(routing=싼 내장 predictor). dense FFN은 predictor 훈련 대형 연구(modest payoff·cost trap).
- 대안 시드: unified-mem 계층/weight 상주(①⑤) or q3 온건(측정된 실 sub-q4) or decode 대역폭 천장 수용.
- **정직**: decode는 근본 bandwidth-bound(⑳)이고, 그 분모(bytes)를 줄이는 두 길(q2 양자화·dense sparsity) 모두 measured로
  막힘/modest. llama 갭(3-way 0.28×)의 실제 성질 = 대역폭(둘 다 같은 273GB/s·bytes 근본)·커널/형상/sparsity로는 modest.

## 산출물
- `tools/sparsity_diag.py`(FFN 활성 sparsity 진단)·docs/c-sparsity-diagnosis.md.
