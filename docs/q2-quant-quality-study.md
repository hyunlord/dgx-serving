# q2 극한양자화 — weight 오차 study: naive q2 붕괴(measured), 속도 2×는 품질 조건부

> decode가 bandwidth-bound(⑳) → bytes↓(q2)가 measured로 지목된 lever. `속도 ≈ 대역폭÷bytes` → q2(2bit)=q4(4bit)의
> 절반 bytes = decode 이론 ~2×. ★단 게이트 전환: bit-id(값 불변) 불가 → **품질 저하 measured**가 본체(속도는 부수).
> ★결론: **naive q2는 품질 붕괴(weight rel L2 오차 47.6% = q4의 5.1×). group size로 못 구함(2-bit 근본적 coarse).
> 속도 2×는 품질 유지 조건부 = naive q2로는 미실현.** 실현 경로 = q3(중간) or 정교 q2(importance/mixed=연구). 정직.

## STEP 0 — 게이트 전환 + 측정 체계

- ★근본 전환: 지금까지 최적화(형상·접근)는 **bit-identical**(값 불변). q2는 **값을 의도적으로 변경**(2bit=4단계) →
  bit-id 불가 → 품질 저하 불가피 → **"얼마나 잃나" measured**가 게이트.
- **이번 측정 = weight 재구성 오차**(fp weight → 양자화 → dequant, rel L2). 품질 프록시(weight 오차↑=출력 오차↑=품질↓).
  tractable(Python, q2 커널 불요). full 품질(argmax/perplexity via q2 forward) = focused 후속.
- 대칭 b-bit 그룹양자화: scale=absmax/(half−0.5), q=clip(round(w/scale+half−0.5),0,2^b−1). bytes/w = bits + 32/group.

## STEP A — 재구성 오차 measured (Qwen3.5-0.8B, MLP gate/up/down)

| scheme | bits/w | rel L2 오차 | vs q4 |
|---|---|---|---|
| q4-g32 | 5.0 | **9.35%** | 기준(실사용 OK — argmax 보존, W4A8 골든 PASS) |
| **naive q2-g32** | 3.0 | **47.6%** | **5.1× 나쁨** |
| q2-g16 | 4.0 | 41.3% | 4.4× |
| q2-g8 | 6.0 | 34.5% | 3.7× (bits는 q4보다 많은데!) |
| q3-g32 | 4.0 | 20.1% | 2.1× |

- ★**naive q2 붕괴 measured**: 47.6% = q4의 5.1×. (q4 9.35%는 실사용 OK — 오차가 matmul서 평균화+argmax 보존.
  47%는 argmax 붕괴 예상 = 품질 붕괴.) "속도는 쉽고 품질이 관건" 확증.
- ★**group size가 q2 못 구함**: q2-g8은 6bit(q4 5bit보다 많음)인데도 34.5%(여전 3.7× q4). **2-bit=4단계가 근본적으로
  너무 coarse** — scale 세분화로 안 됨. (q2-g32 3bit=목표 절반bytes지만 47% 붕괴.)
- **q3-g32(4bit)=20.1%**: q4와 q2 중간. 4bit(q4 5bit의 0.8×)에 오차 2.1× — 현실적 중간 후보.

## STEP C — 판정 (품질↔bytes, 정직)

★**압살(decode 2×) 판정**: q2 속도 이득(bytes 절반)은 **품질 유지 조건부** — **naive q2는 그 조건 실패**(47% weight
오차). 무조건 2×가 아님.
- **실현 경로(measured가 가리킴)**:
  1. **q3(0.8× bytes·20% 오차)**: 온건한 byte↓(decode ~1.25×)·품질 손실 중간. 가장 안전한 sub-q4.
  2. **정교 q2(importance-weighted GPTQ/AWQ + mixed precision)**: naive 47%를 회복 시도 — 단 group size로는 안 되고
     (measured) calibration 기반 중요도 가중 + 민감 층 q4 유지 필요 = **연구 프로젝트**(다중 세션).
  3. 순수 naive q2 = **미실현**(품질 붕괴 measured).
- **모델 특수성**: 26b는 q4서도 fragile(M4: HF 자신 flip) → q2 더 심할 것(fp 미접근으로 이번 미측정, 정직).
- **overclaim 없음**: "q2로 2× 압살"은 품질 미해결이라 아직 **주장 불가**. 실현 = 정교 q2 연구 or q3 절충.

### 다음 (deliberate 후속)
- **full q2-forward 품질**: q2 커널(gemv q2) + argmax/perplexity 측정 = 실제 품질 확정(weight 오차는 프록시).
- **정교 q2 연구**: importance-weighted + mixed precision(민감 층 measured) — naive 47% 회복 시도. 베이지안 탐색
  (품질↔bytes 다목적, M3 연결). = 별도 focused 세션(큰 연구·헌법 "큰 변경 신중").
- **또는 대역폭 돌파(C, 2번째 기둥)**.

★핵심: **속도(bytes↓)는 쉽고 품질이 본체 — measured로 naive q2 붕괴 확인**(47% = 5× q4). group으론 안 됨. 실 sub-q4는
q3(온건) or 정교 q2(연구). 이번은 tractable 오차 기반(품질 프록시)·full 품질은 q2 커널 후속.

## 산출물
- (scratchpad) q2_werr.py 오차 하네스. docs/q2-quant-quality-study.md(측정 체계·오차 표·판정).
