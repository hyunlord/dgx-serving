# 정교 q2 연구 — measured: 압살(2×) 미실현, q3가 실 sub-q4 (설계 0·한 기법씩)

> naive q2 붕괴(464× ppl) → 정교 기법으로 q4 근접 시도. ★설계 0·한 기법씩 소거법(어느 기법이 진짜 일하나).
> ★결론: **importance(AWQ) 단독 미미(464→323×, 여전 gibberish)·혼합 정밀도로 coherent(1.8×) 되려면 75% 층 q4 필요
> =4.62 bits/w=q4의 0.92× → decode 겨우 +8%(2× 아님). q2 half-bytes는 usable 품질과 근본 양립 불가(tractable 기법).
> q3(4bit·1.74×·coherent)가 mixed-q2보다 byte-효율 우위 = 실 sub-q4.** 압살 via q2 = 미실현(measured).

## STEP 0 — 중요도 가중(AWQ) 단독 (한 기법)

AWQ 개념(calib 활성 스케일 s로 salient 채널 보호: quant(W·s)/s, s=act^α). PyTorch 시뮬 forward(0.8b, 품질 먼저).
- q2-naive 464× → **q2-AWQ(α=0.5) 323~372×**(argmax 여전 ~10%·gibberish). α=1.0은 343132×(악화). 
- ★**AWQ 단독은 2-bit 회복 미미**(20~30% ppl↓만). literature 정합: AWQ는 4-bit용, 2-bit엔 QuIP#/AQLM(lattice codebook+
  incoherence processing) 무거운 기법 필요. → 다음 기법(혼합) 필요 measured.

## STEP A — 혼합 정밀도 (민감 상위 K% q4·나머지 q2-AWQ)

민감도=importance(act)×q2오차×파라미터. 품질↔bytes Pareto:
| q4% | bits/w | ppl(vs fp) | argmax | 생성 |
|---|---|---|---|---|
| 0% (AWQ만) | 3.00 | 323× | 11% | gibberish |
| 25% | 3.59 | 71× | 19% | broken |
| 50% | 4.16 | 10× | 48% | semi-coherent |
| **75%** | **4.62** | **1.8×** | 79% | **coherent** ("the capital of France...") |
| 100%(=q4) | 5.00 | 1.1× | 90% | "Paris" ✓ |

## STEP B — 판정 (품질↔bytes, measured)

★**압살(decode 2×) via q2 = 미실현**:
- 순수 q2(3bit=대역폭 절반=2×)는 broken(323× even AWQ). **coherent 품질(1.8×)엔 75% 층 q4 필요=4.62 bits/w** =
  q4(5.0)의 0.92× → decode 겨우 **+8%**(2×와 거리 멂). q2의 half-bytes 이점이 품질 유지선 사라짐.
- **각 기법 기여도(소거법)**: AWQ 단독 464→323×(미미)·혼합이 실제 회복 축(75% q4=1.8×)이나 그건 사실상 "대부분 q4".
- ★**q3(4bit)가 mixed-q2(4.62bit)보다 우위**: q3 uniform 1.74× ppl(coherent)·4 bits < mixed-q2 4.62 bits·1.8×.
  = **실 sub-q4 = q3**(0.8× bytes·decode ~+25%·coherent), q2 정교보다 byte-효율.
- **실용 서빙 판정**: usable 품질(coherent)의 최소 bits/w ≈ **4(q3) — q4의 0.8×** = decode ~+25% 상한(2× 아님).
  q2(3bit)는 tractable 기법(AWQ+mixed)으로 usable 불가.

### 다음 (measured가 정함)
- **압살(2×) 포기 or SOTA 2-bit**: usable 2-bit는 QuIP#/AQLM(lattice+incoherence·무거운 구현·별도 대형 연구) 필요 —
  AWQ+mixed로는 미달(measured). 큰 리서치 프로젝트(헌법 "큰 변경 신중").
- **q3 실용화(온건 실현)**: 4bit·1.74× ppl·coherent·decode ~+25%. 바로 가능한 sub-q4(in-engine q3 커널+골든).
- **또는 대역폭 돌파(C, 2번째 기둥)**.
- 모델 caveat: 0.8b(작음=q2에 더 fragile). 27b는 덜 극단일 수 있으나 2-bit half-quality 트렌드는 유지 예상. 26b별도(fragile).

★핵심: 설계 0·한 기법씩 measured로 **q2 압살은 tractable 기법으로 미실현**(4.62bit=q4근접해야 coherent) 확정. 실 sub-q4=q3.

## 산출물
- `tools/q2_awq.py`(AWQ)·`tools/q2_mixed.py`(혼합 Pareto)·docs/q2-sophisticated-measured.md.
