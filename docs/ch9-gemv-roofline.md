# 챕터9 — gemv_w4a8 roofline de-risk: "21=극한"은 미입증 (상대 아닌 절대)

> 사용자 지적(정확): "llama도 72%니 우리 극한"은 **상대** 비교(llama도 sub-optimal일 수). 극한 입증 = **절대 roofline**
> (이론 하한 대비 + stall 분해). 측정 결과: 갭은 **load-latency(고칠 수도)**, unpack-불가피 아님 → **"21=커널 극한" 미입증.**
> 단 격리 bench가 in-engine과 괴리 + GB10 dram% n/a → 고칠 수 있음의 in-engine 크기는 미확정(추가 실험 필요).

## ① 이론 하한 (절대 roofline)
base 천장 = bytes 14.4GB ÷ peak: @251(read-STREAM)=57.4ms=**17.4 t/s**, @273=52.7ms=19.0 t/s. 우리 11.0(90.6ms) =
하한의 **1.58×(@251)~1.72×(@273)**. 이 갭이 극한 여부의 핵심.

## ② gemv 단독 대역폭
- warm cudaEvent(R=200, FFN gate 44.6MB): 189 GB/s = **69% of 273(75% of 251)**.
- in-engine aggregate(TPOT): 159 GB/s = **63%**.
- ncu cold(`--cache-control all`, 단일 launch): Memory 9.99%, **latency-bound** — 단 cold-start/격리 아티팩트(8547us wall).

## ③ ncu stall 분해 (cache-무관 패턴 지표)
| 지표 | 값 | 의미 |
|------|-----|------|
| sectors/request (global ld) | **4.40** | **완전 coalesced** (X uncoalesced 전례 아님) |
| stall long_scoreboard | **67.04** | global load 지연 (지배적) |
| stall mio_throttle | 0.02 | **unpack/LSU 아님** |
| stall short_scoreboard | 0.22 | shared 아님 |
| Compute (SM) % | 15.5 | **unpack-compute 아님** |
| Achieved Occupancy | 74% | 정상(병목 아님) |

## 판정 — 사용자 decision tree
- **"① unpack ALU 불가피"** = **기각** (mio_throttle 0.02, compute 15% — unpack은 병목 아님).
- **uncoalesced** = 기각 (4.4 sectors/req = 최적).
- **occupancy** = 기각 (74%).
- → 갭은 **global-load latency**(long_scoreboard). 사용자 tree상 **"고칠 수 있음" 쪽**(unpack 불가피 아님) → **"21=극한" 미입증.**

## 단 — 미해결(정직): 격리 ≠ in-engine + dram% n/a
- 격리 ncu(cold 10%, latency-bound)와 in-engine aggregate(63%)가 **크게 괴리** — CLAUDE.md "격리 벤치 ≠ in-engine(X-shared 전례)".
  격리는 cold-start/단일-커널-비오버랩으로 과-비관, in-engine은 63%(파이프라인). 어느 게 gemv의 "진짜 효율"인지 격리론 불명.
- **GB10 dram% n/a** → "DRAM 포화(=bandwidth-bound, 불가피) vs 미포화(=MLP 부족, 고칠 수)"를 직접 못 가름.
- 두 해석 공존: (A) in-engine 63-72%가 진짜 = quantized gemv 실용 천장(long_scoreboard=정상 BW-bound, llama도 72%) →
  21≈극한. (B) latency-bound로 MLP 헤드룸 있음 → base lever. **격리+dram%로는 (A)/(B) 못 가름.**

## 결정적 해소 = 경험적 (in-engine 실험)
격리·이론으론 미해결 → **gemv_w4a8 MLP 변형(warp당 outstanding load↑ / 2행/warp / wider load) 시도 + in-engine t/s 측정.**
빨라지면 (B) base lever 확정(21 아님), 평탄하면 (A) BW-bound 확정(21≈극한). bit-exact 게이트 필수. 이게 "고칠 수 있나"의
유일한 결정적 답(상대·격리 아닌 in-engine 경험). 챕터7은 prefill MMA micro-lever만 평탄 확인 — decode-gemv MLP/load-latency는 직접 미시도.

## 정직 정정
"21 t/s = 커널 극한"(직전 결론)은 **상대 비교(llama 72%)+warm 측정 기반 = 미입증.** 절대 roofline은 1.58× 갭이 load-latency
(unpack 아님). 극한 확정하려면 in-engine gemv-MLP 실험 필요. (2bit는 별개로 품질 NO-GO 유지.)
