# 챕터7 STEP 0 (FA.2c) — Flash-Decoding (KV 분할, decode 병렬성)

> 측정이 decode 먼저 정당화(ncu): decode(Sq=1) **Waves Per SM 0.17 = GPU 83% 놂**(grid 24 CTA, compute 2.9%).
> KV 분할 안 해 head당 1 CTA. → KV split로 GPU 채움(Flash-Decoding). 단 둘 다 추월까지(decode 후 prefill FA.2b.2).

## 0. 측정 근거 (ncu, FA.2b 커널)

| | prefill(Sq256) | decode(Sq1,base8192) |
|---|---|---|
| Grid Size | 384 CTA | **24 CTA** |
| Waves/SM | 충분 | **0.17 (GPU 83% idle)** |
| Compute(SM) | 17.9% | **2.9%** |
| 병목 | warp stall 69.6%(shared 적재) | **병렬성 부족** |

decode는 "GPU 비어있음"(grid 24), prefill은 "차 있는데 기다림"(stall). decode가 더 큰·근본적 비효율 +
격차 6.7×(퇴보) + 긴-context throughput 지배 → decode 먼저. prefill(FA.2b.2 softmax+cp.async)은 다음.

## 1. 2-패스 Flash-Decoding (atomic 단일패스 거부 — 결정성)

단일 쿼리의 긴 KV[0..attendlen)를 NS개 split으로 분할, 여러 CTA 병렬:
- **패스1**(split별): CTA가 KV 구간 [s·L..(s+1)·L) 부분 online softmax → 부분 (m_s, l_s, acc_s[HD]) 저장(중간버퍼).
- **패스2**(결합): split들의 (m,l,acc) merge → 전역 max m*, 각 split rescale exp(m_s−m*), Σ → O.
- **atomic 단일패스 거부**: split들 동시 atomicAdd = 더하는 순서 비결정 → fp 누적 실행마다 다름 → spec/청크
  일관 깸([[w4a8-chunk-prefill-divergence]] Phase C race 교훈). **2-패스는 패스2가 결정적 순서 결합 → race 없음.**

## 2. ★ Sq 일관 (핵심 위험 — hybrid가 test_promote 깬 패턴 재발 금지)

FA.2c(decode Sq=1)가 FA.2b(prefill Sq>1)와 **다른 결과면 spec decode 깨짐**(verify Sq>1 ↔ sequential Sq=1).
hybrid(prefill FA.2b / decode FA.1)가 정확히 이걸로 test_promote FAIL. 단발/청크도 같은 위험.

**해결 — split 수 = f(KV 길이)만, Sq/query 독립:**
- `NS = ceil(attendlen / SPLIT_KV)` (SPLIT_KV 예 2048). **query/Sq 무관** — 같은 KV 길이면 같은 split.
- query 독립(각 query 자기 attend, FA.2b처럼) → Sq=1 N번 == Sq=N 한 번 (같은 split, 같은 결합 순서) → bit 일치.
- **split=1(짧은 KV) → 결합 trivial → FA.2b와 동일 출력.** prefill(base 작음 → 대개 split=1) == 단발/청크 일관.
- decode/verify(base 큼 → split>1) → 자기끼리 같은 split → 일관. **prefill(split=1)/decode(split>1)는 경로 다르나
  test가 직접 비교 안 함**(chunk=prefill끼리, promote=verify·decode 둘 다 같은 base→같은 split).
- 즉 **split을 grid/SM이 아닌 attendlen으로 결정**해야 Sq 독립. grid로 정하면 청크 크기가 split 바꿔 깨짐(주의).

구현: FA.2c가 모든 Sq 처리(decode Sq=1, verify Sq=K+1, 가능하면 prefill도) — Sq는 grid.y, split은 KV. 단
prefill 큰 Sq는 이미 grid 충분(split=1=FA.2b)이라, FA.2c를 launch_flash_attn 단일 진입점에 통합(Sq·split 둘 다).

## 3. 검증 계획 (헌법 + 강조)
1. **CPU 오차0 — 여러 split 결합**(강조1): 일부러 작은 SPLIT_KV로 split 다수 → split마다 m 다를 때 rescale
   정확 검증(1 split은 trivial). == CPU naive rel<1e-2(bf16). GQA + 긴 KV 누적(vLLM 함정).
2. **결정성**(2-패스 race 없음): N회 bitwise. 패스2 결합 순서 고정.
3. **직교성 + Sq 일관**(강조2): q4·W4A8 둘 다 + **FA.2c(Sq=1) == FA.2b(Sq>1)** 같은 결과(spec/청크 — test_promote,
   test_chunk_prefill 통과). 한 쪽만 되면 멈추고 보고.
4. **ncu**: Waves 0.17→? GPU 채움. occupancy/compute.
5. **3-way**: decode 1.8→? llama 12 추격. + prefill 회귀 없음(FA.2b 유지).

## 4. 단계
- **FA.2c.0**: 패스1(split별 부분 m/l/acc) + 패스2(결합) 커널. 작은 split부터. CPU 오차0(다중 split).
- **FA.2c.1**: 엔진 통합(launch_flash_attn에 split 통합, Sq 일관). 스위트/promote/chunk/직교성.
- **FA.2c.2**: SPLIT_KV/NS 튜닝(ncu Waves). 3-way decode 측정.

## 5. 안전·구조
- 중간버퍼(부분 m/l/acc) = O(NS·NQ·HD) 작음(split·head·dim). 메모리 가드.
- FA.2c가 FA.2b 대체/통합(decode 경로) — 커널 수 관리. 작은 split/base부터.
- prefill FA.2b·GDN·matmul 안 건드림(이번 decode만). prefill 추월은 다음 FA.2b.2.

---
*2-패스 Flash-Decoding, split=f(attendlen)(Sq독립) → CPU 오차0(다중split 결합)+결정성+직교성+Sq일관 →
ncu(Waves 0.17→)+3-way(decode 추격). decode flat 후 prefill FA.2b.2로 둘 다 llama 추월. 단계마다 같이.*
