# 챕터9 chunked prefill — 서빙 aggregate + TTFT lever

> SMAX 확장(42b557f) 후 e2e 발견: **1토큰/step prefill이 벤치 지배**(커널 효율 ~815 tok/s가 aggregate에 안 드러남).
> prefill은 병렬 가능(프롬프트 다 앎)인데 1토큰씩 = 병렬성 버림. → **청크(≤512) 병렬 prefill**로 회복.

## STEP A — de-risk (기존 qwen35_decode Sq>1 = full-MMQ 병렬 prefill 재사용) `tools/probe_chunked_prefill.cu`
0.8b W4A8, P=512 프롬프트를 청크 크기별 prefill:
| chunk | prefill_ms | speedup | argmax |
|---|---|---|---|
| 1 (현행) | 2818 | 1.0× | 314 |
| 128 | 68.8 | **41×** | 314 |
| 512 (full) | 45.4 | **62×** | 314 |
- **청크화 62× (W4A8)**, argmax 동일(토큰). max\|Δ\| 0.365(chunk>1 vs 1)는 gemv(Sq1) vs MMQ(Sq>1) 커널차(graph-flip 계열,
  chunk간엔 bit-identical). 1토큰 prefill 2818ms(512토큰!) = 벤치 병목 확정. lever 실재·거대.

## STEP B — prefill_batch_slot (커널 재사용, 새 커널 아님)
batch slot b의 KV/GDN 서브영역([B,...]의 b번째 = single과 **동일 연속 레이아웃**)에 alias한 `DecodeState` view로
기존 `qwen35_decode`(Sq>1 full-MMQ) 재사용(models/qwen35_batch.cu). device 메모리 공유, host 포인터 배열만 임시.
청크(≤512)로 loop → slot b KV[0..P-1]+GDN 채움 → pos=P부터 batch decode 합류.
- **mism=0(`tests/test_chunkprefill.cu`)**: chunked prefill + batch decode == 1토큰/step prefill + batch decode
  (prefill 방식만 차이, 이후 decode 동일) → **fp32·W4A8 0/10 토큰 동일**(chunk 16·40). prefill_batch_slot 정확.

## STEP C — e2e 서빙 (`bench_serve`, c1tok=1토큰 vs c-ch=chunked)
**짧은 프롬프트(P≤32), 0.8b W4A8 Bmax=8:**
| mode | λ=4 TTFT p95 | λ=4 tok/s | λ=16 TTFT p95 | λ=16 tok/s |
|---|---|---|---|---|
| c1tok | 0.305 | 214.8 | 2.913 | 451.3 |
| **c-ch** | **0.021 (14×↓)** | 215.6 | 2.157 | **496.2 (+10%)** |
| stat | 3.327 | 190.7 | 9.090 | 252.9 |

**긴 프롬프트(P≤100, avg~54), λ=8:**
| mode | tok/s | TTFT p95 | e2e p95 |
|---|---|---|---|
| c1tok | 283.3 | 3.180 | 4.378 |
| **c-ch** | **389.2 (+37%)** | **0.719 (4.4×↓)** | 2.444 (1.8×↓) |
| stat | 165.0 | 10.871 | 11.987 |

- **TTFT 14×↓(짧음)~4.4×↓(김)** — prefill이 ~P스텝 → 1스텝(병렬). 서빙 반응성 핵심.
- **aggregate 이득 프롬프트 길이 따라 커짐**: 짧음 +10% → 김 +37% → 긴 프롬프트일수록 prefill 지배 제거 효과↑
  (probe 62× 커널이 e2e로 발현). chunked-continuous가 static 대비 **2.4× throughput**(389 vs 165).
- **TPOT 약간↑**(0.0126→0.0146) — chunked는 admit 즉시 prefill로 slot이 빨리 차 avgB 분포 달라짐(미미).

## 통합 (scheduler)
admit 시 `prefill_batch_slot`(1스텝, 측정) → slot pos=P, 첫 토큰=prefill 로짓(TTFT). 후 batch decode(ragged pos).
prefill(single-path, per-slot)과 decode(batch-path) 각자 globals → 충돌 없음. admission guard(KV 예산) 유지.

## 정직 좌표 + 남은 것
- **순 이득**: TTFT 4~14×↓, aggregate +10~37%(프롬프트 길수록↑), mism=0, 커널 재사용(새 커널 0). static 대비 2.4×.
- **caveat**: rope 테이블 128행 → 현 벤치는 pos>128 timing-only(양 모드 동일=공정). **긴 context 정확 서빙엔 long-rope
  테이블(≥max_len) 필요**(별도, 현 자산은 128). prefill은 P≤128서 rope 정확. single-seed, 토큰선택 제외(forward 지배).
- **남은 것**: ① long-rope 테이블(≥max_len) — 긴 context 정확성 ② 매우 긴 프롬프트(P>512) 청크 interleave(decode 안
  굶게 한 청크/iteration — 현재는 prefill 통째 1스텝이라 P>>512면 잠깐 독점) ③ 27B 재측정 ④ prefill/decode 진짜
  piggyback(한 forward mixed Sq — 최고급, mixed-batch 커널 필요).
- 자산: prefill_batch_slot(qwen35_batch.cu, DecodeState slot view), probe_chunked_prefill, test_chunkprefill, bench_serve(c-ch).
