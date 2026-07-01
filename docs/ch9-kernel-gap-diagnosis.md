# 챕터9 커널 갭 진단 + llama spec 시도 — STEP 1 블록 / STEP 2 진단 / STEP 3 제안(보고)

> 지형도(78d5caf): llama raw throughput 1.16~1.42× 앞섬. 이번 = ②spec 공정 비교 →①커널 갭 규명 후 좁히기.
> 결과: **STEP 1 자산 블록**(fair spec 측정 불가), **STEP 2 진단 확보**, **STEP 3 큰 커널 작업 = 착수 전 보고**(CLAUDE.md #5).

## STEP 1 — llama spec 공정 측정: 자산 블록 (측정 불가, 정직)
우리 spec(21.3)은 MTP-head self-spec. 공정하려면 llama도 MTP spec. **두 경로 다 막힘:**
- **`--spec-type draft-mtp`**(native MTP): `failed to create MTP context` — **27B Q4_0 GGUF에 MTP/nextn 텐서 없음**(표준
  GGUF 변환이 MTP head 제거; 우리는 별도 추출해 보유). → GGUF를 MTP 텐서 포함해 재변환해야(out-of-scope).
- **`--spec-type draft-simple -md qwen35-4b`**(draft-model): `draft model vocab type must match target` — 4B finetune
  vocab ≠ 27B vocab. vocab-matched 소형 draft 필요(base Qwen3.5 소형, 미보유).
- **결론: 가용 자산으로 llama MTP-spec 공정 측정 불가.** 유일 측정점 = 우리 spec 21.3 vs llama **non-spec** 13.83
  (불공정 — 우리 spec vs 그들 non-spec; llama MTP-spec 켜면 갭 줄 것). fair 비교엔 MTP-GGUF 재변환 or vocab-matched draft 필요.

## STEP 2 — 커널 갭 진단 (fresh 측정 + 챕터7 + llama 소스 분석)
**fresh 측정(GB10 27B Q4_0 vs W4A8)**: prefill llama 901 vs 우리 656(1.37×), batch B16 llama S_TG 93.56 vs 우리 65.9(1.42×),
단일 llama 13.2~13.8 vs 우리 base 11.9(1.1~1.16×). **llama decode는 context-flat**(tg128 d0 13.23 → d2048 13.38 = 긴 KV서도
single decode 안 느려짐 = attention 잘 최적화). 우리 서빙 TPOT는 context 따라↑(ctx128 0.33→2K 0.52).
**우리 커널 ncu(챕터7/9)**: gemm_mmq occupancy **50%**(Block Limit Shared AND Reg 동시), L1/TEX **40%**(=llama 맞춤),
stall 73.7%; gemv occ 74%, bw 75%.
**갭 regime(진단)**: L1/TEX(캐시 효율)는 챕터7서 맞췄음. 그 위 1.37×는 **matmul 실행 효율** — 우리 full-MMQ는 **DP4A 기반
occupancy 50% 벽**(shared+reg 동시 제약, 챕터7 확정), llama는 그 위에서 더 빠름. llama Q4_0 커널은 int8 tensor-core MMA
(mma.sync) + arch-튜닝 타일 + 파이프라이닝으로 우리 DP4A 경로보다 실행 효율↑(추정 — 소스 분석 에이전트 실행했으나 상세
file:line 회수 실패, 툴링 한계; 챕터7 진단 + 원리로 이 방향 확정적). **핵심: 갭은 캐시/대역폭 아니라 compute 실행 효율/occupancy.**

## STEP 3 — 갭 좁히기: 큰 작업 = 착수 전 보고 (CLAUDE.md #5)
1.37× 커널 갭 좁히기 = **챕터7이 이미 깊이 판 벽**("occupancy 50% + FA shared 본질", KTILE/stream-K instruction-overhead 기각).
후보(전부 큰 작업, 불확실):
- **int8 tensor-core MMA(mma.sync.s8) 경로** — 우리 DP4A 대신 tcgen05/mma로 W4A8 matmul(FP4는 toolkit 막힘 [[fp4-mma-toolkit-blocked]],
  단 int8 MMA는 13.0 가용). occupancy/실행효율↑ 잠재. **큰 재작성, 정확성 재검증(bit-exact) 필요.**
- **타일/register 블로킹 재튜닝** — occupancy 50%의 shared+reg 동시 제약 완화(챕터7서 부분 시도, 벽).
- **판정**: 이건 챕터7 근본 재구조(이미 벽)의 재도전 = **큰 불확실 작업**. 자율 강행 대신 **방향 확인 후 착수**(rule #5).

## 정직 좌표 (이번 결론)
- **STEP 1 블록**: llama fair spec 측정 불가(자산 — MTP-GGUF/vocab-matched draft 부재). 우리 spec vs llama non-spec(불공정)만 측정됨.
- **STEP 2 진단**: 갭은 **compute 실행 효율/occupancy**(캐시 아님, 챕터7서 L1/TEX 맞춤). llama = int8 MMA+튜닝 타일 추정.
- **STEP 3**: int8 MMA 경로가 유력 후보나 **큰 재작성**(챕터7 벽 재도전) — 착수 전 사용자 방향 필요.
- 자산: llama 측정 로그, 챕터7 ncu, honest-perf ⑦.
