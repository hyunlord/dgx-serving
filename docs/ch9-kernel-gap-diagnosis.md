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

### STEP 3 de-risk 결과 — GO (int8 MMA가 occupancy 벽 완화 확정, `tools/probe_mma_occ.cu`)
int8 WMMA(tensor core s8) 최소 GEMM 커널 측정:
| 커널 | regs | shared | occupancy |
|---|---|---|---|
| 우리 gemm_mmq (DP4A) | 72 | 26.3KB | **50%** (Block Limit Shared AND Reg) |
| int8 WMMA 32×32~64×64 | **39-40** | **2-8KB** | **100%** |

- **occupancy 50→100% 완화.** regs **72→40**(사용자 우려[MMA가 reg 더]와 정반대 — tensor core가 누적 담당), shared **26.3→2-8KB**(WMMA는 DP4A의 큰 X-shared staging 불필요). → **occupancy 벽이 int8 MMA로 완화됨 = 갭의 근본 원인 표적 적중.**
- **correctness 0 mismatch**(int8 MMA == scalar, sm_120 ✓).
- **bit-exact by construction**: int8·int8→int32 누적은 정수=순서무관 정확 → MMA와 DP4A 같은 정수 dot → dequant 동일 적용 시 **MMA W4A8 == DP4A W4A8 bit-exact**(드롭인, 정확성 보존).
- **→ GO: 정식 int8-MMA W4A8 GEMM 재작성 정당**(occupancy 벽 완화 measured, bit-exact 보장). caveat: 100% occ는 필요조건 — 실 speedup은 tensor core util·int4 unpack·memory feed에 의존(정식 구현서 측정).

### ★ 중대 정정 — 우리 prefill은 이미 MMA (STEP 2 진단 오류 수정, 커널 실사)
정식 착수 전 커널 실사: **gemm_mmq(prefill)는 이미 `mma.sync.aligned.m16n8k32.s8.s8.s32` 사용**(line 93) — int8 tensor-core MMA, DP4A 아님. **STEP 2의 "우리=DP4A, llama=MMA" 진단은 prefill엔 틀림.** 실사 결과:
| 우리 커널 | 연산 | 경로 |
|---|---|---|
| gemm_mmq | **MMA** (m16n8k32.s8) | prefill Sq>8 |
| gemm_batch_w4a8 | **DP4A** (float acc[B]) | batch B=16 |
| gemv_w4a8(+sb) | **DP4A** | decode B=1 + small-batch B≤15 |

**정정된 갭 분해:**
- **prefill(901 vs 656)**: 둘 다 MMA. 갭 = **타일/occupancy 튜닝**(우리 128×128 타일 → 72 regs·26.3KB·occ 50%; de-risk의 작은 타일 MMA는 100% occ지만 weight-reuse↓ 트레이드 — 챕터7 이미 탐색한 영역). MMA 부재 아님.
- **decode B=1(13.8 vs 11.9)**: GEMV(M=1), MMA 적용 불가(M≥16 필요). memory-bound(75% bw). MMA N/A.
- **batch B=8-16(93.56 vs 65.9, 1.42×)**: 우리 DP4A(gemv_sb/gemm_batch), M=B=16이 **MMA m16n8k32에 적합** → **MMA가 진짜 빠진 유일한 곳**. llama 배치는 MMA 추정.

**→ 재초점: int8-MMA 기회 = 배치 경로(B=8-16), prefill 아님(이미 MMA)·decode 아님(GEMV).** 단 caveat: 배치가 부분 memory-bound(weight 대역폭 1적재)면 MMA(compute) 이득 제한 — llama B=1→16 스케일 6.97×(≠16×)=부분 compute-bound라 MMA 여지 있으나 불확실. **정식 대상 = MMA 배치 GEMM(gemm_batch DP4A 대체), payoff 불확실 → 착수 전 방향 확인.**

### batch compute/memory-bound de-risk — NO-GO for MMA (`tools/ncu_sb16.cu`)
MMA 빌드 전 ncu로 배치(gemv_w4a8_sb, B=16, 27B shape) 병목 판별:
| 지표 | 값 | 해석 |
|---|---|---|
| Compute (SM) Throughput | **32%** | compute 병목 아님 |
| Memory Throughput | **32%** | 대역폭 포화도 아님 |
| Achieved Occupancy | **80%** | occupancy 이미 높음(lever 아님) |
| stall long_scoreboard | **72.8%** | **L1TEX load 지연 = memory-latency-bound** |

- **NO-GO for MMA**: compute 32%(병목 아님) → **MMA(compute 가속)는 이득 없음**. occupancy 80%(이미 높음) → occupancy도 lever 아님. 병목 = **load-latency**(우리 decode gemv와 동일 계열, [[mtp-spec-decode-verify-bandwidth]] long_scoreboard 67). throughput 32%만 = 지연 미은닉.
- **latency-hiding도 소진**: SB-MLP(outstanding load U-unroll) 이미 시도→역효과(SB는 S-accumulator로 ILP 충분, U-unroll이 occupancy만↓, 챕터9). → 배치 latency wall은 하드.
- **결론 = ③ 정직 수용**(브리프 결정규칙: memory-bound→③). **de-risk-first가 헛빌드 방지** — MMA 배치 커널 지었어도 compute 병목 아니라 무이득이었을 것.

## load-latency in-engine 공격 (사용자: ③은 도망 — 계속 판다)
long_scoreboard 72.8% 직접 공격. 우리 batch/decode = cp.async 0, 완전 동기 load(확정). llama = register-ILP(decode)+shared 더블버퍼(batch, docs/llama-mmvq-latency-analysis.md).
- **격리 de-risk 신뢰성 문제(CLAUDE.md #3)**: decode gemv U-sweep 격리 U=1 최속(ILP 역효과)인데 챕터9 in-engine U=2 +8%=정반대. 격리 warm-cache가 latency-hiding 오도. → in-engine 측정 필수.
- **register-prefetch(2-stage) in-engine A/B(test_batch_decode 0.8b, SB 경로)**: B=12 759.4→753.6(SB_PIPE), B=15 752.7→750.3 = **~중립(무이득)**. 격리(0.867×)보다 낫지만 in-engine도 이득 0. bit-exact.
- **구조 규명(NO-GO 근거)**: SB compute-per-load ~32cy ≪ load latency ~400cy → **2-stage 파이프는 1 chunk분(~32cy)만 겹침 = 400cy 못 숨김**(~12-stage 필요). register/cp.async 무관(둘 다 2-stage 더블버퍼). occ 이미 80%(더 못 올림). → **2-stage pipelining은 lever 아님(측정+구조).** cp.async 2-stage도 동일 한계 → 프록시상 NO-GO.
- **실제 lever(둘 다 major)**: ① deep multi-stage(~12) 파이프(shared 12×, 복잡) ② llama식 big-tile 배치 GEMM(compute-per-load↑로 2-stage로 충분, MMQ 함정 회피 설계). 둘 다 큰 불확실 재작성.

## big-tile 배치 GEMM 실현성 de-risk — GO (설계+측정, `tools/probe_mmq_m16.cu`)
2-stage NO-GO 후 근본 정답(llama big-tile) 실현성 값싸게 규명(major 빌드 전).
- **설계 계산**: batch M=16 intensity ~16 MAC/weight-byte(고정). big-tile 16×BN(BN=128,KT=64)은 W[128,64] shared 1적재→16×128×64 MAC=32 MMA≈**512cy 컴퓨트/tile-load** > 400cy latency → 2-stage 더블버퍼로 숨김(SB의 32cy와 결정적 차이 — load 입도↑). shared ~10KB(gemm_mmq 26KB보다 적음).
- **★측정 확증(gemm_mmq BM=128 at M=16, 27B shape)**: **Warp Cycles/Inst 7.23 vs SB 28.23** = **big-tile 더블버퍼가 latency 이미 숨김**(4× 효율). gemm_mmq 0.94×(sb보다 느림)·util 22%는 **오직 empty-row 낭비**(BM=128에 M=16=7/8 빈 행). bit-exact gemm_mmq≈sb 2e-5(fp dequant 순서, argmax-safe).
- **함정 = empty-row뿐(근본 아님)** → **BM=16(=M, MMA m16 1타일, empty 0)** 으로 고침. gemm_mmq의 cp.async 더블버퍼+MMA+scale 재사용, warp 레이아웃만 재배치(8 warp 전부 col 방향, A-tile 16행). few-blocks(BM=16→OUT/BN 블록=40 vs 48 SM)는 더블버퍼가 intra-block 숨겨 minor(BN 튜닝 가능). **stream-K 전면 재구현 불필요.**
- **판정: GO — 지을 만함(bounded 빌드).** BM=16 배치 GEMM = gemm_mmq 리타일. 더블버퍼 latency-hiding은 measured(7.23), 함정(empty-row)은 BM=16이 제거. 기대: 배치 raw wall(SB 72.8% long_scoreboard) 제거, throughput↑(정확 이득은 빌드서 측정). bit-exact 게이트(정수 dot + 동일 dequant).

## 최종 정직 좌표 (커널 갭 3-축 종합)
- **prefill(901 vs 656, 1.37×)**: 둘 다 MMA. 갭 = 128×128 타일 occupancy 50% 트레이드(챕터7 근본 벽, 재구조 필요).
- **decode B=1(13.8 vs 11.9, 1.16×)**: GEMV memory-latency-bound(long_scoreboard), MMA N/A, gemv-MLP +8% 소진.
- **batch B=8-16(93.56 vs 65.9, 1.42×)**: DP4A지만 **memory-latency-bound(de-risk 확정)** — MMA·occupancy·latency-hiding 다 소진/무관.
- **종합**: 우리 raw 커널은 3축 다 **구조적 한계 근접**(prefill MMA+타일, decode/batch load-latency). **llama ~1.4× 우위 = 단일 미보유 기법(MMA 등)이 아니라 broad 커널 성숙도**(latency-hiding·스케줄링, llama가 년단위 튜닝). 우리 차별점 = niche(bit-identical 정확성·GDN 네이티브·연속배칭 서빙 스택·spec), 범용 raw throughput 아님. **honest-perf ⑦ 확정.**

## 정직 좌표 (이번 결론)
- **STEP 1 블록**: llama fair spec 측정 불가(자산 — MTP-GGUF/vocab-matched draft 부재). 우리 spec vs llama non-spec(불공정)만 측정됨.
- **STEP 2 진단**: 갭은 **compute 실행 효율/occupancy**(캐시 아님, 챕터7서 L1/TEX 맞춤). llama = int8 MMA+튜닝 타일 추정.
- **STEP 3**: int8 MMA 경로가 유력 후보나 **큰 재작성**(챕터7 벽 재도전) — 착수 전 사용자 방향 필요.
- 자산: llama 측정 로그, 챕터7 ncu, honest-perf ⑦.
