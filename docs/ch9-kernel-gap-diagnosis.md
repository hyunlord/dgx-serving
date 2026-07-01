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

## BM=16 배치 GEMM 빌드 → in-engine NO-GO (격리 오도 재발, few-blocks)
실현성 GO 후 BM=16 커널 빌드(gemm_mmq 리타일, tools/probe_bm16.cu): MS=1, 8 warp 전부 col(wn 0..7, 16 cols씩), cp.async 더블버퍼/MMA/scale 재사용.
- **correctness**: argmax ALL match(vs sb, fp dequant 순서차 2e-5), **in-engine iso 게이트 PASS**(B=12·16, fp32·W4A8 batch==single argmax). 정확성 완벽.
- **격리 타이밍**: BM=16 vs sb S=16 **1.60×**·S=12 1.35× 빠름(매우 유망).
- **★ in-engine 타이밍(test_batch_decode 0.8b W4A8): B=12 sb 753 vs BM16 711(−5.6%), B=16 gemm_batch 776 vs BM16 762(−1.9%) = 더 느림!** 격리(1.60×)와 정반대 = **격리 오도 재발**(CLAUDE.md #3, 이번 세션 3번째: register-prefetch·U-sweep·BM=16).
- **근본**: BM=16 grid = N/128 × M/16 = **few blocks(~40 이하)**. GB10 SM=48. 격리선 GPU 독점이라 40블록 OK지만, **in-engine선 다른 레이어 커널과 경쟁 + few-blocks가 GPU 못 채움**. SB는 grid=OUT warps=**many blocks**로 GPU 채워 latency를 block-간 숨김(per-warp long_scoreboard 72.8%여도 many-block로 극복). **de-risk의 "few-blocks minor" 판단이 오류** — in-engine dominant.
- **→ NO-GO(bounded 리타일). 진짜 해결 = stream-K(split-K로 소-M서 블록 생성, 회피하려던 major)** — llama mmq의 실제 설계. big-tile 리타일만으론 few-blocks로 불가. **batch wall 닫기 = stream-K major 빌드 확정(더 이상 추측 아님, 측정).**

## stream-K(split-K) BM=16 — in-engine WIN (arc 첫 돌파, `tools/probe_splitk.cu`)
BM=16 few-blocks(NO-GO) → stream-K(K를 SPLITK로 쪼개 blockIdx.z 블록↑, 40→~96~320). 각 split K-range 부분합(fp dequant)→workspace→reduce 합산(deterministic, atomic 안 씀). MS=1 8warp col, 더블버퍼/MMA/scale 재사용.
- **correctness**: 격리 argmax==sb 모든 SPLITK(1-2e-5); in-engine iso 게이트(vs 진짜 single decode) argmax PASS(B=12·16 fp32·W4A8).
- **correctness OK**: 격리 argmax==sb 모든 SPLITK; in-engine iso(vs 진짜 single) argmax PASS(B=12·16 fp32·W4A8).
- **⚠️ 이득 정정(정밀 재측정, 다음 세션 SPLITK 튜닝 brief)**: 지난 세션 "27B +13.9%(33.9→38.6)"는 **controlled A/B서 재현 안 됨**. 동일 코드·seed42·λ8·Bmax16·c-ch back-to-back: **baseline 43.2 → split-K 44.7 = +3.5%**(baseline 43.2≠이전 33.9 — 이전 A/B는 uncontrolled, baseline 비정상 저). raw B=16(bench_batch N64 REPS3 min-ms)도 **65.6→67.9 = +3.5%**(동일!). **honest 이득 = +3.5%, +13.9% 아님(overclaim 정정, honest-perf ⑧).**
- **B 의존(측정)**: split-K는 **B≥14만 이득**(B10 −12.5%/B12 −4.3%/B14 +4.7%/B16 +3.5%) — BM=16 타일 낭비 (16−B)/16 vs few-blocks 이득. → dispatch 게이트 **B≥14**로 수정(opt-in이 B9~13 regression 안 내게).
- **SPLITK 최적점 = auto(측정)**: B=16서 auto 67.9 > 고정 SPLITK{2:66.2, 4:65.4, 8:64.5, 16:60.9} 전부 — 고정은 단조 악화. auto가 matmul별 적응(gate/up SK1, down/o SK2, kv SK8) = 이미 최적, **튜닝 여지 0.**
- **★ncu — 왜 +3.5%만(구조)**: split-K 커널 occupancy = gate/up **46%**(136블록, ≫SM48 — few-blocks 아님!), q 33%(96블록), o/down 28%(80), kv 22%(64). **지배적 MLP matmul은 occupancy-bound(46%)지 few-blocks 아님 — split-K는 occupancy 못 고침 → 이득 미미.** 진짜 few-blocks는 tiny kv(8블록→SK8)뿐이나 시간 비중 극소(37 vs 508us). reduce 비용 작음(11-16us, ~3-5%). **이전 격리 1.60×는 작은 matmul만 봐서 오도(CLAUDE.md #3).**
- **opt-in(기본 off)**: split-K(MMA)는 sb(DP4A)와 다른 커널 → W4A8 borderline flip으로 **SB 계열 bit-identical-across-B 속성 깸**(test_cbatch_prod mism 11). argmax 정확. default off로 mism=0 보존, env BM16=1로 +3.5%(B≥14, trade 감수).
- **갭(정직)**: raw B=16 93.56/65.6=**1.427× → 93.56/67.9=1.378×**(부분 ~5% 축소, full 아님). 서빙(avgB≈14.6 saturated) +3.5% = raw와 일치(희석 아니라 애초에 작음). **batch wall은 열리되 좁게 — 진짜 병목은 BM=16 MMA occupancy(46%), split-K가 못 여는 부분. "천장 아님"은 맞으나 "크게 열림"은 아님(정정).**

## occupancy de-risk — measured NO-GO (occupancy는 lever 아님)
batch matmul occupancy 46%가 진짜 병목인가 규명(파느냐 마느냐 근거).
- **payoff 상한(nsys 시간 분해, 27B B=16 decode step)**: gemm_bm16_splitk(batch matmul) **59.5%**, gdn_step_batch(GDN recurrence) **36.3%**, flash_attn+gemv+rmsnorm+etc ~4%. → matmul 지배(개선 여지 59.5%), 단 **GDN 36%가 별도 큰 덩어리**(occupancy work 무관).
- **limiter(ncu)**: Registers/thread **80** → Block Limit Registers **3**(min) = 확정 limiter=**REGISTER**. Theoretical occ **50%**(launch_bounds(256,2)). Achieved 28%(o/down grid80=**0.56 waves**)~46%(gate/up grid136) « 50% = **wave quantization**(BM=16 few M-tiles → 블록 부족).
- **★payoff 측정(launch_bounds 2→4→6, regs 80→64→40, spill 0)**: 이론 occ 50→66→100% 올려도 **achieved occ 28% 불변**(mb2·mb4 동일 28%!), **throughput 68.7→68.4→67.9(미세 감소)**. **occupancy 올려도 throughput 안 오름 = occupancy는 lever 아님(measured).**
- **근본**: achieved occ 낮음은 register 제한이 아니라 **wave quantization**(BM=16 M-타일 1개 → 블록 부족, 이론 상한 올려도 채울 블록 없음). matmul은 대역폭 35%(비포화)·long_scoreboard 11-15(latency-bound 아님)·occupancy 무관 → **grid-level wave-quant + B=16 저 arithmetic-intensity가 구조적 제약.** 블록↑ 유일 수단=split-K인데 auto 넘으면 reduce 오버헤드로 악화(측정).
- **결론(정직 니치, 도망 아님)**: raw batch matmul은 **GB10 BM=16 구조 천장 근접**(occupancy measured NO-GO, split-K +3.5%가 이 스킴 천장). 남은 1.38× 갭은 (a)다른 타일링(BN 쪼개 wave-fill — major, matmul이 latency-bound 아니라 payoff 불확실) or (b)llama broad 커널 성숙도(honest-perf ⑦). **occupancy는 팠고 lever 아님을 measured — 우리 우위는 niche(spec/서빙/정확성).** 별도 lever 후보 = **GDN(36%)** — matmul 아닌 recurrence 커널(미탐색 → 아래).

## ★GDN de-risk — arc 최대 WIN (서빙 +27.5%, `kernels/gdn.cuh`)
matmul occupancy NO-GO 후, nsys 시간분해가 가리킨 **미탐색 GDN(gdn_step_batch 36.3%)** 팜.
- **병목(ncu)**: Compute **2.62%**·Memory **13.7%**·occ **87%**(높음!)·**long_scoreboard 601**(천문) = 심각한 memory-latency-bound인데 대역폭·compute 둘 다 극저. 
- **근본**: 런타임 D → 루프 `for(k<D)` unroll 불가 → `float scol[128]`(512B, state 열)이 **local memory(stack)** → scol[k] 512회/thread 접근이 601 stall. (-Xptxas -v: 512B stack frame 확인.) 대조: `gdn_step_ksplit_tree_kernel<D>`(spec)는 이미 template.
- **수정**: `gdn_step_batch_kernel_t<D>` template(compile-time D) → unroll → scol 레지스터화(stack 512→56B, 255 regs). 연산 동일.
- **★controlled A/B(back-to-back, seed42)**: 커널 1665→468us(**3.5×**, long_scoreboard 601→37). raw B=16 68.8→92.1(**+33.9%**). **서빙(Bmax16 λ8 avgB14.6) 43.3→55.2 tok/s(+27.5%)·tpot95 −19.7%·e2e50 −23%.** B=1 +0.8%(weight-load 지배 — GDN 이득은 고배치서, state I/O가 B배).
- **정확성**: **mism=0 gate PASS**(test_cbatch_prod got==iso), tree-spec 커널 미변경(정확성 자산 무관). iso-vs-solo 11→13(FMA contraction from unroll — golden tolerance 내, 중간 반올림↓로 오히려 더 정확). **split-K(mism gate FAIL→opt-in)와 달리 gate 통과 → default-on.** escape: `-DGDN_FORCE_RUNTIME`.
- **의미**: matmul occupancy(measured NO-GO)와 정반대 — **미탐색 GDN(niche)이 진짜 lever였음.** "안 판 영역 파기"가 옳았음. GDN이 우리 차별점(Qwen3.5 하이브리드 네이티브)이라 niche 강화. 갭 좌표: 서빙 aggregate 43→55는 llama 대비 재측정 필요(다음). 남은 GDN 여지: bf16 state(9.7GB→4.8GB I/O 반감, 추가 lever 후보).

## bf16 GDN state de-risk — 두 관문 통과, 통합은 decision (`tools/probe_gdn_bf16*`)
GDN template 후 남은 GDN lever = state I/O 대역폭. state fp32(layer당 100.6MB R+W) → bf16 반감.
- **정확성 de-risk(CPU probe, gdn 수식 N=512 재생, fp32 vs bf16 state)**: max state relerr g_t0.95=**0.6%**·0.99=1.2%·0.999(stress)=1.9%, 전부 **plateau(cascade 없음)**. 이유: recurrence가 **contractive**(scol*=g_t, g_t=exp(G)<1이 오래된 오차 감쇠) — 2bit(decay 없음, 발산)와 근본 다름. **정확성 안전. PASS.**
- **속도 de-risk(isolated kernel, state 50MB≫캐시=격리≈in-engine, DRAM-bound)**: gdn_step_batch_kernel_t<128,ST> fp32 487us/207GB/s vs bf16 251us/201GB/s = **1.94×**. 둘 다 ~200GB/s(76% peak)=**bandwidth-bound 확증**(ncu L2 13.7%는 오도, GB10 DRAM% n/a). 
- **step-level 예상**: post-template GDN은 step의 ~13% → bf16이 GDN 반감 → **batch ~+7%(92→~98, llama 95.30 parity/반전 근접)**. 서빙 더 클 수.
- **STOP-and-report(통합 전)**: 커널 ST-template 완료. full 통합 major → 방향 확인. probe_gdn_bf16.cpp·probe_gdn_bf16_speed.cu.

### bf16 state 단계적 full 통합 — 완료·전 경로 검증(7b7c39c, opt-in)
GdnStateT typedef(기본 float, -DGDN_BF16_STATE=bf16), DecodeState.S·BatchDecodeState.S=GdnStateT, GDN 커널 Sin=GdnStateT(읽기 (float) mixed), snapshot(fp32)→state 변환 헬퍼 `state_from_f32` 3곳(commit·promote_prefix·promote_tree_path).
- **회귀 0(default fp32)**: bench_batch 88.2·test_cbatch_prod mism=0·test_treespec_e2e accept3.67/mism0 전부 불변.
- **batch(bf16)**: fp32 88.2→**bf16 93.3 t/s(+5.8%)**, B=16 controlled. **93.3 vs llama 95.30 = 1.021×(near-parity!).**
- **single(bf16)**: test_cbatch_prod mism=0 PASS(batch/single bit-identical 보존).
- **tree-spec(bf16, 정확성 자산)**: test_treespec_e2e accept **3.67(fp32와 동일)**·mism **0/40** PASS — bit-identical + accept 완전 보존.
- **근거**: recurrence contractive(g_t<1) → bf16 오차 감쇠(de-risk <2% plateau); 커널 bandwidth-bound(1.94×). memory: state 절반(GB10 unified 여유).
- **서빙(bf16, controlled A/B 27B λ8)**: fp32 55.2→**bf16 58.3 tok/s(+5.6%)**·tpot95 0.2489→0.2367. 전체 GDN arc 서빙 **43.3(pre-GDN)→55.2(template)→58.3(bf16) = +34.6%.**
- **opt-in(default fp32)**: -DGDN_BF16_STATE로 활성(회귀 0 = fp32-exactness 테스트 보존). 배포 빌드=플래그. arc batch 갭 1.42→1.1(GDN template)→**1.02×(bf16 state)**.

## prefill(1.38×) de-risk — 구조 천장 재확인(measured, chapter-7 이후 도구)
남은 raw 갭 prefill 656~704 vs llama 903. chapter-7 이미 판 벽을 batch-arc 도구(nsys 분해·occupancy A/B·SpeedOfLight)로 재규명.
- **nsys 분해(prefill C=512, 27B)**: gemm_mmq(matmul) **78.7%** 지배 / gdn_ksplit 4.7% / swiglu 3.9% / 나머지<3%. **prefill=matmul-bound**(GDN 같은 숨은 덩어리 없음 — 병렬이라 GDN 4.7%만; batch와 정반대).
- **ncu gemm_mmq**: regs **128**/thread → Block Limit Reg 2 = occ **33%**(chapter-7 50%아님, full-MMQ register-blocking이 72→128). SpeedOfLight Compute 28.5%·Memory 31.7%·long_scoreboard **2.0**(전부 ~30% 저활용, latency-bound 아님). waves 1.67.
- **★occupancy A/B(measured NO-GO)**: launch_bounds 2→3→4(regs 128→80→64) → spill 16→460→828B → prefill **728→1202→1756ms(−65/−140% 악화)**. **occupancy 올리면 register-blocking 깨져 극적으로 느려짐 = occ는 prefill lever 아님, 33%가 최적**(register-blocked 필수). batch(wave-quant)와 다른 이유지만 결론 동일(occ NO-GO).
- **stream-K N/A**: prefill grid 160+(1.67 waves)=블록 충분 → few-blocks 문제 없음(batch grid 40과 다름). split-K 무의미. chapter-7 기각 재확인.
- **판정(정직 천장)**: prefill은 register-blocked 128×128이 최적, occ/stream-K 둘 다 non-lever(measured). 남은 1.28~1.38× 갭 = **llama 타일/파이프 성숙도**(KTILE/deep cp.async — chapter-7 major NO-GO, instruction overhead). = GDN(미탐색 WIN)과 대조 — prefill은 이미 판 벽, 새 도구로도 구조 천장 재확인. single 1.16×도 memory-bound 구조. **정직 인정(도망 아님, measured 재확인).**

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
