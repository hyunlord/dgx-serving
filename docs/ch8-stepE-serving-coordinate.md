# 챕터8 STEP E — 서빙 엔진 좌표 (vLLM 비교, 27B GB10)

> 목적: 지금까지 비교는 llama.cpp(B=1)뿐. 진짜 "서빙 엔진" 좌표는 **llama(B=1 latency) + vLLM(batch
> throughput)** 둘 다 필요. 강점 축이 다르므로 혼동 없이 분리 측정한다. (측정-우선·정직.)

## 두 축 (혼동 금지)

| 축 | 의미 | 일반적 강자 | 우리 무기 |
|---|---|---|---|
| **축1: B=1 latency** | 단일 요청 decode t/s | llama.cpp (lean B=1) | MTP spec decode |
| **축2: batch throughput** | 동시 요청 total output t/s | vLLM (continuous batching) | W4A8 batch 커널 |

## 측정 환경
- 모델 Qwen3.5-27B (dense, 64층 = 48 GDN + 16 full-attn), GB10 sm_121.
- 우리: W4A8 (~0.5 byte/weight). llama.cpp: Q4_0 (~0.5). vLLM: bf16 (2 byte/weight) — vLLM 0.17.0 docker, `vllm bench throughput`, CUDA graph on, `language_model_only`.
- 공정성: 양자화 bytes 다름(명시). vLLM은 GB10×GDN 미최적(아래 caveat).

## 축1 — B=1 latency (27B decode t/s)

| 엔진 | quant | b/weight | decode t/s | vs vLLM |
|---|---|---|---|---|
| **우리 MTP spec** | W4A8 | ~0.5 | **16.9~24.6** (프롬프트별) | **3.8~5.6×** |
| 우리 plain | W4A8 | ~0.5 | 12.0~12.1 | 2.7× |
| llama.cpp | Q4_0 | ~0.5 | 13.7 | 3.1× |
| **vLLM** | bf16 | 2 | **4.4** | 1× |

→ 축1은 우리(+llama) 압도. vLLM 4.4 t/s는 GB10×GDN 미최적(아래).

## 축2 — batch throughput (27B, total output t/s)

| B (동시요청) | 우리 W4A8 | vLLM bf16 | 비고 |
|---|---|---|---|
| 1 | 12.3 | 4.4 | |
| 8 | **58.4** (4.75×) | **4.7** | 우리 스케일 vs **vLLM flat** |
| 16 | **65.6** (5.3×) | 미측정(OOM) | **우리 B=16 = vLLM B=8의 ~14×** |

→ **핵심: vLLM continuous batching이 GB10×GDN서 스케일 안 함**(B=1 4.4 → B=8 4.7, 거의 불변) — GDN/fla 커널이
per-token 병목이라 동시요청 amortize 못 함. **우리는 스케일함**(12.3→58.4→65.6, B=16서 plateau — per-seq
attention 루프 + 대역폭 한계). **우리 B=1(12.3)이 이미 vLLM B=8(4.7)의 2.6×, 우리 B=16(65.6)은 vLLM B=8의 ~14×** —
vLLM의 batch 강점이 이 niche서 완전 무력화. (우리 batch도 B=16 plateau = 개선 여지: per-seq attn batch화 + SB 통합.)

## 소-batch 커널 (축2 lever) — SB vs 기존 gemm_batch (de-risk)

27B FFN matmul GB/s (높을수록 좋음), SB == gemv bit-exact:

| S | gemv(naive) | MMQ | gemm_batch(기존) | **SB(신규)** |
|---|---|---|---|---|
| 2 | 110 | 112 | 207 | **222** |
| 3 | 75 | 105 | **75 (홀수 붕괴)** | **213** |
| 5 | 45 | 113 | **45 (홀수 붕괴)** | **192~200** |
| 8 | 29 | 112 | 202 | **185~202** |

→ **기존 `gemm_batch_w4a8`는 짝수 S만 빠르고 홀수 S서 대역폭 붕괴**(S=3:75, S=5:45). **신규 `gemv_w4a8_sb`는 전 S
일관(185~222 GB/s)** = batch 경로의 홀짝 cliff 제거 후보(우리 batch throughput 개선 lever).

## 공정성 caveat (필수)

1. **vLLM GDN/fla 미최적(GB10 sm_121)**: 로그 `fla/ops/utils.py:113 UserWarning: seq_len(16) < num_heads(48)
   format mismatch` + `Not enough SMs for max_autotune_gemm`. vLLM의 GDN(linear attn) fla 커널이 GB10서 비효율
   경로 → 4.4 t/s. **vLLM 느림 = GB10×GDN niche 미튜닝**이지 우리 순수 커널 우위 단정 못 함.
2. **양자화 bytes**: 우리 W4A8(~0.5) vs vLLM bf16(2). 단 우리 **fp32(4 b/w)도 vLLM bf16(2)의 ~15×**(70 vs 4.5) →
   양자화 단독으로 설명 안 됨(프레임워크/커널 오버헤드가 지배).
3. **fp8/B=16 미측정**: vLLM fp8는 OOM(이전 bf16 메모리 미해제, 27GiB 점유 → 95.76GiB 불가)으로 sm_121 지원
   여부 판정 불가. B=16도 OOM. SGLang 미측정(GDN fla 공유 추정 → 동일 한계 가능성).

## 정직한 결론

**GB10 × Qwen3.5(GDN) 이 niche에서 우리 특화 엔진이 양 축 모두 vLLM을 크게 앞선다** — 축1(latency) 우리 spec
16.9~24.6 vs vLLM 4.4 (3.8~5.6×), 축2(throughput) 우리 B=1 12 이미 vLLM B=8 4.7 초과. **단 이는 "범용 프레임워크
추월"이 아니라 vLLM이 이 특정 조합(GB10 sm_121 × GDN)에 미튜닝됐기 때문** — 바로 CLAUDE.md 정체성("단일 장비 ×
특정 모델군 최적화")이 빛나는 지점이다. llama.cpp(B=1 13.7, 동일 niche에 잘 튜닝됨)가 vLLM보다 우리의 진짜
경쟁자이며, 우리 MTP spec(16.9~24.6)가 그 llama를 1.25~1.80× 추월한 것이 핵심 성과(STEP D).

**판정 보류(다음 결정):**
- 우리 batch 경로 스케일링 약함(per-seq attention 루프) → continuous batching/paged-KV + SB 커널 통합으로 축2
  강화 가능(큰 작업). vLLM이 이 niche서 약하므로 우선순위는 낮음(우리가 이미 앞섬).
- K self-tuning(축1 보강) — **결론: static K=3~4가 near-optimal, adaptive 불필요.** 고정 K 스윕(27B): code K=3 **23.2 t/s**(accept 3.00) / K=6 **23.5**(accept 5.29 — K↑로 accept↑ 확인되나 **t/s plateau**: draft chain overhead가 추가 accept 상쇄). prose K=3 **16.5**. adaptive K(accept EMA) 구현 시도했으나 ① 버그(d_tok memcpy가 가변 Kc 미반영 → mism 36/40) ② 고쳐도 static K=3과 동급(이득 없음) → **기각·revert**. 즉 prompt 무관 static K=3~4면 충분(code는 K=6도 동급).
