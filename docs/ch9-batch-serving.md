# 챕터9 batch serving — odd-batch 붕괴 수정 (SB 커널 통합)

> 단일 decode 21.3 t/s = latency 천장(커널·무학습). batch = 별개 축(throughput, memory-bound라 weight 1운반=B토큰).
> STEP A measure: odd/non-pow2 batch 붕괴 + B=16 peak 확인. STEP B: SB 커널로 odd 붕괴 수정(bit-exact).

## STEP A — batch throughput 곡선 (27B fp16s W4A8, `tools/bench_batch.cu`)
gemm_batch_w4a8 경로(현). B별 throughput(B·N/t):

| B | baseline t/s | vs B=1 | 비고 |
|---|--------------|--------|------|
| 1 | 12.5 | 1.0× | |
| 2 | 23.0 | 1.84× | 특수화 |
| **3** | **14.9** | 1.19× | **붕괴** |
| 4 | 40.3 | 3.22× | 특수화 |
| **5/6/7** | **15.4/15.5/15.5** | ~1.24× | **붕괴** |
| 8 | 59.3 | 4.74× | 특수화 |
| 16 | **65.4** | 5.24× | 특수화(peak) |
| **12/24/32** | **15.6/15.7/15.7** | ~1.25× | **붕괴** |

**원인(코드):** `launch_gemm_batch_w4a8_t`는 B∈{1,2,4,8,16}만 특수 커널(weight 1적재), 나머지(3,5,6,7,12,24,32)는
**default = B번 B=1 호출 = weight B× 재적재** → throughput ≈ B=1로 붕괴. "B=16 plateau"는 16이 마지막 특수화라서.

## STEP B — SB 커널 통합 (odd 붕괴 수정)
`gemm_batch_w4a8<B>`와 `gemv_w4a8_sb<SMAX=8>`는 **동일 구조**(1 warp/o, weight 1적재, B accumulator DP4A) — 차이는
gemm_batch가 컴파일타임 B(특수화만), SB는 런타임 S≤8(임의 1적재). qwen35_batch.cu: **B≤8 → SB**(odd 포함 1적재), B>8 → gemm_batch.

| B | baseline | **SB** | 수정 |
|---|----------|--------|------|
| 3 | 14.9 | **31.5** | 2.1× |
| 5 | 15.4 | **46.5** | 3.0× |
| 6 | 15.5 | **52.5** | 3.4× |
| 7 | 15.5 | **56.1** | 3.6× |
| (2,4,8,16) | 동일 | 동일 | (SB≈gemm_batch) |

→ **throughput 곡선 매끄러운 단조 증가**(11.8→65.9 @B=16). 임의 batch(continuous batching) 가능 — 이전엔 pow2만 효율.

**정확성:** `gemv_w4a8_sb == gemm_batch_w4a8 bit-exact 0.000e+00`(B=2~8 격리 검증) → batch 출력 불변. batch==single
**argmax ALL match**(토큰 동일). (W4A8 batch-vs-single float max-abs 0.32 = bf16 attn+W4A8 batch 양자화 계통오차,
fp32 0.009 대비 큼 — **pre-existing, 내 swap 무관**[MLP·SB 둘 다 원본과 bit-exact], 토큰 불변. 별도 W4A8-batch 정밀도 이슈.)

## 정직 좌표 (throughput 축)
- **batch throughput: B=1 12.5 → B=16 65.9 t/s (5.3×), 이제 odd 포함 매끄러움.** ~66서 plateau(weight B=16 amortize 후
  활성/compute-bound 근접 = 실용 throughput 천장 근접).
- 단일 latency(21.3)와 **별개 축** — throughput은 서빙(동시 요청) 지표.
- **vLLM 비교 보류(정직)**: STEP E vLLM 4.4는 미튜닝. 공정 비교엔 GB10-튜닝 vLLM 필요(미측정) — "압도" 주장 안 함([[honest-perf-coordinates-not-overtake]]).
- 남음: B>8(12,24,32) 아직 default-collapse(SB SMAX≤8) — SMAX 확장 or gemm_batch 케이스 추가. 단 plateau(~66)라 throughput 이득은 작음.

## 다음
- continuous batching(가변 B 요청 큐) — 이제 odd-B 효율적이라 가능. p95 latency 분포.
- B>8 SMAX 확장(필요시). attention batch는 이미 동작(per-seq KV).
