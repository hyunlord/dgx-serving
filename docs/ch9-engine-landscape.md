# 챕터9 엔진 지형도 — GB10 × Qwen3.5-27B (정직, "압도" 금지)

> 27B 서빙 좌표(5d02313) = 비교 기준선. **STEP E "vLLM 4.4 압도"=미튜닝 과장 교훈 → 이번은 "이겼다" 아니라 정직한 지형도.**
> 결론 선언: **raw 커널 throughput은 llama.cpp가 우리보다 ~1.4× 빠름**(챕터7 미봉 갭). 우리 niche = spec decode 단일 latency + W4A8 bit-identical + GDN.

## STEP A — 엔진 가용성 (실사)
| 엔진 | GB10 상태 | 측정 |
|---|---|---|
| **dgx-serving** (ours) | 네이티브 sm_120, W4A8 | ✓ 측정 |
| **llama.cpp-mtp** | 빌드됨, GB10 CUDA(compute 12.1 확인), Q4_0 GGUF | ✓ 측정 |
| **vLLM** | **미설치** — sm_121 FA/CUTLASS 미지원, custom build 필요(out-of-scope) | ✗ 미측정 |
| **SGLang** | **미설치** — 동일 sm_121 장벽 | ✗ 미측정 |

vLLM/SGLang 설치는 sm_121서 깊은 rabbit hole(경쟁엔진 설정=우리 범위 밖) → **미측정, 포럼 숫자를 2차 참고(라벨)**. 완벽 4-way 불가.

## STEP B — 축별 측정 (llama fresh, 우리 좌표; 양자화 Q4_0 ≈ W4A8 둘 다 4bit weight)
| 축 | dgx-serving (ours) | llama.cpp Q4_0 | 비교 |
|---|---|---|---|
| **단일 decode (non-spec)** | 11.9 t/s (base, gemv-MLP) | **13.83 t/s** (tg128) | **llama 1.16× 빠름** |
| **단일 decode (spec)** | **21.3 t/s** (tree+MTP q4) | 13.83 (non-spec) | ours 1.54× — 단 spec-vs-nonspec* |
| **prefill** | 656 t/s (chunk512) | **901 t/s** (pp512) | **llama 1.37× 빠름** |
| **batch decode B=16** (GDN template 전) | 65.9 t/s | **93.56 t/s** (S_TG) | llama 1.42× |
| **batch decode B=16** (GDN template 후, cda5ea9) | **84.9 shipping / 88.5 +split-K** (N=128) | **95.30 t/s** (재측정) | **llama 1.08~1.12× (near-parity!)** |
| **batch B=8** | ~53 t/s | 68.6 t/s | llama 1.3× |

**★★bf16 GDN state(7b7c39c, opt-in) — batch near-parity 1.02×**: GDN template 위에 state fp32→bf16(대역폭 반감, 커널 1.94×) → batch B=16 fp32 88.2→**bf16 93.3 vs llama 95.30 = 1.02×**. 전 경로 mism=0(batch/single/tree-spec accept 3.67 불변). arc batch 갭 **1.42(65.9)→1.1(GDN template)→1.02×(bf16 state)** — near-parity. opt-in(-DGDN_BF16_STATE, default fp32 회귀 0).

**★GDN template 재좌표(cda5ea9 이후)**: gdn_step_batch D-template화(커널 3.5×)로 **batch B=16 갭 1.42×→~1.1×(near-parity)**. 우리 84.9(shipping)/88.5(+split-K opt-in) vs llama 재측정 95.30(≈이전 93.56). GDN이 batch decode의 36.3%였고 그걸 3.5× 줄임 → batch 축 거의 따라잡음. caveat: W4A8 vs Q4_0(activation int8 vs fp16), 우리 context(4-132)<llama(128-256)라 약간 유리(실제 ~1.08-1.15×). single(1.16×)·prefill(1.38×)은 불변(GDN 무관 축). **서빙(연속배칭) llama-server 공정 매칭은 out-of-scope(Poisson/admission 차) — batch 축이 공정 비교.**

\* llama.cpp-mtp도 MTP spec 가능성(미측정) — 켜면 llama 단일도↑. 우리 21.3은 spec, llama 13.83은 표준 non-spec 기준선.

## 정직한 서술
- **raw 커널 throughput은 llama가 우리를 ~1.16~1.42× 앞섬**(단일 non-spec·prefill·batch 전부). 이유 = llama 커널이 더
  잘 튜닝됨(KTILE/stream-K), **챕터7서 우리가 못 좁힌 갭**(L1/TEX 효율은 맞췄으나 그 위 1.37×). 이건 **STEP E "vLLM 압도"의
  정반대 — llama가 우리를 이김, 정직히 인정.**
- **우리 강점(niche)**: ① **speculative decode**(tree+MTP)로 단일 latency 21.3 > llama non-spec 13.83(1.54×) — 단 llama-mtp
  spec 켜면 갭 줄 수(미측정 caveat) ② **W4A8 + bit-identical 정확성**(spec==plain, batch==iso, mism=0 검증 — 우리만의 보장)
  ③ **GDN(linear attn) 네이티브**(Qwen3.5 하이브리드 특화; vLLM/SGLang은 GDN 미최적) ④ **연속배칭 서빙 스택**(admission
  guard+chunked prefill+long-rope, 5-STEP).
- **못 미치는 영역(정직)**: 포럼 DFlash int4 code **65~94 t/s**(단일) = **별도 학습 draft 모델 + repository-completion 조건**
  (커널 범위 밖, [[honest-perf]]서 규명). 우리 커널 최적화로 도달 불가 = 모델/학습 영역.
- **vLLM/SGLang**: 미측정(sm_121 미튜닝). 포럼 2차 참고: vLLM FP8 ~12~14(GB10 튜닝 시), NVFP4-MTP 19.7. **"우리가 vLLM
  이김" 주장 안 함**(STEP E 교훈) — 그들 GB10 셋업/튜닝은 별도 작업.

## STEP C — 지형도 결론
- **"이겼다" 아님. 지형:** raw 커널은 llama가 앞서고(1.4×, 챕터7 갭), 우리는 **spec decode 단일 latency + bit-identical 정확성
  + GDN 네이티브 + 연속배칭 서빙 스택**이 차별점. GB10×Qwen3.5(GDN 하이브리드)라는 **좁은 niche**서 통합 서빙(정확성 보장 +
  긴 context + 연속배칭)이 강점이지, 범용 throughput 우위 아님.
- **정직 caveat**: 양자화 Q4_0 vs W4A8(둘 다 4bit weight, 품질 근사 매칭이나 정확 동일 아님); 우리 spec vs llama non-spec
  (llama-mtp spec 미측정); vLLM/SGLang 미측정(sm_121); 단일-seed 서빙 좌표. **이 표 = 향후 vLLM(GB10 튜닝 시)/Gemma4 비교
  기준선.**

## 측정 로그 (재현)
- llama: `llama-bench -m Qwen3.5-27B-Q4_0.gguf -ngl 999 -p 512 -n 128` → pp512 901·tg128 13.83; `llama-batched-bench ... -npl 1,2,4,8,16` → B16 S_TG 93.56.
- ours: probe_chunked_prefill FULLONLY(27B) → 656 t/s; 단일 21.3/base 11.9(문서); batch 65.9(bench_batch, 2959c17).
