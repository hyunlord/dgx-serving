# 챕터8 STEP 0 — MTP 0% accept 해결 + 공정성 + fp16-scale de-risk (decode 대역폭 천장 돌파)

> 2026-06-30. decode ≈ bandwidth ÷ bytes/token. 대역폭 분자는 천장(gemv 72%≈llama) → bytes/token(분모) 줄여
> decode 추월. 이번 세션: MTP(곱셈자)의 0% 막힌 이력 해결 + fp16-scale(base 갭 lever) de-risk. **정식화는 다음 세션.**

## 1. MTP 0% accept — 근본원인 규명·해결 (de-risk 성공)

장기 미해결 "Python 재현 0% accept" → **근본원인 = Qwen3.5 RMSNorm 식 불일치.**
- `transformers Qwen3_5RMSNorm`(modeling_qwen3_5.py:736-749): `output = x*rsqrt(mean(x²)+eps) * (1.0 + weight)`
  (Gemma식, weight=0 초기화 델타). 과거 재현(gb10 mtp_v5 등)은 `* weight`(직접) — **`(1.0+)` 누락** →
  enorm/hnorm/q/k/post/head 모든 norm 틀림 → 0%. (`Qwen3_5RMSNormGated`(GDN gate)는 plain weight — 두 컨벤션 공존.)
- 검증(오차0): 수정 rms(pre_norm_hidden, model.norm.weight) @ EMBED.T == model.logits **max-abs 0.000e+00**
  (tie_word_embeddings=True). authoritative 소스 대조(추측 아님).
- **accept: 0% → code 63% / prose 21%** (콘텐츠 의존, MTP head 건강 작동). 그래프는 처음부터 llama.cpp
  (qwen35.cpp:488-643)와 단계별 일치였음 — 유일 버그가 RMS식.

## 2. 공정성 — MTP는 fundamental 추월이 아님 (정직 좌표)

- **측정:** 표준 llama 27b Q4_0 GGUF에 **MTP head 0개**(nextn 텐서 없음) → 우리 비교 baseline decode 13.7은
  **MTP-OFF**. (27b HF·0.8b GGUF엔 head 있음 — 우리·llama 둘 다 통합 가능.)
- **분석:** MTP speedup ≈ (1+accept)는 **모델-속성 곱셈자**(accept는 head 속성, 엔진 무관) → 양쪽 똑같이 곱함
  → base decode 갭(우리 10.1 vs llama 13.7 = 1.36×) **안 닫음**. 공정(둘 다 MTP-on): llama 22 vs 우리 16 = 여전히 뒤짐.
- **실용 win은 있음:** 표준 llama 배포본 MTP 미탑재 → 우리가 켜면 실제 llama(13.7)보다 빠름(코드 16). "기법 매칭 +
  실용 win + 2nd 기둥"이지 fundamental 추월 아님.
- (llama MTP 직접 t/s 측정 시도 → llama-cli/bench가 GB10 이 세션서 행, ollama serve GPU 경합. 곱셈자 결론은
  accept=모델속성이라 견고.)

## 3. base 갭의 진짜 원인 = bytes/token → fp16-scale de-risk (성공)

gemv 효율은 이미 72%≈llama인데 decode t/s 1.36× 느림 = **같은 효율로 더 많은 바이트.**
- 산수: 우리 W4A8 weight = int4(4b) + **fp32 scale/32(=1b) = 5 b/weight**. llama Q4_0 = 4b + **fp16 d/32(=0.5b)
  = 4.5 b/weight**. → 우리 1.11× 무거움(scale이 트래픽 ~20%).
- **fp16-scale de-risk(`tools/ncu_gemv_fp16s.cu`):** scale fp32→fp16.
  - 정확성: fp16 vs fp32 scale rel **9.3e-4**(gate)/1.4e-3(down) — int4 quant 1.5e-2 대비 무시(llama도 fp16).
  - t/s: gate **1.109×**, down **1.126×** (격리라 L2로 과소평가 가능 — 실 e2e ≥).
- fp16시 4.5 b/weight = **llama 포맷 동일 = 핸디캡 제거.** gemv 91%에 → e2e decode ~1.10×, base 갭 1.36→~1.23×.

## 4. 다음 세션 (큰 빌드 — 신선한 세션서)

**A. fp16-scale 정식화 (base 갭 직접):** `extract_weights_qwen35.py` scale fp32→fp16 + `gemv_w4a8.cuh`·
`gemm_mmq.cuh` fp16 읽기 + 27b 재추출 + e2e bench. 게이트: 오차0(==gemv ref) + argmax/golden + 결정성 +
Sq일관 + decode t/s(base 갭 닫나).
**B. MTP CUDA 통합 (base 위에 곱셈자):** draft→verify(Sq=K, B 커널 Sq일관)→accept. fp16로 base 닫은 뒤 MTP 곱하면
둘 다 효과. 게이트 동일 + accept율 e2e.
**C. 잔여:** 작은 projection grid-starve(occ 40%) split-K(GDN k-split식). RMSNorm (1+w) — 엔진 다른 norm도
같은 식 쓰는지 점검(본체는 argmax/golden 통과하나 확인 가치).

de-risk 도구: `tools/ncu_gemv_fp16s.cu`. Python MTP 수정본: scratchpad(미커밋). 메모리:
[[mtp-rmsnorm-1plus-w-root-cause]], [[perf-baseline-vs-llamacpp-vllm]].
