# 챕터9 — 단일 decode 정직 좌표: 18.8 vs 포럼 65~94 격차 분해 (측정)

> 사용자 지적: "18.8 = 천장"을 너무 빨리 매듭졌다. 내가 증명한 건 **단일 MTP head accept 천장(2.9)** 이지
> **decode 속도 천장**이 아니다. `decode t/s ≈ base × accept`인데 base(~10)를 고정으로 봤다. 재측정·정직 분해.

## 측정 (최적 bench.cu, harness 아님)
- **plain decode: 9.7 t/s** (TPOT 103.4ms, 27B W4A8 ctx512, cudaEvent). (harness 10.1은 per-token malloc 오버헤드.)
- **bytes/token = 16GB**: int4 weight 12.8GB + **scale fp32 3.2GB(=트래픽 20%)**. (du 측정.)
- **달성 대역폭 = 16GB / 103ms = 155 GB/s = DRAM peak(273)의 57%.**
- spec(tree leaf-alt + MTP q4): 18.8 t/s = accept 2.27의 1.94×.

## base 하드 천장 (물리 — 어떤 커널도 못 넘음)
27B **dense**(MoE 아님 — manifest에 experts 없음) W4A8 = 16GB/token. GB10 273 GB/s ÷ 16GB = **~17 t/s가 B=1 base 절대 천장.**
우리 9.7 = 57%. → **65~94 t/s는 27B-dense-W4A8-GB10 B=1서 base만으론 물리적으로 불가**(17 초과 못 함). speculation accept로만 가능:
65/17 = **accept 3.8 이상 필요.**

## 격차 분해 (18.8 → 65~94, 물리 기반)
`decode = base × accept(amortize)`. 65~94이 이 모델·HW라면:
- **base 항**: 우리 9.7 → 천장 17 = **1.75× headroom**. 단 lever 작음:
  - scale fp32→fp16: 트래픽 16→14.4GB(-10%) → ~10.7. **검증됨**(gemv_w4a8/sb가 `scale_dtype=1` __half 지원, 27B dump만 fp32). 쉬움.
  - matmul→DRAM peak: 커널 구조 한계(int4 unpack+DP4A+GDN 48층 serial). 챕터7 micro-lever 소진. ~25% non-matmul(GDN serial)도 k-split 이미 적용. **나머지는 hard.**
- **accept 항**: 우리 2.27~2.9(단일 MTP head, STEP6서 천장 2.9 증명). 4-5는 **별도 draft 모델**(Eagle/DFlash식) 필요 = 모델·학습 영역, 커널 아님.
- **곱해도 부족**: 우리 최대(base 17 × accept 2.9) ≈ 49. 65~94엔 미달 → **나머지는 측정조건**(다른 모델 변종·NVFP4·context·code-cherry — STEP E vLLM 미튜닝 오판과 동형 위험).

## 정직 결론
1. **"18.8 = 천장"은 과장이었다**(사용자 지적 맞음). 우리 자산 reachable: scale-fp16 + best-accept(2.86) ≈ **~25 t/s**.
2. **하지만 65~94도 단일 미발견 lever 아님** — base(커널-hard, 천장 17) × accept(draft 모델 = 범위 밖) × 측정조건의 곱.
3. **base는 우리가 llama보다 빠름**(챕터7: 우리 91% vs llama 72% of gemv ceiling) — base 격차는 DFlash가 우리보다 base가 빠르단 증거 없음(같은 273 대역폭 한계). 격차의 진짜 큰 항 = **accept(별도 draft 모델)** + **측정조건**.
4. batch serving은 **별개 축**(throughput) — 이 단일-decode 격차(latency) 안 풂.

## 미해결(측정 불가/범위 밖)
- 포럼 65~94의 정확 조건(모델·양자화·HW·context) = 외부 출처 필요. 27B-dense-W4A8-GB10이면 위 물리상 accept 3.8+ 필수.
- 별도 draft 모델 feasibility(accept 4-5) = 모델 자산/학습. 우리 q4/tree/verify 토대 재사용 가능성은 별도 de-risk 대상.
