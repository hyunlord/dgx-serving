# 챕터9 — base decode 손실 분해: 2bit 효과 + GDN lever 판정 (측정)

> 사용자 질문: base 57%(미포화)의 손실이 ① 대역폭(→2bit 직결) 인지 ② 비-대역폭(GDN serial/unpack/launch →2bit 무용) 인지.
> 둘 다 "base가 대역폭-bound냐"로 귀결. **측정 결론: bandwidth-bound 확정 → 2bit가 base lever, GDN은 lever 아님.**

## ① GB10 실용 대역폭 한계 (STREAM 실측)
microbench(4GB>L2, cudaEvent): **read-only 251 GB/s (92% of 273)**, copy R+W 224 (82%). fp32 lm_head gemv도 259(95%, 03a).
→ **weight-load(read-heavy) 실용 peak ≈ 251 GB/s.** GB10은 clean read서 포화 — "GB10이 peak 못 냄"이 아님. (사용자 추정 190-230보다 높음.)

## ② decode 손실 분해 (nsys, 27B W4A8 fp16-scale 현 상태)
| 커널 | % (decode-only ~보정) |
|------|------|
| **gemv_w4a8 (matmul weight-load)** | **~85%** |
| flash_attn (full-attn 16층) | ~6% |
| GDN 전체 (ksplit 1.4 + in_proj gemv_f32 2.1 + conv 0.3 + qknorm/decay) | **~4%** |
| rmsnorm | ~2% |
| (gemm_mmq 8.3%는 prefill, 제외) | |

- **decode = ~85% matmul(memory-bound), GDN serial ~4%, launch 무시(GPU 96% busy, ch7).** → **bandwidth-bound 확정.**
- **사용자 우려(손실이 GDN serial/unpack) 기각** — GDN 4%(이미 k-split 최적화, ch7 63ec59c). 손실은 matmul 자체가
  ~72% 대역폭(=llama 73%, ch7 FFN)으로 도는 것 = quantized gemv 구조적(int4 unpack+per-block scale+DP4A, llama와 동급).
- 우리 aggregate ~159 GB/s = read-peak 251의 **63%** (headroom 1.58× to 실용 peak). gemv 72%(FFN) > aggregate 63%
  (소-projection grid-starve + GDN/attn/norm이 적은 bytes 비효율로 끌어내림).

## 판정 — 2bit / GDN
- **2bit = 진짜 base lever (bandwidth-bound이므로).** bytes/token W4A8 14.4GB(int4 12.8+scale fp16 1.6) → W2 8.0GB
  (int2 6.4+scale 1.6 고정) = **1.8× 감소**(2배 아님, scale 고정). base 천장 17.5→**32 t/s**. 실현 base ~11→**15-20**
  (×1.4-1.8, int2 unpack이 72% util 일부 잠식). spec(×~1.9 amortize) → **~28-35 t/s** 잠재 (FP8+DFlash 18-20 넘고
  DFlash int4 65-94 접근하나 미달).
- **GDN 최적화 = lever 아님** (4%, 이미 k-split). 테이블에서 제외.
- **2bit의 게이트 = 속도 아니라 품질**: 2bit(4 레벨)는 거칠어 순진 양자화는 품질 붕괴 → importance-aware(GPTQ/AWQ,
  메모리 "q2 품질 목표") 필요 = research-grade·큰 작업·불확실. **de-risk: 순진 2bit 품질(vs fp32) + importance-aware
  회복 가능성을 작은 단위로 먼저**(int2 커널 풀빌드 전 — 모든 큰 빌드와 동일 패턴).

## 결론
base는 **bandwidth-bound**(85% matmul memory-bound). → 2bit가 유일한 실질 base lever(~1.5-1.8×, spec ~28-35 잠재),
단 **품질-게이트**(importance-aware quant, 큰 작업). GDN/launch는 lever 아님(측정 4%/무시). 다음 = 2bit 품질 de-risk
(착수 시) or 현 21 t/s 좌표 수용. 도구: tools/bw_stream.cu(GB10 대역폭 characterization).
