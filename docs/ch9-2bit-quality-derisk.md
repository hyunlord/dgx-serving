# 챕터9 — 2bit 품질 de-risk → NO-GO (feasible 블록단위 2bit는 품질 붕괴)

> "극한"의 마지막 축 = bytes(②). base는 bandwidth-bound 확정(85% matmul) → 2bit가 직결 base lever(1.8× byte↓).
> 단 게이트는 품질. de-risk(reconstruction relerr, forward 불요 = GDN fla·108GB 회피): **순진 2bit 붕괴, 최선 scheme도
> q4의 4× — feasible 블록단위 2bit는 byte win과 품질을 동시에 못 줌. int2 커널 풀빌드 NO-GO.**

## 측정 — 실제 27B weight reconstruction relerr (`tools/probe_q2_relerr.py`)
relerr = ||W - dequant(quant(W))|| / ||W||. 대표 6개 big matrix(GDN in_proj, FFN gate/down, attn q/o, lm_head) 평균:

| scheme | relerr | ×q4 | 유효 bit/w(fp16 scale) | byte↓ vs q4 |
|--------|--------|-----|------------------------|-------------|
| q4-sym b32 (현 엔진) | **0.098** | 1.0× | 4.5 | — |
| q2-sym b32 (naive, 엔진 스킴 2bit) | **0.648** | **6.6×** | 2.5 | 1.80× |
| q2-minmax b32 (best 2bit scheme) | 0.391 | 4.0× | 3.0 | 1.50× |
| q2-minmax b16 (finer block) | 0.325 | 3.3× | 4.0 | 1.12× |

## 판정 — NO-GO (feasible 2bit)
- **순진 2bit = relerr 65%** (q4의 6.6×) = 가중치 거의 파괴. end-to-end 붕괴 확실(64층 누적).
- **최선 단순 scheme(minmax) = 39%** (q4의 4×) at 1.5× byte win. q4의 10%(end-to-end OK)와 차원이 다름 — 40% 가중치
  오차는 64층 통과 시 출력 붕괴 거의 확실.
- **품질 회복(finer block) = byte win 소멸**: b16 minmax = 33% relerr인데 **4 bit/w = q4와 동급**(byte 이득 1.12× 무의미).
- **본질: 2bit(4레벨)는 coarse — byte win(1.8×)과 q4급 품질을 동시에 못 준다.** bits 더하면(scale/outlier/finer) win 사라짐.

## 정직 결론 — "극한"은 ~21 t/s (세 축 다 구조 천장)
| 축 | 상태 | 한계 사유 |
|----|------|-----------|
| ① bw 효율 | 우리 63%, gemv 72%=llama급 | 커널 천장(챕터7 micro-lever 소진, int4 unpack 구조) |
| ② bytes | 4bit | **2bit 품질 붕괴(이 de-risk) — 4bit가 품질-최소.** 그 이하 영역 밖 |
| ③ accept | 단일 MTP 2.25~2.9 | 단일 head 천장(STEP6). draft 모델=학습 영역 |

→ **decode ~21 t/s가 우리 장비×모델의 커널/양자화 구조 천장.** 세 축 다 닿음. 그 이상은 전부 **영역 밖**:
- 별도 draft 모델(accept 4-5) = 모델/학습.
- research-grade 2bit(QuIP#/AQLM = 벡터양자화+incoherence+fine-tune) = 모델/학습(블록단위 ≠, 거대 작업). FP4 = CUDA 13.0 toolkit 막힘.
- → 21 t/s = NVFP4-MTP 19.7/FP8+DFlash 18~20 넘음, llama 13.7 추월. DFlash int4 65~94는 위 영역-밖 lever들(draft 모델 등)의 곱.

de-risk가 int2 커널 풀빌드(거대)를 품질 벽 앞에서 차단 — STEP6 full-forking NO-GO, 챕터7 cp.async/FP4 기각과 동형(측정-우선).
