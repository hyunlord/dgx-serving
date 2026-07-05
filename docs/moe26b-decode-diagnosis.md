# 26B-A4B MoE decode "24% 미스터리" 진단 — measured가 전제를 교정

> 가설: 3-way에서 26B single 14.7 t/s = roofline ~24%로 비정상 저조 → 26B decode 커널이 튜닝 안 됨(malloc churn 등).
> ★measured 결론: **24%는 raw 커널이 아니라 서빙 스케줄러 좌표.** raw 커널 single decode는 이미 **36%**(정상 MoE 범위).

## STEP 1 — 측정 인프라 (bench에 26B MoE decode 추가)

기존 `bench_gemma4`는 gemma4_moe.cu 미링크 → 26B MoE single decode raw 측정 불가(3-way 14.7은 HTTP 서버=스케줄러 포함).
`tools/bench_gemma4.cu`에 `is_moe`(num_experts>0) 분기 추가 + `gemma4_moe.cu` 링크 + active-bytes roofline 계산.
31B(num_experts=0)는 else로 기존 경로 불변(additive·무회귀).

## STEP 2 — baseline (measured, 26B W4A8, warm, ngen=64)

| 항목 | measured | 비고 |
|---|---|---|
| **single decode (W4A8)** | **45.43 t/s** | roofline 126.0의 **36.1%** |
| single decode (q4) | 41.50 t/s | 32.9% |
| batch B1/B8/B16 aggr | 45.8 / 156 / 179 t/s | amortization 작동 |
| prefill (B=1, 256tok) | 171.55 t/s | 참고치 |
| **roofline 상한** | **126.0 t/s** | active-bytes÷255.8GB/s (dense∥MoE 병렬 포함) |

★roofline 상한이 앞선 추정(50+)이 아니라 **126.0**: 26B는 dense FFN + 8 expert **병렬**(dense∥MoE 제3 구조)이라
active-bytes가 더 큼(H2816·dense FFN2112·moe 8×704). 정확 계산은 bench `moe_single_roofline_ts`.

## ★핵심 반전 (measured — 프롬프트 전제 교정)

- **3-way HTTP 14.7 t/s ≠ raw 커널.** raw single = **45.4 t/s**. **14.7 / 45.4 = 서빙이 raw의 32%.**
- 즉 "3배가 테이블에 남았다"의 진짜 범인 = **서빙 스케줄러**(HTTP 온라인 배칭 경로), 커널 아님.
  → 35b MoE에서 이미 measured한 패턴과 **동일**(서빙 63 ≪ raw 195 = chunked prefill이 decode 차단·단순 continuous 스케줄러).
- raw 36%는 27B dense 62.5%보다 낮으나 **MoE 구조적**(토큰당 640 expert-gemv launch·tiny 704폭 matmul·dense∥MoE 이중 MLP) —
  이미 알려진 MoE 낮은 달성률(24.2~29.8% 기록과 정합, 여기선 fused로 36%). **24% 아님**.

## STEP B — 소거 후보 산술 판정 (measured-먼저)

프롬프트가 지목한 후보들을 measured 산술로 사전 판정:

| 후보 | 산술 | 판정 |
|---|---|---|
| **persistent scratch**(malloc/free churn) | single 22ms/token(1408ms÷64) 중 22 malloc + 22 free ≈ 220us/token = **~1%** | 24% lever 아님(예상 소폭). 27B가 이미 쓰는 `ws()` 이식은 가능하나 이득 미미 예상 |
| dense∥MoE 중복 read | roofline 이미 병렬 포함(126)·달성률 36%라 과잉 read 아님 | 범인 아님 |
| 서빙 스케줄러 | raw 45.4 → HTTP 14.7 = **-68%** | ★진짜 범인(별도 트랙) |

★"malloc churn이 24% 범인" 가설은 **measured 산술로 약화**(single 커널 22ms 중 malloc ~1%). raw가 이미 36%면
malloc은 큰 lever 아님. (batch에선 GPU 오버랩으로 이득 0이었고, single도 산술상 ~1% — persistent scratch는
정확성 위험 대비 이득 낮아 보류. 원하면 measured 확인 가능하나 24% 답 아님.)

## 판정 / 다음

- **"24% 미스터리" 해소(measured): raw 커널은 36%(MoE 정상), 24%(14.7)는 서빙 스케줄러 오버헤드.**
  커널 튜닝(persistent scratch 등)은 24% lever가 아님 — measured가 프롬프트 전제를 교정.
- **진짜 lever = 서빙 스케줄러**(raw 45.4 → HTTP 14.7, -68%): chunked prefill이 decode 차단·단순 continuous
  스케줄러. 35b MoE와 동일 패턴 → serving-scheduler 트랙(별도·이미 알려진 갭).
- raw 36% → 더 올리려면 MoE 구조적 launch 감축(CUDA graph·device-fused 심화) — 별도.
- 산출물: bench_gemma4 26B MoE 측정 경로(measured 인프라)·이 문서. 코드 정확성 변경 0(측정만·31B 무회귀).

## 후속 규명 — 서빙 갭의 정확한 메커니즘 = online 루프 Bmax 전폭 (active-width로 해소)

위 진단은 "서빙 경로가 범인"까지 맞았으나 메커니즘은 chunked prefill로 추정했다. 후속 측정으로 정확히 규명:

**3점 격리(measured, 26b W4A8 warm decode-only tpot)**:
| 측정점 | t/s | 의미 |
|---|---|---|
| raw bench (1슬롯) | 45.4 | 커널 상한 |
| daemon-direct (python IPC + serving_online.h 루프) | 18.2 | ★여기서 60% 손실 |
| HTTP full (+ 서버 threading) | 18.4 | HTTP·python 계층 ≈0 |

- daemon-direct(18.2) ≈ HTTP(18.4) → HTTP/python 오버헤드 무시. 갭 전부 serving_online.h 루프.
- **Bmax=1 재측정 = 43.1**(raw의 95%) → 확정: online 루프가 **single 요청도 Bmax=8 전폭 decode**(free 7슬롯 더미) → batch-8 per-slot 처리량(156/8≈19.5)만 나옴. TTFT/prefill/chunked 아님(긴 생성 TTFT 2.9%).
- detok 격리: incremental HF decode 0.013 ms/token(무시).

**수정(active-width)**: serving_online.h decode 폭 = 최대 active 슬롯+1(W). single(slot0만)=W1=1슬롯 커널=raw.
gemma4 backend에 `s->B=Bmax` 추가(qwen 대칭 — 미설정 시 alloc Bmax 고정으로 stale 슬롯 illegal access).
- **측정: single-stream 18.2 → 45.76 t/s(raw 완전 회복, 2.5×)**. HTTP 전체회계(브라우저) 18 → 43(긴 생성).
- 정확성 불변: 각 슬롯 KV/state 독립 → active row는 W와 무관하게 bit-id(test_moe26b_online PASS).
- batch(다 active)면 W=Bmax(무영향). qwen daemon도 s->B=Bmax 이미 설정이라 재빌드만으로 적용.

★교훈: "서빙 스케줄러"라는 첫 추정(chunked prefill)은 메커니즘 오귀속이었고, **3점 격리 + Bmax 스윕**이 진짜 원인(고정폭 decode)을 규명. 측정-먼저가 추정을 교정한 사례.
