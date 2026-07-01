# 챕터9 실부하 서빙 벤치 — production scheduler 지표 (연속 vs 정적, 부하 스윕)

> production scheduler(c8dd3df: admission guard + slot 재사용 + 큐, iso mism=0) 위에 **제대로 된 실부하 측정**.
> R=6 검증부하 아니라 **Poisson 요청 스트림 + 가변 길이 + warm-up 버림 + 정적 대비 + 부하 스윕**.
> 하네스: `tools/bench_serve.cu` (`make tools/bench_serve`; [w][g][uq][Bmax][nreq][seed][λ...]).

## 하네스 설계 (제대로)
- **요청 스트림**: Poisson 도착(λ req/s, 지수 inter-arrival) + 가변 프롬프트 U[8,32] + **bimodal gen**(short U[16,48] 70% / long U[96,192] 30%, 평균 ~66토큰). seed 고정 trace(재현).
- **sim clock**: 실제 스텝 wall-clock(std::chrono) 누적. idle면 다음 도착까지 fast-forward(GPU 놀림 정직 반영).
- **warm-up**: 완료 순 앞 20% 제외(cold·batch 덜 참 구간 버림), steady state만 집계.
- **정적 baseline**: 같은 trace, 배치를 채워 launch하면 **전원 완료까지 slot 붙잡음**(head-of-line — 짧은 요청도 긴 것 끝까지 대기). 연속과 back-to-back 동일 프로세스 → 타이밍 bias 상쇄, **비율이 robust 신호**.
- **측정 제외**: 토큰 선택(argmax/sampling) + logit readout — GPU 병렬 op라 직교(forward가 지배). gen 길이는 trace 사전결정.
- **avgB** = 평균 active 배치 = 활용 proxy(dead-slot 유휴 포착). **maxQ** = 최대 큐 깊이 = 부하 지표.

## 결과 1 — 0.8b W4A8, Bmax=8 (부하 스윕)
| mode | λ | req/s | tok/s | e2e p50 | e2e p95 | e2e p99 | ttft p95 | tpot p95 | avgB | maxQ |
|------|---|-------|-------|---------|---------|---------|----------|----------|------|------|
| cont | 1 | 0.835 | 55.2 | 0.56 | 1.81 | 2.12 | 0.30 | 0.010 | 1.38 | 0 |
| stat | 1 | 0.825 | 54.5 | 4.36 | 12.81 | 14.63 | 12.43 | 0.011 | 3.97 | 0 |
| cont | 4 | 3.24 | 214.2 | 0.58 | 1.93 | 2.19 | 0.31 | 0.010 | 2.87 | 0 |
| stat | 4 | 3.07 | 203.3 | 2.09 | 4.32 | 4.57 | 3.33 | 0.011 | 3.97 | 3 |
| **cont** | **12** | **7.09** | **479.1** | 1.95 | **3.01** | 3.42 | 1.99 | 0.012 | 6.86 | 18 |
| stat | 12 | 3.78 | 250.0 | 7.88 | 10.10 | 10.83 | 9.65 | 0.011 | 3.97 | 40 |
| cont | 32 | 6.53 | 455.1 | 4.49 | 6.31 | 6.94 | 5.68 | 0.012 | 7.23 | 50 |
| stat | 32 | 3.64 | 240.4 | 10.55 | 14.39 | 15.32 | 13.97 | 0.011 | 3.97 | 56 |

- **포화**: 연속 ~479 tok/s(7.1 req/s) @λ=12, 정적 ~250 tok/s(3.8 req/s). **연속 1.9× throughput.**
- **꼬리 지연**(λ=12): 연속 e2e p95 3.0s vs 정적 10.1s = **3.4×↓**. TTFT p95 2.0 vs 9.6s.
- **avgB**: 연속 1.4→7.2(부하 따라 slot 채움), 정적 **3.97 고정**(짧은 요청 끝나도 배치 붙잡음=dead-slot 유휴 = 정적 비효율의 기전).

## 결과 2 — 27B W4A8, Bmax=8 (실제 서빙 모델)
| mode | λ | req/s | tok/s | e2e p50 | e2e p95 | ttft p95 | tpot p95 | avgB |
|------|---|-------|-------|---------|---------|----------|----------|------|
| cont | 0.3 | 0.202 | 15.8 | 8.52 | 28.8 | 4.7 | 0.150 | 3.32 |
| stat | 0.3 | 0.211 | 15.5 | 29.6 | 58.2 | 36.8 | 0.153 | 3.90 |
| cont | 0.8 | 0.364 | 30.2 | 16.5 | 33.9 | 16.4 | 0.154 | 6.20 |
| stat | 0.8 | 0.228 | 16.8 | 64.9 | 97.1 | 81.3 | 0.153 | 3.90 |
| **cont** | **1.5** | **0.393** | **32.6** | 28.8 | 50.4 | 35.1 | 0.153 | 6.72 |
| stat | 1.5 | 0.228 | 16.8 | 75.6 | 115.7 | 99.3 | 0.153 | 3.90 |
| cont | 3 | 0.397 | 32.9 | 35.9 | 63.4 | 48.7 | 0.153 | 6.78 |
| stat | 3 | 0.228 | 16.8 | 83.5 | 126.5 | 109.6 | 0.153 | 3.90 |

- **포화**: 연속 ~32.9 tok/s(0.40 req/s) @λ≥1.5, 정적 ~16.8 tok/s(0.228 req/s). **연속 1.96× throughput.**
- **꼬리**: 연속 e2e p95 50s vs 정적 116s(2.3×↓). 저부하(λ=0.3)서도 정적 TTFT 36.8s(배치 채우려 대기) vs 연속 4.7s.
- TPOT ~0.153s(슬롯당 ~6.6 tok/s) 양 모드 동일 — **per-token decode 속도는 배칭무관, 이득은 순전히 배칭 효율(활용·꼬리).**

## 결과 3 — Bmax=16 collapse (연속배칭이 B>8 못 쓰는 이유, 직접 측정)
0.8b λ=32: **Bmax=16 → 416 tok/s, TPOT 0.0223s, avgB 11.45** vs **Bmax=8 → 455 tok/s, TPOT 0.0115s, avgB 7.23.**
- Bmax=16이 avgB 더 높은데 throughput **더 낮고 TPOT 2배** → 9~15 active B가 `gemm_batch_w4a8` 비특수화(특수화 {1,2,4,8,16}만)라 **weight B× 재적재**. memory-bound서 weight 재적재 = TPOT 2배 = 정확한 예측.
- **함의: 연속배칭은 active B가 매 스텝 요동 → B≤8 SB 영역(런타임 S 1적재, 매끄러움)이 올바른 운영점.** 정적 B=16 peak(bench_batch 65.9 tok/s, 전 16슬롯 상주)은 **연속모드서 도달 불가** — gemv_w4a8_sb SMAX 8→16 확장이 전제(별도 커널 과제).

## STEP C — 정직 좌표
- **연속 vs 정적 이득**: throughput **~1.9~2.0×**(0.8b·27B 일치), 꼬리 지연 **2.3~3.4×↓**, TTFT 압도(정적은 batch-fill+head-of-line 대기). 기전 = 연속은 slot 즉시 재사용(avgB↑) + 짧은 요청이 긴 것에 안 막힘.
- **포화점**: Bmax=8서 0.8b ~479 tok/s(λ~12), 27B ~33 tok/s(λ~1.5). 그 이상 λ는 큐만 쌓임(maxQ↑), throughput 평탄+미세 저하.
- **admission guard**: 예산 내 Bmax 사전할당 → 고부하는 큐(host, tiny)만 증가, KV oversubscribe 없음 = 시스템 다운 없이 안정(스트레스는 c8dd3df서 Bmax=5000·maxlen=500000 거부 확인).
- **단일 latency(21.3 t/s)와 별개 축** — 이건 throughput(동시요청 amortize) 축.
- **vLLM 비교 보류**: GB10 튜닝 필요(STEP E 미튜닝 4.4 교훈). 공정 비교하려면 GB10-튜닝 vLLM(spec/NVFP4/fla) 별도 셋업.
- **정직 caveat**: single-seed trace(비율이 robust 신호, 절대 tok/s는 warm-up/타이밍 변동); 토큰선택 제외(forward 지배); 1토큰/step prefill(긴 프롬프트 TTFT 부풀림 — chunked prefill 미구현). Bmax=8 = SB 매끄러운 영역(B>8 collapse).

## 남은 것
- gemv_w4a8_sb SMAX 8→16 확장 → 연속모드 B=16 활용(정적 peak 65.9 접근). chunked prefill(TTFT↓). 우선순위 스케줄링/preemption.
