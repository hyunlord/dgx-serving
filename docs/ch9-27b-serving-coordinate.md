# 챕터9 27B 서빙 종합 좌표 — 모든 lever 통합 (최종)

> 서빙 축 5-STEP(production scheduler·실부하벤치·SB16·chunked prefill·long-rope)을 **27B에 모두 켜고** context 스윕.
> = 엔진 "어디까지 왔나" 최종 좌표 + 향후 vLLM/Gemma4 공정 비교 기준선. bench_serve(Poisson·warm-up·정적대비).

## 27B W4A8 실서빙 좌표 (모든 lever: 연속배칭 + SMAX16 + chunked prefill + long-rope ext)
| context | Bmax | λ | req/s | tok/s | TTFT p95 | e2e p95 | TPOT | avgB | per-seq KV |
|---------|------|---|-------|-------|----------|---------|------|------|-----------|
| **128** | 16 | 1 | 0.415 | **33.4** | **13.6s** | 53.8 | 0.33 | 10.4 | 180MB |
| **2K**  | 8  | 1 | 0.248 | 14.7 | 15.6s | 52.4 | 0.52 | 4.75 | 306MB |
| **8K**  | 4  | 0.5| 0.083 | 7.0 | 27.9s | 60.4 | 0.32 | 2.55 | 708MB |

(latency 초. 토큰선택/readout 제외=forward 지배. single-seed=비율 신호. Bmax는 KV 예산이 제한 — 긴 context일수록↓.)

## ctx=128 비교 (모든 lever의 효과, avg 프롬프트 ~68토큰, λ=1)
| mode | req/s | tok/s | TTFT p95 | e2e p95 | avgB |
|------|-------|-------|----------|---------|------|
| c1tok (1토큰 prefill) | 0.253 | 20.2 | 66.5s | 93.8 | 11.4 |
| **c-ch (chunked, 모든 lever)** | 0.415 | **33.4** | **13.6s** | 53.8 | 10.4 |
| stat (정적 배칭) | 0.161 | 12.9 | 163.5s | 195.5 | 7.9 |

- **chunked prefill**: c-ch vs c1tok = throughput **+65%**(20→33 tok/s), TTFT **4.9×↓**(66.5→13.6s). 27B서 1토큰 prefill은 브루탈(~68 forward/req).
- **연속 vs 정적**: c-ch vs stat = throughput **2.6×**(33.4 vs 12.9), TTFT **12×↓**(13.6 vs 163.5s). 0.8b(1.9~2.4×)와 일치, 27B 확인.
- **SMAX16 활성**: c-ch avgB 10.4(9~15 영역) — Bmax=16 요동 매끄러움(collapse 없음). stat avgB 7.9(dead-slot).

## 관찰 (context 스케일)
- **모든 lever 27B서 통합 동작 확인**: 연속배칭·SMAX16(avgB 10, ctx128)·chunked prefill(TTFT bounded)·long-rope(pos≥128 정확, 8K까지 crash 없음).
- **TTFT**: context 커져도 chunked가 bounded(13.6→15.6→27.9s). 1토큰 prefill이면 8K서 ~8192 forward = 수십분(불가) — chunked가 긴 context 서빙을 **가능하게** 함.
- **throughput**: context 커질수록↓(33→14.7→7.0 tok/s) — ⒜ 긴 KV = decode당 attention 느림 ⒝ KV 예산이 Bmax 제한(128:16 → 8K:4)으로 동시성↓. 둘 다 근본적(memory-bound + unified 예산).
- **avgB(활용)**: 128=10.4(SMAX16 영역), 8K=2.55(KV-limited). 긴 context = 배칭 여지 작음(per-seq KV 큼).
- **admission guard**: per-seq 180MB→708MB(context 따라), Bmax×per-seq ≤ 0.7·free 정확 계산, 다운 없이 안정(8K도).

## 정직 좌표 (최종)
- **단일 decode(21.3 t/s)와 별개 축.** 이건 throughput(동시요청 amortize) 축 = ctx128 33.4 tok/s(aggregate, avgB 10).
- **caveat**: ⒜ single-seed(비율이 신호, 절대 tok/s 타이밍 변동) ⒝ 토큰선택/logit readout 제외(forward 지배) ⒞ context별 Bmax·λ 다름(KV 예산·부하 — 완전 apples-to-apples 아님, 각 행은 그 조건의 좌표) ⒟ static 비교는 ctx128만(긴 context static은 1토큰 prefill이라 스킵 — 이득은 ctx128서 2.6× 확인) ⒠ 정확성은 pos<128 HF골든 + pos≥128 rope 증명(1 ULP)+엔진 자기일관(test_longrope) — **절대 e2e >128 transformers 골든은 미측정(별개)**.
- **vLLM 비교 보류**: GB10 튜닝 필요(STEP E 미튜닝 4.4 교훈). 이 좌표 = 공정 비교 기준선(같은 하네스로 GB10-튜닝 vLLM 측정 시).
- **8K < attention 12K shared 벽**([[attn-12k-shared-wall]]): 8K 동작하나 native 262K엔 FA 확장 필수(전제).

## 서빙 축 5-STEP 요약 (챕터9)
production scheduler(iso mism=0, admission) → 실부하 벤치(연속 2×/꼬리 3×) → SB16(B9~15 collapse 제거 2×) →
chunked prefill(TTFT 4~14×↓, aggregate +10~65%) → long-rope(pos≥128 정확 1 ULP) → **27B 통합 좌표(ctx128 33 tok/s·TTFT 13.6s, 8K 7 tok/s·TTFT 27.9s)**.
- 자산: bench_serve(SKIP1TOK/SKIPSTAT env로 mode 선택), gen_rope_long.py, 5-STEP 커널/graph.
