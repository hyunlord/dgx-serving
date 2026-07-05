# per-(N,K) 형상 dispatch — 격리 표는 차이 실재, in-engine 이득은 무시가능 (measured)

> 256×64 in-engine +5% 후속. 감쇠(14.5→5%) 원인 ② = 다중 (N,K)(엔진 mmq는 층당 여러 matmul, 256×64는 FFN만 최적).
> 이번 = ②를 per-(N,K) 형상 dispatch로 직접 공략. ★결론: **격리 표는 (N,K)별 최적 형상이 실제로 다름(확인) —
> 그러나 per-NK의 in-engine 이득은 단일 256×64 대비 무시가능(+0.1~0.3%)**. FFN matmul이 지배·단일이 이미 그걸 잡음.
> mmq 형상튜닝은 ~+5% prefill서 포화(Amdahl +11% 미달 — 달성가능 형상-win이 믹스로 제한). bit-id 불변.

## STEP 0 — 27b prefill matmul (N,K) 인벤토리

| matmul | N | K | mmq FLOP 비중 |
|---|---|---|---|
| qkv | 8192 | 5120 | 12% |
| o | 5120 | 6144 | 9% |
| gate/up | 17408 | 5120 | 각 25% (2×) |
| down | 5120 | 17408 | 25% |

## STEP A — (N,K)별 형상 스윕 → 표 (M=512, TFLOP/s)

| matmul (N,K) | 128×128 | 256×64 | 최적 |
|---|---|---|---|
| qkv (8192,5120) | **47.4** | 45.1 | **128×128** |
| o (5120,6144) | **50.8** | 44.9 | **128×128** |
| gate/up (17408,5120) | 42.1 | **48.6** | **256×64**(+15%) |
| down (5120,17408) | 46.1 | 46.5 | 256×64(동급·noise) |

★**sm_121 형상 표(신규 자산)**: **256×64는 large-N/K FFN(gate/up/down)만 최적. qkv/o는 128×128이 더 빠름**(256×64는
qkv −5%·o −12% 손해). = 감쇠 원인 ②의 정확한 실체(단일 256×64가 qkv/o를 최적서 끌어냄). heuristic: `N≥16384||K≥16384`
→ 256×64, else 128×128(4개 (N,K) 전부 스윕-최적 매핑·FFN-스케일 일반화).

## STEP B — per-(N,K) dispatch 배선 + in-engine A/B

- `launch_gemm_mmq_w4a8_auto` MMQ256=2: (N,K) heuristic 룩업(FFN→256×64·qkv/o→128×128). MMQ256=1=전 256×64·0=128×128.
- ★bit-id: MMQ256=2 == baseline(test_decode argmax ALL match·PASS) — 형상 무관 계산 불변.
- ★in-engine A/B(27b prefill, warm·back-to-back, prefill tok/s):

| ctx | 0: 128×128 | 1: 단일 256×64 | 2: per-NK | per-NK vs 단일 |
|---|---|---|---|---|
| 512 | 678.8 | 706.6 (+4.1%) | 707.4 (+4.2%) | **+0.1%** |
| 2048 | 646.9 | 680.2 (+5.1%) | 682.1 (+5.4%) | **+0.3%** |

## STEP C — 판정 (measured)

★**per-(N,K)는 단일 256×64 대비 in-engine 이득 무시가능(+0.1~0.3%)**:
- **왜(measured)**: mmq FLOP의 ~75%가 FFN(gate/up/down = 256×64 최적) → 단일 256×64가 이미 지배 부분을 잡음.
  per-NK가 개선하는 qkv/o(128×128 최적)는 mmq의 ~21% + 256×64 페널티도 작음(qkv −5%·o −12%의 작은 base) →
  TTFT 복구 ~0.1-0.3%. 산술 예측(mmq 층시간 단일 7387→per-NK 7256=−1.8% mmq)이 TTFT선 더 희석(비-mmq + qkv/o 소).
- **감쇠 ②는 실재하나 영향 작음**: 격리 표는 (N,K)별 최적 다름(확인) but FLOP 믹스가 FFN-지배라 단일로 충분.
- **mmq 형상튜닝 포화 ~+5%**: 단일·per-NK 모두 +4-5%. Amdahl 상한(+11%) 미달 = mmq 자체 형상-win이 믹스로 제한
  (gate/up만 +15%·qkv/o/down은 형상 거의 무관) → 완전 최적화해도 mmq +8% 남짓 → TTFT +5%대.

### 정직 결론 + 다음
- **per-NK는 27b prefill mmq엔 불필요**(단일 256×64 대비 +0.1-0.3%, 무의미) — 복잡도 대비 이득 없음. 단일 256×64로 충분.
- **단 표/방법은 자산**: sm_121 (N,K)→형상 표는 (a)FLOP 믹스가 다른 커널(gemm_batch/moe_fused — matmul 크기 분포 다름)
  (b)다른 모델서 유용. per-NK 인프라(heuristic dispatch)는 그때 재사용.
- **overclaim 없음**: per-NK 이득 무의미가 measured. mmq 형상 최적화의 실 엔진 천장 = ~+5%(단일 256×64로 달성).
- 현 opt-in 유지(MMQ256, 기본 0). 256×64 단일이 실용 최적(prefill +5%·bit-id).

## 산출물
- `kernels/gemm_mmq_tmpl.cuh`(launch_gemm_mmq_w4a8_auto MMQ256=2 per-NK heuristic). docs/per-nk-shape-dispatch.md(표+판정).
