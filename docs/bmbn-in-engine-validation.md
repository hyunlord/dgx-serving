# 256×64 in-engine 검증 — 격리 +14.5% → 엔진 +5% (survives, bit-id)

> BM/BN 형상 WIN(256×64 +14.5% 격리 벤치) 후속. ★X-shared 전례(격리 1.44×→엔진 bank conflict/occupancy 붕괴→
> net 1.0× 폐기)라 "격리≠in-engine"이 헌법. 이번 = 256×64 엔진 배선 + prefill 유지되나 measured 검증.
> ★결론: **엔진 +5% prefill(27b, ctx 512/2048, warm·back-to-back). bit-id 확정. X-shared 붕괴 아님(survives).**
> 격리 +14.5% → 엔진 +5%(감쇠: Amdahl + 엔진 mmq 다중 (N,K)). 실 엔진 이득이나 격리 헤드라인보단 작음(정직).

## 배선 (STEP 0/A)

- `launch_gemm_mmq_w4a8_auto`(gemm_mmq_tmpl.cuh): env `MMQ256=1`→256×64(tall), 기본 0→128×128(원본·무회귀).
- `qwen35_decode.cu` prefill mmq 2곳 → auto 배선(bench.cu B=1 경로). **다른 경로 불변**.
- ★**출력 bitwise 동일**: 256×64와 128×128은 같은 K-reduction 순서(타일 형상은 블록↔출력 매핑만 바꿈, 특정 (m,n)의
  K-누적 순서 불변) → 계산 불변.

## ★in-engine bit-id 검증 (STEP A)

`test_decode`(qwen35_decode 골든 경로, argmax+MEAN-rel):
| | logits max-abs | MEAN-rel | argmax | 판정 |
|---|---|---|---|---|
| MMQ256=0 (128×128) | 4.488e-02 | 9.264e-04 | ALL match | PASS |
| MMQ256=1 (256×64) | **4.488e-02** | **9.264e-04** | ALL match | PASS |

→ 마지막 자리까지 **완전 동일** = 256×64 엔진 출력 bitwise 동일. 정확성 불변 확정("빠른데 틀림" 0).

## ★in-engine perf A/B (STEP B — 유지되나)

27b prefill, warm, back-to-back (각 2회·2번째):
| ctx | 128×128 TTFT | 256×64 TTFT | prefill tok/s | 이득 |
|---|---|---|---|---|
| 512 | 756.4ms (676.9) | 719.3ms (711.8) | +34.9 | **+5.2%** |
| 2048 | 3162ms (647.6) | 3015ms (679.2) | +31.6 | **+4.9%** |

- **+5% prefill 유지**(양 ctx 일관·back-to-back). **X-shared식 붕괴 아님**(양수·occupancy 붕괴 없음 — 붕괴면 net≤0).
- **격리 +14.5% → 엔진 +5% 감쇠 이유**(measured 해석):
  1. **Amdahl**: mmq는 prefill의 ~78%(ch9 ⑩), 나머지(attention·norm·rope 등)는 형상 무관 → mmq +14.5%면 TTFT 상한 ~+11%.
  2. **다중 (N,K)**: 엔진 mmq는 층당 여러 matmul(qkv/o/gate/up/down = 각 다른 N,K) 호출. 256×64는 FFN-up(N=17408,
     격리 dims)만 최적 — 다른 matmul엔 덜 최적/무관 → **평균 mmq 이득 < 14.5%**. → 순 +5%.

## 판정 (STEP C) + 다음

- **✅ 격리 WIN이 엔진서 survives(+5%, bit-id)**: X-shared 전례(폐기)와 달리 이번 형상 WIN은 엔진서 살아남음.
  방법론(재구조화+게이트+스윕+**in-engine 검증**)이 **실 엔진 이득**을 냄(격리 맹신 아님·measured).
- **정직 감쇠**: 엔진 +5% ≪ 격리 +14.5%(Amdahl+다중 N,K). overclaim 없음 — 엔진 숫자는 +5%.
- **다음(더 회수)**:
  1. **per-(N,K) 형상 dispatch**(autotuner): 각 matmul에 최적 형상 → 14.5%의 더 많은 부분 회수. 이게 CUTLASS/llama 방식.
  2. **4모델 확장**: 이번은 27b 2 ctx. gemma4/35b/26b도 A/B(형상 이득 모델별 확인) 후 default 결정.
  3. 현재 **opt-in 유지**(MMQ256, 기본 0=무회귀) — 4모델 검증 전 blanket-default 금지(정직).

## 산출물
- `kernels/gemm_mmq_tmpl.cuh`(launch_gemm_mmq_w4a8_auto 토글)·`qwen35_decode.cu`(prefill 배선). MMQ256=0 기본=무회귀·bit-id 검증.
