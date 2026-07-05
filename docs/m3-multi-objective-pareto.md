# M3 — 다목적 축 정식화 + 모델별 파레토 프론트 (기존 measured 재구성, 제품 실물)

> 성능 탐사 완결(c-bandwidth-seeds-verdict) 후, 남은 큰 일은 성능이 아니라 **제품**이다.
> M3 = "당신의 서버에서, 당신의 목적대로" 서빙 세팅을 고르게 하는 실물.
> ★이 문서는 **새 측정이 아니라 기존 measured의 재구성**(regime 테이블·recipe claims·roofline). 빈 곳은 정직히 "미측정".

## STEP 1 — 다목적 축 정식화

서빙 최적화는 단일 스칼라(throughput)가 아니라 **여러 축의 트레이드오프**다. 이 엔진에서 measured로 실재하는 축:

| 축 | 정의 | measured 근거 | SLO 후보? |
|---|---|---|---|
| **throughput** | 집계 tok/s(전 요청 합) | regime 테이블·recipe claims | (목적) |
| **tpot** | per-token latency(생성 지연, s/token) | regime 테이블 tpot95(B8 0.11 vs B16 0.25 등) | ✅ 상한 SLO |
| **TTFT** | time-to-first-token(프리필 지연) | bench e2e50/95/99·chunked prefill(serving-scheduler-step0) | ✅ 상한 SLO |
| **메모리** | 모델 weight + KV + scratch(128GB unified 예산) | per_slot_bytes admission(autotune 메모리 필터) | 하드 제약(가드) |
| **워크로드** | spec accept 민감(반복 6.99 / 산문 3.5)·mix(short_ratio) | spec accept 민감도·regime mix 지배 | (조건 라벨) |
| **품질** | 출력 정확도 | **q4 단일(measured 결론)** | 고정(트레이드 없음) |

### ★품질 축은 q4 단일 — 정직 (measured 결론)

다른 서빙 엔진은 "품질↔속도" 트레이드를 양자화 비트폭으로 제공하나, **이 하드웨어에선 그 옵션이 measured로 없다**:
- **q2**: naive 완전 붕괴(ppl 464×·gibberish)·정교 기법도 coherent엔 ~q4 bits 필요 → 순 절약 미미(소거).
- **q3**: 정확성 BIT-ID 성공이나 **속도 역행**(3bit 비정렬 로드 2× — 격리 q4의 1.2× 느림, 소거).
- **∴ 이 엔진의 품질 축 = q4 고정**(w4a8). "품질을 낮춰 속도를 산다"는 파레토 지점은 **불가**(bytes↓ sweet spot = 4bit,
  근거: c-bandwidth-seeds-verdict STEP E). 정직 = 품질을 트레이드 축으로 광고하지 않음.

### 메모리 축 — 짧은 context에선 비병목 (정직)

- admission = MemAvailable×margin ÷ per_slot_bytes. GB10 128GB unified(oversubscribe=시스템 다운 전례라 하드 가드).
- **실측: 짧은 context(maxlen 256~384)에선 메모리 비병목** — gemma4 per-seq 692MB → Bmax_hi≈96(레시피 B16보다 훨씬 큼).
  즉 현 운영점(짧은 range)에선 **throughput↔tpot 트레이드가 지배**, 메모리는 여유.
- 메모리가 binding 되는 곳 = **긴 context(>1024)**: sliding KV 링버퍼 미구현([[attn-12k-shared-wall]])·KV read 비중 증가.
  이 지점은 별도 트랙(장문 서빙). 현 파레토는 짧은 range 조건.

## STEP 2 — SLO-제약 튜닝 (페널티 → 제약)

기존 autotune 목적함수 = "throughput 최대 + tpot **soft 페널티**"(`score = tok_s − 100·max(0, tpot95−cap)`).
M3에서 이를 **"SLO 제약 하 throughput 최대"**로 정식화(예: "tpot95 ≤ 0.15s에서 throughput 최대"):
- `autotune.py --slo-mode`(opt-in): tpot95 ≤ cap 만족 config 중 throughput 최대(초과=infeasible). 기본값은 soft 페널티 보존(무회귀).
- 프레임워크 신설 X — 기존 `score()`/`--tpot-cap` 확장만(YAGNI). 상세 = docs/autotune-harness.md.
- **레시피 SLO 필드 = v0 스키마 내 수용**: `target.user_constraints.tpot95_cap_s`(이미 존재) + claims의 regime별 `tpot95`(measured)
  = "이 SLO엔 이 config"의 재료가 v0에 이미 있음. 명시적 SLO regime 축(`when.tpot_target_s`)은 v1 후보(기록만).

## STEP 3 — 모델별 파레토 프론트 (SLO 지점 선택 = "목적대로")

각 지점 = (config → throughput, tpot95)·**measured 조건 명시**(nreq/seed/maxlen/워크로드). 사용자가 "내 SLO"로 지점 선택.
tpot95를 SLO 상한으로 스윕한 파레토(같은 워크로드 조건끼리만 비교 — 정직).

### 3-1. Qwen3.5-27B (spec, MTP) — 가장 풍부한 파레토

조건: nreq60, seed42, w4a8+mtp_q4+concat-mmq(Nn>8). 워크로드 mix가 파레토를 **지배**(long-heavy가 큰 B의 곱 실현).

| tpot95 SLO 상한 | 최적 config | throughput | 워크로드 조건(정직) |
|---|---|---|---|
| ≤ 0.15s | B4 K7W2 | 33.2 | λ4, short-heavy(0.9) — 슬롯 회전 빠름·큰 B 큐잉 손해 |
| ≤ 0.19s | B6 K7W3 mw1 | 36.4 | λ1, short-heavy(0.9) |
| ≤ 0.23s | B10 K8W2 mw1 | 47.8 | λ8, long-heavy(0.3) 대안 |
| ≤ 0.32s | B16 K7W2 | 51.6 | λ1, long-heavy(0.3) |
| ≤ 0.35s | B16 K7W2 | 51.7~53.2 | λ4/λ8, long-heavy(0.3) — throughput 최대 |

- single(비배치) spec = 21.3 t/s(tree K4W3 accept 2.27)·non-spec 11.9. **파레토 밖(대역폭 벽)**: "single 50 t/s"는 불가(정직).
- spec의 차별점 = **워크로드 민감**: 반복 워크로드(accept 6.99)면 지연이 크게 유리·산문(3.5)은 완만. plain 대비 1.21×(같은 Bmax).

### 3-2. Gemma4-31B (plain dense) — Bmax 단일 축

조건: nreq48, seed42, maxlen384, w4a8, gen mix 70/30 고정(short_ratio 미스윕=한계).

| tpot95 SLO 상한 | 최적 config | throughput | 조건(정직) |
|---|---|---|---|
| ≤ 0.12s | B4 | 23.0~23.8 | 저지연 |
| ≤ 0.14s | B8 | 35.6~38.3 | ★knee(무릎) — 최고 효율점 |
| ≤ 0.25s | B16 | 40.3~41.3 | throughput 최대(λ≥4) |

- **knee = B8**: B4→B8은 tpot +0.02s에 throughput +60%(큰 이득), B8→B16은 tpot +0.09s에 throughput +8%(체감).
  저부하(λ≤1)는 B16 역효과(tpot 2× + throughput↓) → B8. single decode 9.28(roofline 17.4의 53.3%).
- avgB<Bmax(λ16 B16서 12) → 서빙 41 < raw same-length batch 76(정직 — 서빙 큐 미포화).

### 3-3. Qwen3.5-35B-A3B (plain MoE) — B8이 명확한 운영점

조건: nreq40, seed42, maxlen256, fused-moe+w4a8+gdn.

| tpot95 SLO 상한 | 최적 config | throughput | 조건(정직) |
|---|---|---|---|
| ≤ 0.12s | B8 | 62.7~62.9 | ★지배점 — λ4서 이미 포화(avgB 6.9), λ4~32 평탄 |
| ≤ 0.25s | B16 | ~65 | throughput +3%에 tpot 2×(0.113→0.251) — latency 무가치 |

- **B8이 파레토 지배**: B16은 throughput +3%뿐인데 tpot 2× → 거의 항상 B8. 고throughput만 절실하면 B16 옵션.
- single decode raw 51.8(roofline 165.3의 29.8% — MoE 구조적 낮은 달성률). 서빙 ~63 ≪ raw batch aggr 195(B16)/169(B8):
  ★큰 격차 원인 = **chunked prefill이 decode 차단**(스케줄러 여지 큼, 커널 아님) — serving-scheduler 트랙.

### 3-4. Qwen3.5-26B-A4B (plain 병렬 MoE) — roofline만 (서빙 파레토 = 후속)

- roofline active 3.68B → single 상한 **139.7 t/s**. decode graph/서빙 backend 구현 완료(4모델 완성)이나
  **서빙 bench(throughput/tpot 실측)는 미완** = 파레토 프론트 **미측정**(정직 갭).
- 다음: bench_serve 26B regime 스윕 → 파레토 산출(35b MoE와 유사 형태 예상 — B8 knee 가설, 미검증).

## ★정직 라벨 정리 (파레토 밖 = 불가)

- **single 고throughput(예: 50 t/s single)은 불가** — 대역폭 벽(273GB/s × 토큰당 bytes). 배치(B)로만 집계 throughput ↑,
  단 per-request tpot는 그대로(오히려 큐잉으로 악화). 즉 **"낮은 tpot + 높은 single throughput" 동시 불가**.
- 파레토 지점은 **워크로드 조건부**(27b: mix가 지배·spec accept 민감). 다른 mix/λ의 수치를 섞어 비교 금지.
- 품질 트레이드 축 없음(q4 고정). context는 짧은 range(256~384) 조건 — 장문(>1024)은 별도 트랙.
- 26B 서빙 파레토는 미측정(roofline만).

## 산출물 / 다음
- 이 문서(measured 재구성)·autotune.py `--slo-mode`(STEP 2)·recipe SLO 필드 v0 수용 확인(recipe-schema.md).
- 다음: 26B 서빙 bench로 파레토 완성 · 레시피에 SLO 판(latency_alt 이미 병기) 노출 · 장문 트랙(sliding 링버퍼).
