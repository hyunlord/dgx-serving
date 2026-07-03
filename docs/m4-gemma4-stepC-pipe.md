# M4 STEP C — Gemma4 파이프 완주 (해부→전략→튜닝→레시피, 모델 2호 first end-to-end)

> 파이프(plan→autotune→recipe)가 두 번째 모델에서 처음 돈 기록 + **사람 개입 지점 지도**(M7 제품화 입력).
> 정직 스코프: plain 서빙(spec 無), context maxlen=384(링버퍼 전), short_ratio 미스윕(gen mix 고정).

## C-1 — plan.py 전략 (spec 無 공간 정직 축소)

profile에 `mtp` 플래그 추가 → strategy가 `spec_capable` 분기. **Gemma4(mtp=false)**:
- 제거된 축: K/W/DRAFT_*/MTP_Q4/accept (spec 경로 부재 — 정직 축소, "removed_axes" 필드에 명시).
- 남은 축: **Bmax(★주 lever)** + 경로별 양자화 mix(프리필 q4·decode W4A8) + scheme_note(SMAX 버킷).
- fanova: 27b(spec)의 중요도(DRAFT_SEQ 0.33 등)는 spec축이라 transfer 불가 명시, bmax(0.31)만 후보.
- **CYC 정합 검증**: 전략이 이식 세션 실측과 일치(Bmax 주 lever=sb B4포화 돌파, 프리필 q4 75≫W4A8 12.7). qwen 회귀 OK.

## C-2 — 튜닝 (Gemma4 plain 서빙 스윕, 진짜 엔진 위에서)

autotune.py는 bench_serve_spec(spec)에 깊이 결합 → plain 전용 `tools/bench_serve_gemma4`(모델-무지 serving_core
+ Gemma4Backend + dummy rope[현실 길이] + MemAvailable 메모리 필터) 신설. warm 프로토콜.

**Bmax × λ 스윕 (W4A8 sb, nreq48, maxlen384, tpot95 cap 0.35s 내 전부):**

| Bmax | λ1 tok/s | λ4 | λ16 | tpot95 | avgB(λ16) |
|---|---|---|---|---|---|
| 4 | 23.8 | 23.0 | 23.0 | 0.116 | 3.84 |
| 8 | 35.6 | 38.1 | 38.3 | 0.137 | 7.01 |
| 16 | 34.7 | 40.3 | 41.3 | 0.224 | 12.06 |

- 메모리 필터: per-seq 692MB(maxlen384) → Bmax_hi≈96 = 메모리 **비병목**(짧은 context, 링버퍼 전에도 여유).
- **regime→Bmax**: λ≤1→**B8**(35.6 — B16이 저부하선 역효과: tpot 2×+throughput↓), λ≥4→**B16**(40-41).
- ★avgB<Bmax(λ16 B16서 12) → 서빙 tok/s(41)는 raw same-length batch(76)보다 낮음(정직 — 서빙은 큐 미포화).

## C-3 — cross-model warm-start 관찰 (축 대조)

27b 레시피(spec)를 Gemma4(plain) prior로 볼 때 transfer 여부(관찰 — 라이브 warmstart A/B는 autotune plain 포트 필요=갭):

| 27b 축 | Gemma4 공간 | transfer |
|---|---|---|
| K, W, DRAFT_MW, MTP_Q4 (spec) | 없음(spec 부재) | **0% — 전부 out-of-space(스킵)** |
| bmax | 있음(주 lever) | **부분** — 고λ 천장 B16 일치, 저λ 최적 shift(27b 6 → Gemma4 8) |

**관찰 결론(정직)**: spec 모델→plain 모델 warm-start는 거의 transfer 안 됨. 27b 중요도의 대부분(DRAFT_SEQ 0.33
+K/W)이 공간 밖이고, 유일 공유축 bmax(0.31)도 저부하 최적이 커널-형상/GDN-부재 차로 이동(6→8). = **cross-model
transfer는 아키텍처가 축을 공유할 때만 유효** — 이질 모델(spec↔plain)엔 한계. (안 통해도 가치: 이 데이터가 결론.)

## C-4 — 첫 레시피 + verify

`recipes/gemma4-31b-gb10.yaml`(schema dgx-recipe/v0): plain config=bmax 단일, scheme=W4A8 sb/gemm_batch(버킷 명시),
claims=plain(spec 비교 없음), backend.quant_path(프리필 q4·decode W4A8). recipe.py verify plain-tolerant화
(per_slot_bytes_auto=KV-only, 미니벤치=bench_serve_gemma4, bit-id=오프라인 test_gemma4_batch 버킷).

**verify --full 결과 PASS**: 지문 정상 · 메모리 3 config OK/0 거부(로컬 재계산 84.65GB) · 미니벤치 재현율
λ16 B16 claim 41.3 vs local 41.2 = **99.8%**, λ1 B8 claim 35.6 vs local 36.3 = **102.0%** · bit-id=오프라인
(test_gemma4_batch SMAX 버킷). → 레시피 export→verify 라운드트립 성립(재현율 99.8~102%).

## 파이프 end-to-end 리포트 — 사람 개입 지점 지도 (M7 입력)

모델 2호에서 파이프가 돈 기록. **어디가 자동 / 어디서 손이 갔나**:

| 단계 | 자동 | 사람 개입(자동화 갭) |
|---|---|---|
| 해부(anatomy) | profile 있으면 전략 자동 | ★anatomy --compare가 엔진 manifest(HF config 아님)서 KeyError — profile 수동 작성 |
| 전략(strategy) | ✅ spec_capable 분기로 자동 축소 | mtp 플래그 profile 수동 주입(1회) |
| 튜닝(autotune) | — | ★autotune.py가 bench_serve_spec(spec)에 깊이 결합 → plain 전용 bench_serve_gemma4 신설(포트) |
| 커널 dispatch | 이식 후 자동(sb) | ★27b 축1 결정표 수동 이식(이식 세션) + 프리필/decode 양자화 mix 수동 결정 |
| warm-start | M2 스킵보고 설계 존재 | ★plain 라이브 warmstart는 autotune plain 포트 필요 — 축 대조 관찰로 대체 |
| 레시피 export | recipe.py export 존재 | ★export가 spec 스키마(concat-mmq/K/W) → plain yaml 수동 작성 |
| 레시피 verify | ✅ plain-tolerant화 후 자동 | per_slot_bytes_auto·plain 미니벤치 분기 추가(포트) |

**핵심 갭(M7)**: 파이프의 **전략·verify는 profile 분기로 모델-무지化 성공**했으나, **튜닝 하네스(autotune.py)와 export는
27b spec 스키마에 하드코딩** — 새 모델 유형(plain)마다 backend 신설+수동 작업 필요. 즉 자동화는 "같은 아키텍처
계열 내"에선 강하나 "새 유형(spec↔plain, dense↔MoE)"엔 backend 추상화가 미완. M7 = 이 backend 계층을 일반화.
