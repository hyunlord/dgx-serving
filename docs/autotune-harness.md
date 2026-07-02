# auto-tune 하네스 (Phase 1) — 장비×모델 config 자동 최적화

> 비전: "사용자 장비×모델 자동 최적화"의 첫 조각. 수동 스윕(챕터10 Phase 6/7)이 한 일을
> 자동화 — 4모델 확장 시 프로파일만 교체해 재사용. llama.cpp(휴리스틱)/vLLM(수동 가이드)과
> 달리 **실측 + 정확성 게이트(ISO)** 기반 오토튜닝이 차별점.

## 구성 (tools/autotune/)

- `autotune.py` — 하네스 본체. Optuna TPE + sqlite(중단·재개), trial 이력 전량 보존.
- `profiles/gb10.json` — device 프로파일(대역폭, unified 메모리, 가드 방식, toolkit 가용 — FP4 blocked 명시).
- `profiles/qwen35-27b.json` — model 프로파일(GDN/tied/quant + 가드 공식에 필요한 형상).
- `artifacts/` — best config 아티팩트(JSON, provenance 포함). `autotune.db`는 커밋 안 함.

## 설계 원칙 (헌법)

1. **config-주도**: 커널 불변. bench_serve_spec의 argv(Bmax/K/W/λ/seed)와 env(MTP_Q4)만 조작.
2. **메모리 사전 필터**: trial 실행 전 admission 예산을 Python으로 재계산(bench C++ 가드와 같은 공식,
   MemAvailable 기반) → 초과 config는 **실행 없이 prune**. GB10 unified oversubscribe = 시스템
   전체 다운 전례라 이중 가드(하네스 사전 + bench 내부).
   - 공식 재현 검증: 27b K6W3/K7W3/K6W4 per-slot이 bench 로그와 0.1MB 단위 일치.
3. **결정적 목적함수**: 같은 seed/trace(controlled), λ4 단일점, nreq 40.
   score = tok/s − 100·max(0, tpot95−0.35) (tpot soft 제약 — 지연 폭주 config 배제).
4. **canonical 제약**: Nn=K+W>8만 탐색(concat-mmq 곱 전 구간 + 단독 spec 자동 일치 — mism 원천 차단).
   ≤8은 per-slot 폴백(곱 손실)이라 prune.
5. **ISO 2단**: trial은 성능만, **최종 best만 full ISO(12/12)** — 실패 시 아티팩트 무효.
6. **과적합 가드**: best를 다른 seed trace로 재측정해 병기.
7. **BO 재발명 금지**: 탐색은 Optuna(TPE). 우리 몫 = 도메인 래퍼/필터/게이트/아티팩트.

## 사용

```bash
# 27b, 30 trials (GPU 1개 — 직렬 실행 전제)
~/github/emissary_llm/.venv/bin/python tools/autotune/autotune.py \
    --model tools/autotune/profiles/qwen35-27b.json --trials 30 --seed 42 --lam 4 --nreq 40
# 중단 후 재실행하면 sqlite에서 재개(완료 trial은 캐시 재사용)
```

## Phase 1 검증 — 27b ground truth 대조 (결과)

Phase 7 격자 실측(ground truth): 최적 **B12 K6W3 = 48.1 tok/s(λ4)**, B16 K6W3 48.7(tpot 악화),
K7/K8 전 B 무득, plain 1.21×.

**결과(2026-07-02, 30 trial): ✅ ground truth 정확 재발견 — BEST = B12 K6W3 (trial#29, 50.1 tok/s)**

- **trial 예산 소비: 실측 19 + prune 11**(canonical Nn≤8 / 중복 캐시 — 실행 없이 스킵). 수동 격자
  (Phase 6 스크리닝 6점 + Phase 7 격자 9점 + λ/B 스윕)와 동급 커버리지를 실측 19회로.
- **수렴 궤적**: 랜덤 탐색(#0–9, 33~47) → K6·저W 영역 학습(#10–17) → B12 K5W4 = 49.9(#20, tpot 페널티
  경계의 동급점 발견) → **#29에서 B12 K6W3 = 50.1 도달**. tpot soft 페널티가 B14/K8(고tpot) 영역을
  올바르게 배제(예: #1 B14K8 45.8 tok/s → score 35.8).
- **검증 프로토콜 전부 통과**: 다른 seed(777) 재측정 47.5 tok/s(과적합 아님 — 48대 유지, trace 난이도
  차이), **full ISO 12/12 PASS**, CYC 분해 draft 18~19%/verify 68~72%(Phase 7 계측과 정합).
- 50.1 vs Phase 7의 48.1 차이는 nreq(40 vs 60) — 같은 조건이면 동일점(주장은 재발견까지, 성능 신기록 아님).
- **동급점 발견(보너스, 정직 기록)**: B12 K5W4(Nn=9) = 49.9 — K6W3과 사실상 동급(같은 Nn, accept
  4.20 vs 4.37). Phase 6은 K5W4를 B8에서만 봤음(44.9) — B12에서 동급임은 하네스가 처음 관찰.
- **판정: 하네스 정합성 검증 성공** — 4모델 확장(프로파일 교체)에 사용 가능.

## Phase 2 — 축 확장 + warm 서버 + 120 trial (결과)

**논거**: 기존 정착값들은 다른 운영점에서 OFAT로 정착(SPLITK auto=plain B16, 경계 M≤16=B=2 실험,
chunk 512=pre-GDN) — 현 운영점(spec 서빙)에서 전체 공간을 튜너에 넣고 측정이 판정.

- **warm 서버(TRIALS=1)**: bench에 stdin trial 루프 추가(모델 1회 로드, 한 줄="Bmax K W λ nreq seed").
  trial당 2.5분 → **핫 축 ~40초**. 오염 게이트: 같은 config 재실행 4수치 전부 동일(결정적) 확인 후 본 실행.
  env 축(bench static 캐시)은 재시작 그룹 — env key 변경 시만 서버 재시작.
- **축 실사(정직 기록)**: 후보 10+축 중 spec 경로에 실효인 env 축은 **MTP_Q4/DRAFT_GEMV/DRAFT_SEQ 3개뿐**.
  SMAX·GDN state 타입=컴파일 타임(Phase 2b), chunk·canonical 경계 M·split-K 게이트=env 손잡이 부재,
  BM16/SPLITK=plain batch-decode 전용(spec 경로 미실행). → 6축(B/K/W + env 3).
- **120 trial(완료 82 + prune 38, TPE multivariate)** 판정 = **⒜ 새 최적 발견: Bmax 14** — Phase 7 격자(B8/12/16)가
  안 본 점. nreq60 확정치(같은 seed42, controlled):
  | 후보 | nreq60 | tpot95 | alt-seed(777) | ISO |
  |---|---|---|---|---|
  | **B14 K6W3** | **50.4** | 0.353 | 50.0 | 12/12 PASS |
  | B14 K5W4 | 50.2 | 0.343 | 51.7 | 12/12 PASS |
  | B12 K5W4 | 48.8 | **0.309** | 49.1 | 12/12 PASS |
  현행(Phase 7) B12 K6W3 = 48.1 → **B14 K6W3 = 50.4 (+4.8%)**. tpot 0.32→0.35 소폭 악화(정직 병기) —
  지연 민감하면 B12 K5W4(48.8, tpot 0.309)가 대안. CYC: verify 68~74%로 비중 구조 불변(B14도 B-불변 법칙 유지,
  "왜"=B12→14 곱 잔여 스케일, B16은 스케줄 오버헤드·tpot에서 손해).
- **env 축 전부 현행 기본값 재확인(measured 확정)**: MTP_Q4=1(off는 26.5), DRAFT_GEMV=0(gemv 폴백 31.6),
  DRAFT_SEQ=0(순차 32.5) — P4/P5의 개별 결론을 튜너가 상호작용 포함 공간에서 독립 재확정.
- **축 중요도(fANOVA)**: DRAFT_SEQ 0.33 > Bmax 0.31 > DRAFT_GEMV 0.19 > MTP_Q4 0.13 ≫ K 0.03 > W 0.01
  — draft 경로 커널 스킴이 지배, 트리 형상은 미세 튜닝(4모델 확장 시 공간 축소 prior).

## 한계 / 다음

- 목적함수가 λ4 단일점(Phase 7에서 λ 포화 확인된 조건) — 부하 프로파일이 다르면 목적 재정의 필요.
- 컴파일 타임 축(SMAX/GDN state/타일)은 재빌드 루프 필요 — Phase 2b 후보.
- Phase 3 = 4모델 확장(0.8b/기타 프로파일 추가), 장비 프로파일 자동 측정(STREAM/SM probe), 다목적(λ1 지연 병기).
