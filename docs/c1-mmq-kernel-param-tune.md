# C-1 — mmq 커널 파라미터 극한 최적화 (개념증명, measured)

> 진단(사용자): llama 열세는 "베이지안 최적화 반쪽" — dispatch(B/K/W)만 튜닝, **커널 내부 파라미터 미탐색**.
> 이번 = 순수 CUDA 유지 + mmq 커널 파라미터 직접 탐색(compile→gate→measure 루프). ★정확성 게이트 최우선.
> 결론 선요약: **루프+게이트는 작동 증명(AP=8 "빠른데 틀림" 거부). 단 탐색 가능 파라미터(KT/AP/BPAD)는
> 이미 손-튜닝 최적 = 유의미 개선 없음(measured). 진짜 미탐색 lever = BM/BN 타일(재구조화 = 다음 POC).**

## STEP 0 — mmq 해부 + premise 정정

`kernels/gemm_mmq.cuh` (챕터7 full-MMQ W4A8 GEMM, 128×128 타일·8워프 2×4·MMA m16n8k32·ldmatrix·cp.async):

| 파라미터 | 현황 (★premise 정정) | 탐색 여지 |
|---|---|---|
| KT (K-타일) | **이미 `template<int KT>`** + 스윕 인프라(`launch_gemm_mmq_w4a8_kt<KT>`). **{32,64} 제약**(128=48KB 정적 shared 초과, dynamic shared 필요) | 좁음(2값) |
| launch_bounds (occupancy) | **이미 measured de-risk**(주석 기록): occ↑(minblk 3/4)→register spill→prefill 728→1756ms(−140%). 33%/minblk2 최적. batch도 wave-quant로 occ 무관 | 거의 없음 |
| BM/BN (128×128) | 8워프 2×4 레이아웃 + MMA shape에 고정(MS=BM/32, NS=BN/32) | **최대 미탐색·재구조화 필요** |
| AP/BPAD (shared 패딩) | 미탐색(bank-conflict 축) | 후보(스윕함) |

**★사용자 premise "커널 내부 0 튜닝"은 부분적으로 부정확**: 최대 노브 2개(KT·occupancy)는 이미 탐색/measured됨.
단 **체계적 Optuna 조인트 탐색은 안 함**(단일축 ad-hoc만) — 그래서 이번에 탐색 가능 축을 실제 루프로 측정.

**baseline (27B MLP up prefill 512×17408×5120, warm SM 2418MHz)**: KT=64/AP=0/BPAD=4 = **2146us · 42.5 TFLOP/s**,
게이트 PASS(mmq vs gemv_w4a8 rel 3.45e-6 < 1e-5 · 결정성 30회).

## STEP A/B — 파라미터화 + 컴파일→게이트→실측 루프

- `MMQ_AP`/`MMQ_BPAD`를 `#ifndef` 가드(컴파일-타임 상수, `-D` 오버라이드 = 템플릿 동치·기본값 불변·엔진 무회귀).
  KT는 이미 템플릿. (신호 있으면 정식 `template<int>` 승격 계획이었으나 — 아래 null 결과로 불요.)
- **`tools/bench_mmq_tune.cu`**: 조합별 ★정확성 게이트(mmq vs gemv_w4a8 rel<1e-5 + 결정성 30회) → warm(50 rep 자체워밍) → 타이밍.
- 스윕 KT{32,64} × AP{0,8} × BPAD{0,4,8} = 12 조합, 각 nvcc 컴파일→게이트→실측.

## STEP C — 판정 (measured)

| KT | AP | BPAD | time_us | TFLOP/s | vs baseline | 게이트 |
|---|---|---|---|---|---|---|
| 64 | 0 | 0 | 2141.8 | 42.6 | **+0.2%**(잡음) | PASS |
| 64 | 0 | 4 | 2146.1 | 42.5 | baseline | PASS |
| 64 | 0 | 8 | 2145.4 | 42.5 | +0.03% | PASS |
| 32 | 0 | 0/4/8 | 2213~2234 | 41.0 | **−3%** | PASS |
| * | 8 | * | — | — | — | ★**FAIL(rel≥1e-5)** |

1. **★게이트 작동 증명**: AP=8(sA K-stride 패딩)은 ldmatrix 16B 정렬을 깨 **"빠른데 틀린 답"** — 게이트가 전부 거부.
   "성능만 변·답 불변"을 실제로 강제함(POC의 핵심 안전 장치 검증). AP는 구조적으로 0 고정(자유 파라미터 아님).
2. **KT=64 최적 확인**: vs KT=32 +3% 빠름. 엔진 손 선택이 옳음(128은 shared 초과로 불가).
3. **BPAD 무영향**: 0/4/8 전부 2141~2146us(±0.2%, 잡음 내). sB 패딩은 bank-conflict lever 아님(현 dim서).
4. **최선 = +0.2%(잡음) = 유의미 개선 없음.**

### 정직 결론 (안 되면 왜 — measured, 도망 아님)
- **탐색 가능 커널 파라미터(KT/AP/BPAD)는 이미 손-튜닝 최적** — 체계적 스윕이 free lunch 못 찾음.
  premise("커널 내부 미탐색→큰 이득")는 **이 파라미터들엔 measured-반증**(KT·occupancy는 애초에 탐색됨, AP는 구조
  고정, BPAD는 무영향).
- **llama 갭은 KT/패딩 미튜닝 탓이 아니다** — 이 축들은 이미 최적. 갭의 출처는 딴 데(BM/BN 타일 or FP4 or 커널 성숙도).
- **진짜 미탐색 lever = BM/BN 타일 형상**(현 128×128은 손-선택, 한 번도 탐색 안 함 — CUTLASS/llama가 극한 튜닝하는 바로
  그 축). 단 8워프 2×4 레이아웃 + MMA m16n8k32에 깊이 묶여 **커널 재구조화 필요**(단순 파라미터 추가 아님) → 다음 POC.
- FP4 텐서코어는 여전히 toolkit-blocked(별개).

### 헤드룸은 구조적(compute-bound + 기존 결론과 정합)
- baseline은 **compute-bound**: W(17408×5120 q4=44.5MB) read floor ~163us(273GB/s) ≪ 실측 2146us. 즉 메모리 아님·연산 지배.
- 42.5 TFLOP/s가 int8 TC peak 대비 얼마인지는 GB10 int8 TC 스펙+ncu SM% 필요(별도). 단 **스윕 파라미터(KT/AP/BPAD)가
  전혀 안 움직임** = 헤드룸이 있다면 이 축들이 아님.
- **★기존 결론과 정합**: ch9 ⑩(prefill de-risk)이 이미 measured로 "prefill 갭 = llama **타일/파이프 성숙도**,
  occupancy는 lever 아님(A/B NO-GO), 128×128 register-blocked 최적"이라 결론. 이번 C-1이 **KT/패딩도 lever 아님**을
  추가 확인 → 잔여 헤드룸은 전부 **BM/BN 타일 형상 + 파이프 깊이**(CUTLASS식 딥 최적화)에 몰림. 일관.

### 개념증명 판정
- **✅ 방법론 증명**: compile→게이트→measure 루프 작동. ★게이트가 "빠른데 틀림"(AP=8) 실제 차단 = 안전한 커널 탐색 가능.
- **❌ 이 축 이득 없음**: KT/AP/BPAD는 손-튜닝 이미 최적(measured, +0.2% 잡음). overclaim 없음.
- **다음**: BM/BN 타일 형상 재구조화(template<int BM,int BN> + 워프 레이아웃 도출) = 진짜 미탐색 축·최대 잠재 lever.
  이게 되면 gemm_batch/moe_fused/attn로 확장. 단 재구조화라 correctness 게이트 하에 신중히(별도 세션).

## 산출물
- `tools/bench_mmq_tune.cu`(파라미터 스윕 벤치·게이트 내장)·`kernels/gemm_mmq.cuh` AP/BPAD `#ifndef` 가드(무회귀).
- 방법론(루프+게이트)은 BM/BN·다른 커널에 재사용 — POC의 실 자산.
