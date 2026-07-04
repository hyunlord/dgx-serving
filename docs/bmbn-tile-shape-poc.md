# BM/BN 타일 형상 재구조화 POC — 첫 커널-파라미터 탐색 WIN (measured)

> C-1(KT/패딩=free lunch 없음) 후속. 소거법이 가리킨 진짜 미탐색 축 = **BM/BN 타일 형상**(CUTLASS/llama 극한
> 튜닝 축, 현 128×128은 손 선택·미탐색). 재구조화(파라미터 추가 아님) + 정확성 게이트 + 형상 스윕.
> ★결론: **256×64 타일 = +14.5%(격리 벤치, M=512 prefill, back-to-back·gate PASS). 커널-파라미터 탐색의 첫 실 WIN.**
> 손 선택 128×128은 sm_121 최적이 아니었음. 단 격리 벤치 승리 → in-engine 검증 필요(다음).

## STEP 0 — ncu 헤드룸(재구조화 정당화)

baseline mmq(128×128 KT64, 512×17408×5120, warm): **42.5 TFLOP/s**. ncu:
- SM 처리량 **36%**·issue active **37%**(63% 사이클 stall)·occupancy **31%**(reg 128·smem 3블록). tensor IMMA util=n/a(GB10).
- → compute-peak 아님, **stall/latency-bound**. 헤드룸 실재. 재구조화 정당(measure-first가 큰 삽질 방지 확인).

## STEP A/B — 재구조화 + bit-id 게이트 (구조 옳은지 먼저)

`gemm_mmq_tmpl.cuh`: `template<int BM,int BN,int KT,int WM,int WN>`. 원본(128×128·워프 2×4 하드코딩)의 **충실한 일반화**:
- 파생: MS=(BM/WM)/16, NS=(BN/WN)/8, NW=WM*WN, NT=NW*32. 치환: 64→MS*16, 32→NS*8, 256→NT, wm/wn=warp/WN·warp%WN.
- 계산(MMA 시퀀스·누적 순서) 불변 → **구조만 일반화**. 격리(신규 파일·프로덕션 gemm_mmq.cuh 불변).

★게이트(test_mmq_tmpl, 커밋 6682125): tmpl<128,128,64,2,4> == 원본 gemm_mmq_kernel<64> **BITWISE**(diff=0/131072
max|Δ|=0.0, 3 dim 배수+비배수 200×300 전부) → 재구조 틀 정확. + 5 형상 vs gemv rel<1e-5 + 결정성 = 형상 수용.

## STEP C — 형상 스윕 판정 (measured)

27B MLP up prefill(M=512, N=17408, K=5120), warm, 각 형상 컴파일→게이트(rel<1e-5+결정성)→실측:

| 형상 | 워프 | KT | TFLOP/s | vs 128×128 |
|---|---|---|---|---|
| **256×64** | **4×2** | **64** | **49.1** | **+14.5%** ★최적 |
| 128×128 (baseline) | 2×4 | 64 | 42.9 | — |
| 128×64 | 4×2 | 64 | 40.5 | −5% |
| 256×64 | 4×2 | 32 | 37.3 | −13%(K-reuse↓) |
| 64×64 KT128 | 2×2 | 128 | 35.6 | −17% |
| 64×64 / 64×128 / 64×256 | | 64 | 24~25 | −42%(reuse↓) |
| 512×64 | 4×2 | 32 | 18.7 | −56%(acc 128reg spill) |
| 256×128 / 128×256 | | 64 | 15 | −65%(smem/occ 붕괴) |

- **★256×64 KT64 = +14.5%**(back-to-back 재현: 48.0→49.1, 잡음 ±0.2%보다 훨씬 큼 = 실 WIN). gate PASS·결정성.
- **메커니즘(measured 정정)**: 처음 가설 "작은 타일→occ↑"는 **틀림**(64×64 −42%). 실제 = **tall 타일이 W 재읽기↓**:
  W(44.5MB)를 M/BM번 읽음 → 128×128=4패스(178MB), 256×64=2패스(89MB, 절반). W가 X(2.6MB) 지배 → 메모리 stall↓.
  BM=512(1패스)는 과도 — acc MS=8→128 레지스터 spill로 붕괴(18.7). sweet spot = **256×64**(W 절반 + acc 64reg 유지 + KT64 K-reuse).
- **★sm_121 sweet spot**: 손 선택 128×128은 이 형상(M=512)서 최적 아님 — 256×64가 +14.5%. **BM/BN이 진짜 lever였음**
  (C-1의 KT/패딩 null과 대조 — 소거법 확증).

## 정직 캐비엇 (overclaim 금지)

- **격리 벤치 승리**: +14.5%는 격리 mmq 벤치. 헌법 교훈(격리≠in-engine·X-shared 전례) — **in-engine 검증 필수**(엔진
  prefill 경로에 256×64 배선 + 서빙 실측 + bit-id). 아직 "엔진 +14.5%"라 주장 안 함.
- **size-specific**: M=512 최적이 256×64일 뿐, 다른 M(prefill 길이·batch)엔 다른 최적 — **이게 요점**(per-(M,N,K) 형상
  선택 = CUTLASS/llama가 하는 것). 프로덕션 = 형상 오토튜너(다음).
- **mmq(prefill) 한정**: decode(gemv_w4a8/moe_fused)는 다른 커널·다른 축. prefill throughput 개선(26b decode 24%와 별개).
- **정확성 불변**: 전 형상 gate PASS(rel<1e-5·결정성) — "빠른데 틀림" 0. 재구조판 128×128=원본 bit-id.

## 판정 + 다음

- **✅ 커널-파라미터 탐색 첫 실 WIN**: 재구조화+게이트+스윕이 손 선택 대비 **+14.5%**(측정·정직). "반쪽 최적화" 가설이
  BM/BN 축에서 **검증됨**(C-1의 쉬운 축 null과 대조 = 진짜 lever는 형상).
- **다음**: (1) 256×64 in-engine 배선+서빙 검증(격리≠in-engine) (2) 형상 오토튜너(per-(M,N,K) — Optuna, mmq POC 하네스
  재사용) (3) gemm_batch/moe_fused/attn로 확장(decode 축). 되면 llama 갭 좁힘 첫 실 데이터.

## 산출물
- `kernels/gemm_mmq_tmpl.cuh`(형상 파라미터화·bit-id)·`tests/test_mmq_tmpl.cu`(게이트)·`tools/bench_mmq_shape.cu`(형상 벤치).
- 방법론(재구조화+게이트+스윕) = gemm_batch/moe_fused/attn 재사용 자산.
