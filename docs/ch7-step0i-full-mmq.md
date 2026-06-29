# 챕터7 STEP 0i — full-MMQ W4A8 GEMM (prefill 추월)

> **측정으로 기전 확정(추측 아님)**: prefill 병목 = W4A8 GEMM(72.5%). 우리 matmul은 **L1/TEX 84% 포화**, llama는 **39% 무포화** → llama **3.2×**(= prefill 3.4× 갭 일치). micro-lever(BM·ldmatrix·register-block·bank-conflict) 전부 측정 기각 — **패키지 전체라야 L1/TEX 내려감.**

## 0. 측정 입력 (확정)

**llama vs 우리, 동급 FFN matmul (N=17408, K=5120):**

| | M | duration | L1/TEX | compute | occupancy |
|---|---|---|---|---|---|
| llama mul_mat_q (128×128) | 512 | 1.44ms | **39.5%** | 40% | 29% (reg 120) |
| 우리 gemm_mma_w4a8 | 256→512환산 ~4.64ms | | **84%** | 48% | 64% |

→ **3.2× 차이 = L1/TEX 효율.** llama는 shared 트래픽 ~½·무포화(40% 균형=latency-bound, 여유 있음).

**micro-lever 측정 기각(2.46→2.32ms, ~6%, L1/TEX 82% 불변):** BM 64→256(reuse L1TEX wavefront −10%만, occupancy 붕괴 64→17%), ldmatrix-A(명령↓이나 대역폭 무변 0%), register-block(+2.4%), 4B-unpack(+3.3%), bank-conflict 11.1M(구조적·BM무관·벽 주범 아님). → **현 구조 천장. 개별 교체 무효.**

## 1. 기전 — L1/TEX 효율 (occupancy 아님)

llama occupancy도 낮음(29%, reg 120) = 우리 BM=128과 **동일**. 그런데 3.2× 빠름 → **차이는 occupancy도 tile 크기 단독도 아니라 L1/TEX 효율(39 vs 84%).** llama는 동일 FLOP을 절반의 shared 트래픽으로 → 포화 안 됨 → 빠름. 우리는 스칼라 shared load·QBLK=32 sync(160회)·conflict로 L1/TEX 포화.

## 2. 패키지 구성 (전부 sm_121 가용, **전부 함께라야 L1/TEX↓**)

| 요소 | L1/TEX 효과 | 비고 |
|---|---|---|
| **128×128 타일**(BM·BN 둘 다 128) | A·B 재read/재write ↓ (BN↑가 A cp.async write [(N/BN)·M·K] 직접 절감 — 미측정이나 dominant) | reg/acc↑(occupancy 29% 수용) |
| **MMQ_ITER_K=256** | K 256씩 적재, 8 MMA-k substep/sync → **sync 160→20회**, compute/shared 비↑ | shared 적재 stall 은닉 |
| **ldmatrix** | fragment 1명령 적재 = L1TEX wavefront↓ | **de-risk 통과**(test_ldmatrix 오차0). 16B 정렬 stride 필요 |
| **shared 레이아웃**(swizzle/pad) | bank conflict 제거 = 트랜잭션↓ | q8 act + q4 weight, llama load_tiles 참고 |
| **stream-K** | K분할+fixup 리덕션 → grid 채움/load-balance | fixup 커널 별도(결정성 가드 필요) |

핵심: **개별로는 L1/TEX 안 내려감**(나머지가 포화 유지) — ldmatrix만/BM만 측정 0%가 그 증거. 조립해야 39% 도달.

## 3. 정직한 범위·위험

- **별도 큰 커널**(~300줄, gemm_mmq.cuh). piecemeal 측정 불가 → **조립 후라야 perf 측정**(de-risk는 correctness 단위로).
- occupancy 낮음(~29%) 수용 — L1/TEX 효율이 본질(llama도 그러함).
- **B(int4) ldmatrix**: ldmatrix는 byte 적재라 int4→shared int8 unpack 먼저 필요(or llama load_tiles_q4_0 방식으로 q4를 shared에 두고 mma 레이아웃 적재). 설계서 해결.
- **추월 보장 아님**: 목표 = llama L1/TEX ~40% 재현 → ~3.2×. 못 미치면 어느 요소가 부족한지 ncu로 진단(측정-먼저).
- ldmatrix 교훈: 큰 가정(이 패키지가 L1/TEX 39% 낸다)은 **조립 최소본으로 먼저 측정**, 통과 후 최적화.

## 4. 단계 (각 게이트 통과 후 다음)

- **S0 ✓** ldmatrix A-fragment de-risk — `test_ldmatrix` 오차0 (완료).
- **S1** B 적재 de-risk: int4→shared int8 unpack + ldmatrix B == 스칼라 오차0 (microtest).
- **S2** 단일 타일 MMQ 커널(128×128, MMQ_ITER_K=256, ldmatrix A+B, 블록 스케일) → **== gemv_w4a8 오차0**(oracle, 비배수 bounds 포함) + 결정성 30×.
- **S3** 조립 최소본 ncu: **L1/TEX 84→~40% 내려가나?**(핵심 가정 측정). 안 내려가면 어느 요소 부족 진단 — full 최적화 전 게이트.
- **S4** stream-K(필요시, 결정성 race 가드) + 최적화 → duration ~llama 추격.
- **S5** 엔진 통합(use_q4 prefill 경로 교체) — **합의 후**. 직교성 q4·W4A8, Sq일관(단발==청크), 전체 스위트.

## 5. 검증 게이트 (헌법 불변 — 재작성이라도)

- **오차0**: == gemv_w4a8 rel<1e-5, 비배수 bounds(M/N/K 비배수).
- **결정성**: 30× bitwise(cp.async + stream-K 리덕션 race 가드).
- **Sq·split 일관**: 단발==청크(B 단일커널 정신).
- **직교성**: q4·W4A8 둘 다.
- **ncu**: L1/TEX 84→~40%(핵심), occupancy는 무관 확인, duration vs llama 1.44ms(M512).
- **prefill t/s**: 240 → llama 806 추격(엔진 통합 후).
- **안전**: 별도 커널 gemm_mmq.cuh, 현 gemm_mma(2.32ms) 롤백 가능. GB10 메모리 가드.

---
*측정 체인: 대반전(prefill=GEMM) → BM 기각 → ldmatrix 기각 → bank-conflict 기각 → reuse de-risk(~10%) → **llama 프로파일(L1/TEX 39 vs 84%, 3.2×, 패키지가 기전)**. de-risk가 매 단계 추측 차단. full-MMQ = 측정 정당화된 추월 경로(sm_121 달성가능).*
