# 챕터4 설계안 — gemm 커널 극한 (mmvq multiple-output) STEP 0 (게이트)

> 챕터3: B=8서 llama.cpp 추월(980 vs 912). 양옆 뒤짐: B=1 215 vs 277, B=16 1016 vs 1266.
> 둘 다 gemm 구조. 이 문서(mmvq 정독 + occupancy 전략 + 검증) 합의 후 STEP 1.

## 1. mmvq 정독 — "occupancy 안 깨는 multiple-output"의 정체

`ggml-cuda/mmvq.cu mul_mat_vec_q`. **2d/챕터3 우리 실패와 정반대 구조:**

| | 우리(2d MO / 챕터3 batch gemm) | llama.cpp mmvq |
|---|---|---|
| 한 출력행 담당 | **1 warp** (MO 출력을 1 warp가 다 듦) | **nwarps warp 협력**(IN을 나눠) |
| 메모리 병렬성 | warp 1개분(부족 → latency 노출) | nwarps×32 thread분(높음) |
| 누적기/thread | MO(또는 B)개 → register 압박 | ncols×rows_per_block(작음) |
| 축약 | warp_reduce(1 warp) | **cross-warp shared 축약**(nwarps→1) |

핵심: **"multiple-output-per-warp"가 아니라 "multiple-WARPS-per-output"**.
- `nwarps = calc_nwarps`: B(=ncols_dst) ≤4 → **4 warp**, 5~8 → **2 warp**, >8 → 1.
- `rows_per_cuda_block(MO) = calc_rows`: B=1 → 1, B 2~8 → 2, >8 → 1.
- `tmp[ncols][rows]` 누적기(작음). IN을 `blocks_per_iter = vdr·nwarps·warpsize/qi`로 **블록 전체 thread가 분할**.
  partial sum → `tmp_shared[nwarps-1][ncols][rows][warpsize]`로 cross-warp 축약 → 1 warp가 최종.
- ⇒ register는 ncols×rows(작음) 유지, 메모리 병렬성은 **warp 수↑**로 확보. occupancy 안 깨고 latency 숨김.

## 2. 우리 현 gemm 진단 (ncu)

`gemm_batch_*`: 1 warp/output, 누적기 B개. ncu:
- **B=1**: reg 40, occ 65~75%, **compute/mem 27~34%, long_sb stall 22~40 = latency-bound**(1 warp/output라 in-flight 부족).
  = 챕터2 벽. mmvq 4-warp/output이면 in-flight 4× → stall↓.
- **B=16**: reg **48**(2d MO=4 함정 영역), per-seq −43% saturation. 누적기 16개가 register 천장.

## 3. B별 격차의 정체 + mmvq의 답

- **B=1~8 (B=1 215<277)**: 우리 1 warp/output = under-parallel(latency). **답 = multi-warp-per-output**
  (nwarps=4/2, IN 분할 + shared 축약). register 안 늘리고 메모리 병렬성↑.
- **B=16 (1016<1266)**: mmvq는 ncols **≤8만** 처리(MMVQ_MAX_BATCH_SIZE=8). **B>8은 MMQ**(tensor-core tiled gemm)
  별도 경로 → 그래서 그들이 B=16서 잘 scale. 우리 단일 B=16 커널은 reg48 천장.
  **답 후보 (둘)**: (가) B=16 = **2×(B=8)** 분할 — STEP1의 최적 B=8 커널 2회(register 천장 회피, 간단). 
                  (나) MMQ식 tiled gemm(tensor-core/shared tiling) — 크고 별도. 먼저 (가) 시도.

## 4. 설계 — multi-warp-per-output gemm (STEP1 핵심)

block = nwarps warp. block이 `rows_per_block` 출력행 × B열 담당.
- IN을 block 전체 thread(nwarps×32)가 stride 분할 → 각 thread `tmp[B][rows]` partial.
- cross-warp 축약: `tmp_shared[nwarps-1][B][rows][32]` → warp0이 최종 합 + 출력.
- nwarps: B≤4 → 4, B≤8 → 2 (mmvq 테이블 차용). rows_per_block: B=1 → 1, B 2~8 → 2.
- vec_dot은 경로별: fp32(float MAC), q4(unpack+MAC), W4A8(DP4A). 기존 gemm_batch의 inner를 재사용.
- **register 예산**: 누적기 B×rows (B=8,rows=2 → 16) + 축약 — 2d reg46~48 아래 유지가 목표. ncu 감시.
- ⚠️ shared 축약은 X-shared와 다름(weight/X 로드는 그대로 global, shared는 partial sum만) → bank conflict 위험 낮음.
  단 ncu로 shared bank conflict + occupancy 확인.

## 5. 검증 계획
- **정확성(절대)**: 새 multi-warp gemm == 기존 gemm_batch (fp32/q4/W4A8 × B) **오차0**(test_gemm_batch 확장).
  + in-engine batch decode == B single 유지 + spec on==off.
- **처리량**: B=1,2,4,8,16 in-engine 전후. microbench 금지(챕터3서 이긴 함정).
- **ncu**: occupancy 안 깨졌나(reg<46), long_sb stall↓(latency 숨김), compute/mem%↑, shared bank conflict.

## 6. 작업 순서 (하나씩)
- **STEP 1**: multi-warp-per-output 커널(fp32 먼저 정확성, 그다음 q4/W4A8). B=1~8 측정. occupancy 추적.
- **STEP 2**: B=16 = 2×B8 분할 (또는 막히면 MMQ 보고). per-seq 회복 측정.
- **STEP 3**: 전 구간(B 1~16) × 정밀도 재측정 + llama.cpp 재비교(전 구간 추월?).

## 7. 하지 않을 것
- vLLM(2단계), continuous batching(3단계). naive MO 재시도(occupancy 붕괴 확정). 정확성 변경. microbench 채택.

## 8. 사용자 결정 필요 (게이트)
1. **B=1 답 = multi-warp-per-output**(nwarps warp가 IN 분할 + shared 축약, register 안 늘림) — mmvq 정독대로. 동의?
2. **B=16 답 = 먼저 2×(B=8) 분할**(간단, register 천장 회피), MMQ tiled gemm은 막히면 — 동의? 아니면 바로 MMQ?
3. nwarps/rows 테이블 mmvq 차용(B≤4→4warp, B≤8→2warp; rows B≥2→2) — 동의?
4. 검증 = 새 커널 == 기존 gemm_batch 오차0 + B 스윕 in-engine + occupancy ncu — OK?
