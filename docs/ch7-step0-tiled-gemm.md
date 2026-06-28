# 챕터7 STEP 0 — 진짜 tiled q4 GEMM (prefill 25× 갭 좁히기)

> 3-way 측정이 정한 #1 우선순위: prefill이 llama.cpp/vLLM에 25~38× 뒤짐. 둘 다 cuBLAS/tensor-core급 GEMM.
> 우리 Phase-1(gemm_batch_q4<16>)=1.84 TFLOP/s — gemv 4.1×나 cuBLAS 25× 부족. ncu: Phase1 후 compute-bound.

## 왜 Phase-1로 부족한가 (ncu 확정)
- gemm_batch_q4<16>: weight row 1회 read → 16행 재사용(레지스터 acc[16]). 그러나:
  - **X를 출력당 재read**(shared 안 씀) → ncu: gemm_batch 11.49M sec/tok, mw 6.26M인데 시간 동일 = compute/util bound.
  - util 1.84 TFLOP/s = peak의 ~4% → **연산 자체가 비효율**(타일 작음, 레지스터 제약 B≤16, tensor-core 미사용).
- = 진짜 GEMM(shared-mem X·W 타일 + tensor-core/DP4A MMA)이 아니면 cuBLAS 못 따라감.

## 설계 — 타일드 W4A8 GEMM (tensor-core 경로)

### 핵심: shared-mem 양 타일 + MMA
- 블록이 출력 타일 [TILE_M, TILE_N] 계산. K(=IN)를 TILE_K씩 루프:
  - X 타일 [TILE_M, TILE_K] → shared (1회 로드, TILE_N 출력에 재사용).
  - W 타일 [TILE_N, TILE_K] → shared (dequant 또는 int4 그대로, TILE_M에 재사용).
  - **양쪽 재사용** = Phase-1의 "X 재read" 제거.
- **GB10 Blackwell tensor core**: int8 MMA(W4A8 = int8 X × int4→int8 W, DP4A 또는 mma.sync). q4는 int4→int8 unpack.
  - W4A8 경로(use_q4=2, 챕터2b) 활용: X int8(per-block-32) × W int4. tensor-core int8 MMA가 핵심 가속.
- bank conflict 회피: X-shared 패딩/swizzle(★ 우리가 여러 번 데인 영역 — 챕터3 X-shared 교훈, ncu bank_conflict 0 확인 필수).

### Phase 분리 (위험 관리)
- **Phase A**: shared-mem 타일 GEMM(dequant→fp32 누적, tensor-core 없이). X·W 양 재사용으로 util↑.
  검증 쉬움, bank conflict만 주의. 목표: Phase-1의 4% → 수십% util.
- **Phase B**: tensor-core int8 MMA(W4A8). 최대 throughput. mma.sync/wgmma 복잡 → Phase A util 부족 시.
- 타일 크기(TILE_M/N/K)는 occupancy 스윕(ncu) — 챕터3/4 "2d MO, B별 occupancy" 교훈.

### 어디 쓰나
- prefill 전용(S 큼). mm()이 S>임계면 tiled GEMM, S=1 decode는 gemv 유지.
- 청크 prefill과 결합(이미 있음): 청크 내부 행렬곱이 tiled GEMM.

## 검증 계획 (헌법 — 특히 정확성·ncu)
1. **정확성**: tiled GEMM == gemv_q4 오차0 (작은 Sq, 안전). test_prefill_gemm 확장.
2. **bank conflict 0**: ncu l1tex bank conflict 지표 = 0 확인(X-shared 함정).
3. **util↑**: ncu sm__throughput / tensor active. 목표 Phase-1 4% → Phase A 수십% → Phase B cuBLAS 근접.
4. **트래픽↓**: X 재read 제거 확인(global load sectors).
5. **회귀 0** + 27B/35B forward + **재측정으로 prefill tok/s 격차 축소 확인**(llama.cpp 대비 25× → ?×).
6. **안전**: 단발 Sq 상한 유지, 작은 Sq부터. 큰 측정은 정확성 통과 후.

## STEP 0 게이트 — 결정 필요
1. **방향: 타일드 W4A8 GEMM(shared X·W 타일 + tensor-core)** 동의? prefill 25× 갭의 진짜 해법.
2. **Phase A(shared 타일, tensor-core 없이) 먼저** → 부족하면 Phase B(tensor-core MMA). 동의?
   (Phase A로 util 4%→수십%면 큰 진전. tensor-core는 복잡·위험이라 필요 확인 후.)
3. **X-shared bank conflict를 ncu로 0 확인**(과거 데인 지점) 필수 게이트. 동의?
4. 구현: STEP1(Phase A 타일 GEMM + 오차0) → STEP2(occupancy/bank ncu 튜닝) → STEP3(재측정 격차 확인) → (필요시) Phase B.
   동의? 아니면 처음부터 tensor-core?
