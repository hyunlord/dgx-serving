# 챕터7 STEP 0b — Phase B: tensor-core int8 MMA prefill GEMM (11× 갭 좁히기)

> Phase A(software tiled): util 4%→79%, 갭 25×→11×. 한계 도달 — 남은 11×의 대부분이 "tensor-core 미사용".
> llama.cpp 900 t/s = int8 tensor-core MMA. software dequant→fp32로는 못 넘음. 하드웨어로 넘어야.
> **검증: int8 MMA `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` 가 sm_120(GB10) 컴파일 성공.** 가용 확정.

## operand 포맷 — 이미 있음 (챕터2b W4A8)
- **X → int8**(`quantize_x_q8`, per-block-QBLK=32 symmetric): xq[S,IN] int8, xs[S,IN/32] scale fp32.
- **W int4**(per-block-32, zero-point 8) → unpack `(wq−8)` signed int8.
- dot = Σ(xq · (wq−8)) × xs·sw  (블록당). int8×int8→int32 정확(MMA 무손실), 양자화 오차만(W4A8 = 챕터2b).

## 핵심: K-tile = QBLK = MMA k = 32 정렬 (우아함)
- MMA `m16n8k32`: k=32 = 정확히 한 양자화 블록. → 블록당 int32 누적 후 ×(xs·sw) → fp32 누적.
- 블록 경계 = MMA 경계 = 스케일 경계. scale broadcast/누적 단순.

## 설계 (warp-level MMA 타일)
- **operand**: A = X int8 [M16, K32] (row), B = W int8 [N8, K32] (col, unpack된 (wq−8)). C = int32 [M16, N8].
- **CTA 타일**: warp 그리드로 [WM×16, WN×8] 출력. 각 warp이 mma.sync 누적.
  - K 루프: 블록(32)씩 — cp.async로 X int8·W packed→shared, ldmatrix로 fragment 로드, mma.sync.s8 → int32 누적,
    ×(xs[blk]·sw[blk]) → fp32 acc. (또는 int32를 블록끝까지 모았다 스케일 — 블록=k=32라 매 MMA마다 스케일.)
- **W unpack**: int4 packed → shared에 (wq−8) int8로 적재(2× shared지만 int8 OK). 또는 ldmatrix 전 레지스터 unpack.
- **fragment 레이아웃**: m16n8k32 s8 — A는 lane별 4×int8(int 1개)×4, B는 2×int. ldmatrix.x4(A)/x2(B) 또는 수동 로드.
- **cp.async**(sm_121 `cp.async.ca.shared.global`) 2-stage double-buffer(load/compute 겹침). Phase A K-tiling 유지.
- bank conflict: shared 패딩 + ldmatrix(자체 swizzle). Phase A 교훈(글로벌 coalescing 우선) 유지.

## 정확성 전략
- int8 MMA = int8×int8→int32 **정확**(하드웨어, 라운딩 없음). → MMA 결과 == `gemv_w4a8`(DP4A) **오차0**(둘 다 int8 dot).
- 따라서 Phase B 검증 = **새 MMA GEMM == gemv_w4a8 오차0** + golden(W4A8 비트) + 27B/35B forward(use_q4=2) 회귀0.
- 단발 == decode 누적 동등성. (W4A8 양자화 오차는 기존과 동일 — Phase B는 속도만 바꿈.)

## 작업 순서 (위험 큰 작업 — 단계별 de-risk)
- **STEP B.0**: 단일 warp-tile MMA 격리 — int8 A·B → int32 == CPU int dot 오차0. fragment 레이아웃 확정.
- **STEP B.1**: 블록 스케일 통합(int32 × xs·sw → fp32) → 단일 CTA 타일 == gemv_w4a8 오차0(작은 M).
- **STEP B.2**: K-tiling + cp.async double-buffer + ldmatrix 확장. 큰 N/K.
- **STEP B.3**: ncu(tensor active %, bank conflict, util) + 정확성 + 회귀.
- **STEP B.4**: 27B prefill 재측정 — 11× → ?×. cuBLAS(llama 900) 근접도.

## 하지 말 것 / 위험
- **성급한 full 커널** — 단일 warp-tile MMA 오차0(B.0)부터. fragment 레이아웃 틀리면 전부 틀림 → 격리 먼저.
- mma.sync.block_scale(mxfp4, X도 E2M1) — 더 복잡. 우선 s8.s8.s32(검증된 W4A8 operand). 부족하면 그때.
- 작은 M부터(크래시 교훈), 메모리 가드 유지. bank conflict ncu 확인.
- **막히면 보고**: fragment 레이아웃 안 맞음, 스케일 통합 오차, MMA가 cuBLAS에 여전히 한참 못 미침.

## STEP 0b 게이트 — 결정
1. **방향: s8.s8.s32 int8 MMA(검증된 W4A8 operand) Phase B** 동의? (mxfp4 block_scale은 나중.)
2. **K-tile=32=QBLK=MMA k 정렬** 동의? (블록당 스케일.)
3. **STEP B.0(단일 warp-tile MMA 오차0)부터 격리 de-risk** → 확장. 동의?
4. 정확성 레퍼런스 = gemv_w4a8(DP4A, 둘 다 int8 dot → 오차0 기대). 동의?
