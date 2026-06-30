# 챕터9 STEP 0 — FP4 텐서코어 de-risk (NO-GO: toolkit-gated)

## 요약

가장 큰 미사용 compute lever = **FP4 텐서코어**(우리는 int4 weight를 DP4A 스칼라 dot으로 처리, GB10 네이티브
FP4 MMA 미사용 — 포럼 NVFP4가 우리보다 빠른 핵심 이유). 챕터7 방식(투자 전 컴파일 de-risk)으로 가용성 규명:
**HW엔 있으나 우리 CUDA 13.0 toolkit이 못 emit → 현재 NO-GO.**

## HW: FP4 MMA 있음 (sm_121/GB10)

- warp-level `mma.sync` (tcgen05 아님 — 그건 sm_100). 명령(forum, PTX ISA 9.3 Fig 93-96):
  `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0`
- A=4×u32(e2m1 FP4, .u4/.s4 m16n8k64 레이아웃), B=2×u32, D=4×f32, scale=ue8m0(16-element micro-block, scale_vec::2X).
- FP4 게이트 = `sm_121f`(family-mode). nvcc 13.0이 `-arch=sm_121f` 컴파일 + `.target sm_121f` 전파까지는 함.

## Toolkit이 막음 (측정 확정)

| 검증 | 결과 |
|---|---|
| nvcc/ptxas 버전 | CUDA **13.0** (시스템에 13.0만, 13.1+ 없음) |
| ptxas 최대 PTX `.version` | **9.0** (`.version 9.3` 명시 거부: "current version is 9.0") |
| mxf4 필요 PTX ISA | **≥9.2(web)/9.3(forum)** |
| `kind::mxf4` block-scale on sm_121f/9.0 | ❌ "Instruction 'mma with block scale' not supported" + "Feature '.kind::mxf4' not supported" |
| `kind::mxf4` 비-block-scale on sm_121f/9.0 | ❌ "Instruction 'mma with FP6/FP4' not supported" + "Illegal modifier '.kind::mxf4'" |
| llama.cpp #19662 (동일 에러) | CUDA **13.1.115도 실패** 보고 — 13.1조차 불충분(해결 미확정) |

→ 우리 ptxas(9.0)는 어떤 FP4 mma도 미구현. `sm_121f` 타겟은 전파되나 ptxas가 mxf4 미지원이라 무의미.

## 판정 & 다음

**FP4 lever = toolkit-gated NO-GO(현재).** 쓰려면:
1. CUDA **13.1+/13.2 업그레이드** (시스템 변경, 사용자 결정) — 단 13.1도 실패 보고라 **13.2+ 필요할 수, 불확실**.
2. 업그레이드 후 재-de-risk: 그 toolkit이 실제 mxf4 sm_121f 컴파일하나(작은 커널) → 정확성(FP4 e2m1 weight가 golden
   유지하나, block-scale 16) → 속도(FP4 MMA vs 우리 DP4A) → gemv/gemm 포팅.

**교훈:** 챕터7 TMA/tcgen05 규명과 동형 — HW 기능이 있어도 toolkit/PTX 버전이 게이트. ~10분 컴파일 de-risk로
큰 커널 작업 헛수고 차단(NO-GO 확정). 추측·착수 전 가용성부터.

**다른 길(업그레이드 안 하면):** 잔여 lever 대부분 micro-lever 천장(perf-baseline). 후보 = MTP accept↑, batch
plateau(per-seq attn 배치화), 또는 정확성 차별화(우리 mism=0 vs DFlash int4 품질 불명). 단 FP4만큼 크진 않음.

probe: scratchpad `/tmp/fp4mma_probe.cu`(미커밋). 메모리 [[fp4-mma-toolkit-blocked]].
