# 챕터7 STEP 0 (Phase C) — MMA double-buffer 파이프라인 설계

> 측정이 결정한 설계. 추측 없음. 입력 = nsys prefill 분해 + MMA 커널 ncu(이 세션).
> 선행: Phase A(software tiled q4, X 재read 제거) → Phase B(tensor-core int8 MMA) → **Phase C(파이프라인 겹침)**.

## 0. 왜 Phase C인가 — 측정 근거 (추측 아님)

**순수 prefill 분해 (nsys, 2048 토큰, W4A8, decode 격리):**
| 구성요소 | 비중 | 커널 |
|---|---|---|
| **matmul** | **74.4%** | gemm_mma_w4a8_tiled |
| attention O(Sq²) | 14.9% | attn_decode |
| GDN | 7.7% | gdn_step |
| 나머지 | ~3% | swiglu/quant/norm/rope/add |

→ 남은 5×의 #1 = matmul. (attention은 O(Sq²)라 8K/32K선 비중↑ — 그땐 FlashAttention. 지금 2048선 matmul이 5배 큼.)

**MMA 커널 ncu (왜 util 36%):**
- Memory 41.7% / Compute(SM) 35.4% — **둘 다 비포화 → 자원-bound 아님.**
- Achieved Occupancy 63.6%(이론 66.7%) — occupancy 양호, 문제 아님.
- **Warp stall 59.9% = "shared 적재 대기 scoreboard 의존성"** (23.9 사이클 중 14.3 stall).

진단: load→`__syncthreads`→MMA→`__syncthreads` **직렬화**. K=5120 → blk당 160 배리어, double-buffer 없음.
텐서코어(35%)와 메모리(42%)가 겹치지 못하고 번갈아 논다 = **latency-bound**.
→ 흔한 추측("util 낮으니 ldmatrix/큰 타일")은 틀림(자원 비포화). **진짜 레버 = 겹침(double-buffer).**

## 1. 처방 (측정에서 직접)

1. **cp.async 2-stage 더블버퍼 (핵심)** — 다음 K-타일을 `cp.async.ca.shared.global`로 prefetch하며
   현재 타일 MMA 실행 → scoreboard stall 은닉, mem 42% ↔ compute 35% 겹침.
2. **큰 K-타일** — 스테이지당 KSTAGE=2~4×QBLK 처리 → 배리어 160→80→40, cp.async 배칭↑, scale 분할 상환.
3. **scale 호이스트** — 현 blk마다 xs/sw global read + fp32 mul 누적을, K-타일 단위로 묶어 정리.

cp.async 컴파일 확인됨(sm_120, `cp.async.ca.shared.global` 16B + commit_group/wait_group OK).

## 2. 구조 (현 gemm_mma_w4a8_tiled_kernel 기반)

현재 (Phase B, gemm_mma.cuh):
- CTA 타일 MMA_BM=64 × MMA_BN=64, 8 warp(2×4 그리드), warp당 4 MMA 타일(2 ms × 2 ns).
- K-루프 nblk=K/QBLK 회: `load sX[64][32]+sW[64][32](W unpack) → sync → MMA → sync`.
- shared: sX[64][36]+sW[64][36] int8 = 4.6KB (작음 — 더블버퍼 여유 충분).

Phase C 변경:
- **더블버퍼 shared**: `sX[2][BM][KTILE+PAD]`, `sWp[2][BN][KTILE/2]`(W는 packed int4 그대로 staging).
  KTILE=QBLK×KSTAGE. smem 2버퍼라도 ~10–20KB (99KB 한참 안 — occupancy 영향 없음 확인).
- **W는 cp.async로 packed int4 그대로 적재**, MMA fragment build 때 unpack((q-8)). compute-store-sync 제거.
  (현재는 적재 시 unpack→sW. Phase C는 packed staging + read-time unpack로 cp.async 순수화.)
- **파이프라인**: prefetch stage0 → for k: { cp.async stage k+1 → commit; wait_group; sync; MMA(stage k); sync }.
  2-stage면 wait_group 1(다음 것 1개 in-flight). 잔여 stall 보이면 3-stage.

### W unpack 위치 결정
- 현재: 적재 루프가 int4→(q-8) int8 unpack해서 sW에 씀 (compute + sync 필요 → cp.async 불가).
- Phase C: packed int4를 `sWp`에 cp.async(raw bytes), **fragment build 때** 2바이트(4 nibble) 읽어
  (q-8) int8 4개로 풀어 int 레지스터 구성. unpack ALU가 텐서 파이프에 겹쳐 숨음. X는 int8이라 순수 cp.async.

## 3. 검증 계획 (헌법 — 정확성 먼저, ncu, 막히면 보고)

1. **정확성 오차0**: 새 커널 == `gemv_w4a8`(기존 검증된 W4A8) — `tests/test_gemm_mma.cu` 같은 케이스
   (M/N/K 배수·비배수: {64,128,512}{200,128,512}{256,130,256}{129,256,1024}) rel < 1e-5. 통과 전 통합 금지.
2. **동등성**: 엔진 통합(use_q4=2) 후 prefill 단발 == 기존 경로, 청크 prefill 누적 오차0, 0.8B 회귀 0.
3. **ncu (핵심 — 처방이 들었나)**: warp stall scoreboard 59.9% → 감소 확인. Compute(SM)/Memory util 상승
   (35%/42% → 겹침으로↑). 안 줄면 보고(가설 틀림 — 3-stage or 다른 stall).
4. **성능**: prefill t/s (현 148–187 → ?). matmul 74% 비중이라 커널 N× → 전체 prefill ~0.74×(1−1/N) 개선 기대.

## 4. 안전 (크래시 교훈)
- 작은 Sq부터(64~256) 정확성 통과 후 상한↑. 단발 prefill Sq 상한 + 메모리 가드 유지([[gb10-unified-memory-crash]]).
- smem 더블버퍼가 occupancy 안 깨는지 ncu 확인(현 occ 63.6% 유지/개선 목표).

## 5. 하지 말 것
- attention/GDN 손대기(이번은 matmul). 긴 context FlashAttention은 별도(8K/32K 타깃 시).
- ldmatrix/큰 타일 단독 추구(ncu가 자원 아닌 겹침 문제라 했음 — double-buffer 먼저).
- decode 커널(1.4× 갭은 별도 fused q4).
- 성급한 큰 Sq, 성능 위해 정확성 양보.

---
*Phase C.1(cp.async 2-stage 더블버퍼) 구현 → 오차0 → ncu scoreboard stall 감소·util 상승 → prefill t/s.
부족하면 C.2(3-stage / 큰 K-타일 / scale 호이스트). 단계마다 같이 본다.*
