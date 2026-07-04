# 파이프라이닝 깊이(multi-stage) POC — 측정 null (depth-1이 이미 최적)

> BM/BN 형상 WIN(256×64 +14.5%) 후속. 요청: "llama/CUTLASS는 조각크기+파이프라이닝 둘 다 튜닝, 우리는 절반만" →
> 파이프라이닝(겹치기) 추가. ★결론: **전제가 measured-반증** — mmq는 **이미 depth-1 파이프라이닝**(더블버퍼 prefetch),
> 더 깊은 파이프(STAGES≥3)는 **이득 없음**(smem-blocked on 챔프 + latency 여유 없음). 파이프 축 = null(C-1 KT/패딩과 같은 정직 null).

## STEP 0 — 전제 정정 2건 (측정/코드로)

1. **승자 형상 정정**: 실측 승자 = **256×64(tall) +14.5%**(49.1 vs 128×128 42.9), "64×256(wide) +8~9%" 아님
   (64×256은 실제 −42%). tall이 이긴 이유=W 재읽기 절반, wide는 정반대.
2. **★파이프라이닝은 이미 존재**: `gemm_mmq_tmpl.cuh`(및 원본)는 `sA[2]`/`sB[2]`(더블버퍼) + `load(0,0)` 후
   `load((it+1)&1, it+1)`(다음 prefetch) + `cp.async.commit/wait_group`(비동기) = **depth-1 파이프라이닝 완비**.
   "지금은 순차(불러오기→기다림→계산)"는 사실 아님. → 진짜 미탐색 = **깊이(depth 2→3+)**.

## STEP 0 ncu (재구조화 정당화 — 256×64)
- L1 49%·L2 51%(메모리 파이프 **미포화**)·long_scoreboard stall 2.05~2.69(latency 대기)·SM 42%·occ 32%. dram%=n/a.
- → latency-bound(bandwidth 미포화) = deeper 파이프 **도울 여지 있음**(weak green). 단 256×64 win이 traffic서 왔으니
  bandwidth도 요인 = deeper null 가능(measure 필요).

## STEP A/B — multi-stage 구현 + bit-id 게이트

`gemm_mmq_ms.cuh`: `template<...,int STAGES>` — sA/sB `[STAGES]` 링버퍼·프롤로그 STAGES-1 prefetch·매 스텝
`wait_group(STAGES-2)`(가장 오래된 것만 대기 = STAGES-1개 항상 in-flight)·매 스텝 1 commit(tail=empty commit로 count 유지).
- ★"안 온 데이터 계산" 차단: wait_group(STAGES-2)가 tile it 도착 보장 후 compute. 경계(tail) empty-commit 안전.
- ★게이트 PASS: **ms<STAGES=2> == tmpl<...> BITWISE**(계산 불변·깊이만 일반화) + 전 형상 vs gemv rel<1e-5 + 결정성.

## STEP C — 형상×STAGES 스윕 판정 (measured)

27B MLP up(512×17408×5120), warm, minblk=2:

| 타일(warp) | S2 | S3 | S4 | S6 |
|---|---|---|---|---|
| 256×64 (4×2) | **43.9** | ✗ smem 55KB>48 | — | — |
| 128×128 (2×4) | 37.7 | 37.2 | — | — |
| 128×64 (4×2) | 33.4 | 33.8 | 35.3 | — |
| 64×64 (2×2) | 23.8 | 23.4 | 24.0 | 23.2 |

(※ ms 커널은 STAGES 일반화 오버헤드로 tmpl보다 느림: 256×64 S2 ms 43.9 vs tmpl 49.1 — bit-id지만 extra 분기. 챔프는 여전히 tmpl 256×64=49.1.)

**★판정: deeper 파이프(STAGES≥3)는 lever 아님(measured null):**
1. **어느 STAGES≥3 조합도 256×64 못 이김**. 깊이 이득: 128×128 S3<S2(−1%)·64×64 flat(23~24)·128×64 S4 겨우 +5.7%
   (그래도 33→35, 256×64의 44/49엔 한참 못미침).
2. **챔프(256×64)는 깊어질 수 없음**: STAGES=3 = smem 55KB > 48KB 정적 한계. **48KB가 큰 타일↔깊은 파이프 상충**.
3. **깊이 이득 없음의 이유**: depth-1이 이미 latency 대부분 은폐 + 깊이↑ = smem↑ → occupancy↓로 상쇄. 메모리 win은
   이미 **타일 형상(256×64 traffic 절반)**이 가져감 → deeper가 덮을 추가 latency 적음.

### 정직 결론 (전제 반증 · 도망 아님)
- **"절반만 튜닝(파이프 미탐색)"은 measured-반증**: 파이프 half는 **이미 있었음**(depth-1). deeper는 GB10 48KB smem +
  낮은 latency 여유로 **null**. BM/BN(형상)이 진짜·유일 lever였음(+14.5%). C-1(KT/패딩 null)·이번(깊이 null) = 소거법으로
  **형상만 lever** 확증.
- **overclaim 없음**: 깊이 이득 없음이 measured. 방법론(multi-stage 재구조화+게이트, STAGES=2 bit-id) = 자산(다른 커널/
  size서 깊이가 도울 수 있으면 재사용). 격리 벤치·prefill 한정.

### 다음
- 형상(256×64)이 확정 lever → **in-engine 배선+서빙 검증**(격리≠in-engine)이 진짜 다음 관문.
- 파이프 깊이는 GB10서 tapped-out — 재방문은 smem 여유 큰 작은 타일 or 다른 아키(48KB↑) 한정.

## 산출물
- `kernels/gemm_mmq_ms.cuh`(multi-stage·STAGES=2 bit-id)·`tools/bench_mmq_ms.cu`(게이트+타이밍). null POC 자산.
