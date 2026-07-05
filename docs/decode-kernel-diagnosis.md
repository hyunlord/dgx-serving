# decode 커널 진단 — kernel lever 소진, bandwidth-bound (measured → q2가 진짜 lever)

> 형상 트랙(prefill) 완결 후 실서빙 진짜 병목 = decode(memory-bound). measure-first 진단: decode가 대역폭 얼마 뽑나,
> kernel lever 남았나. ★결론: **decode 커널(gemv_w4a8_mlp)은 lever 소진(coalesced+occupancy 90%+U 최적, latency-bound
> ~59-75% BW). 남은 kernel 여지 없음** → decode는 근본 bandwidth-bound(토큰당 13.5GB weight read) → **q2(bytes↓)/
> 대역폭 돌파가 진짜 lever**(measured 확인). overclaim 없음.

## STEP 0 — 대역폭 달성률 + ncu 진단 (27b dense, warm)

- single decode = **11.9 t/s**. 이론 상한 = BW 273 ÷ (27B×0.5B q4 = 13.5GB/tok) ≈ 20 t/s → **~59%**(메모리 ⑤ 다른
  분모로 ~75%). 높은 편.
- ★ncu(gemv_w4a8_mlp<2>, S=1 decode 커널):
  - **sectors/req = 4.40**(clock-무관) = **완전 coalesced**(128B 로드=4섹터). weight read 최적 — coalescing lever 없음.
  - **occupancy(warps_active) = 90%** — warp 최대(더 못 늘림).
  - **long_scoreboard stall = 51**(매우 높음) = **latency-bound**(coalesced인데도 메모리 대기 — 토큰당 13.5GB read latency).
- 진단: coalescing 최적 + occupancy 최대 + latency-bound. 남은 후보 = outstanding-load U(latency 은폐).

## STEP A — outstanding-load U 스윕 (진단이 가리킨 lever)

`gemv_w4a8_mlp<U>`: U청크 먼저 issue 후 compute(long_scoreboard 은폐). compute 순서 baseline 동일 = **전 U bit-exact**.
env MLP_U 토글(기본 2=엔진값). 27b decode t/s(warm):

| MLP_U | decode t/s |
|---|---|
| **2** | **11.9** ★최적 |
| 3 | 11.5 |
| 4 | 11.7 |
| 6 | 11.4 |

- ★**U=2 이미 최적**(U↑는 오히려 느림 — 레지스터/발사 압력↑로 스케줄 악화). bit-id: MLP_U=4 == baseline(argmax ALL·PASS).
- → **outstanding-load lever도 소진**(U=2가 sweet spot, 메모리 ⑤ 재확인).

## STEP C — 판정 (measured)

★**decode 커널(gemv_w4a8_mlp)은 모든 micro-lever 소진**:
- coalescing 최적(4.40 sectors/req)·occupancy 최대(90%)·outstanding-load 최적(U=2)·latency-bound(long_scoreboard 51).
- 남은 kernel 여지 **없음**. decode 갭은 kernel 비효율이 아니라 **근본 bandwidth**(토큰당 전 weight read = 13.5GB @ 273GB/s).

### decode 상한의 의미 → 진짜 lever
- **prefill(형상)로는 서빙 갭 못 메움**(형상은 compute-bound prefill만·decode M=1엔 안 걸림) — 확인됨.
- **decode도 kernel로는 천장**(coalesced·occupancy·U 다 최적, latency-bound) — 확인됨.
- → **진짜 decode lever = ①q2(bytes↓: 4bit→2bit = weight read 절반 → 대역폭 상한 ~2×) ②대역폭 돌파(C, 2번째 기둥)**.
  이게 llama 갭(3-way 26b single 0.28×)의 실제 성질 = 대역폭(둘 다 같은 BW지만 llama가 더 성숙히 뽑거나 bytes 적음).
- **MoE decode(26b/35b, ~24-40% roofline)**: dense보다 낮으나 병목은 **구조적**(토큰당 640 expert-gemv launch+tiny matmul
  +GDN 순차) — moe_fused가 launch 이미 잡음(42.8→51.8). 남은 것도 kernel 미세opt 아닌 구조(8-expert 병렬 한계).

### 정직 결론 + 다음
- **decode 커널 lever 없음**(measured — coalescing/occupancy/U 다 최적). 도망 아님 — 진단이 "kernel 천장"을 가리킴.
- **다음 = q2 극한양자화**(bytes↓ = decode 대역폭 상한 직접 ↑, 실서빙 갭의 진짜 공략)·또는 대역폭 돌파(C, 2번째 기둥).
- MLP_U 토글은 진단 자산(기본 2=무회귀). overclaim 없음(U 여지 없음 measured).

## 산출물
- `kernels/gemv_w4a8.cuh`(launch_gemv_w4a8_mlp_auto env MLP_U 토글·기본 2 무회귀)·qwen35_decode 배선. docs/decode-kernel-diagnosis.md.
