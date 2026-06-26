# 2d 설계/측정 — speculative 성능 최적화 (수준2)

> 2c(정확성·상태관리 완성, 1.17× 가속) 위에서 **성능만**. 정확성 불변이 절대 조건.
> 헌법: microbench 신뢰 금지 — in-engine end-to-end t/s + ncu/nsys로 판정. 하나씩 측정.

## B0 — Baseline 고정 (재현 가능)

명령: `./tests/test_spec artifacts/qwen35-0.8b/weights artifacts/qwen35-0.8b/mtp_golden 80 4`
(prompt=15, gen=80, K=4). 3회 재현 안정:

| 지표 | 값 |
|---|---|
| plain t/s | 37.0 |
| spec t/s | 43.1 |
| **speedup** | **1.165×** |
| avg accept | 3.76 / 4 (94%) |
| cycles | 17 (80토큰) |
| phase (wall) | draft 27% / verify 68% / catchup 5% |

정확성: spec on==off greedy mismatch 0 (절대 조건, 매 최적화 후 재확인).

## nsys 절차 (per-kernel "왜")

```
mkdir -p $SP/nvtmp
TMPDIR=$SP/nvtmp nsys profile -t cuda --stats=true -o $SP/sp --force-overwrite true \
  ./tests/test_spec <w> <g> <N> <K>
```
(주의: nsys는 `/tmp/nvidia` 권한 거부 → TMPDIR 지정 필수.) ncu도 동일하게 TMPDIR.

### B0 nsys 발견 (N=24, K=4) — GPU 커널 시간 분포
| 커널 | GPU time % |
|---|---|
| **gemv_f32_kernel** | **90.1%** |
| gdn_step_kernel | 6.9% |
| rmsnorm_zc | 1.2% |
| attn_decode | 0.5% |
| 나머지(add/conv/swiglu/rope/embed/gate) | 각 <0.3% |
- cudaLaunchKernel 오버헤드: 6284 launch, ~4.8us/launch = 30M ns.

**해석 (중요 — 우선순위 재조정):**
- GPU는 **압도적 gemv-bound (90%)**. 모든 matmul(qkv/o/gate/up/down/lm_head/MTP)이 naive gemv(1 warp/output row, X 재사용 없음, W coalescing 나쁨).
- wall-clock phase(draft 27% 등)와 GPU-time(gemv 90%) 갭 = **host argmax(V=248320 D2H + CPU 루프) + launch 오버헤드**.
- → **PHASE A(gemv 최적화)가 진짜 GPU 레버.** PHASE B(device argmax)는 wall-clock host/sync 오버헤드 제거 (갭 축소).
- lm_head는 full-V(248320×1024 dot) 계산 자체가 inherent — argmax하려면 전 V 필요(B2 "계산 축소"는 근사 없이는 불가; fuse만 가능).

## 최적화 로그 (각: 정확성 불변 + t/s 전후 + nsys/ncu + 해석)

### B1 — device argmax (draft) [측정 완료 → 무효, revert]
- 변경: mtp_step draft argmax를 host(V D2H + CPU루프) → device(block-reduce 커널, kernels/argmax.cuh).
- **결과: spec 43.1 → 42.0 t/s (−2%, 역효과).** 정확성 불변(mismatch 0). persistent 버퍼로도 동일(−2%).
- **원인 (데이터)**: ① 단일 블록(256 thread) argmax가 V=248320 underutilize → host DMA+CPU루프보다 느림.
  ② 근본: draft 비용은 **lm_head gemv(full-V 계산)**지 argmax/host전송이 아님 (nsys gemv 90%).
  argmax는 V를 계산하는 것 대비 싸므로 어떤 argmax 최적화도 무효.
- **조치: revert** (host argmax 유지). 의뢰서 "안 되는 최적화 억지로 밀지 않는다" 준수. argmax.cuh는 보존.

### B2 — lm_head 축소 [분석상 무효]
- draft lm_head = embed weight **1GB(248320×1024×4B) 읽는 memory-bound** gemv. exact argmax엔 전 V 계산
  =전 weight 읽기 불가피. 근사 없이는 축소 불가 (근사는 정확성 깨므로 범위 밖). → 무효.

### 대역폭 분석 (ncu 막혀 타이밍 기반)
- body ≈ 751M params = **3.0GB fp32**. plain 37 t/s = 27ms/token → **달성 111 GB/s = peak(273)의 ~40%**.
- 대역폭 상한 @273GB/s = **91 t/s**. → **포화 아님, 헤드룸 있음.** 갭(40%)은 launch 오버헤드(6284 launch×4.8us)
  + 多수 tiny 커널 + gemv 비효율. 즉 PHASE A(gemv) + launch 축소가 잠재적으로 유효.

### ⚠️ 블로커 — ncu 권한 (ERR_NVGPUCTRPERM)
- `ncu`가 GPU 성능 카운터 접근 권한 없음(admin 필요). dram%, bank conflict, sectors/req, occupancy 측정 불가.
- nsys(타이밍)는 동작 → 커널 시간 분포는 얻음. 그러나 **헌법 "성능 판단은 ncu 지표 동반"을 ncu 없이 못 지킴**.
- PHASE A(gemv 최적화)는 정확히 X-shared 함정 영역(microbench↑ ≠ in-engine↑) — ncu 진단 없이 진행은 위험.
- 해결: admin이 `NVreg_RestrictProfilingToAdminUsers=0` + 드라이버 리로드, 또는 ncu sudo 실행.

## PHASE A — STEP A0 (ncu 진단, 권한 해결 후)

**환경 변화**: ncu 권한 작업 시 GPU 클럭도 해제된 듯 — baseline t/s ~2× 상승.
새 baseline: **plain 65.3 / spec 80.1 t/s = 1.227×** (accept 3.76, mismatch 0). 비율은 1.165→1.227로 개선.

### gemv_f32 ncu 지표 (여러 shape)
| gemv (grid) | Mem TP% | Compute% | sectors/req | Occupancy | Dur | 실제 BW(역산) |
|---|---|---|---|---|---|---|
| 768 (in_proj_qkv S=1) | 12% | 7.9% | 4.0 | — | 107us | 235 GB/s* |
| 11520 (in_proj_qkv S=15) | — | — | 4.0 | — | 410us | ~61 GB/s (25MB W) |
| 448 (FFN) | 11% | 7% | 4.0 | 89% | 70us | — |
| 256 | 9.7% | 6.2% | 4.0 | 98% | 46us | — |
| 30/2 (in_proj_a/b OUT=16) | 0.6% | 0.1% | 4.0 | 21% | 6-16us | 순수 오버헤드 |

(*GB10은 `dram__throughput` 미노출(unified LPDDR5X). ncu "Memory TP%"는 L1/L2 파이프 기준이라 실제 DRAM과
 다름. 실제 BW는 duration+W바이트로 역산.)

### 진단 (확정, 추측 아님)
1. **coalescing 최적**: sectors/req=4.0 (float 32thread×4B=128B=4sector). 접근 패턴 문제 없음.
2. **compute-bound 아님**: SM throughput 3-8%.
3. **대역폭 포화 아님**: 큰 gemv도 ~22-86% 변동, 디코드 shape(작은 S)는 22% 수준. tiny gemv는 순수 launch/latency.
4. **latency/MLP-bound**: occupancy 높아도(89-98%) BW 못 채움 → naive(1 warp/output + 끝 warp-reduce 직렬)가
   메모리 latency 미은닉, warp당 outstanding load 부족.

### ★ 전략 분기 (게이트 결정)
- **"비효율 존재" 분기 확정** — fp32 gemv 커널 헤드룸 있음.
- **q4 아직 무효**: BW의 ~22%만 쓰는 latency-bound 커널은 weight 1/8(q4) 줘도 안 빨라짐. **fp32 gemv 먼저.**
- **A1 방향(ncu 지목)**: warp당 다중 output(mMO식, X 재사용 + outstanding W load↑) + float4 vectorized load → MLP↑.
  학습 자산 reference의 m1/mMO 변형 참고(단 X-shared 함정: in-engine으로만 판정).

### STEP A1 — gemv MO + float4 [측정 완료, 한계]
변경: gemv_f32 → warp당 MO output + float4 load (X는 global/L1, smem 안 씀 = bank conflict 회피).
정확성: 전 스위트 오차0, spec on==off mismatch 0 (불변).
| 버전 | plain | spec | ncu |
|---|---|---|---|
| baseline(naive) | 65.3 | 80.1 | occupancy 98%, reg 36 |
| MO=4 | 65.5 | 81.3 | **occupancy 98%→20% 붕괴**, reg 46 (X-shared 함정 재현) |
| MO=2 (채택) | 66.7 | 81.4 | reg ~40, occupancy 손실 완화 |
- **결론: gemv MO 최적화 = +2% in-engine (미미).** decode shape가 작아 MO가 병렬성을 latency 은닉 한계 아래로
  떨굼(occupancy collapse). gemv-internal 헤드룸 거의 없음. MO=2는 작은 실이득이라 유지(정확성 안전).

### ★ nsys가 드러낸 더 큰 레버 (gemv 아님)
N=80 spec+plain nsys:
- GPU 커널: gemv 94%(2038M ns), gdn_step 3%, 나머지 <1%. **lm_head gemv 1콜 = 59ms (1GB/59ms = 17GB/s 비정상)**.
- CUDA API: **cudaMalloc 337M ns(7856콜) + cudaFree 73M = 0.41s** ← mtp_step/decode가 매 호출 scratch malloc/free.
  cudaLaunchKernel 138M(45100콜), cudaMemcpy 134M.
- **발견: A1 gemv-internal(+2%)보다 (a) malloc churn 0.41s 제거, (b) lm_head gemv(17GB/s) 수리가 훨씬 큰 레버.**
  둘 다 gemv 커널 내부 튜닝이 아님 → 전략 재조정 필요. (malloc=낭비제거, lm_head=특정 gemv shape 비효율.)

## PHASE A 게이트 — 결론 (A1 후)
- gemv MO/float4 = +2% (occupancy-limited, 헤드룸 소진).
- 진짜 레버 = ① **scratch malloc churn 제거(persistent 버퍼, 0.41s)** ② **lm_head gemv 비효율 진단/수리(17GB/s)**.
- → 사용자 결정: A1 종료하고 ①②로 갈지 / q4로 갈지 / 종료.

## PHASE B 게이트 — 결론
- B1 무효(−2%, draft는 gemv-bound), B2 무효(lm_head memory-bound 불가피). **PHASE B 헤드룸 없음.**
- 진짜 레버 = PHASE A(gemv, 40%→상향 여지) 또는 q4(8× weight 트래픽↓, 범위 밖이나 대역폭의 본질 레버).
- **단 PHASE A는 ncu 필요 (X-shared 함정). ncu 블로커 해결 전엔 헌법 위반 위험.** → 사용자 결정 필요.
