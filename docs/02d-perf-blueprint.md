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

## PHASE B 게이트 — 결론
- B1 무효(−2%, draft는 gemv-bound), B2 무효(lm_head memory-bound 불가피). **PHASE B 헤드룸 없음.**
- 진짜 레버 = PHASE A(gemv, 40%→상향 여지) 또는 q4(8× weight 트래픽↓, 범위 밖이나 대역폭의 본질 레버).
- **단 PHASE A는 ncu 필요 (X-shared 함정). ncu 블로커 해결 전엔 헌법 위반 위험.** → 사용자 결정 필요.
