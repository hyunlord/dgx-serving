# 챕터8 STEP C+D — MTP speculative decode로 27B decode 추월

## 요약

fp16 base(decode 11.6 t/s, 갭 1.18×) 위에 **MTP speculative decode**를 통합하고, verify 병목을
**소-batch W4A8 커널**로 제거해 **27B decode 추월**을 달성했다.

- **27B W4A8 spec K=3 = 16.9 t/s** = plain 12.1의 **1.40×** = **llama.cpp 13.7도 1.23× 추월(실용)**.
- 정확성: spec 출력 == plain greedy 출력 **bit-identical (mism=0)** — 0.8B(fp32/q4/W4A8) + 27B(W4A8).
- 결정성 30회 고유 출력 1, 직교성(fp32/q4/W4A8) 전부 통과.

**정직한 좌표:** fundamental 추월은 아니다 — MTP는 곱셈자라 llama도 켜면 곱한다. 그러나
**표준 llama 배포(MTP 미탑재) 대비 실용 추월**이며, 2번째 기둥(대역폭 한계 돌파 기법)이다.

## STEP C — MTP CUDA 통합 (정확하나 미추월)

엔진엔 이미 speculative decode 스캐폴딩이 있었다(이전 "2c STEP 4"):
- `qwen35_decode(... d_prenorm, d_postnorm, commit, va ...)` — pre/post-final-norm hidden 탭 + commit 분리.
- `VerifyArtifacts` + `alloc_verify_artifacts` — verify용 GDN per-token snapshot.
- `promote_prefix(c, st, va, m)` — accept된 m 토큰 승격(KV는 verify 중 기록, GDN state=snap[m-1]).
- `tests/test_spec.cu` — draft chain(MTP head, 자체 KV) → verify Sq=K+1 batch → accept longest-prefix → promote → catch-up. 정확성 게이트 = spec==plain(mism=0).

27B 연결: `tools/dump_mtp_weights.py`로 safetensors `mtp.*` → fp32 .bin 덤프(body forward 불필요 →
fla triton 크래시 회피). rope/token_ids는 0.8B와 동일(θ=1e7·tokenizer 공유)하므로 복사 재사용.
`test_spec.cu` 수정: ① `launch_gemv_q4`에 `scale_dtype` 전달(27B fp16 scale OOB 수정), ② fp32 body는
`bq4==0`일 때만 로드(27B fp32 54GB 회피, embed는 항상 fp32).

**측정(STEP C, verify=MMQ 경로):** 27B W4A8 spec가 **모든 K에서 plain보다 느림**:

| K | spec t/s | plain | 비율 | accept |
|---|---|---|---|---|
| 2 | 10.1 | 12.1 | 0.83× | 1.16/2 |
| 3 | 11.7 | 12.1 | 0.97× | 1.73/3 |
| 4 | 10.8 | 12.0 | 0.90× | 1.73/4 |

**Root cause(측정):** verify가 spec 시간의 69~79%. plain Sq=1 = **163 GB/s**인데 verify Sq=5 = **79 GB/s
(절반)**. verify Sq=K(2~8)의 W4A8 matmul이 prefill용 MMQ 128×128-tile로 가서 소-M에 행 대부분(123/128)을
비워 → 블록 136개뿐(GPU 미충진) → 대역폭 반토막. accept도 단일 MTP head라 ~1.73서 포화.

프리미스(11.6×1.63≈18.9)는 과낙관이었다 — 실제 accept 1.73(27B 다양 출력) + verify amortize 약함.

## STEP D — 소-batch W4A8 verify 커널 (추월 달성)

**설계:** `gemv_w4a8_sb_kernel`(kernels/gemv_w4a8.cuh) — warp 1개 = 출력 o 1개. weight `w[o]`를
**1회만 적재(DRAM)**하고 S개 활성화 컬럼에 대해 S개 accumulator로 DP4A. weight DRAM 트래픽 = Sq=1
gemv와 동일(=1회) → 대역폭 최적. 활성화(S×IN int8 ≈ 수십 KB)는 L2 상주라 o마다 재읽기해도 DRAM 아님.
누적 순서가 `gemv_w4a8_kernel`과 동일(lane-minor 블록 순회) → 수치 bit-동일.

(대비: Sq=1 gemv를 S번 부르면 같은 o weight를 s마다 재적재=S× 트래픽; MMQ 128-tile은 소-S에 비효율.)

**de-risk(`tools/bench_w4a8_sb.cu`, 27B FFN gate 5120→17408, fp16 scale):**

| S | naive-gemv | MMQ(현 경로) | **SB(신규)** | SB==gemv |
|---|---|---|---|---|
| 1 | 241 | — | 239 | maxae=0 |
| 2 | 118 | 106 | **224** | maxae=0 |
| 5 | 47 | 113 | **208** | maxae=0 |
| 8 | 29 | 106 | **179** | maxae=0 |

SB는 Sq=1 대역폭(239) 유지 = weight 1회 적재 amortize 성공. **MMQ 대비 ~2×**, gemv와 bit-exact.

**ncu(SB vs MMQ, S=5 부근):** SB duration 366us vs MMQ 687us(**1.88×**), SB 2176블록 vs MMQ 136(MMQ
128-tile이 소-M에 GPU 못 채움), sm_throughput SB 50% vs MMQ 30%. (GB10 sm_121은 `dram__bytes` n/a —
"1회 적재"는 커널 구조 + 대역폭 + gemv bit-exact로 확정.)

**통합:** `qwen35_decode` `mm()` + `MMs()` W4A8 dispatch 3-way:
`Sq==1 → gemv_w4a8` / `2≤Sq≤8 → gemv_w4a8_sb`(verify) / `Sq>8 → gemm_mmq_w4a8`(prefill).

**e2e 측정(27B W4A8, SB verify):**

| K | spec t/s | plain | vs plain | vs llama 13.7 | accept |
|---|---|---|---|---|---|
| **3** | **16.9** | 12.1 | **1.40×** | **1.23× 추월** | 1.73 |
| 4 | 15.3 | 12.1 | 1.26× | 1.12× | 1.73 |
| 5 | 14.9 | 12.1 | 1.23× | 1.09× | 1.93 |

verify% 73→60%(K=3) — SB가 verify 시간 절반으로. spec 0.97×→**1.40× flip**. de-risk 예측(~17 t/s) 적중.
(0.8B W4A8도 spec 286 vs plain 201 = 1.42×, verify 63→43%.)

**다양 프롬프트 accept 분포(K=3, 27B W4A8) — 체리피킹 아님:**

| 프롬프트 | spec t/s | accept | vs plain | vs llama 13.7 |
|---|---|---|---|---|
| code (구조적) | **24.6** | 3.00/3 | 2.03× | 1.80× |
| fact | **19.8** | 2.08/3 | 1.64× | 1.44× |
| prose (개방형) | **17.1** | 1.80/3 | 1.41× | 1.25× |

**전 프롬프트가 llama 13.7 추월(1.25~1.80×)** — 가장 예측 어려운 개방형 산문도 추월. accept는 프롬프트
예측성에 의존(code 3.00 vs prose 1.80)하나 하한(prose)도 1.41×.

## 검증 게이트 (전부 통과)

- **정확성**: spec==plain mism=0 — 0.8B(fp32/q4/W4A8) + 27B(W4A8).
- **결정성**: 0.8B W4A8 K=3 30회 실행, 고유 출력 1(bitwise 동일).
- **직교성**: fp32/q4/W4A8 body 각각 spec==plain(mism=0).
- **Sq 일관**: SB는 Sq 무관 per-output 동일(gemv와 bit-exact) → 단발(Sq=1)==verify(Sq=K). prefill(Sq>8)은
  MMQ 유지(영향 없음).

## 부가가치

`gemv_w4a8_sb`는 **batch serving 병목(동시 요청 = 소batch)과 동일**한 문제를 푼다 — MTP와 무관하게
소-batch W4A8 추론에 가치. verify(Sq=K) = 소-batch decode와 같은 커널.

## 다음

- K self-tuning(accept 분포별 최적 K — 현재 K=3 고정 최적; code는 K↑가 더 이득일 수 있음).
- 잔여 base lever: 작은 projection grid-starve split-K.
- 정식 엔진 API화(현재 test_spec.cu가 spec 루프 보유 — engine.h로 승격).
