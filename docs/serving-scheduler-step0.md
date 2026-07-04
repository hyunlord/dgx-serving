# 서빙 스케줄러 STEP 0 — 갭 계측 de-risk (측정 먼저, 3모델 공통)

> "서빙 throughput ≪ raw batch"(35b 63 vs 195 / gemma4 41 vs 76)의 **출처를 계측으로 분해**해
> 유력 lever를 measured 확정한다. 추측 lever 강행 금지(헌법). 계측은 **스케줄링 무변경 = 출력 bit-id 불변**.
> **결론: 큰 lever로 추정한 idle/underfill은 신기루. 진짜 lever = chunked prefill(2/3 모델 이미 보유, gemma4는 신규 graph 필요).**

## 계측 도구 (accuracy-invariant)

`runtime/serving_core.h`의 `sc_run_continuous`에 갭 분해 계측 추가(타이밍 누적 + active 히스토그램만 — 스케줄링/계산 무변경):
- `[gap] makespan = prefill(%) + decode(%) + idle(%) | avgB/Bmax underfill≈ stepN` — makespan을 세 버킷으로 귀속.
- `[ahist] active 분포(step수)` — 매 step의 active 배치 크기 히스토그램(avgB 정체 기전 규명).

★주의(귀속 해석): **c1tok(chunked=false) 모드에선 프롬프트를 1토큰/step ingest** — 이 ingest step들이 `[gap]`의
**"decode" 버킷에 은닉**된다(출력 없는 낭비인데 decode로 계상). 따라서 c1tok의 "decode 99%"는 순수 decode가 아님.
c1tok vs c-ch 직접 대조가 이 은닉 비용을 노출한다(아래).

## 측정 결과 (seed 고정 trace, warm-up 20% 제외)

### 27b (동일 trace, c1tok vs c-ch 직접 대조 — 결정적)
| mode | λ | tok/s | avgB | `[gap]` |
|---|---|---|---|---|
| c1tok | 16 | **41.9** | 6.93/8 | prefill 0% + decode 99.4% + idle 0.6%(★ingest가 decode에 은닉) |
| c-ch | 16 | **46.6** | 6.91/8 | prefill 12.1% + decode 87.2% + idle 0.7% |
| c1tok | 4 | 41.1 | 6.80/8 | 〃 |
| c-ch | 4 | 46.2 | 6.84/8 | 〃 |

→ **chunked prefill(c-ch)이 c1tok보다 +12%**(46.6 vs 41.9). 차이 = c1tok이 프롬프트를 1토큰/step ingest하며
낭비한 step을 c-ch은 1스텝(512청크)에 처리. `[ahist]` active 지배=8(314~394/약410~530 step = 85% 배치 full).

### gemma4 31B (c1tok 전용 — prefill_batch_slot 부재)
| λ | tok/s | avgB | 비고 |
|---|---|---|---|
| 4 | 41.1 | 6.80/8 | `Gemma4Backend.prefill_slot` → `exit(2)`(미구현), chunked=false 강제 |
| 16 | 41.9 | 6.93/8 | 27b c1tok과 동급 — **c-ch로 갈 여지(+12%)를 못 먹음** |

### 35b-a3b (구조적 동일 — 재측 생략)
35b는 27b와 **동일 QwenBackend + serving_core chunked 경로** 사용(같은 `prefill_batch_slot`). 27b가 qwen-chunked
클래스를 대표, gemma4가 c1tok 클래스를 대표 → **두 스케줄러 모드 모두 커버**. 이전 측정(avgB~6.9, idle 무시)과 정합.

## measured lever 우선순위 (3모델 공통)

| 후보 lever | 측정 | 판정 |
|---|---|---|
| **idle(요청 대기)** | 0.2~0.9% (전 λ) | ✗ **신기루** — 포화 시 무시. |
| **배치 underfill(avgB<Bmax)** | ~13~15% but active 지배=8(85% full) | ✗ **신기루** — 대부분 ramp/drain 아티팩트(현실 연속 서버엔 drain 없음). 저-λ의 낮은 avgB는 부하-바운드(동시요청 부족)이지 스케줄러 버그 아님. |
| **chunked prefill** | c-ch > c1tok **+12%**(27b 확증) | ✓ **진짜 lever** — c1tok의 1토큰 ingest 낭비. 이 벤치는 프롬프트 U[8,32]로 작아 +12%; **현실 프롬프트(100s~)면 c1tok는 파국적**(프롬프트 길이만큼 step 낭비) → 과소평가. |

**즉 유일한 fixable 스케줄러 throughput lever = chunked prefill.** idle/underfill은 착수 가치 없음(de-risk 성과 — 억지 lever 헛수고 회피).

## 3모델 STEP A 실행 가능성 (measured)

- **27b / 35b (qwen-chunked)**: chunked prefill **이미 구현**(c-ch). 권장 서빙 모드 = c-ch. 이 축에서 **스케줄러 근-최적** —
  추가 easy lever 없음(prefill 12%는 필수 compute, idle/underfill은 신기루).
- **gemma4 31B**: chunked prefill = **신규 graph 필요**(`gemma4_prefill_batch_slot` 아날로그 — sliding-window
  multi-token attention + dual RoPE + GeGLU + sandwich 4-norm + softcap). `serve_gemma4.cu:6,46`에 이미 의도적
  deferred+보고. **정확성 게이트 필수**(chunked 출력 == c1tok 출력 bit-id, mism=0 — sliding-window KV 정합).
  → **substantial**(헌법 "큰 변경"). 리스크 = sliding-window multi-token KV를 1-tok ingest와 정확히 일치시키기.

## 정직한 결론 (overclaim 금지)

- **measure-first de-risk 성공**: 헤드라인(대량 underfill/idle)은 신기루, 진짜 lever는 chunked prefill.
- **"서빙 ≪ raw B16" 갭의 지배 요인**: (1) **부하-바운드 동시성**(sustainable λ에서 avgB<Bmax — raw full-B16과
  apples-to-oranges), (2) **필수 prefill compute**. **2개 chunked 모델엔 fixable 스케줄러 비효율 아님**(근-최적).
- **gemma4의 actionable lever**(chunked prefill)는 **신규 graph**를 요함 = 사용자 판단점(정확성-게이트 graph 작업이
  +12%/대형-프롬프트 이득 대비 가치 있는가). 부재 중 정확성-리스크 신규 graph 강행 금지(헌법).
- 26B-A4B는 decode 그래프 미빌드 → 이 lever 대상 밖(chunked prefill 구현 시 공통 수혜, 나중 합류).

## STEP A — gemma4 chunked prefill 구현 + 실측 (measured, 정직)

STEP0 lever(chunked prefill)를 gemma4에 착수. **어댑터 `gemma4_prefill_batch_slot`**: KV 레이아웃 `[B,max_len,nkv,hd]`에서
슬롯 b 서브영역이 곧 단일-시퀀스 KV → 검증필 `gemma4_decode`(Sq=P)를 슬롯-b 뷰로 재사용(신규 커널 0·③=0).

**① 정확성 게이트 `tests/test_gemma4_prefill_slot`** (chunked vs c1tok 1-tok ingest):
- ★**W4A8(서빙 모드): 완전 bit-id** — KV Δ=0 **전 60층**·decode 4토큰 Δ=0·argmax 4/4. **chunked=순수 스케줄링**(브리핑 전제 충족).
- q4(uq=1): scheme-internal(KV Δ≤5.9e-4·decode 4e-3·argmax 보존 — tiled vs batch gemm 누적차, 서빙 미사용).
- ragged 무회귀 PASS(bit-id·argmax 15/15). → **정확성 완벽.**

**② e2e 실측 (bench_serve_gemma4, pmax=32 Bmax=8 λ4): ★NET-LOSS**
| mode | tok/s | tpot50 | prefill% | makespan |
|---|---|---|---|---|
| c1tok | **38.8** | 0.132 | 0% | 81s |
| c-ch(chunked) | **22.1** | 0.274 | 54.3% | 142s |

c-ch가 **-43%**(prefill ~1.6s/req, P≤32에 비정상). **chunked prefill이 gemma4에선 net-loss** — 정확성은 완벽하나 성능 역행.

**③ 진단 (커널 근거 확정, 추측 아님):**
- `gemma4_decode`의 W4A8 mm(`gemma4_decode.cu:52`)은 **plain `launch_gemv_w4a8`** 사용 = weight를 P행에 amortize 안 함
  (tiled-gemm `:53`은 q4 전용). 반면 amortized 커널 `gemv_w4a8_sb`(`gemv_w4a8.cuh:126` "weight 재적재 없음")은
  **decode(mmB)만** 사용. → 프리필 P토큰이 비-amortized gemv로 P× 메모리 트래픽 → c1tok의 배치 ingest보다 느림.
- 대조: qwen `prefill_batch_slot`(`qwen35_batch.cu:13`)은 청크 multi-token + amortized 매트멀 → +12% WIN.

**④ 수정 경로 (substantial — 사용자 판단):** gemma4 프리필이 **`gemv_w4a8_sb`(amortized, SMAX≤16 → P를 청크)**를 쓰도록
새 chunked prefill forward 작성(qwen 아날로그). decode와 **같은 SB 커널이라 bit-id 유지 가능성 높음**(재게이트 필수).
substantial + 정확성-민감 → 헌법(막히면 보고·큰 변경 강행 금지)대로 방향 확인 후 착수.

**정직 결론:** chunked prefill lever는 27b/35b엔 유효(amortized prefill 보유)나, **gemma4는 amortized W4A8 프리필 커널
경로 부재로 naive 구현이 net-loss**. bit-id 어댑터+게이트는 올바른 자산(빠른 버전의 기반). 26B는 대상 밖(decode 미빌드).

## 자산
- 계측: `runtime/serving_core.h`([gap]/[ahist], accuracy-invariant 영구 측정 도구).
- 벤치: `tools/bench_serve.cu`(27b/35b, c1tok+c-ch 스윕), `tools/bench_serve_gemma4.cu`(c1tok+c-ch 양모드 스윕).
- 어댑터: `models/gemma4_batch.cu` `gemma4_prefill_batch_slot`(bit-id, 현재 net-loss). 게이트: `tests/test_gemma4_prefill_slot.cu`.
