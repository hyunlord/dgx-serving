# 3a 설계안 — q4 × speculative 지도화 (대역폭 돌파 챕터1, STEP 0)

> 2a~2c(fp32 본체+MTP+spec, 1.227×) + 2d(gemv 진단) 위에서 q4×spec 결합을 측정 지도화.
> 멘탈모델: decode 속도 ≈ 대역폭 ÷ 토큰당 바이트. q4=분모↓(weight 8×), spec=분자↑(amortize).
> **이 문서 합의 후 STEP 1.** 정확성(구현 오차0)과 품질영향(측정기록)을 절대 분리.

## 1. q4 적용 지도 (어느 gemv에)

본체(qwen35_decode)는 `mm()` dispatch로 use_q4 토글 — **QMat matmul 전부 q4 가능**:
- q4 대상(big-weight): q/k/v/o_proj, in_proj_qkv/in_proj_z/out_proj, mlp gate/up/down, lm_head.
- fp32 유지(작음): in_proj_a/b(16×1024), conv1d, 모든 norm. (2a 정책 그대로.)
MTP(mtp_step)는 현재 fp32만 — q4 적용엔 **mtp.* q4 추출 필요**(STEP 2).

## 2. 인프라 현황
- ✅ body q4: `qwen35_decode(use_q4=1)` + body q4 weight(151개) + golden_q4 존재. test_spec에서 q4=1 로드만.
- ⚠️ MTP draft q4: mtp_step이 `_f32.bin`만. mtp.* q4 추출(extract 재사용) → STEP 2.
- gemv_q4 커널(2a): naive 1-warp/output, q4 packed(0.5B/weight) → fp32 대비 weight read 8×↓.

## 3. 측정 조합 (4점) × 3축

| 조합 | 본체 | draft | 출력 |
|---|---|---|---|
| (a) baseline | fp32 | fp32 | fp32 출력 |
| (b) body-q4 | **q4** | fp32 | **q4 출력**(바뀜) |
| (c) both-q4 | q4 | q4 | q4 출력 |
| (d) draft-q4 | fp32 | q4 | fp32 출력(불변) |

3축(매 조합): **① t/s**(in-engine, 다회) **② 출력 일치도**(consistency/quality) **③ accept rate**.

## 4. 정확성 기준 — 둘로 분리 (q4 핵심)

1. **구현 정확성 (오차 0, 절대 조건)**: q4 경로 출력 == golden_q4. (STEP1: q4 incremental decode == golden_q4.)
2. **q4-spec 일관성 (mismatch 0, 절대 조건)**: q4-spec-on 출력 == q4-plain-decode 출력.
   draft 정밀도 무관(본체가 verify) → (b)(c) 출력은 q4-plain과 같아야. 깨지면 **상태관리 버그**(q4 손실 아님).
3. **품질 영향 (측정·기록, 오차0 아님)**: q4-plain 출력 vs fp32-plain 출력 — 몇 번째 토큰부터 갈리나 + accept 변화.
   "나빠짐"을 숫자로. 지도의 한 축.

## 5. 대역폭 측정 (ncu)
- 2d: GB10은 `dram__throughput.pct` = n/a(unified LPDDR5X). `dram__bytes_read.sum` 시도; 안 되면
  duration 기반(같은 shape q4 gemv vs fp32 gemv 시간비 = 실 speedup).
- 핵심 검증: 2d서 fp32 big-gemv가 latency-bound(BW 22%)였음 → **q4로 mem-bound 되며 빨라지는지**, 아니면
  여전히 latency-bound라 q4 효과 작은지. ncu occupancy/duration + 토큰당-바이트 산출.

## 6. 대역폭 돌파 지표 (STEP 3 산출)
- 토큰당 weight 바이트 = (조합별 weight read bytes) / (사이클당 토큰). fp32-single 대비 q4×spec가 몇 배↓.
- 유효 t/s ÷ 물리 대역폭 = "물리 한계 대비 효율" — q4×spec가 fp32-single 한계를 어디까지 넘는지.

## 7. 예상(2d 근거) — 측정으로 검증/기각
- q4 body: gemv가 latency-bound라 **8× 안 나올 수 있음**(modest). 큰 gemv(lm_head)는 weight-read 지배라 더 이득.
- draft q4: 작은 gemv 위주라 속도 이득 작고 accept↓ 가능 → 순이득 불확실. 측정이 답.

## 8. 작업 순서
STEP1 body-q4(b): use_q4 배선 + q4-incremental==golden_q4 + 3축 + ncu 대역폭 → 게이트 보고.
STEP2 draft-q4: mtp.* q4 추출 + draft q4 + accept 영향 측정.
STEP3 4조합 지도표 + 토큰당바이트/유효대역폭. STEP4(선택) llama.cpp 비교.

## 9. 사용자 결정 필요
1. q4 대상 범위: QMat 전부(lm_head 포함) q4 — 동의? (lm_head 1GB가 최대 레버.)
2. 측정 조합 (a)(b)(c)(d) 순서 — body-q4(b) 먼저 동의?
3. 품질 측정 방식: 같은 프롬프트로 fp32-plain vs q4-plain 토큰열 비교(첫 divergence index) + accept — OK?
