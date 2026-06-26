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

---

## STEP 1 결과 — 본체 q4 (combo a vs b) [완료]

검증 환경: clocks 해제(RmProfilingAdminOnly=0), prompt=15tok, N=96 gen, K=4. in-engine t/s만 사용.

### 구현 정확성 (절대조건, 오차 0) ✅
- `test_decode use_q4=1`: q4 incremental decode == q4 prefill (logits 7.6e-6, pre-norm 1.7e-6, argmax ALL match).
- 체인: q4-prefill == golden_q4 (test_qwen35, 2a) ⇒ **q4 decode == golden_q4**. 절대조건 충족.
- `make test` 12/12 PASS (회귀 0).

### 3축 측정
| 조합 | plain t/s | spec t/s | accept | q4-spec==q4-plain | 품질(vs fp32) |
|---|---|---|---|---|---|
| (a) fp32 body | 66.8 | 80.9 (1.21×) | 3.80/4=95% | — | 기준 |
| (b) q4 body | **176.0 (2.63×)** | ~122 (net-NEG) | 3.80/4=95% | **mismatch 0** | **토큰 96/96 완전일치** |

- **① t/s**: q4 body plain = **2.63× fp32 plain**. q4 plain(176) > fp32 spec(80.9) — 대역폭 돌파 1차 실증
  (q4 한 레버가 fp32+speculative 결합보다 빠름).
- **② consistency/품질**: q4-spec == q4-plain mismatch 0 (상태관리 무결). 품질축 — q4-plain 토큰열이
  fp32-plain과 96/96 **완전일치** (lm_head q4 포함해도 argmax 0 divergence). 이 프롬프트에선 q4 품질대가 0.
- **③ accept**: 3.80/4=95%, (a)(b) 동일 — draft가 양쪽 fp32라 불변(예상대로).

### ncu/nsys per-shape (decode S=1, 트래픽 8×↓가 시간↓로 실현됐나)
| gemv shape | out | fp32 dur | q4 dur | 실 speedup | 비고 |
|---|---|---|---|---|---|
| **lm_head** | 248320 | **3921us** | 728us | **5.4×** | fp32 1.017GB→**259 GB/s ≈ BW한계(95%)**; q4 ~185 GB/s |
| FFN gate/up | 3584 | 66us | 16.6us | 4.0× | |
| FFN down·q·o | ~1024 | 48us | 19.8us | 2.4× | 작은 shape일수록 speedup↓(고정비) |
- q4 lm_head ncu: mem 61% / compute 63% / occ 92% — int4 unpack 추가로 mem·compute 균형(완전 8× 대신 5.4×).

### 2d 멘탈모델 정정 (중요)
2d는 fp32 lm_head를 "17 GB/s, 6%, latency-bound"로 기록했으나, 클럭 해제 후 측정하니
fp32 lm_head는 **이미 BW-bound (259 GB/s, 273의 95%)**. 2d 수치는 클럭락 상태 측정 오류로 판단.
⇒ q4의 레버는 "latency-bound를 mem-bound로 바꿔서"가 아니라 **이미 BW-bound인 lm_head의 트래픽을
8× 줄여 시간을 5.4× 단축**하는 것. big-weight일수록 효과 큼(lm_head 5.4× > FFN 4.0× > 작은 gemv 2.4×).

### spec net-negative 정체 규명
combo(b)에서 spec(122) < plain(176). nsys: body_q4=1 run에 **fp32 mo_kernel(grid 15520, 3.96ms)**
= MTP draft의 fp32 lm_head가 그대로 돎. 본체가 q4로 빨라지자 fp32 draft chain(특히 draft lm_head
3.96ms/step)이 병목. ⇒ **STEP 2(draft q4)의 직접 동기**. 본체만 q4면 speculative는 손해.

**STEP 1 결론**: q4 body는 big-weight gemv(특히 lm_head)에서 진짜 레버(2.63× plain, 품질대가 0 this prompt).
단 fp32 draft가 spec를 무력화 → 다음은 draft q4(STEP 2)로 분자(amortize)를 살려야 q4×spec 곱이 성립.

---

## STEP 2a 결과 — draft lm_head q4 (tied 재사용) [완료]

draft 비용의 94%는 lm_head(254M param=1GB fp32). lm_head는 embedding과 tie_word_embeddings →
**body의 이미 검증된 `w.lm_head` q4를 draft lm_head 투영에 재사용**(test_spec mtp_step, dq4 토글, argv[6]).
추가 추출/golden 불필요 — 동일 q4 weight·동일 gemv_q4 커널(2a·STEP1 검증)이라 **구현 정확성 상속**.

### 4조합 지도 (N=96, K=4, clocks 해제, 다회 안정)
| 조합 | body | draft | plain t/s | spec t/s | spec/plain | accept | mismatch |
|---|---|---|---|---|---|---|---|
| (a) fp32 | fp32 | fp32 | 65.5 | 80.9 | 1.24× | 3.80/4 | 0 |
| (b) body-q4 | q4 | fp32 | 176.0 | 124.5 | **0.71× NEG** | 3.80/4 | 0 |
| (c) **both-q4** | q4 | q4 | 176.4 | **191~193** | **1.08× POS** | 3.80/4 | 0 |
| (d) draft-q4 | fp32 | q4 | 65.7 | 102.0 | 1.55× | 3.80/4 | 0 |

### 핵심 결과
1. **q4×spec 곱 실증 (#1 질문 답).** (c) both-q4 spec = 191~193 t/s로 **net-POSITIVE 회복**
   (= fp32 plain 65.5의 **2.9×**, fp32 spec 80.9의 **2.36×**). draft도 q4여야 곱이 성립 — 대역폭 돌파 공식 그대로.
   (b)의 net-negative(0.71×)는 fp32 draft lm_head(3.96ms)가 병목이었고, draft lm_head q4(728us)가 제거.
2. **accept 전 조합 3.80/4 불변 — draft q4의 accept 대가 0.** lm_head q4 노이즈가 draft argmax를 안 바꿈
   (STEP 1 body lm_head q4와 동일 현상: argmax가 q4 logits 노이즈 견딤). draft lm_head q4 = **순수 속도 이득**.
3. **consistency 절대조건 충족.** mismatch 0 전 조합 — draft 정밀도는 최종 출력에 영향 없음(본체 verify가 진실원).
   (d)는 body fp32라 출력이 fp32-plain과 동일(불변), draft q4로 spec만 1.55× 가속.
4. **phase breakdown이 기전 확증.** draft(MTP chain) 비중: (b) 50% → (c) **25%** (draft lm_head 절반↓).
   (c)에서 verify(body batch) 60%가 재차 지배 — 다음 레버는 배칭/verify(별도 챕터).

### draft lm_head이 BW-bound인가 (예측 검증)
draft lm_head는 body lm_head와 동일 weight(1GB)·동일 커널 → STEP 1의 per-shape 그대로 적용:
fp32 3.96ms(259 GB/s BW한계) → q4 728us(5.4×). 사용자 예측("draft lm_head도 BW-bound면 q4 효과 클 것") 적중.

### STEP 2a 결론 + 미측정 범위
lm_head-only draft q4가 **draft-q4 Pareto 최적 후보**: 지배비용(94%) 제거, accept 대가 0, 신규 추출/리스크 0.
미측정: **MTP 본체 gemv(fc/Wq/Wk/Wv/Wo/Wg/Wu/Wd) q4**는 아직 fp32. 이들은 작은 latency-bound gemv(각 2~4M,
합 ~15M ≪ lm_head 254M)라 속도 이득은 작을 것이나, draft hidden을 바꿔 **accept를 실제로 떨어뜨릴 수 있음**
(work order의 "draft 품질 손실→accept↓" 측정 대상). 이건 mtp.* q4 추출+golden(2a 방식)이 필요 → STEP 2b 후보.
