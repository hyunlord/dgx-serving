# 성능 (dgx-serving on GB10) — 측정 좌표·정직 벤치·알려진 한계

> 모든 수치는 GB10(sm_121, ~273 GB/s unified) 실측이다. **결과와 측정 조건만** 적는다.
> ★원칙: **이기는 좌표도 지는 좌표도 그대로** — overclaim 0. 이 엔진의 신뢰가 곧 제품.

## 1. 서빙 처리량 좌표 (모델별, 연속 배칭, warm)

| 모델 | 구조 | 대표 운영점 | throughput | tpot95 | 조건 |
|---|---|---|---|---|---|
| Qwen3.5-27B | GDN+MTP(spec) | B16 K7W2 (λ4, long-heavy) | ~51.7 t/s | ~0.34s | nreq60·seed42·w4a8+mtp_q4 |
| Gemma4-31B | dense | B8 (★knee) | 36~38 t/s | ~0.14s | nreq48·maxlen384·w4a8 |
| Gemma4-12B | dense (31B same-family) | B8 (전 regime 지배) | ~96 t/s (aggr) | ~0.05s | nreq48·maxlen384·w4a8·c1tok |
| Gemma4-12B **spec** | +0.4B assistant(EAGLE류) | K=4 single-stream | **46.7~50.5 t/s** | — | HTTP client-observed·출력==plain greedy·drafter W4A8 |
| Qwen3.5-35B-A3B | MoE(256e top8)+GDN | B8 (지배점) | ~62.7 t/s | ~0.11s | nreq40·maxlen256·fused-MoE |
| 26B-A4B | 병렬 MoE | single-stream(HTTP) | **43~46 t/s** (W4A8) | — | active-width 수정 후 · raw 45.4 회복 |

- single(비배치) decode: 27b 11.9 · gemma4-31b 9.28 · **gemma4-12b 23.3** · 35b MoE 51.8 · **26b MoE 45.4 t/s(raw 커널)**. **single 고throughput은 물리적으로 불가**
  (대역폭 벽 — 배치로만 집계 throughput↑, per-request 지연은 그대로).
- ★**26b 서빙 single-stream 갭 해소(measured, active-width)**: 3-way HTTP 14.7/18.2 t/s는 online 루프가 **single 요청도 Bmax=8 전폭 decode**(7슬롯 더미 → batch-8 per-slot 처리량만) 탓. 3점 격리(raw 45.4 / daemon-direct 18.2 / HTTP 18.4)로 HTTP·python 계층 오버헤드 ≈0 확정 → serving_online.h를 **active-width**(active 슬롯 폭만 decode)로 수정 → **single-stream 18.2 → 45.76 t/s(raw 완전 회복, 2.5×)**. bit-id 유지(test_moe26b_online PASS). → [`docs/moe26b-decode-diagnosis.md`](./moe26b-decode-diagnosis.md).

### 회계 정의 (표기 통일 — 혼동 방지)
- **raw**: bench decode-only 정상상태(1슬롯, prefill·IPC 제외). 커널 상한 척도.
- **decode-only tpot**: HTTP 벽시계에서 TTFT 뺀 것 = (n−1)/(elapsed−TTFT). 수정 후 26b = 45.76(raw 근접).
- **전체회계 t/s**: HTTP 벽시계 전체 ÷ 생성토큰(TTFT 포함) = 브라우저 체감. 짧은 생성일수록 TTFT 비중↑(26b 긴 생성 43.0, 짧은 32.7).
- spec 곱(27b): chain K=3 **1.40×**(plain 대비)·tree K4W3 1.11×·plain 배칭 1.21×. **워크로드 민감**(accept 6.99 반복 ~ 3.5 산문).

## 2. roofline 달성률 (weight-이동 상한 대비 — 상한이지 예측 아님)

| 모드 | 상한(t/s) | 실측 | 달성률 |
|---|---|---|---|
| 27b single non-spec | 19.0 | 11.9 | 62.5% |
| 27b spec serving (B16 K7W2) | 304.5 | 51.7 | 17.0% |
| gemma4-31b single | 17.4 | 9.28 | 53.3% |
| gemma4-12b single (W4A8 sb) | 40.4 | 23.3 | 57.6% |
| 35b-a3b MoE single | 165.3 | 51.8 | 29.8% (MoE 구조적으로 낮음) |

하회 요인(measured): batch=GDN state per-slot 운반, spec=draft/MTP 오버헤드+accept<천장, MoE=토큰당 다수 expert-gemv launch.

> 상한 basis 주의: 27b/31b 행은 273(nominal)기준 historical 좌표. gemma4-12b 행은 **255.8 GB/s(probe 실측 read) + active-bytes**
> (k_eq_v V-proj 공유 반영 → bench_gemma4 config-주도 산출, 하드코딩 제거). 같은 basis로 31b는 15.7이 되나 historical 표기 유지.

## 3. 정직 벤치 — vs llama.cpp (Gemma4-26B-A4B, 공정 2-way, HTTP, warm, ~q4급)

같은 26b MoE·같은 프롬프트/max_tokens·같은 동시성. HTTP 클라이언트 관측(배포 현실).

| 동시 N | dgx t/s | llama t/s | dgx/llama |
|---|---|---|---|
| 1 | 14.7 | 52.6 | **0.28×** (3.6× 느림) |
| 4 | 48.4 | 65.3 | 0.74× |
| 8 (=Bmax) | 80.6 | 168.3 | **0.48×** (2.1× 느림) |
| 16·32 (초과구독) | ~81 포화 | ~145~163 | 0.50~0.56× |

- ★**후속(active-width 수정 §1·7cc75ba): N=1 재측정 = 43~46 t/s(전체회계~decode-only) = 0.82~0.87×.** 표의
  14.7(0.28×)은 수정 전 서빙 루프 고정폭 버그의 좌표(역사적 기록으로 유지). N=4·16·32 행은 수정 전 측정
  (N=8 전폭 활성은 설계상 무영향) — 재측정 안 함(정직).
- **dgx가 진다(measured).** raw-속도로 이기는 좌표는 없다 — 숨기지 않는다.
- **원인(정직)**: 26b는 최신·미최적 MoE(decode graph 갓 구현·perf 튜닝 0), llama는 수년 성숙한 fused/tiled 커널 =
  **커널 성숙도 갭**(엔진 천장 아님 — 27b dense는 roofline 62.5% 달성). 남은 lever = per-token 커널 속도(FP4·fused·tiled).
- **배칭은 작동**: dgx 연속 배칭 N=1→8 near-linear 5.5×(14.7→80.6)·초과구독을 큐잉으로 흡수(크래시 없음).
- vLLM 3rd leg = 후속(GB10 구동 실증됨·26b 체크포인트+docker HTTP만 남음, 정직히 범위 밖).

### 3b. 정직 벤치 — vs llama.cpp (Gemma4-12B, **plain vs plain**, single decode, warm)

★**회계 명시(중요)**: 아래는 **spec OFF plain끼리** 공정 비교다. llama의 12B **speculative 포함 66.9 t/s와 직접 비교 금지**
(plain roofline 40.4 위 = spec 없인 물리 불가). llama-bench(tg64)=speculative 없는 순수 decode.

| 척도 | dgx plain (W4A8 sb) | llama plain (Q4_K_M, spec OFF) | dgx/llama |
|---|---|---|---|
| single decode | 23.3 t/s | 26.8 t/s (llama-bench tg64, ngl99) | **0.87×** (1.15× 느림) |
| roofline 달성률 | 57.6% | 66.4% (동일 40.4 상한) | — |

- **near-parity plain(measured)** — 26b MoE(0.28~0.48×)와 대조적으로 12B dense는 llama plain에 **1.15× 갭**까지 근접.
- **원인(정직)**: 12B는 27b와 same-family dense W4A8 gemv 경로(성숙 자산) 재사용 → 남은 갭=작은 잔여 커널 성숙도(fused/tiled).
- **★spec 서빙 GB10-vs-GB10(캡처값 검증 완료·진짜 좌표)**: MTP fork llama-server를 **같은 조건**(같은 프롬프트·
  K4·client-observed·warm·single-stream·같은 공식 0.4B assistant[f16 GGUF])으로 GB10 직접 벤치
  (`tools/bench_llama_gb10.py`) — 우리 spec 서빙(drafter W4A8·출력==plain greedy 불변식 게이트)과 나란히:

  | 12B spec (K4·128tok·client-observed) | llama-GB10 | dgx | dgx/llama | spec 곱 (llama/dgx) |
  |---|---|---|---|---|
  | plain | 25.6 t/s | 22.0~22.3 | 0.86× | — |
  | spec chat | 56.2 | **50.5** | **0.90×** | 2.20× / **2.30×** |
  | spec code | 69.6 | 46.7 | 0.67× | 2.72× / 2.12× |
  | spec prose | 45.6 | 31.9 | 0.70× | 1.78× / 1.43× |

  기존 "66.9 대비 0.70~0.75×"의 66.9는 캡처값이었고 **GB10서 재현 확인**(code 69.6) — 진짜 좌표는 워크로드 분해:
  **chat 0.90×(근접)·code 0.67×·prose 0.70×**. spec 곱 자체는 chat서 우리 우세(2.30 vs 2.20). ★새 함의: llama도
  4bit(Q4_K_M) target인데 code accept가 높음 → accept 갭 규명의 "q4 조건화"가 정밀화됨 — **양자화 스킴 품질**
  (K-quant super-block vs 우리 blueprint q4 rel 0.1016)이 accept에 직결 = **수정 가능 축으로 승격**. 조건 차 명시:
  target Q4_K_M(6.9GB) vs w4a8(동급 4bit·다른 스킴)·drafter f16 vs W4A8(llama에 고정밀 유리 제공).
  accept 갭 규명(진단·measured): 활성화·rope·궤적 가설 전부 기각, code 갭=target 양자화 조건화 확정(fp32-target
  1.74→2.56≈HF). ★K-quant 품질 주입 실험(구현 0): 가중치 32% 정확(rel 0.0695 vs 0.1016)해도 code accept는
  소폭만 회복(1.91 — **code 벽=4bit 자체·스킴 아님**)·prose는 HF 동률 완전 회복 → K-quant 커널 구현 보류
  (q2 패턴). ★적응적 draft 길이 접힘(measured — 호스트 정책 3종 삼중 판정): margin 조기중단 code −9%·연장
  전용 +1.4%(잡음권)·HF 동적 K −13% — 원인은 비용 구조(draft 2.2ms ≪ 수락 토큰 가치 18ms → 절단 손익분기
  P(acc)<13%인데 실측 최저-확신 구간도 20~38% accept = 어떤 문턱도 손해·llama의 적응 이득은 비싼-draft 전제의
  동형 함정). ★그 부산물(verify Sq 둔감 46.5/49/63ms)은 **verify concat 분할상환으로 실현**(gemma4_verify_concat
  — 활성 슬롯들의 verify를 한 SB 호출로·신규 커널 0·스킴 경계 구성적 회피[전 M을 ≤16행 SB 그룹으로만]·
  concat==per-slot 로짓 Δ=0 M 경계 전수 게이트): spec 서빙 다중 슬롯 aggregate **N2 +39~41%·N4 +33~36%**
  (code 44.9→62.5·chat 53.9→74.8·prose 31.4→44.2, daemon-직결 warm), 구 per-slot은 N에 완전 평평(verify
  직렬화 measured)·단일 스트림 불변(±2% 잡음권)·spec==greedy 재PASS. 남은 lever: **draft 페이즈 concat**
  (assistant 배칭 — 다중 슬롯 라운드 ~20%)·(잔여 후보: K-quant bytes 10%↓=plain 소폭·bf16-반올림 정합)
  → [`gemma-assistant-spec-measured.md`](./gemma-assistant-spec-measured.md).
- 프리필: dgx B1 통짜 35.5 t/s vs llama pp256 1971 t/s = 큰 갭(별도 lever·chunked prefill/TTFT 트랙, plain-decode 헤드라인 아님).

## 4. 대역폭 지형도 (측정 결론 — "이 장비에서 가능한 것")

decode 속도 ≈ 대역폭 ÷ 토큰당 이동 bytes. 2×급 돌파 후보를 전수 측정 판정:

| lever | measured 판정 |
|---|---|
| 형상 튜닝(mmq 256×64) | prefill +5%(구현·bit-id) |
| spec 곱 | 단일 지연 1.40×(반복 워크로드) — 구현 |
| MoE(내장 sparsity) | active-bytes 8.7× roofline — 구현 |
| L2 persist·KV 압축·dense sparsity·prefetch | 소거/modest(비중 없음·예측비용 함정) |
| q2/q3 양자화 | 소거(q2 품질 붕괴·q3 속도 역행 — **4bit이 bytes↓ sweet spot**) |
| FP4 텐서코어 | toolkit-blocked(CUDA 13.2+ 대기) |
| SOTA 2-bit·draft 모델 | 엔진 밖(대형 연구·모델 자산) |

**결론: 273 GB/s × 토큰당 bytes가 근본 벽.** 엔진 내부 커널/config 축은 measured로 소진 — 큰 도약은 FP4(대기) 또는 엔진 밖.

## 5. 알려진 한계 (로드맵 — 숨기지 않음)

- **26B 서빙 파레토 미측정**(roofline 상한만).
- **spec 서빙 경계 미정**(plain backend 훅 밖 — 별도 추상화 필요).
- **Gemma draft-spec: 구현·서빙 완료(4관문)**: E2B는 접음(~1.0×, [`gemma-draft-spec-verdict.md`](./gemma-draft-spec-verdict.md))
  → 공식 0.4B assistant로 재판정 GO → verify bit-id·assistant HF골든·사이클 e2e·서빙 게이트 전부 PASS
  ([`gemma-assistant-spec-measured.md`](./gemma-assistant-spec-measured.md)). 서빙 실측 1.40~1.57×(§3b). 남은 lever
  = drafter 양자화(fp32→bf16/q4)·accept 갭(W4A8 조건화 가설)·verify concat 분할상환·spec 배칭.
- **긴 context(>1024)**: gemma4 sliding KV 링버퍼 미구현.
- **FP4**: toolkit-gated(CUDA 13.2+ 대기) — 최대 미사용 lever.
- 벤치는 warm·측정 조건 명시분에 한함(cold 클럭 교란 제외). 3-way의 vLLM leg는 후속.
