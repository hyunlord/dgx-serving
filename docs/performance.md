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

> **품질 조건 다름(llama Q4_K_M)**: 아래 속도 비교는 동일 품질을 보장하지 않는다. 엔진의 RTN 대칭 per-32 q4와 llama Q4_K_M은 서로 다른 양자화 스킴이며, 엔진의 품질 기준선은 §3c에 별도로 기록한다.

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
  동형 함정). ★그 부산물(verify Sq 둔감 46.5/49/63ms)은 **verify concat + draft concat으로 실현 = 다중 슬롯
  사이클 완전 배칭**(gemma4_verify_concat + gemma4_assistant_step_batch — 활성 슬롯들의 verify/draft를 각각 한
  호출로·신규 커널 0·스킴 경계 구성적 회피[전 M을 ≤16행 SB 그룹으로만]·concat==per-slot 로짓 Δ=0 게이트 양쪽):
  spec 서빙 다중 슬롯 aggregate **누적 N4 +47~51%**(per-slot 순차 기점: code 44.6→66.0·chat 54.1→79.3·prose
  31.2→47.0, daemon-직결 warm — verify concat 몫 +33~36%p·draft concat 몫 +10%p로 N4가 N2 따라잡음/추월,
  피크 chat N3 80.4), 구 per-slot은 N에 완전 평평(verify 직렬화 measured)·단일 스트림 불변(±1~2% 잡음권)·
  spec==greedy 재PASS(bit-id 게이트 3종: verify M 경계 전수·draft 재귀/ragged/KV 매핑·서빙 동시==솔로).
  남은 잔여 후보: K-quant bytes 10%↓(plain 소폭)·bf16-반올림 정합(chat accept)
  → [`gemma-assistant-spec-measured.md`](./gemma-assistant-spec-measured.md).
- 프리필(2026-10 측정, 12B·fp32 scale·B1·P=512): MMQ **715.835029564~721.244315927 tok/s**(교차 5회) vs 최신 llama pp512 **1678.33 tok/s**. 같은 날 GB10, SM 2080~2093MHz 조건이며, MMQ 프리필 구간 중앙값은 5회 모두 2080MHz였다. 품질 조건 다름(llama Q4_K_M vs 엔진 W4A8). 프리필은 여전히 llama보다 느리다. 991토큰 요청의 client TTFT는 SB 약 **7.6초 → MMQ 2.47초**(각 3회, ComfyUI 상주·유휴). 반복 event 시간은 host gap을 포함한다. 7월 수치는 클럭 조건 미상이며 이 좌표와 직접 비교하지 않는다.

### 3c. 품질 기준선 (Gemma4-12B, 2026-10-26)

고정된 UltraChat `test_sft` 채팅 **36대화·assistant 응답 16,201토큰**을 채점했다. 대화마다 독립 실행하고 선두 BOS를 정확히 1개 유지했으며, 역할·턴 제어 토큰은 채점에서 제외했다. 모든 경로에 같은 토큰·길이·마스크를 사용했다. ComfyUI 상주(유휴) 조건이며, 클럭은 실제 forward 구간의 100ms 표본 중앙값이다.

| 경로 | PPL | SM 클럭 중앙값 |
|---|---:|---:|
| HF bf16 원본 | 28.547566147882474 | 2080MHz |
| HF fp32 + 엔진 q4 복원 weight | 39.80235741370706 | 2086MHz |
| 엔진 uq=1 (q4 weight + fp32 활성화) | 39.798947748384805 | 2086MHz |
| 엔진 uq=2 (W4A8, MMQ OFF) | 40.097271756992043 | 2080MHz |
| 엔진 uq=2 (W4A8, MMQ ON) | 40.253322798171865 | 2086MHz |

**HF q4 복원 / 엔진 uq=1의 PPL 비율은 1.0000856722480154**로, 사전 등록한 일치 범위 0.95~1.05를 충족했다. 복원한 q4 선형층 weight만 HF 모델에 넣고 나머지 텐서는 HF 원본으로 유지했으며, 임의 3개 텐서의 복원값은 독립 C 참조와 비트 일치했다. RoPE 테이블과 sliding mask 경계도 대조를 통과했다.

이 12B 평가에서 HF 원본 대비 손실 원인은 **weight 양자화(RTN 대칭 per-32)**로 판정했다. uq=2 / uq=1 PPL 비율은 1.0074957762826617로, 추가 int8 활성화 양자화 비용 5% 초과 조건에는 해당하지 않았다. 이는 해당 입력·경로에 대한 원인 분리 결과이며, llama Q4_K_M과의 품질 동등성이나 MMQ·fp16 스케일 채택을 뜻하지 않는다. fp16 스케일은 별도 품질 결과가 나오더라도 기본 적용하지 않는다.

별도 MMQ 기본화 게이트에서는 fp32 scale ON/OFF PPL 차이 **+0.3891811944851664%**로 사전 등록한 절대 상대 차이 0.5% 미만을 통과했다. 평균 KL 0.064412201173875513, KL 99백분위 1.1483002789553447, top-1 일치율 0.9314857107585951은 보고용 지표이며 판정에는 쓰지 않았다. fp16 측정 행렬은 의도적으로 생략했고 채택·기본 적용하지 않는다.

### 3d. QAT / PTQ 품질 회복 (Gemma4-12B, 2026-10-27)

§3c와 같은 UltraChat test_sft 36대화·assistant 16,201토큰, 동일 토큰·길이·마스크,
ComfyUI 상주(유휴). 신규 엔진은 uq=2, MMQ ON, fp32 scale이며 기존 RTN 행은
역사값 MMQ OFF다. 실제 forward 구간 SM 클럭 중앙값을 병기한다.

| 경로 | PPL (로그 원값) | SM 중앙값 |
|---|---:|---:|
| HF bf16 원본 | 28.547566147882474 | 2080MHz |
| QAT qat-ident (maxabs / 8.05) | 28.904909148350214 | 2086MHz |
| 원본 clip-search | 30.338353224952133 | 2086MHz |
| 기존 RTN (MMQ OFF) | 40.097271756992043 | 2080MHz |

QAT는 사전 등록한 PPL ≤29.98 기준을 통과했다. QAT 전용 HF FP32 골든에 대한
기존 테스트도 최대 층별 상대오차 1.476e-2 < 1.5e-2, argmax 5/5로 통과했다.
이는 위 평가와 해당 골든에 대한 결과이며 모든 입력에서 HF 원본과 동일함을 뜻하지 않는다.
기존 RTN 골든으로는 0/5였던 기록을 보존한다. 스케일만 후변환한 fp16은 PPL
29.089064374088686 (+0.6371070906790344%)로 별도 0.5% 게이트에 실패했다.
스케일을 먼저 fp16으로 반올림한 뒤 q를 다시 배정한 변형도 동일한 PPL
29.089064374088686으로 게이트에 실패했다. fp16 교차 속도 측정은 조건에 따라 생략했다.
QAT 권장 추출 경로는 fp32 scale이며 serving-setup.md를 따른다. 레시피·프로파일 경로는 아직 RTN이다.

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
