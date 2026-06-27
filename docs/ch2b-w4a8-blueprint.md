# 챕터2b 설계안 — W4A8 (weight q4 + activation int8) STEP 0 (게이트)

> 챕터2 STEP1(uint4 wide load) 기각: weight wide-load ⊥ X-coalescing, fp32-X 트래픽 4× 천장.
> llama.cpp q4 우위 일부 = int8(q8_1) activation 경로. W4A8로 전환, golden 재정의, llama.cpp 추월 노림.
> **이 문서(품질 사전측정 + 설계) 합의 후 STEP 1.** 품질 손실 크면 멈추고 재논의.

## 1. 원칙 변경 — 무엇이 바뀌나
- "fp32 대비 오차0"은 이미 q4에서 포기(2a). 유지하는 건 "**golden 대비 구현 오차0**".
- W4A8 = 양자화를 activation까지 확장 + **golden을 int8-X×q4-W로 재정의**. 정확성 포기 아님(llama.cpp도 q8_1 내 구현정확).
- **구현(새 golden 오차0)과 품질(fp32-X 대비 토큰 변화)을 분리** — q4 때와 동일 원칙.

## 2. STEP 0 품질 사전측정 결과 (torch, 모든 nn.Linear 입력 int8 round-trip, greedy 4프롬프트×64토큰)

`tools/probe_a8_quality.py`. **핵심: scale granularity가 품질을 좌우.**

| scale 방식 | 토큰 변경 | 프롬프트별 first_divergence |
|---|---|---|
| **per-token** (전체 행 1 scale) | **53.9%** (138/256) | AI=0, galaxy=29, code=6, France=64 — **파국** |
| **per-block-32** (llama.cpp q8_1식) | **22.7%** (58/256) | AI=64, galaxy=64, **code=6**, France=64 |

- **per-token은 파국**(outlier 채널 1개가 scale 지배 → 대부분 underflow). 쓰면 안 됨.
- **per-block-32는 3/4 프롬프트 완전 동일**, 코드 프롬프트만 token 6에서 flip. 22.7%는 그 1개의 **greedy cascade**가
  과장한 것(1개 뒤집히면 이후 전부 "다름"). 진짜 신호 = first_divergence: 대부분 없음, 코드류만 조기 flip.
- 판정: **per-block-32 int8 activation = 품질 손실 "작지만 0은 아님"**(프롬프트 의존, 코드/정밀 생성에서 조기 flip 가능).
  파국 아님 → 진행 가능. 단 "토큰 완전동일"은 아님 — 품질축에 기록될 대가.

## 3. DP4A / sm_121 확인 — ✅ 동작
`__dp4a(int8x4, int8x4)` sm_120 컴파일·GB10 실행 정상(테스트 result=70 정확). int8 dot 하드웨어 경로 가용.

## 4. 설계 결정 (측정 근거)

### 4a. scale granularity = **per-block-32** (필수)
- X: IN 방향 32블록마다 symmetric int8 scale (per-token 금지 — 파국). weight 기존 q4 per-block-32 scale 재사용.
- dot: blockwise Σ(int8_x · int4_w) → int 누적 후 (x_scale · w_scale)로 dequant. (llama.cpp q8_1×q4 구조와 동일.)

### 4b. X 동적 양자화 (런타임)
- weight는 static(미리), X는 매 gemv 전 런타임 양자화. 비용 = X 1패스(max+round+pack). 트래픽 절감(fp32 4B→int8 1B) 대비
  순이득 측정 필요(STEP 1 ncu). X가 작아(IN~1024) 양자화 커널은 가벼울 것.

### 4c. int8 X가 X-shared 함정을 푸나
- X가 1 byte(fp32 4B의 1/4) + DP4A로 int 누적 → **X coalesced 로드 + weight wide-load 양립 가능**(챕터2 STEP1 회귀의 원인 해소).
  단 측정으로 확인(ncu sectors/req, occupancy).

## 5. 새 golden 체계
- `tools/`에 int8-X(per-block-32)×q4-W 레퍼런스 생성(torch). 기존 q4 golden과 **별도 트랙**(golden_w4a8).
- 커널 출력 == golden_w4a8 오차0(구현 정확성, 절대조건). + q4-spec 일관성(spec on==off).

## 6. 작업 순서
- STEP 1: X int8 양자화 커널(per-block-32) + golden_w4a8 생성 + 구현 오차0 검증 + 양자화 비용 ncu.
- STEP 2: W4A8 gemv(int8 X × int4 W + DP4A). in-engine t/s, X-shared 함정 회피 확인(ncu). 본체 큰 gemv부터.
- STEP 3: 품질(fp32-X 대비 토큰, 위 22.7% 재확인) + accept + llama.cpp 재비교(비등/추월?). 3축 지도에 W4A8 점.

## 6ب. 빈틈 보완 — llama.cpp와 품질 같이 뽑기 (3-way)

우리가 잰 건 "우리 fp32-X vs 우리 int8-X". 진짜 비교 = "우리 W4A8 vs llama.cpp W4A8". 같은 엔진 내
F16 vs Q4_K_M로 cross-engine 토큰 confound 회피하고 코드 프롬프트(token6 flip)를 검증:

- **llama.cpp F16 vs Q4_K_M (fibonacci): 발산함.** thinking 텍스트가 초반에 갈림
  (F16 "incomplete, need correct implementation" vs Q4_K_M "common beginner question, I should: 1.2.3.").
  → **성숙한 llama.cpp의 W4A8(K-quant W + q8_1 X)도 코드류에서 자기 F16과 갈린다.**
- **결론: 코드 프롬프트 flip은 양자화 본질 한계, 우리 scale 처리 미흡 아님.** 우리 int8 = llama.cpp와 같은 급.

### q8_1 정독 (우리 설계 검증)
`quantize.cu`: `QK8_1=32`(per-block-32), `d = amax/127`(symmetric), `ds = (d, sum)` 저장
(sum = 블록 내 Σx). **우리 per-block-32 설계와 정확히 동일 스킴.** sum은 우리 int4 zero-point-8 보정
`Σ xq·(wq−8) = Σxq·wq − 8·Σxq`의 `8·Σxq` 항에 그대로 필요 → **블록당 (scale, sum) 저장 확정.**
dot = sx·sw·(dp4a(xq,wq) − 8·sum_b). DP4A로 int 누적 후 블록 scale 곱.

## STEP 1 결과 — X int8 양자화 커널 + W4A8 gemv + golden_w4a8 [완료]

구현: `kernels/quantize_x.cuh`(quantize_x_q8: per-block-32 → xq int8, xs scale, xsum=Σxq int32),
`kernels/gemv_w4a8.cuh`(dot = Σ_b xs·sw·(Σxq·wq − 8·xsum), STEP1은 정확성 우선 1warp/output 스칼라),
`tests/test_w4a8.cu`(in-test fp64 golden_w4a8 = per-block-32 int8 X dequant × q4-W 정확 dot).

### 구현 정확성 (오차0, 절대조건) ✅
| shape | kernel==golden_w4a8 | xq_mis | xsum_mis |
|---|---|---|---|
| IN1024×OUT256 | 5.3e-6 | 0 | 0 |
| IN1024×OUT2048 | 6.7e-6 | 0 | 0 |
| IN3584×OUT1024 | 1.9e-5 | 0 | 0 |
| IN1024×OUT4096 S5 | 7.0e-6 | 0 | 0 |
오차는 fp32-vs-fp64 누적뿐(<2e-5). quantize 출력(xq/xsum)이 CPU와 bit-일치. make test 13/13 PASS.

### 품질 델타 (gemv 레벨, W4A8 vs W4-fp32A) — 측정·기록
int8 X가 fp32 X 대비 gemv 출력에 더하는 상대오차: **rel 0.35~0.44%** (max |Δ| 0.15~0.26, 출력 max 33~74).
→ gemv 단위에선 작음. STEP0 토큰 측정(3/4 동일, 코드류 조기 flip)과 일관: 대부분 argmax 견딤, 누적 시 가끔 flip.

### 양자화 비용 (ncu) — 순이득 경고
quantize_x_q8 duration ~**8.9~12.3us/call** (compute 0.3~0.5%, grid 작아 **launch-overhead 지배**). gemv_w4a8(STEP1 naive)는 22~185us.
- 우려: 토큰당 quantize 호출 多(layer마다 attn입력/mlp입력 ≈ 2회 × 24층 + lm_head ≈ ~50회). 10us×50 ≈ 500us/token =
  현 5.68ms/token의 ~9%가 launch 오버헤드. **순이득(X 트래픽 4×↓) 갉아먹을 수 있음** → STEP 2에서 in-engine으로 판정,
  필요시 quantize를 직전 커널에 fuse하거나 X 1회 양자화→여러 gemv 재사용으로 호출수↓.

### STEP 1 결론
quantize + W4A8 gemv 수치 정확(golden_w4a8 오차0), 품질델타 gemv 0.4%. DP4A는 아직 미사용(STEP2 최적화).
다음 STEP 2: DP4A W4A8 gemv + in-engine 통합(X-shared 함정 회피 확인) + 양자화 호출수/비용 최적화 + in-engine t/s.

## STEP 2 결과 — W4A8 in-engine 통합 + 전제 기각 (중요) [부분완료]

DP4A gemv_w4a8(`kernels/gemv_w4a8.cuh`: lane-minor 파티션 + (wq−8) signed dp4a, xsum 보정 불요) +
mm() use_q4==2 통합(qwen35.cu/qwen35_decode.cu, per-call quantize). golden_w4a8 err 2.8e-6, 13/13 PASS(q4/fp32 불변).

### 통합 정확성: 토큰 레벨 OK
- W4A8 spec==plain **mismatch 0**, 토큰열 == q4 (이 프롬프트). 커널 golden_w4a8 오차0.
- 단 decode==prefill **logit err 0.28**(argmax ALL match): 양자화 round 경계가 decode/prefill 본질 fp미세차(q4서 5e-6)를
  증폭. 버그 아님(argmax 견딤) — 단 "logit 오차0 자기일관성"은 양자화-activation 경로엔 부적용. 정합성은 golden_w4a8(커널)+토큰으로.

### in-engine t/s: **W4A8 plain 180.6 vs q4 176 = +2.5%뿐.** per-call quantize가 그마저 갉음.

### 전제 기각 (산술, 측정) — 핵심
**batch-1 decode에서 X 트래픽이 weight 대비 negligible:**
| gemv | weight(q4) | X(fp32) | X/weight | int8-X 절감 |
|---|---|---|---|---|
| q_proj | 1.05MB | 4.1KB | 0.39% | weight의 0.29% |
| FFN_down | 1.84MB | 14.3KB | 0.78% | 0.59% |
| lm_head | 127MB | 4.1KB | **0.002%** | 0.002% |

→ **의뢰서 전제("int8-X로 X 트래픽 4×↓ → llama.cpp 추월")는 batch-1에서 기각.** decode gemv는 weight read가
지배(OUT≫8이라 weight=OUT·IN·0.5 ≫ X=IN·4). int8-X는 트래픽을 거의 안 줄임. **llama.cpp q4 우위는 X 트래픽 아님.**

### 재해석 — int8-X의 진짜 가치 = wide weight load 활성화
- 챕터2 STEP1 교훈: 진짜 레버는 **wide weight load**인데 fp32-X에선 X-uncoalesce로 막혔음(176→104 회귀).
- **int8-X(1byte)는 lane-minor로 coalesce되므로 wide weight load와 양립** → fp32서 불가능했던 최적화를 푼다.
  즉 int8-X는 "트래픽 절감"이 아니라 "**wide weight load의 enabler**".
- 현 gemv_w4a8는 weight를 2byte/lane으로 읽어 아직 wide-load 미실현 → +2.5%는 q4와 사실상 동일 메모리패턴 + quantize 오버헤드.
- **다음 진짜 레버**: gemv_w4a8를 wide weight load(uint32/uint4, DP4A lane-minor)로 → weight 명령수↓·latency↓가 q4를 넘나.
  + shared-X quantize로 호출수↓(오버헤드 제거). 둘 다 in-engine 판정.
- 트래픽 관점 int8-X 이득은 **batch>1**에서 비로소(X가 batch로 커짐) — batch 챕터의 일.

## STEP 3 결과 — wide weight load + shared-X → q4 못 넘음, (C) 정리 [완료]

(A) 재해석 트랙: int8-X enabler 위에 wide weight load + shared-X quantize 얹어 q4(176) 넘나 검증.

### 구현
- gemv_w4a8: 8원소 chunk(uint32 weight 1로드 → 2 dp4a), weight 로드 명령 4×↓. golden_w4a8 오차0 유지(2.7e-6).
- shared-X quantize: norm 1회 → q/k/v(또는 in_qkv/in_z) 공유, n2 1회 → gate/up 공유 (qwen35_decode.cu).
  호출수↓. spec==plain mismatch 0, 토큰==q4, 13/13 PASS(q4/fp32 불변).

### in-engine t/s: **W4A8 180.5 vs q4 176 = +2.5%뿐.** wide-load+DP4A+shared-X 다 얹어도 q4 못 넘음.

### ncu 규명 — 왜 안 넘나 (결정적)
| 지표 | gemv_q4(byte) | gemv_w4a8(wide) |
|---|---|---|
| **sectors/req** | 2 | **4.40** ✅ wide load 실현 |
| memory throughput | 15~61% | 4~10% |
| **long_scoreboard stall** | 10~20 | **44~78** (오히려↑) |
| occupancy | 32~100% | 34~86% |

- **질문 ① YES**: int8-X가 X-coalesce 안 깨고 wide weight load 가능(sectors/req 2→4.4). STEP1 fp32 uint4 실패(30 파국)를 우회.
- **그러나 여전히 latency-bound**: stall 44~78 지배, mem throughput 4~10%(compute도 아님). wide load는 sectors/req만 고치고
  **memory-level parallelism(latency 숨김)은 못 고침**. 1-warp/output·작은 IN 구조라 in-flight 요청 부족 → DRAM latency 노출.
- **= 2d fp32 gemv가 부딪힌 벽과 동일**(2d MO/occupancy +2% 헤드룸). wide-load/DP4A/int8-X가 다른 병목을 고친 셈.

### 결론 — (C) 정리
- **q4 격차(176 vs llama.cpp 266)의 본질 = gemv 구조(memory-level parallelism), weight-load 폭도 정밀도도 아님.**
  llama.cpp mmvq는 multiple-output/warp 등으로 latency를 더 숨김(더 많은 in-flight). 우리 naive 1-warp/output이 한계.
- W4A8 인프라(quantize_x, gemv_w4a8 DP4A wide, shared-X, golden_w4a8)는 **정확·검증 완료, use_q4==2로 격리 보존**.
  batch-1 decode에선 q4 대비 이득 없음(+2.5%) — 단 **batch>1에서 X 트래픽·DP4A 밀도가 의미** → batch 챕터 자산.
- 다음 진짜 레버 = **gemv 구조 재설계(multiple-output/warp, register pressure 관리)** 또는 **배칭**. 둘 다 별도 챕터.

## 7. 사용자 결정 필요 (게이트)
1. **품질 판정**: per-block-32에서 3/4 프롬프트 동일, 코드류 1/4 조기 flip(22.7%, cascade 과장). 이 정도 품질대가를
   감수하고 W4A8 진행 — 동의? (아니면 일부 민감 Linear는 fp32 유지하는 혼합도 가능 — 그건 STEP에서.)
2. **scale = per-block-32 고정**(per-token 파국이라 배제) — 동의?
3. **golden_w4a8 별도 트랙 생성**(기존 q4 golden 유지) — 동의?
4. (선택) 민감도 완화: lm_head 또는 attention은 fp32 X 유지하는 혼합 W4A8을 STEP에서 비교할까? (코드 flip 줄 수 있음.)
