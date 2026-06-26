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

## 7. 사용자 결정 필요 (게이트)
1. **품질 판정**: per-block-32에서 3/4 프롬프트 동일, 코드류 1/4 조기 flip(22.7%, cascade 과장). 이 정도 품질대가를
   감수하고 W4A8 진행 — 동의? (아니면 일부 민감 Linear는 fp32 유지하는 혼합도 가능 — 그건 STEP에서.)
2. **scale = per-block-32 고정**(per-token 파국이라 배제) — 동의?
3. **golden_w4a8 별도 트랙 생성**(기존 q4 golden 유지) — 동의?
4. (선택) 민감도 완화: lm_head 또는 attention은 fp32 X 유지하는 혼합 W4A8을 STEP에서 비교할까? (코드 flip 줄 수 있음.)
