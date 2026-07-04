# M5 API 성능 측정 — API 계층 오버헤드 (3-way 선결)

> 프레이밍(honest-perf ⑦): "dgx-serving vs vLLM"이 **엔진 비교**가 되게 하려면, 우리 API 계층
> (HTTP·HF 토큰화·IPC 파이프·Python 스레딩)이 엔진 위에 얹는 오버헤드를 **measured로 확인·제거**해야
> 한다. 안 그러면 인프라 미최적을 엔진 탓으로 오귀속하게 된다. 이 문서는 그 측정 결과다.
> 스코프: gemma4-31B plain·greedy. 목적=성능 측정(정확성은 이미 PASS — bit-id 유지만 확인).

## 방법 (measure-first·controlled·warm)

`tools/bench_api.py`: 동일 워크로드(같은 pre-tokenized 프롬프트·gen 길이·동시성)를 두 경로로 warm·back-to-back:
- **L2 daemon-direct**: `serve_daemon_gemma4_online`에 pre-tokenized 태그드 요청을 파이프로 직접 주입
  (HTTP·HF 토큰화 없음) = **엔진 + IPC**.
- **L3 full API**: `serve_api.py`에 동시 HTTP 스트리밍 요청(HF 토큰화 왕복·Python 스레딩 포함) = **전체 API 스택**.
- 갭(L2→L3) = **API 계층 오버헤드** 그 자체(같은 워크로드라 엔진 계산은 동일 — greedy bit-id 교차확인).

지표: throughput(집계 tok/s)·TTFT(첫 토큰 지연)·tpot(토큰간 간격 = 배치 스텝시간).
warm 프로토콜(honest ⑪): throwaway 워밍 런으로 SM 클럭 208MHz(idle)→2398MHz 승격 확인 후 측정.

## 결과 (G=64, warm, 두 동시성)

| N | 경로 | throughput | TTFT mean | tpot mean | **API 갭(throughput)** |
|---|---|---|---|---|---|
| 8 | L2 daemon-direct | 55.2 tok/s | 610ms | 137.5ms | — |
| 8 | L3 full API | 54.7 tok/s | 588ms | 139.2ms | **+1.0%** (잡음 내) |
| 16 | L2 daemon-direct | 65.8 tok/s | 1153ms | 228.8ms | — |
| 16 | L3 full API | 65.5 tok/s | 1193ms | 229.2ms | **+0.4%** (잡음 내) |

**견고성**: 동시성 N=8→16(스레드·GIL 압박↑)에도 API 갭은 +1.0%→+0.4%로 **안 커짐** — 스레드가 GPU-대기로
블록(GIL 해제)이라 스레딩 병목 **경험적 기각**. (N=16서 throughput↑·tpot↑·TTFT↑는 배치 스케일링/순차 prefill 탓 =
L2·L3 공통 엔진 특성, API 오버헤드 아님.) warm 확인(208→2437MHz)·bit-id 교차확인 PASS.

**STEP A 분해 (토크나이저 micro-bench, measured):**
- `tokenize(prompt)` = 0.012 ms/call (Rust fast tokenizer — 무시 가능).
- streaming detok(현 O(G²) 누적 재디코드, G=64) = **0.39 ms/req 전체** (토큰당 0.006ms).
- 단일 detok(1회) = 0.010 ms.
- → **GPU decode 스텝(137ms)이 벽시계의 ~99% 지배**. 토큰화·detok·IPC·스레딩 합쳐 <0.3%.

## 결론 (정직)

**API 계층 오버헤드 = +0.4~1.0% (측정 잡음 내·N=8/16 견고). API는 이미 엔진 천장에 있다.** GPU decode 스텝이
압도적으로 지배(99%)하며, 우리가 코드리뷰에서 "명백한 병목"으로 의심한 **streaming detok O(G²)조차 measured
0.39ms/req(벽시계의 0.004%)** = 비-지배. 여기를 "최적화"하면 honest ⑦ / cudaFree 교훈(nsys API%≠critical-path)의
전형적 오귀속이 된다.

**STEP B — 제거할 지배 오버헤드 없음.** 측정이 "이미 근접"을 가리킨다(STEP 0의 명시적 예상 결과 "작으면 이미
근접"). O(G²) detok는 알고리즘상 준최적이나 fast tokenizer 상수가 극소라 measured-negligible이고, incremental
detok으로 바꾸면 (다중토큰 UTF-8/SentencePiece 병합 탓) text bit-id 리스크가 실측 이득 0.004%를 위해 발생 —
measure-first는 "손대지 말라"이다. **근본 재설계(front C++화)는 스코프 밖**(명백한 것만 — 판단 보고).

**참조 — 엔진 기준선**: `bench_serve_gemma4`(기록) λ16 B16 ≈ 41 tok/s. L2(55.2)가 이보다 높은 건 **워크로드 차이**
(내 하네스=8슬롯 full-batch 버스트·avgB=8 상수 / bench=Poisson λ·avgB<Bmax·ramp/drain·mixed gen). 다른 워크로드라
직접 비교 아님(정직) — 유효한 비교는 오직 **동일 워크로드 L2 vs L3 갭(+1%)**.

## 3-way 인프라 재확인 (STEP C)

**여전히 인프라-gated**(ch9/M4 그대로):
- vLLM/SGLang/flash-attn/flashinfer **미설치** (sm_121 FA/CUTLASS 미지원, 커스텀 빌드 out-of-scope).
- llama-server 바이너리 부재.
- **gemma4-31B(E4B) gguf 부재** — 로컬엔 gemma-4-12B·26B-A4B gguf만 존재(우리가 서빙하는 31B dense 아님).

→ 3-way 실측은 인프라 선결(다음). **단 이 측정의 값은 delivered**: 우리 API가 엔진 천장(+1%)임을 measured로
확인 = 인프라 준비 시 비교가 **엔진 대 엔진**이 됨(우리 미최적 API 탓 오귀속 방지, honest ⑦).

## 자산
- `tools/bench_api.py` — API 계층 오버헤드 측정 하네스(L2 daemon-direct vs L3 API, warm·controlled).
- 재현: `<venv>/python tools/bench_api.py [N] [G]`.
