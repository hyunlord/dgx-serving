# dgx-serving — 프로젝트 헌법 (CLAUDE.md)

> 이 문서는 매 세션 자동 로드되는 프로젝트 헌법이다. 모든 작업은 여기에 종속된다.
> 큰 설계 결정은 claude.ai 설계 대화에서 내려와 이 문서로 전달된다.

## 프로젝트 정체성

**dgx-serving** 은 NVIDIA DGX Spark **GB10 (sm_121)** 전용 오픈소스 LLM 추론 엔진이다.
범용 엔진(vLLM, llama.cpp)이 아니라 **단일 장비(GB10) × 특정 모델군**에 대해 최적화된 서빙 엔진을 지향한다.

GB10 하드웨어 특성 (모든 설계의 전제):
- sm_121 (Grace-Blackwell), 컴파일은 `-arch=sm_120` (sm_121 바이너리 호환) 또는 `sm_121a`
- ~128GB unified LPDDR5X, 대역폭 **~273 GB/s** (이게 병목의 근원)
- 핵심 멘탈모델: **decode 속도 ≈ 대역폭 ÷ 토큰당-이동바이트**. 모든 가속은 분모를 줄이는 것.

두 기둥 (프로젝트 목표):
1. **장비×모델 최적 서빙** — GB10에서 vLLM/llama.cpp 대비 우위
2. **대역폭 한계 돌파 기법** — 연구/차별점 (MTP, MoE sparsity, 양자화 등)

## 아키텍처 원칙 (가장 중요 — 위반 금지)

이 엔진은 "모델별 통짜 코드"가 아니라 **공용 커널 라이브러리 + 모델별 얇은 graph**로 짠다.
(학습 단계의 통짜 .cu 들은 다른 repo에 있고, 여기선 처음부터 이 구조로 간다.)

1. **커널은 모델 독립적이다.** RMSNorm, quantized matmul(GEMV), RoPE, GQA attention, SwiGLU 등은
   모델 무관한 연산 커널로 `kernels/`에 둔다. 모델이 달라도 이 커널은 공유한다.
2. **모델 차이는 graph 배선으로 표현한다.** 각 모델은 `models/<name>.cu` 에서
   공용 커널을 "어떤 순서로, 어떤 파라미터로" 호출하는 ~150줄 graph 함수다.
   새 모델 추가 = 새 커널이 아니라 새 graph (+ 정말 새로운 연산이 있을 때만 커널 1~2개 추가, 예: GDN).
3. **`#define`으로 모델 상수(H, L, FFN, n_heads)를 박지 않는다.** 런타임 config로 받는다.
   절충: 거의 안 바뀌는 것(head_dim, 양자화 블록 크기 QBLK=32)만 컴파일 타임 허용.
4. **layout/포맷도 config화.** weight 추출 파이프라인(`tools/`)은 특정 모델 하드코딩이 아니라
   config(HF AutoConfig)에서 형상을 읽어 범용 동작해야 한다.

## 검증 원칙 (어기면 디버깅 지옥 — 반드시 준수)

이 프로젝트는 과거 "끝값만 보고 추측 디버깅" 하다 오래 헤맨 적이 있다. 절대 반복하지 않는다.

1. **정확성 먼저, 오차 0이 기준.** 새 커널/graph는 반드시 레퍼런스(HF transformers fp32 출력)와
   수치 대조해 통과시킨 뒤에야 "맞다"고 선언한다. fp32 기준 max abs error < 1e-2 정도를 OK로 본다.
2. **단계별 격리 검증.** forward 전체를 한 번에 맞추려 하지 말고, 각 연산(norm→proj→rope→attn→...)을
   레퍼런스 중간값과 단계별로 대조한다. 어디서 갈라지는지 즉시 드러나게.
3. **시간만으로 성능 판단 금지.** 속도 비교는 항상 ncu 지표(sectors/req, bank conflict, occupancy,
   wavefronts)를 동반한다. 격리 벤치가 in-engine과 다를 수 있음(과거 X-shared 사례)을 늘 의심.
4. **추측 디버깅 금지.** 0%/오답이 나오면 가설을 바꿔가며 찍지 말고, 레퍼런스 중간값(골든)을 떠서
   단계 대조한다. 골든 없이 끝값만 보는 디버깅은 금지.
5. **막히면 멈추고 보고.** 2~3회 시도해도 안 풀리면 자율적으로 큰 변경을 강행하지 말고,
   현재까지의 관찰(어느 단계가 어떻게 틀렸는지)을 정리해 사용자에게 보고하고 방향을 받는다.

## 빌드/환경

- 컴파일러: `nvcc -arch=sm_120 -O2` (또는 `sm_121a`). 빌드 시스템은 Makefile.
- 프로파일러: Nsight Compute (`ncu`) — 성능 분석의 1차 도구.
- Python 레퍼런스 환경: transformers 5.12.1, safetensors 0.8.0, torch 2.12.1 (HF fp32 골든 생성용).
  - venv 위치: `~/github/gb10-cuda-ht/07-engine/.venv` (레퍼런스 환경 재사용).
  - 주의: PyTorch가 sm_121 공식 미지원 경고를 띄우나 동작은 함.
- 검증 모델 자산 (사용자 머신):
  - SmolLM3-3B (HF) — 첫 타깃, Llama 계열. HF 캐시: `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B`
  - Qwen3.5-0.8B (HF safetensors) — MTP 타깃. 본체가 GDN(linear attn)+full 혼합, MTP head weight 보유.
- 학습 repo 자산 (참고/이식 출처): `~/github/gb10-cuda-ht/07-engine`
  - 검증된 통짜 커널: `decode_batch.cu`, `decode_spec.cu`, `full_engine.cu` 등
  - SmolLM3 형상: H=2048, NQ=16, NKV=4, HD=128, FFN=11008, V=128256, L=36, QBLK=32
  - q4 weight 포맷: `ref_full/layers/`에 레이어별 q4.bin+scale.bin, 골든 `ref_full/full_logits.bin`

## 코드/커밋 컨벤션

- 커밋 메시지: 명령형, 간결. 한 커밋 = 한 논리적 변경.
- 새 커널/graph는 반드시 검증 테스트(레퍼런스 대조)와 함께 커밋. 테스트 없이 "동작함" 커밋 금지.
- 큰 바이너리(실행파일, *.bin weight 덤프, .venv)는 절대 커밋하지 않는다 — .gitignore로 차단.
- 문서: 설계 결정과 검증 결과는 `docs/` 또는 worklog .md에 기록.
