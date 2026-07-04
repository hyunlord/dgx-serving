# M5 STEP D — 3-way 공정 비교 하네스 (방법론 설계 — 실측 금지)

> HTTP 서버가 서면(M5) 챕터9부터 인프라-gated로 미룬 3-way(llama-server / vLLM / dgx-serving) 공정 비교가 처음
> 가능해진다. ★그러나 **이번은 방법론 설계 + 셋업 상태 조사만** — 실측 강행 금지(공정 조건·인프라 확인이 선결).
> 선례: `docs/ch6-3way-27b-results.md`(CLI 기반 3-way, 27b) — 이번은 그 위에 **HTTP 서버 기준** 재설계.

## 1. 왜 HTTP 기준인가 (CLI 3-way와 다른 점)
- ch6 3-way는 CLI 벤치(각 엔진 내부 타이머). HTTP 3-way는 **클라이언트가 본 실제 서빙 성능**(네트워크/직렬화/스케줄러
  포함) — 배포 현실에 가깝고, 세 엔진을 **같은 부하 생성기**로 때릴 수 있음(동일 trace → 동일 조건 강제).
- dgx-serving의 정체성(recipe 기반 서빙)은 HTTP에서만 드러남(regime 룩업이 실 요청 부하에 반응).

## 2. 공정 조건 명세 (같은 조건이 "진짜" 같은지)
| 축 | 요구 | 함정 |
|---|---|---|
| **모델/양자화** | q4급 대응(dgx W4A8 ↔ llama.cpp Q4_K_M ↔ vLLM AWQ/GPTQ-4bit) | 양자화 스킴이 정확히 같지 않음 — bit-정확 비교 불가, **품질(perplexity/argmax) 별도 보고**하고 속도만 대조 |
| **프롬프트/부하** | 동일 trace(같은 프롬프트 집합·같은 Poisson λ·같은 도착 시각) | 각 서버 클라이언트가 다르면 부하 불일치 — **단일 부하 생성기**로 세 서버 동일하게 |
| **측정 지점** | HTTP 기준 TTFT(첫 토큰)·tpot(토큰간)·throughput(tok/s)·e2e | 서버 내부 타이머 ≠ 클라이언트 관측 — **클라이언트 측 타임스탬프 통일** |
| **토크나이저** | 같은 tokenizer(HF) — 프롬프트 문자열은 같아도 토큰 수가 엔진별로 다를 수 있음 | dgx=HF, llama=gguf 내장 tokenizer, vLLM=HF — **토큰 수 정규화**(tok/s는 각자 토큰 기준 명시) |
| **동시성/배치** | 같은 max concurrency·같은 batch 정책 노출 범위 | vLLM continuous batching vs dgx regime 룩업 — 정책 차이 자체가 비교 대상(공정≠동일 정책, 동일 부하) |
| **warm-up** | 세 서버 모두 warm(클럭·캐시) 후 측정(dgx warm 프로토콜 준용) | cold 측정은 GB10 클럭 교란(기존 교훈) |

## 3. 하네스 스펙 (설계)
```
[부하 생성기(단일)] — 동일 trace(프롬프트·λ·도착시각, seed 고정) → 3 서버 각각 HTTP 병렬 세션
  → 각 요청의 TTFT/토큰 타임스탬프 수집(클라이언트 측) → TTFT/tpot/throughput 백분위 집계
  → 품질(선택): 같은 프롬프트 greedy 출력의 argmax 일치율(양자화 스킴 차 표면화 — 속도와 분리 보고)
```
- bench_serve의 trace 생성기(Poisson·mix) 재사용 가능 — HTTP 클라이언트로 감싸면 3 서버 공통 부하.
- OpenAI 호환 표면(3 서버 모두 /v1/completions 지원 시) → 단일 클라이언트로 세 서버.

## 4. 셋업 상태 조사 (인프라-gated 재확인 — 실측 전 선결)
| 서버 | GB10 상태 | 갭 |
|---|---|---|
| **dgx-serving** | M5 API Phase 0 골격 중(gemma4 plain) | 온라인 배칭·spec/MoE·실-rope tokenizer 정합 |
| **llama-server** | llama.cpp 빌드됨(artifacts/llamacpp: **0.8b gguf만**·ch6 CLI 비교 이력) | 타깃 모델(27b/35b/gemma4) **gguf 변환 부재** + llama-server(HTTP 모드) 起動 미확인 |
| **vLLM** | **미설치**(venv에 vllm X) | GB10 sm_121 빌드/설치 자체가 챕터9 infra-gated 난제(stage2-vllm-blueprint) — 미해결 |

★결론: **HTTP 3-way 실측은 아직 불가**(vLLM 미설치·llama-server 타깃 gguf 부재·dgx API 골격 단계). 챕터9 인프라-gated
상태 그대로 재확인됨. **방법론(위 §2-3)은 준비 완료** — dgx API 성숙 + llama-server gguf 변환 + vLLM GB10 설치가
각각 선결. 실측은 골격 성숙 후 별도 트랙(강행 금지 — overclaim 방지).

## 5. 다음 (실측 진입 조건)
1. dgx-serving M5 API가 온라인 배칭 + 타깃 모델(최소 gemma4 or 27b) 안정 서빙.
2. llama-server: 타깃 모델 gguf 변환(Q4_K_M) + `--server` 모드 起動 확인.
3. vLLM: GB10(sm_121) 설치 경로 확보(별도 de-risk — 챕터9 blueprint).
4. 셋 다 /v1/completions 뜨면 → 단일 부하 생성기로 동일 trace 실측(TTFT/tpot/throughput + 품질 분리).
