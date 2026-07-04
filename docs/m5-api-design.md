# M5 — Serve API 설계 (Phase 0: 표면 정의 + 최소 골격)

> 파이프의 마지막 단계(Serve API화) + 공개 표면 첫 등장(정립 노트 2A 실물화). **★Phase 0 = 완제품 아님 =
> API 표면 정의 + plain 1모델(gemma4) 최소 골격.** 성능/3-way 실측 금지(설계·골격 동작 확인만).
> M5의 진짜 가치 = **recipe.yaml 로드 → regime 룩업 운영** = "최적화 세팅 기반 서빙"(vLLM 등에 없는 정체성).

## 0. 아키텍처 결정 (2계층 — 자산 최대 활용, "얇게")

C++ 엔진(serving_core + backend)은 **token-ID in/out**만 안다(HTTP·토크나이저 부재 — golden/dummy 모두 pre-tokenized).
OpenAI 호환 API는 **text 프롬프트 + HTTP + SSE**가 필요. 두 세계를 얇게 잇는다:

```
[클라이언트] --HTTP(OpenAI)--> [Python front: tools/serve_api.py]         [C++ daemon: tools/serve_daemon_gemma4]
                                 · stdlib http.server(무설치)              · gemma4 + recipe 로드(1회)
                                 · HF 토크나이저(text↔ids, venv)   --pipe--> · serving_core 훅5 재사용(계산)
                                 · SSE 스트리밍·OpenAI req/resp             · token-ID in → decode → token-ID out(stream)
```

- **Python front = 얇은 표면**: OpenAI 스키마·토큰화·SSE만. 계산 안 함(자산: stdlib http.server + HF transformers, 무설치).
- **C++ daemon = 서빙 코어 감쌈**: 영속 프로세스(weight 1회 로드) + recipe regime 테이블 상주 + serving_core로 decode.
  ★계산은 기존 엔진과 동일 → **API 경유 bit-id**(계층이 계산 안 바꿈 — daemon은 ID를 serving_core에 통과만).
- **왜 2계층**: C++ BPE 토크나이저 이식은 substantial(SentencePiece/BPE from tokenizer.json) → Phase 0 밖. Python HF가 이미 있음.
  대안(단일 C++ httplib+BPE)은 다음 트랙(단일 바이너리 원하면). 대안(FastAPI)은 venv 미설치 → stdlib로.

## 1. STEP A — OpenAI 호환 API 표면 (최소)

### 엔드포인트 (지원/미지원 명시)
| 엔드포인트 | Phase 0 | 비고 |
|---|---|---|
| `POST /v1/completions` | ✅ (골격) | prompt(text)·max_tokens·stream·(temperature=0 greedy만) |
| `POST /v1/chat/completions` | ◐ 설계(다음) | chat 템플릿 적용(HF apply_chat_template) 후 completions와 동일 경로 |
| `GET /v1/models` | ✅ (골격) | 로드된 recipe의 model.id 리턴 |
| SSE 스트리밍(`stream:true`) | ✅ (골격) | `data: {chunk}\n\n` … `data: [DONE]` |
| logprobs·n>1·best_of·stop 시퀀스·tools | ✗ 미지원 | Phase 0 밖(명시적 400 or 무시+경고) |

### 요청→내부 매핑
- `prompt`(text) → **HF 토크나이저**(Python front) → token-ids → daemon.
- `max_tokens` → daemon 생성 상한. `stream` → SSE vs 단일 JSON.
- **temperature/top_p**: Phase 0는 **greedy(argmax)만**(temperature 무시 or 0 요구) — ★결정성/bit-id 게이트의 전제
  (샘플링은 RNG 도입 → bit-id 개념이 "같은 seed"로 바뀜, 다음 단계). 미지원 파라미터는 정직 명시.
- 응답: `choices[0].text`(완성) / `choices[0].delta`(스트림), `usage`(prompt/completion tokens), `model`, `id`.

### ★토크나이저 통합 (핵심 이슈)
- 엔진은 ID만 — text↔ids가 API의 선결. **결정: Python front이 HF `AutoTokenizer`로 처리**(모델별 tokenizer.json,
  HF 캐시 보유: gemma-4·Qwen3.5). detokenize도 HF(스트림 시 incremental decode — 부분 토큰 경계 주의).
- 미해결(다음): (a) C++ 네이티브 토크나이저(단일 바이너리용) (b) chat 템플릿(HF apply_chat_template) (c) BOS/EOS·
  special token 처리 (d) streaming detokenize의 multi-byte/부분 토큰 버퍼링.

## 2. STEP B — 레시피 ↔ 서버 연결 (M5의 차별점)

### 起動 (recipe 로드 + 검증)
```
serve_api.py --recipe recipes/gemma4-31b-gb10.yaml
  1. recipe.py verify 재사용: ①장비 지문 ②로컬 메모리 가드 재계산 ③(선택)미니 재현율 ④ISO/golden
     — 불일치 지문이면 거부 or warm-start 안내(기존 훅). 맹목 적용 금지(헌법).
  2. PASS → daemon 起動(weight 로드) + regime 테이블 메모리 상주.
```

### 런타임 (regime 룩업 운영 = 정체성)
- 요청 부하 추정(최근 도착률 λ + short_ratio mix) → **regime 테이블 정적 룩업**(recipe.py `run`의 λ·mix 최근접
  regime 선택 재사용) → config(bmax 등) 적용. 부하 변하면 regime 재선택.
- **온라인 컨트롤러는 이후**(Phase 0는 정적 룩업 — 요청-경계 핫스위치는 스킴 경계 미해결이라 프로세스/윈도 단위).
- ★이게 "그냥 도는 서버"와 다른 점: 서버가 **측정된 최적 config를 부하별로 적용** = 레시피 기반 서빙.

### 인터랙티브 서빙 갭 (설계 과제)
- 현 `serving_core.sc_run_continuous`는 **trace-기반**(요청 전부 사전 생성). 실 API는 **온라인 도착**(요청이 시간에
  걸쳐 옴) → 요청-구동 루프(도착→admit→decode→stream→완료) 필요 = serving_core의 온라인 변형.
- **Phase 0 골격**: 단순화(단일 요청 or 소규모 동시)로 표면·bit-id·recipe 로드를 먼저 실증. 온라인 연속배칭 통합 = 다음.

## 3. spec 코어 통합 (기록 — 다음)
M4는 **plain 코어만 모델-중립 추상화**(serving_core 훅5), spec(27b MTP)은 스텝 구조 상이(draft→verify 사이클)로 별도.
API가 27b(spec) 서빙하려면 spec 경로를 API 뒤에 어떻게 노출할지 = 설계 과제. **Phase 0는 plain(gemma4) 먼저**,
spec/MoE는 코어 연결 성숙 후(무리한 통합 금지 — 정확성 리스크). 27b spec 서빙 코어는 bench_serve_spec에 이미 있음(통합만).

## 4. 미해결/열린 이슈 (정립 노트 2C/2F 입력)
- 토크나이저: Python HF(Phase 0) vs C++ 네이티브(단일 바이너리) — 배포 형태 정할 때.
- 인터랙티브 온라인 배칭(현 trace-기반 → 요청-구동).
- spec 코어 API 노출(draft/verify 사이클을 completions 뒤에).
- verify 실행 계층 갭(레시피 세션 발견: export/warm-start spec-하드코딩 — recipe-schema.md v1 후보).
- 샘플링(temperature>0) 시 bit-id → seeded 결정성으로 개념 전환.
