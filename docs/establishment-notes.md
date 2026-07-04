# 정립 노트 (Establishment Notes) — 공개 표면·경계·자동화 갭

> 프로젝트가 "실험 코드"에서 "공유 가능한 산출물"로 정립되는 과정의 노트. **M5(Serve API)가 정립 시작점** —
> 처음으로 외부에 노출되는 공개 표면(API·레시피)이 등장하며 인터페이스 안정성·경계·자동화 갭을 명시적으로 정해야 함.
> 이 노트는 설계 대화에서 내려온 트랙(2A 공개 API / 2C backend·spec 경계 / 2F 자동화 갭)의 실물 기록.

## 2A — 공개 API·Serve 표면 (M5에서 등장)
- **레시피 스키마 v0**(`docs/recipe-schema.md`): 이미 공개 아티팩트 표면(공유 recipe.yaml + verify 로드 검증 4단).
  3모델 실사용으로 **v1 후보 9개** 드러남(export/warm-start spec-하드코딩 등 — 아래 2F).
- **Serve API 표면**(`docs/m5-api-design.md`, Phase 0): OpenAI 호환 `/v1/completions`(+chat 설계)·SSE 스트리밍·
  `/v1/models`·`--recipe` 起動. ★차별점 = **recipe 로드 → regime 룩업 운영**("최적화 세팅 기반 서빙" — vLLM 등에 없음).
  Phase 0는 표면 정의 + gemma4 plain 골격. **greedy만**(샘플링·logprobs·n>1 미지원 — 명시).
- 정립 결정 필요(다음): API 버전 정책·안정 표면 범위·토크나이저 배포 형태(Python HF vs C++ 네이티브).

## 2C — backend / spec 경계 (M4 유산 + API 노출 과제)
- **plain 코어는 모델-중립 추상화 완료**(`serving_core.h` 훅5): 27b·35b MoE·gemma4가 QwenBackend/Gemma4Backend로
  코드 변경 최소 서빙(M4 실증). MoE도 이 경계 안(35b fused-MoE = QwenBackend).
- **spec(27b MTP)은 별도 경로**: draft→verify 사이클로 스텝 구조가 plain과 상이 → serving_core 훅에 안 들어감
  (bench_serve_spec 별도). ★**API가 spec을 어떻게 노출**할지 = 열린 설계(completions 뒤에 draft/verify를 감추는 계층).
  Phase 0는 plain(gemma4) 먼저 — spec/MoE API 연결은 코어 성숙 후(무리한 통합 = 정확성 리스크, 금지).
- 경계 정립: "plain backend 훅5"는 안정 인터페이스 후보. spec은 별도 추상화(draft 커널 축)가 필요 — 미정.

## 2F — 자동화 갭 (개입 곡선의 남은 수동 지점)
레시피 3모델 일괄(스냅샷)에서 실사용으로 드러난 파이프 자동화 갭 (`docs/recipe-schema.md` v1 후보 상세):
- **export spec-하드코딩**: `cmd_export`가 K/W/accept/DRAFT_MW 가정 → plain/MoE 레시피는 hand-authored(수동).
- **warm-start 훅 spec-하드코딩**: `recipe_warmstart_configs`가 plain 레시피에 `KeyError 'K'` CRASH(실측) →
  cross-paradigm transfer 코드로 불가(가설은 config-space 구조로 확정).
- **verify 조건 충실도**: 35b repro 168.8%(local 106 vs claim 62.9)를 "PASS"로 리포트 — serving_mode 미기록 +
  repro 상한 미체크(>100% 조건 불일치 못 걸름). = "같은 조건끼리만" 원칙의 실행 계층 갭.
- **인터랙티브 서빙**: serving_core가 trace-기반(사전 생성) → 실 API의 온라인 도착(요청-구동)과 갭.
- 정립 방향: 이 갭들의 **scheme-class 추상화**(export/warm-start가 spec/plain/MoE를 유형으로) = M5/M6 공개 API의 선결.

## 요약 — 정립 시작점으로서의 M5
M5는 단순 기능 추가가 아니라 **공개 표면의 등장**이라 위 세 트랙(공개 API 안정성·backend/spec 경계·자동화 갭)을
명시적으로 정해야 하는 시점. Phase 0는 그 표면을 정의(설계 + 최소 골격)하고, 갭들을 정직하게 목록화(구현은 트랙별
별도). 관련: `docs/m5-api-design.md`·`docs/m5-3way-methodology.md`·`docs/recipe-schema.md`.
