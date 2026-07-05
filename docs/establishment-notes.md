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

## 2C — backend / spec 경계 (★M6에서 확정 — 트리거 ② 충족)
- **plain 코어는 모델-중립 추상화 완료**(`serving_core.h`·`serving_online.h`): 27b·35b MoE·gemma4가
  QwenBackend/Gemma4Backend로 코드 변경 최소 서빙(M4 실증). MoE도 이 경계 안(35b fused-MoE = QwenBackend).
- **★M6 경계 확정(`docs/m6-backend-boundary.md`)**: 코드 실사로 경계 인벤토리(공유 vs family-특수) + 개입 곡선 4점
  정합 확인 + measured 중복 실사. **판정: backend 어댑터 = 25~33줄 얇은 어댑터·추출할 공통 로직 없음**(각 줄이
  family primitive 배선) = 경계가 이미 얇음 = 리팩터 불필요(no-op). 진짜 중복은 전역 boilerplate(CK 매크로 96파일·
  rf() 26파일)뿐 — cosmetic·blast-radius 96파일이라 YAGNI로 안 정리. **★플러그인/레지스트리 신설 안 함**(실존
  family 2개·가상 3호 추상화는 투기). 산출물 = "새 family 추가 가이드"(공짜로 받는 것 vs 써야 하는 것·규모)
  = 공개 확장 문서 원본. **개입 비용 = paradigm 거리 비례**(same-paradigm 3호=0·new-family 2호/new-paradigm 4호=graph+config·커널 0 신규).
- **spec(27b MTP)은 별도 경로(미정 유지)**: draft→verify 사이클로 스텝 구조가 plain과 상이 → serving_core 훅에
  안 들어감(bench_serve_spec·serve_daemon_qwen_spec_online 별도). serving_backend.h [spec 확장점] 주석에 플러그인
  설계만(가변 accept out-배열). ★**API가 spec을 어떻게 노출**할지 = 열린 설계. 무리한 통합 = 정확성 리스크(금지).
- **경계 정립 결론(M6)**: plain backend 훅(ServingBackend 7메서드 / OnlineBackend 8메서드·§경계 규약)은 **안정
  인터페이스로 확정**(불변식: device op backend 소유·슬롯 격리 bit-id·scheme 명시·config 계약). spec은 별도 추상화
  (draft 커널 축) 여전히 미정 — M7+ 과제. **트리거 ②(backend 경계 안정) 충족 = M7(정립·공개) 착수 가능.**

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
