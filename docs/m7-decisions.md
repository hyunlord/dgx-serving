# M7 Phase 0 — 사용자 결정 요청 목록 (executor는 후보+장단만·결정은 사용자)

> ★공개(push)는 사용자 손. 이 문서는 **결정 사항을 표면화**만 한다(자율 확정 금지). 각 항목: 후보 + 장단 + (있으면) 권장.
> 결정 후 → M7 Phase 1(결정 반영·README.md 교체·공개 준비 마무리).

## 1. 이름 (name)

| 후보 | 장 | 단 |
|---|---|---|
| **dgx-serving 유지** | 이미 커밋 이력·문서 전반에 사용·설명적(DGX+serving) | "DGX"는 NVIDIA 상표(오픈소스명에 벤더 상표 = 잠재 마찰)·GB10 특화가 안 드러남 |
| GB10 특화명(예: gb10-serve, spark-serve) | 장비 특화가 이름에 명시 | 리네이밍 = 문서 전반 치환·이력 혼동 |
| 중립 코드네임 | 상표 마찰 회피 | 검색성·설명력 약함 |

- 권장: **dgx-serving 유지**(전환 비용·이력 일관성) — 단 "DGX/Spark는 NVIDIA 상표" 고지 1줄 README에.

## 2. 라이선스 (license) — 후보+함의만·결정은 사용자

| 후보 | 함의 |
|---|---|
| **Apache-2.0** | 특허 grant 포함(기여자 특허 방어)·기업 채택 친화·NOTICE 요구. 커널/추론 엔진 오픈소스 관행(vLLM=Apache-2). |
| MIT | 가장 단순·제약 최소. 특허 grant 없음. llama.cpp=MIT. |
| (미정 유지) | 라이선스 없이 공개 = 기본 "all rights reserved"(사실상 사용 불가) → 공개 시 반드시 택일 필요 |

- 참고: 이 repo는 **가중치를 배포 안 함**(HF에서 사용자가 받음) → 모델 라이선스는 별개(Qwen/Gemma 각자 라이선스 준수는 사용자 몫, README 고지 권장).
- 코드에 llama.cpp/vLLM 소스 **복사본 없음**(분석 문서에 파일 경로 인용만) → 라이선스 상충 없음. 권장: **Apache-2.0**(특허 grant·엔진 관행) — 단 최종 결정은 사용자.

## 3. 공개 범위 (scope) — ★지형도/worklog는 강한 차별 자산

| 자산 | 공개? | 장단 |
|---|---|---|
| 코드(커널·graph·serving·튜너·recipe·API) | (핵심) | 엔진 본체 — 당연 공개 후보 |
| **docs/ 판정·gate 문서 89개** | **결정 필요** | ★지형도(7시드 판정)·honest-perf 방법론·측정 좌표 = **강한 차별 자산**(신뢰·투명). 단 일부는 내부 worklog 톤(정제 필요)·수치 많음 |
| recipes/ 3종 + 스키마 | (자산) | "검증 가능한 config 배포판" = 차별점. 공개 권장 |
| **CLAUDE.md**(프로젝트 헌법) | **결정 필요** | 설계 철학·검증 원칙 = 좋은 소개. 단 **로컬 자산 경로**(venv·HF 캐시·학습 repo 경로) 포함 → 공개 시 scrub 필요 |
| tools/autotune/artifacts/*.json(5개) | 결정 필요 | 튜닝 산출물(regime 테이블) — recipe 재현 스토리엔 자산. 로컬 경로 내장 여부 확인됨(문제 없음) |
| 제외 확정 | 가중치·.bin·골든·.venv·.omc | gitignore로 이미 차단(이력에도 없음 — 검증됨) |

- 권장: **코드 + recipes + 정제된 docs(지형도·방법론·M6 가이드) 공개**(투명성이 차별점). CLAUDE.md는 **경로 scrub 후 공개**(설계 철학 소개로 가치 큼). 내부 worklog 톤 문서는 "docs/worklog/"로 분리하거나 그대로 두되 README에서 "개발 로그(정제 안 됨)" 라벨.

## 4. 포지셔닝 톤 (tone)

| 후보 | 장 | 단 |
|---|---|---|
| **연구 프로젝트**(honest 측정·지형도 중심) | 지는 좌표·소거 결과가 자연스럽게 자산이 됨·overclaim 압력 없음·차별점(방법론)이 전면 | "쓸 수 있는 도구?" 신호 약할 수 있음 |
| 실용 도구(서빙 엔진·레시피 재현) | 채택 유인·API/recipe가 전면 | 3-way 지는 좌표와 "실용 최속" 기대 사이 긴장(overclaim 리스크) |
| 혼합(연구-기반 실용 엔진) | 둘 다 | 메시지 흐려질 수 있음 |

- 권장: **연구 프로젝트 톤 우선 + 실용 표면(API/recipe) 명시**(정직 벤치가 자산이 되는 프레이밍 — "GB10에서 무엇이 가능한지의 측정 지도 + 그 위의 서빙 엔진"). = README 초안(`m7-readme-draft.md`)이 이 톤.

## 5. 공개 시점 (timing) + 부수 결정

- **공개 시점**: 즉시 vs (26B 파레토 측정·spec 경계 정리 등 알려진 갭 일부 해소 후). 권장: **README/라이선스 확정 즉시 공개 가능**(갭은 로드맵으로 정직 명시 — 완벽 대기 불필요). 결정은 사용자.
- **author 이메일**: 커밋 이력 = `hyunlord@naver.com`(개인 이메일 노출). 후보: 그대로 / `git`의 noreply 전환(이력 재작성 필요·이력 공유 시 무거움) / 향후 커밋만 noreply. 권장: **향후 커밋 noreply 정도**(이력 재작성은 과함) — 결정은 사용자.
- **문서 로컬 경로 scrub**: README env(venv 경로)·CLAUDE.md(로컬 자산)·docs/llama-mmvq-latency-analysis.md(`/home/hyunlord/repos/llama.cpp-mtp` 인용)·blueprint 문서들. 공개 범위 확정 후 일괄 scrub(Phase 1). python 툴 2개(HF 캐시 경로)는 **이번 세션 이미 이식성 수정 완료**.

## ★결정됨 (사용자, 2026-07-05 — Phase 1 입력)

| 항목 | 결정 | Phase 1 반영 작업 |
|---|---|---|
| 1. 이름 | **dgx-serving 유지** | README에 "DGX/Spark는 NVIDIA 상표" 고지 1줄 |
| 2. 라이선스 | **Apache-2.0** | `LICENSE` 파일 추가 + 소스 헤더 정책(선택)·README 배지·모델 라이선스 별개 고지 |
| 3. 공개 범위 | **엔진 + 판정 문서**(worklog 선택 공개) | 코드+recipes+정제 docs(지형도·방법론·M6 가이드) 공개·worklog는 선별·CLAUDE.md 경로 scrub 후 포함 |
| 4. 톤 | **연구 프로젝트 톤** | README 초안(m7-readme-draft) 톤 그대로 확정·측정 지형도/방법론 전면 |
| (부수) 시점 | 미명시 | README/LICENSE 확정 즉시 공개 가능(갭=로드맵). push는 사용자 손 |

**Phase 1 계획(다음 — 공개 push 없이 준비 마무리)**: ①`LICENSE`(Apache-2.0) 추가 ②`m7-readme-draft.md` → `README.md` 확정(상표 고지·라이선스 배지·연구 톤) ③문서 로컬 경로 scrub(CLAUDE.md·README env·llama 분석·blueprint) ④worklog 선별(공개/보류 분류) ⑤clean 환경 빌드 스모크 ⑥author 이메일 정책(향후 noreply). **공개(push)는 사용자 실행.**

## STEP 0 위생 요약 (이번 세션 확인·수정)

- ✅ gitignore 견고(weights·.bin·.venv·.omc·골든 차단) · 추적 301파일에 바이너리 0 · **이력에 비밀/가중치 커밋 0**(스캔 확인).
- ✅ Makefile 이식성(NVCC/ARCH 오버라이드 가능·로컬 경로 0) · 코드/빌드 파일 로컬 경로 0.
- ✅ **수정**: `tools/dump_mtp_weights.py`·`tools/mtp_reference.py` 하드코딩 HF 캐시 경로 → `HF_HUB_CACHE`/`HF_HOME`/`~/.cache` 이식성(계산 불변).
- ◐ **남음(공개 범위 결정 후 scrub)**: 문서류 로컬 경로(README env·CLAUDE.md·llama 분석·blueprint) · author 이메일.
- ⚠ fresh-clone 빌드 재현: nvcc+GB10 필요라 이 세션 실빌드 미실행(GPU/toolkit 의존) — Makefile 이식성은 정적 확인. Phase 1에서 clean 환경 빌드 스모크 권장.
