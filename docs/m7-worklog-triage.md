# M7 Phase 1 — worklog 공개 분류 (선별 목록 · 삭제 아님 · 애매하면 보류 기본)

> 결정 3(공개 범위) = **엔진 + 판정 문서**(worklog 선택). 이 문서는 `docs/` 90개를 티어로 **분류**만 한다.
> ★현재 전부 git 추적 중 → **push 시 전부 공개됨**. 보류(Tier 3) 항목을 curated 공개로 원하면 push 전 `git rm`
> (또는 `docs/worklog/`로 이동) = **사용자 결정**. 이 문서는 분류만·자동 삭제 안 함(보류 기본).

## Tier 1 — 공개 확정 (정제된 차별 자산)

verdict·방법론·가이드·스키마·결과 = 프로젝트의 얼굴. README가 직접 링크.

- `c-bandwidth-seeds-verdict.md` — 대역폭 시드 전수 판정("가능한 것의 지도") ★
- `m6-backend-boundary.md` — 설계 철학·경계·새 family 추가 가이드 ★
- `m3-multi-objective-pareto.md` — 다목적 파레토·SLO 튜닝
- `recipe-schema.md` — 레시피 v0 스키마
- `establishment-notes.md` — 정립 노트(4절)
- `m7-public-doc-structure.md` — 공개 문서 구조
- `m5-3way-methodology.md` · `m5-3way-results.md` — 정직 벤치(2-way, 지는 좌표)
- `m5-spec-differentiator.md` — spec 곱 차별점
- `autotune-harness.md` — 자동튜닝(베이지안) 하네스

## Tier 2 — 공개 권장 (worklog이나 "측정-먼저 문화"의 증거 = 정직 자산, "개발 로그" 라벨)

마일스톤 blueprint/기록 — 측정 과정의 정직한 궤적(격리≠엔진·소거도 성과). 정제 비용 낮음(README에서 "worklog" 라벨로 이미 명시). 공개가 신뢰를 더함.

- `01-stage1-blueprint.md`·`02a~02d`·`03a-q4spec-blueprint.md` (초기 커널·MTP·spec·q4 여정)
- `ch2~ch10` 시리즈 (배치·gemm·MMQ·FA·spec·serving 챕터 기록)
- `m4-*` (4모델 확장 — gemma4·MoE 35b/26b phase/serving/recipe)
- `m5-api-*`·`m5-spec-api-design.md` (Serve API 설계·성능)
- `decode-kernel-diagnosis.md`·`serving-scheduler-step0.md`·`ch6-3way-27b-results.md` (진단·선례)

## Tier 3 — 보류 후보 (선별 필요 · 기본=보류 · push 전 사용자가 정리 결정)

중복·중간 POC·내부 스크래치 톤 = 공개 가치 < 다듬기 비용일 수 있음. **기본 보류**(curated 원하면 push 전 제거/이동).

- **q2 중복 3종**: `q2-quant-quality-study.md`·`q2-quality-measured.md`·`q2-sophisticated-measured.md` —
  결론은 verdict(Tier 1)에 흡수됨. 원문은 과정 기록(택1 공개 or 전부 보류).
- **micro-POC**: `bmbn-tile-shape-poc.md`·`bmbn-in-engine-validation.md`·`pipeline-depth-poc.md`·
  `per-nk-shape-dispatch.md`·`c1-mmq-kernel-param-tune.md`·`c-sparsity-diagnosis.md` — 세부 실험(결론은 verdict).
- **derisk 단편**: `ch9-step0-fp4-derisk.md`·`ch9-2bit-quality-derisk.md`·`ch9-ngram-accept-derisk.md`·
  `ch8-step0-mtp-fp16-derisk.md`·`ch5-bitwidth-measurement-memo.md` — 짧은 사전 검토(선택).
- **q3 POC**: `q3-gemv-poc.md` — 결론(소거)은 verdict에. 커널 자산은 코드로 보존(문서는 선택).
- `stage2-vllm-blueprint.md` — vLLM 3rd-leg 미완(후속) 설계 메모.

## m7-* 메타 문서 (공개 전 정리 결정)

- `m7-readme-draft.md`·`m7-decisions.md`·`m7-worklog-triage.md`(이 문서) = **공개 준비 과정 산출**.
  공개 후엔 내부 메모 성격 → **push 전 제거 or docs/internal/ 이동 권장**(README.md·LICENSE로 결과는 이미 반영됨).

## 정리 방침 (권장 — 결정은 사용자)

1. **최소 정리(권장)**: Tier 1+2 그대로 공개(worklog은 정직 자산)·Tier 3와 m7-* 메타만 push 전 검토.
   README가 "worklog=개발 로그(정제 전)" 라벨을 이미 달아 기대치 관리됨 → **과한 정리 불필요**.
2. **curated 공개**: Tier 1만 docs/ 최상위·나머지 `docs/worklog/`로 이동(구조 명확하나 링크 갱신 비용).
- ★어느 쪽이든 **삭제/이동은 push 전 사용자 실행**. 이 세션은 분류만(보류 기본·헌법).
