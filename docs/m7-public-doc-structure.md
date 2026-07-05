# M7 Phase 0 — 공개 문서 구조 아웃라인

> **이것은 M7 Phase 0 아웃라인 — 공개 문서 본문 아님. Phase 1에서 결정 반영 후 실제 본문을 작성한다.**
> 이 문서는 "어떤 공개 문서를 어떤 순서로 낼지 + 각 절이 어느 기존 문서/근거에서 오는지"의 매핑만 다룬다.
> 코드 변경 0, 새 수치 발명 0 — 모든 근거는 기존 문서 인용으로만 채운다.

## 전제

`docs/establishment-notes.md`의 트랙 구조(2A 공개 API / 2C backend·spec 경계 / 2F 자동화 갭)가
M6(`docs/m6-backend-boundary.md`)로 트리거 ②(backend 경계 안정)를 충족해 M7(정립·공개) 착수 가능 상태.
이 노트는 그 트랙들을 **공개 문서 4절**로 확장 매핑한다. 절 번호(2D/2E/확장가이드/2F)는 establishment-notes의
2A/2C/2F 트랙을 이어받는 명명이며, 실제 공개 시 문서 제목·순서는 Phase 1에서 확정한다.

---

## 절 1 — 2D 설계 철학 (Design Philosophy)

**요지**: 이 엔진이 "모델별 통짜 코드"가 아니라 공용 커널 라이브러리 + 모델별 얇은 graph로 짜인 이유와,
그 헌법이 4모델(SmolLM3·Gemma4-31B·Qwen3.5-35B-A3B MoE·Qwen3.5-26B-A4B MoE)에 걸쳐 실측으로 버텼다는 근거.

다룰 내용:
- 커널은 모델 독립적이다 (RMSNorm·GEMV·RoPE·GQA attention·SwiGLU 등 `kernels/`에 모델 무관 연산으로 존재).
- 모델 차이는 graph 배선으로 표현한다 (`models/<name>.cu` = 공용 커널을 어떤 순서·파라미터로 호출하는 그래프).
- `#define`으로 모델 상수를 박지 않는다 — 런타임 config (`ModelConfig`)로 받는다. 예외는 거의 불변인 것
  (head_dim, 양자화 블록 크기 QBLK=32)만.
- layout/포맷도 config화 — weight 추출 파이프라인(`tools/`)은 HF AutoConfig에서 형상을 읽어 범용 동작.
- 레시피(`recipes/*.yaml`) = 이 config-주도 헌법의 배당금: "검증 가능한 주장이 붙은 config 묶음" 배포판.

**근거 문서**:
- `CLAUDE.md` §아키텍처 원칙 (커널 독립·graph 배선·config화·layout config화 — 4개 원문 조항 그대로 인용).
- `docs/m6-backend-boundary.md` §개입 곡선 4점과 대조 — "개입 비용은 paradigm 거리에 비례. same-family/same-paradigm
  = 0(3호 35B-A3B), new-family(2호 Gemma4)/new-paradigm(4호 26B-A4B) = 새 graph + config, **커널은 언제나 0 신규**"
  = 헌법이 4모델 실측으로 버틴 핵심 문장.
- `docs/recipe-schema.md` §구조 — 레시피 5층 YAML 구조(레시피=config 배포판 근거).

---

## 절 2 — 2E 정확성 계약 (Correctness Contract)

**요지**: bit-identity가 무엇을 의미하고(스킴 내 개념, fp32-exact 아님) 무엇을 의미하지 않는지를
공개적으로 명시. 검증 먼저·골든 대조·추측 디버깅 금지의 문화를 사용자가 신뢰할 수 있는 계약으로 번역.

다룰 내용:
- ★scheme-internal golden: bit-identity는 "같은 배치 형상·양자화 dispatch(스킴) 내"에서만 성립하는 개념이지
  fp32-exact 재현이 아님. 스킴 경계 전환(SMAX 버킷 경계·concat-mmq Nn 변경 등)에서는 유효 argmax가 복수이므로
  flip이 허용됨 — 이것은 버그가 아니라 명시된 불변식.
- 모델별 게이트의 fragile 사례: 26B-A4B는 HF 레퍼런스 자신도 특정 위치에서 argmax가 flip하는 경우가 있어
  "완전 일치"가 아니라 "HF와 동일한 fragile 지점에서 동일하게 흔들림"까지가 게이트 기준.
- 검증 문화: 정확성 먼저(오차 0 기준)·단계별 격리 검증(norm→proj→rope→attn→...)·시간만으로 성능 판단 금지
  (ncu 지표 동반)·추측 디버깅 금지(골든 없이 끝값만 보는 디버깅 금지).
- 불변식 목록 공개화: backend가 모든 device op 소유(코어는 모델 무지) / 슬롯 격리(동시 N == 단독 bit-identical) /
  scheme 명시 의무 / config 계약(`#define` 금지).

**근거 문서**:
- `CLAUDE.md` §검증 원칙 (5개 조항 — 정확성 먼저·단계별 격리·시간만 판단 금지·추측 디버깅 금지·막히면 보고).
- `docs/m6-backend-boundary.md` §경계 규약 > 불변식 — "슬롯 격리: 온라인 배칭이 각 요청 계산 안 바꿈 → 동시 N ==
  단독 bit-identical(게이트 기준)" / "scheme 명시 의무: bit-identity는 스킴 내 개념... 스킴 경계 전환은 유효
  argmax 복수라 flip 허용(요청-경계만 ISO)".
- `docs/recipe-schema.md` §claims — "scheme 필수: bit-identity는 스킴 내 개념(유효 argmax 복수) — 스킴 없으면
  주장 무효(로드 거부)".
- 26B fragile 사례: **근거 미확인** — MEMORY.md에 "26b가 HF 자신도 argmax flip하는 fragile 사례"라는 서술이
  있으나 이번 세션에서 원본 게이트 문서(`docs/m4-moe-26b-a4b-gate.md` 등)를 직접 읽어 인용문을 확인하지 못함.
  Phase 1 본문 작성 전 해당 문서를 직접 읽고 정확한 인용으로 대체할 것.

---

## 절 3 — 확장 가이드 (공개판) (Extension Guide)

**요지**: `docs/m6-backend-boundary.md`의 "새 family 추가 가이드"(STEP B)를 그대로 공개 문서로 승격.
새로운 사용자가 자기 모델을 이 엔진에 추가하려 할 때 "공짜로 받는 것 vs 스스로 써야 하는 것(+대략 규모)"과
지켜야 할 경계 규약(불변식)을 안내.

다룰 내용:
- 공짜로 받는 것: 커널 라이브러리 전체 · 서빙 코어(trace `serving_core.h` + 실시간 `serving_online.h`) ·
  스케줄러(continuous/static·admission 가드) · 튜너 · 레시피 export/verify(모델-무지) · API front
  (`serve_api.py --model`) · bench 하네스.
- 써야 하는 것(paradigm당 규모, 2호 Gemma4 실측 기반): graph(~130~400줄) · backend 어댑터(~25~33줄) ·
  loader(weights_<family>) · manifest 파서(config.json→ModelConfig, 새 필드는 공용 config.h에 추가) ·
  daemon 진입(~20줄 setup). **신규 커널은 정말 새 연산일 때만**(예: GDN) — 4모델 실측에서 커널 0 신규.
- 경계 규약 원문: `ServingBackend`(trace-구동, 7메서드: has_spec/per_seq_bytes/alloc_state/free_state/
  reset_slot/prefill_slot/decode_step) · `OnlineBackend`(실시간, 8메서드: vocab/max_len/alloc_state/
  free_state/reset_slot/prefill_ids/decode_step/logits_to_host).
- 개입 곡선 4점 표(1호 SmolLM3 전부 신규 / 2호 Gemma4-31B new-family / 3호 35B-A3B MoE 0개입 / 4호 26B-A4B
  new-paradigm) — "새 모델을 붙이면 내 경우는 이 표의 어디쯤인가"를 가늠하게 해주는 참조표로 그대로 인용.

**근거 문서**:
- `docs/m6-backend-boundary.md` 전체(§STEP 0 경계 인벤토리, §STEP B 새 family 추가 가이드, §경계 규약,
  §개입 곡선 4점 표) — 이 절은 사실상 이 문서를 공개 포맷으로 옮기는 것이므로 원문 그대로 링크/인용.
- `CLAUDE.md` §아키텍처 원칙 — "새 모델 추가 = 새 커널이 아니라 새 graph" 원 문장과의 정합 확인용 교차 참조.

---

## 절 4 — 2F 알려진 갭 (정직 로드맵) (Known Gaps / Honest Roadmap)

**요지**: 완결된 것처럼 포장하지 않고, 현재 미해결/미측정/차단된 항목을 숨김없이 로드맵으로 공개.

다룰 내용:
1. **spec 경계 미정**: 27b MTP(draft→verify→promote)는 plain의 `ServingBackend.decode_step`(+1 훅)에
   구조적으로 안 들어감 — 별도 경로(`bench_serve_spec`·`serve_daemon_qwen_spec_online`) 유지. serving_backend.h에
   [spec 확장점] 설계 주석만 존재, 구현은 M7+ 과제.
2. **26B 서빙 파레토 미측정**: 26B-A4B는 roofline(active-bytes 기반)만 산출됐고 실제 bench_serve regime 스윕에
   의한 서빙 파레토는 아직 없음(35B-A3B MoE와 유사 형태 예상 — B8 knee 가설, 미검증).
3. **Gemma4 sliding KV 링버퍼 TODO**: 현재 sliding window KV는 캡(1024) 방식이고, 긴 context(>1024)에서
   물리 슬롯 재작성이 필요한 진짜 ring buffer 구현·불변식 증명·>1024 golden 검증은 아직 미완(두 Gemma family
   전부 해당, "delicate 집중 후속" 항목으로 별도 표시됨).
4. **verify 자동화 갭**: (a) export/warm-start가 spec 전용 필드(`K`,`W`,`bmax`)를 하드코딩해 plain/MoE 레시피는
   hand-authored로만 존재하고 warm-start는 `KeyError 'K'`로 CRASH — scheme-class 추상화가 선결 과제.
   (b) verify가 `serving_mode`를 기록하지 않아 조건 불일치(35B 사례: repro 168.8% = local 106 vs claim 62.9,
   실측상 c-ch vs c1tok 모드 차이)를 못 걸러내고 "PASS"로 오리포트한 전례 있음 — repro 상한 체크 및
   `serving_mode` 명시 필드가 v1 후보.
5. **FP4 toolkit-blocked**: GB10 하드웨어는 FP4 텐서코어(MMA)를 지원하나 CUDA 13.0 ptxas(PTX 9.0)가 mxf4를
   emit하지 못함(≥9.2/9.3 필요, 13.1도 실패 보고 사례 있음) — toolkit 업그레이드 전까지 착수 불가, 가장 큰
   미사용 lever로 명시.
6. **대역폭 돌파 지형도 완결의 정직한 함의**: 7개 시드(unified 계층/weight 상주·dense sparsity·KV 압축·
   spec 극한·prefetch·L2 persist·q2/q3 양자화) 전부 measured로 소거됨 — "이 장비·이 모델군·tractable 기법으로
   2×급 대역폭 돌파는 measured 불가"가 정직한 결론. 남은 것은 엔진 밖(SOTA 2-bit 연구·draft 모델 업그레이드) 또는
   toolkit-gated(FP4)뿐. 이것을 "실패"가 아니라 "탐사 완결(환상 제거)"로 정직하게 프레이밍.

**근거 문서**:
- `docs/establishment-notes.md` §2C(spec 경계 미정 — "M7+ 과제" 명시), §2F(자동화 갭 4항목 원문).
- `docs/recipe-schema.md` §v1 후보(export/warm-start 하드코딩 상세, 35B 168.8% repro 실측 사례,
  serving_mode 필드 제안) — 항목 4의 1차 출처.
- `docs/m6-backend-boundary.md` §spec 경계(미정) — "plain backend 훅은 안정 인터페이스, spec은 별도 추상화
  미정(M7+ 과제)" 원문 — 항목 1의 1차 출처.
- `docs/m3-multi-objective-pareto.md` — "26B 서빙 파레토는 미측정(roofline만)" · "다음: 26B 서빙 bench로
  파레토 완성 · 장문 트랙(sliding 링버퍼)" 원문 — 항목 2·3의 1차 출처. (sliding 링버퍼 TODO는 MEMORY.md의
  "STEP D(sliding 링버퍼 두 Gemma·긴 context·불변 증명·물리슬롯 재작성·>1024 golden)" 서술과 일치하나, 이
  MEMORY.md 문구 자체는 세션 자동 요약이라 원본 설계 문서 대조는 Phase 1에서 재확인 권장.)
- `docs/fp4-mma-toolkit-blocked.md`(파일명 추정 — MEMORY.md summary `fp4-mma-toolkit-blocked.md` 참조.
  **근거 미확인**: 이번 세션에서 해당 파일을 직접 열람하지 않았으므로 toolkit 버전 조건(13.0/13.1/13.2+)
  정확한 문구는 Phase 1에서 원문 확인 후 인용할 것). 잠정 근거는 MEMORY.md 자동 요약.
- `docs/c-bandwidth-seeds-verdict.md` 전체 — 7시드 판정표(§STEP C)·"2번째 기둥 정직 결론"(§STEP C 하단)·
  q3 STEP D/E("bytes 절약 ≠ 속도" 결론) — 항목 6의 1차 출처, 원문 확인 완료.

---

## 공개 순서 제안 (아웃라인 수준 — Phase 1에서 확정)

1. 절 1 (설계 철학) — 왜 이렇게 짰는가, 독자가 아키텍처를 신뢰하는 진입점.
2. 절 3 (확장 가이드) — "나도 내 모델을 붙일 수 있나?"에 바로 답하는 실용 문서, 설계 철학 직후 배치.
3. 절 2 (정확성 계약) — 확장 시도 전에 "무엇을 보장하고 무엇을 보장 안 하는지" 알아야 하므로 확장 가이드와 인접.
4. 절 4 (알려진 갭) — 마지막, 정직한 로드맵으로 마무리(과장 방지 장치).

이 순서는 잠정안이며 Phase 1에서 대상 독자(오픈소스 기여자 vs 사용자)에 따라 재배치 가능.
