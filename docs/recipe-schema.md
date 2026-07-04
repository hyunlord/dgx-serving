# 레시피 스키마 v0 (`dgx-recipe/v0`) — 공유 가능한 최적화 서빙 아티팩트

> 레시피 = 튜닝 산출물(regime→config **정책**)의 배포판. 받은 쪽이 파일 하나로 같은 서빙 세팅을 재현하고,
> **claims를 자기 장비에서 검증**한다. config-주도 헌법의 배당금: 엔진이 이미 config로 움직이므로
> 레시피는 "검증 가능한 주장이 붙은 config 묶음"이다.

## 구조 (YAML, 5층 + 백엔드 분리)

```yaml
meta:            # schema/created/tool/engine_commit — 어느 코드로 만들었나
target:
  device:        # 지문: name, gpu_probe(nvidia-smi), bandwidth_gbps, mem_total_gb, unified_memory
                 # toolkit_required(cp_async/mma_int8/fp4_mma — 미충족 시 로드 거부)
                 # tolerance(bandwidth_pct/mem_total_pct — 오차 허용, 초과=경고+재튜닝 안내)
  model:         # id/params_b/gdn/tied_lm_head/moe/quant
  user_constraints:  # 튜너 입력이자 명시: quant_bits, tpot95_cap_s 등
settings:
  policy:        # type: static_regime_lookup — regime(when: λ, short_ratio)→config(bmax/K/W)+backend_env
                 # (요청-경계 핫스위치는 미구현 — 사이클 전환은 스킴 경계로 bit-ISO 미보장, Phase 2b 실측)
  backend:       # dgx-serving 고유(engine_bin, fixed_env, variant_axes) — 코어 스키마는 백엔드-중립
claims:          # regime별 실측: measured.screen(nreq12)/final(nreq60) + 조건(nreq/seed) 명시
                 # scheme 필수: bit-identity는 스킴 내 개념(유효 argmax 복수) — 스킴 없으면 주장 무효(로드 거부)
provenance:      # source_artifact+sha256, trial DB 참조, 검증 프로토콜 통과 기록(full ISO 등)
```

## 로드 검증 순서 (`recipe.py verify` — 맹목 적용 금지)

1. **장비 지문 대조**: toolkit 필수 caps 미충족 → **거부**(hard fail). 이름/대역폭/메모리 오차 초과 →
   **경고 + "레시피를 prior로 로컬 재튜닝" 안내**(불일치=실패가 아니라 transfer의 warm-start — v1에서
   autotune enqueue 훅).
2. **로컬 메모리 가드 재계산**(안전 불변식): 레시피의 예산 필드를 신뢰하지 않고 각 regime config를
   이 머신의 admission 공식(MemAvailable×margin)으로 재계산. 초과 config 거부, 전부 초과면 REJECT.
   GB10 unified oversubscribe = 시스템 전체 다운 전례가 이 원칙의 근거.
3. **미니 자체검증(in-engine)**: 대표 regime(최고 throughput + 최저 tpot)을 claims의 **screen 조건
   (같은 nreq/seed/λ/mix)으로 재실측** → 재현율 %를 그대로 리포트(같은 조건끼리만 비교 — 정직).
   `--full`은 final 조건(nreq60)으로 재현.
4. **ISO 스팟**: 승자 config를 해당 스킴으로 12요청 bit-identity 확인.
5. 종합: 재현율 ≥90% + ISO PASS + 지문 경고 없음 → PASS, 아니면 PARTIAL(사유 명시).

## 실행 (`recipe.py run`)

`--lam/--shortp`로 가장 가까운 regime을 정적 룩업 → 실행 직전에도 로컬 메모리 재계산(이중 가드) →
엔진 argv/env 구성 후 exec. 정책 전환은 프로세스 단위(요청-경계 핫스위치는 후속).

## 공유 워크플로

```
[튜닝한 쪽]  autotune --phase2b → regime-table.json → recipe.py export → recipe.yaml 공유
[받은 쪽]    recipe.py verify recipe.yaml   # 지문→가드→재현율 리포트
             ├─ PASS → recipe.py run recipe.yaml --lam 4 --shortp 0.3
             └─ 지문 경고 → 레시피를 prior로 로컬 재튜닝(v1 warm-start) → 자기 레시피 재공유
```

## v0 한계 / v1 후보

- ~~warm-start 훅은 안내 문구 수준~~ → **M2에서 실체화**: `autotune.py --warmstart recipe.yaml`이 regime
  승자 config를 Optuna `enqueue_trial`로 주입(공간 밖 config는 스킵 보고). 효과는 controlled A/B로 정량
  (docs/autotune-harness.md M2 STEP C).
- ~~장비 지문이 프로파일 파일 기반~~ → **M2에서 해소**: `probe.py`가 STREAM/deviceProp/meminfo/toolkit 컴파일
  시험으로 자동 생성, GB10 수동본 대조 PASS. `verify --device <auto.json>`로 그대로 사용 가능.
- 정책이 정적 룩업(요청-경계 핫스위치·온라인 컨트롤러는 스킴 경계 해결 후).
- 서명(무결성)은 provenance sha256 수준(암호학적 서명은 배포 채널 정할 때).

## v1 후보 — 3모델 일괄 스냅샷에서 실사용으로 드러난 한계 (기록만, 이번 구현 X)

> 27b(spec)·35b-a3b(plain MoE)·gemma4 31b(plain dense) 3모델을 한 배치로 담으며 v0의 **spec-편향**이 노출됨.
> 위 §구조/§로드검증 서술은 사실상 27b(export 산출) 형태 — plain/MoE 변형은 hand-author로 메꿔진 상태.
> M5(공개 API)는 아래를 v1 스키마로 정리해야 한다(구현은 별도 트랙).

- **★export가 spec-하드코딩**(`recipe.py cmd_export`): `config={bmax,K,W}`·`claims.accept`·`scheme="…mtp_q4+concat-mmq+draft_…"`·
  `backend.engine_bin=bench_serve_spec`·`variant_axes={DRAFT_MW}`·`provenance{}` 만 생성. → **plain/MoE 레시피(35b·gemma4)는
  cmd_export로 못 만들어 hand-authored**(그 증거: 둘은 `verification{}` 블록 있고 `provenance` 없음, backend 하위구조가 서로 다름).
  v1: scheme-class(spec/plain/plain-moe)별 export 분기 필요.
- **backend 하위구조가 레시피마다 ad-hoc**: 27b=`variant_axes`+`fixed_env(MTP_Q4…)` / 35b=`scheme_detail{moe,quant,attn}`+`MOE_FUSED` /
  gemma4=`quant_path{prefill,decode}`. 표준 없음(각자 발명). v1: backend를 scheme-class로 타입화하거나 공통 `scheme_detail` 스키마 규정.
- **config union에 판별자 없음**: spec `{bmax,K,W}` vs plain `{bmax}`. 레시피 자신은 타입 미선언 —
  `verify`는 **외부 model profile의 `mtp`**로 plain/spec 분기(`plain = not mp.get("mtp")`, recipe.py:233). 레시피가 자기서술 못 함.
  v1: 레시피-레벨 `serving_class` 판별자(verify가 --model 없이도 분기 가능하게).
- **claims.measured 형태 차이**: spec은 `accept`(spec 전용) 포함, plain은 없음. v0 doc는 `screen/final`만 서술 —
  실제로는 plain에서 screen==final(같은 nreq)인 경우도. v1: scheme-class별 optional claim 필드 명시.
- **`verification` vs `provenance` 이원화**: export는 `provenance{source_artifact,sha256,verification문자열}`, hand-authored plain은
  `verification{scope,golden,engine_bin,caveats}`. 목적 겹침. v1: 통합(정합·bit-id 근거+출처 한 블록).
- **model 필드 v0 doc 불완전**: MoE는 `num_experts/top_k/active_params_b` 필요(35b 보유)·`mtp` 필드 필요(분기 축) —
  doc §구조의 model 목록(id/params_b/gdn/tied/moe/quant)은 이들 누락. v1: MoE/mtp 필드 문서화·요구.
- **★파이프-벤치 stdout 컬럼 결합(취약)**: `run_bench_plain`이 bench stdout를 컬럼 위치로 파싱 →
  이번 서빙 변경(bench_serve_gemma4에 c1tok/c-ch **mode 컬럼** 추가)이 gemma4 파서를 깨뜨림(이번 세션 수정: 신포맷+구포맷 fallback).
  v1: bench가 machine-readable(JSON) 출력 or 레시피가 `serving_mode` 기록(claim이 어느 모드 측정인지). 지금 gemma4 claim은 c1tok-era(모드 미기록).
- **★warm-start 훅도 spec-하드코딩**(`autotune.py recipe_warmstart_configs:349`가 `c["K"],c["W"]` 가정 / `enqueue_warmstart:361`이
  `space["K"/"W"]` 가정): **plain 레시피(35b/gemma4)를 prior로 로드 시 `KeyError: 'K'` CRASH**(실측 확인 — 27b spec만 OK).
  → cross-model warm-start "3점"(27b↔gemma4·gemma4↔35b)이 현 코드로 실행 불가. **"축 SET transfer" 가설은 config-space 구조로
  확정**(spec {bmax,K,W} ⊃ plain {bmax}: 이질=bmax만 겹침·spec축 미전이 / 같은 plain paradigm=bmax SET 전이·값 fresh)이나
  코드가 plain을 표현 못 해 실증 실행 불가. v1: paradigm-aware warm-start(config 축 SET을 recipe scheme-class에서 읽어 유연 매칭).

## 파이프 자동화 갭 — 3모델 일괄 스냅샷 리포트 (개입 곡선 연장)

| 단계 | 27b(spec) | 35b·gemma4(plain/MoE) |
|---|---|---|
| plan(strategy·roofline) | ✅ 자동(profile 분기 모델-무지 — 실측 확인) | ✅ 자동(spec축 정직 축소·expert축 추가) |
| tune(autotune regime) | ✅ 자동(spec 축 스윕) | ◐ 자동이나 축=bmax 위주(spec축 제거) |
| **export**(→recipe.yaml) | ✅ 자동(`cmd_export`) | ✗ **수동**(spec-하드코딩 → hand-authored) |
| **warm-start**(prior 로드) | ✅ 자동 | ✗ **CRASH**(spec-하드코딩) |
| verify(지문·메모리·재현율·bit-id) | ✅ 자동(spec ISO) | ✅ 자동(plain-tolerant·offline golden) |

★결론: 파이프는 **spec(27b) end-to-end 자동**이나 **plain/MoE는 export·warm-start가 spec-하드코딩이라 수동 개입 필요**
(plan/verify는 모델-무지 성공). 개입 곡선의 다음 갭 = **export·warm-start의 scheme-class 추상화**(M5 공개 API의 선결).
