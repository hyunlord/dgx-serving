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

- warm-start 훅은 안내 문구 수준(v1: 레시피 config를 Optuna enqueue_trial로 주입).
- 장비 지문이 프로파일 파일 기반(로컬 자동 프로브는 STREAM/SM probe와 함께 v1).
- 정책이 정적 룩업(요청-경계 핫스위치·온라인 컨트롤러는 스킴 경계 해결 후).
- 서명(무결성)은 provenance sha256 수준(암호학적 서명은 배포 채널 정할 때).
