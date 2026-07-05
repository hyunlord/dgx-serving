# M6 — backend family 경계 확정 (정립 노트 트리거 ② · M7 선결)

> M3(다목적 파레토) 완결 후, M7(정립·공개)의 선결 = **backend 경계 안정**. ★M6의 정직한 형태 =
> **플러그인 시스템 신설이 아니라**(YAGNI — 실존 family는 qwen/gemma 둘뿐·가상의 3호 추상화는 투기),
> 개입 곡선 4점이 이미 알려준 경계를 **코드 실사로 확정·문서화**하고 **measured 중복만** 정리하는 것.
> 산출물 = 코드가 아니라 **검증된 경계 문서**(이번 코드 변경 0 = 무회귀 자동).

## STEP 0 — 경계 인벤토리 (현 코드 실사)

### 공유 (모델-중립 — 새 family가 "공짜로 받는 것")

| 자산 | 파일 | 성격 |
|---|---|---|
| **커널 라이브러리** | `kernels/*.cuh`(24개) | ③=0 — 전 모델 공유(rmsnorm·rope·attn·flash·gemv q4/w4a8·gemm mmq/mma/tiled·gdn·moe_fused·router_topk·swiglu·argmax…) |
| **서빙 코어(trace-구동)** | `runtime/serving_core.h`(119줄) | Poisson trace·admit/slot 재사용·continuous/static 스케줄·warm-up·백분위·admission 가드·갭 계측. **모델 분기 0** |
| **온라인 코어(실시간)** | `runtime/serving_online.h`(122줄) | 실시간 stdin reader 스레드+큐·실토큰 prefill·greedy 피드백(argmax)·태그드 스트림 demux. **모델 분기 0** |
| **backend 인터페이스** | `runtime/serving_backend.h`·`serving_online.h`(struct) | 코어↔모델 접촉면(아래 §경계 규약) |
| **spec 코어** | `tools/bench_serve_spec.cu`·`serve_daemon_qwen_spec_online.cu` | draft→verify→promote 사이클(plain과 스텝 구조 상이 — 별도 경로, §spec) |
| **튜너·레시피·API front·bench** | `tools/autotune/*`·`recipes/*`·`serve_api.py`·`bench_serve*` | config-주도라 profile 교체로 재사용(plan/verify 모델-무지 실증) |

### family-특수 (새 family가 "써야 하는 것")

| 자산 | qwen 계열 | gemma 계열 | family 고유가 무엇인가 |
|---|---|---|---|
| **graph(decode/batch)** | `models/qwen35*.cu`·`qwen35_decode.cu`(289)·`qwen35_batch.cu`(391) | `models/gemma4*.cu`·`gemma4_decode.cu`(132)·`gemma4_batch.cu`(288)·`gemma4_moe.cu`(304) | 공용 커널 호출 **순서·파라미터**(GDN 혼합 vs sliding/full·GeGLU·sandwich-norm·dual-rope·softcap) |
| **backend 어댑터** | `QwenBackend`(31)·`QwenOnlineBackend`(25) | `Gemma4Backend`(33)·`Gemma4OnlineBackend`(25) | primitive 배선 + reset_slot(qwen=GDN clear/gemma=no-op) |
| **state 버퍼** | `decode_state.h`(BatchDecodeState — GDN state 항) | `decode_state_gemma4.h`(G4BatchState — dual-rope·sliding KV) | per-seq bytes(admission 예산) 형상 |
| **loader** | `weights_qwen.cu`(prefix·텐서 패턴) | `weights_gemma4.cu`(tied·per_expert_scale) | HF 텐서명 prefix·양자화 텐서 레이아웃 |
| **manifest 파서** | `manifest_qwen.h`(28) | `manifest_gemma4.h`(31) | config.json→ModelConfig 필드 매핑(공용 `config.h`가 필드 보유) |
| **daemon 진입** | `serve_daemon_qwen_online.cu` main | `serve_daemon_gemma4_online.cu` main | setup(rope 레이아웃·device alloc) 후 `run_online_daemon(be,Bmax)` |

### 개입 곡선 4점과 대조 (곡선 = 코드 실사와 정합)

| 호 | 모델 | 개입 | 코드 실사 근거 |
|---|---|---|---|
| 1호 | SmolLM3 | **전부 신규** | 커널+graph+loader 첫 구축(공유 자산이 아직 없음) |
| 2호 | Gemma4-31B | **new-family** | 신규 graph(gemma4.cu) + backend 어댑터 + loader/manifest — **커널 0 신규**(③=0), config 3종 추가(sliding·GeGLU·dual-rope) |
| 3호 | 35B-A3B MoE | **0 개입** | `QwenBackend` 그대로 재사용(qwen-family·MoE는 moe_fused 커널+config `num_experts`로 흡수) — daemon 코드 변경 0 |
| 4호 | 26B-A4B MoE | **new-paradigm** | 신규 graph(dense∥MoE 병렬 = 제3 구조) + config 3종(per_expert_scale·router-norm) — **커널 0 신규**(moe_fused 재사용), Gemma4OnlineBackend에 `moe` 자동분기 1줄 |

**곡선의 함의**: 개입 비용은 **paradigm 거리**에 비례. same-family/same-paradigm(3호) = 0(조립조차 불필요). new-family(2호)/new-paradigm(4호) = 새 graph + config, **커널은 언제나 0 신규**(정말 새 연산이 있을 때만 커널 1~2개 — 아직 발생 안 함). = 헌법("커널 독립·graph 배선·config화")이 4모델 실측으로 버팀.

## STEP A — measured 중복 정리 (판정: 리팩터 불필요 — 경계가 이미 얇음)

실제 복붙 중복을 diff로 실사한 결과:
- **backend 어댑터 = 25~33줄의 얇은 어댑터**. `QwenOnlineBackend`와 `Gemma4OnlineBackend`는 둘 다 25줄로 구조가
  평행하나, **substantive한 모든 줄이 family-특수**(rope 레이아웃 qwen 1테이블[golden ext/long] vs gemma 2테이블[dual
  생성]·state 타입 BatchDecodeState vs G4BatchState·decode fn qwen35_decode_batch vs gemma4_decode_batch·reset_slot
  GDN clear vs no-op). = **추출할 공통 "로직"이 없음** — 각 줄이 다른 primitive를 부른다. 이것이 헌법이 의도한
  "얇은 graph 배선"의 정확한 실현.
- **진짜 중복 = 프로젝트 전역 boilerplate뿐**: `CK` 매크로 **96개 파일**·`rf()` golden float reader **26개 파일**.
  이는 backend 경계 로직이 아니라 관용적 인라인 CUDA boilerplate(1줄 매크로·작은 I/O 헬퍼). 공통 헤더로 추출하면
  **blast-radius 96파일 + 전체 재빌드**인데 payoff는 cosmetic → **YAGNI + 무회귀 위험 > 이득으로 정리 안 함**.

**★판정: STEP A = no-op(정리 불필요)**. "경계가 이미 얇으면 그대로가 정답" — 이것이 오히려 가장 강한 M6 결과다
(measured 중복이 없다 = 아키텍처가 이미 정립됨). 코드 변경 0 → 4모델 게이트 무회귀 자동 성립.

## STEP B — ★새 family 추가 가이드 (M6 산출물 = 공개 확장 문서 원본)

새 모델 family를 붙일 때 **써야 하는 것**과 **공짜로 받는 것**(4모델 실측 기반 — 2호 gemma4가 실례):

### 공짜로 받는 것 (건드리지 않음)
- 커널 라이브러리 전체(`kernels/`) · 서빙 코어(trace `serving_core.h` + 실시간 `serving_online.h`) ·
  스케줄러(continuous/static·연속 배칭·admission 가드) · 튜너(`autotune.py` profile 교체) · 레시피
  export/verify(plan/verify는 모델-무지) · API front(`serve_api.py` `--model` 추가) · bench 하네스.

### 써야 하는 것 (paradigm당 규모, 2호 gemma4 실측)
1. **graph** (`models/<family>_decode.cu`·`_batch.cu`): 공용 커널을 이 모델 순서·파라미터로 호출. **~130~400줄/graph**
   (paradigm당 — same-paradigm이면 미러+graft, new-paradigm이면 새로). ★신규 커널은 정말 새 연산일 때만(예: GDN).
2. **backend 어댑터** (struct : `ServingBackend`/`OnlineBackend`): **~25~33줄**. 아래 §경계 규약의 메서드 구현 +
   family 특수 처리(GDN이면 reset_slot clear·MoE면 num_experts 자동분기).
3. **loader** (`weights_<family>.cu`/.h): HF 텐서 prefix·양자화 레이아웃 매핑. extract 파이프(`tools/extract_weights*.py`)는
   HF config에서 형상 읽어 범용(모델 하드코딩 X).
4. **manifest 파서** (`manifest_<family>.h`): config.json→`ModelConfig`. **새 필드는 공용 `config.h`에 추가**(모델 상수
   `#define` 금지 — 런타임 config). 2호가 추가한 예: sliding_window·geglu·dual head_dim·softcap·layer_scalar.
5. **daemon 진입** (`serve_daemon_<family>_online.cu` main): rope 레이아웃·device alloc setup 후 `run_online_daemon(be,Bmax)`
   1콜. **~20줄 setup**. + `serve_api.py --model` choices에 추가.

### 규모 요약 (new-family 1개 ≈)
graph(paradigm당 130~400) + 어댑터(~60=bench+online) + loader(~200) + manifest(~30) + daemon setup(~40) + config 필드
몇 개. **커널 0 신규**(같은 하드웨어 연산). = 2호 gemma4가 이 규모로 실측됨(무리한 통합 없이).

## 경계 규약 (명문화 — 공개 시 "확장 문서"의 계약)

### ServingBackend (trace-구동 bench·튜닝용, `serving_backend.h`)
`has_spec()`(spec 사이클 유무·plain=false) · `per_seq_bytes(maxlen)`(admission 예산 — GDN 항 vs KV-only) ·
`alloc_state/free_state`(opaque void*·코어 타입 무지) · `reset_slot(state,b)`(재사용 잔재 clear·b=-1=전체) ·
`prefill_slot(state,b,P)`(chunked prefill) · `decode_step(state,h_toks,h_pos,B)`(plain=+1토큰 전진).

### OnlineBackend (실시간 API용, `serving_online.h`)
`vocab()` · `max_len()` · `alloc_state(Bmax)` · `free_state` · `reset_slot`(기본 no-op — GDN family만 override) ·
`prefill_ids(state,b,h_ids,P)`(실토큰 + 마지막 logit) · `decode_step(state,h_toks,h_pos,Bmax)`(ragged 슬롯별 위치) ·
`logits_to_host(h_dst,rows)`(D2H — 루프는 순수 host argmax).

### 불변식 (모든 backend가 지켜야)
- **backend가 모든 device op 소유**(H2D/D2H·rope·weight·state) → 코어는 host 배열만(모델 무지 = ③ 회피).
- **슬롯 격리**: 온라인 배칭이 각 요청 계산 안 바꿈 → 동시 N == 단독 **bit-identical**(게이트 기준).
- **scheme 명시 의무**: bit-identity는 스킴 내 개념(배치 형상·양자화 dispatch가 스킴). 게이트는 **scheme-internal
  bit-id**(SMAX 버킷 내·concat-mmq Nn 고정 등) — 스킴 경계 전환은 유효 argmax 복수라 flip 허용(요청-경계만 ISO).
- **config 계약**: 모델 상수(H·L·FFN·n_heads·num_experts…)는 런타임 `ModelConfig`. `#define` 박기 금지
  (head_dim·QBLK=32 등 거의 불변인 것만 컴파일 타임 허용).

## spec 경계 (미정 — 정직)
spec(27b MTP)은 draft→verify→promote로 **스텝 구조가 plain과 상이** → 현 `ServingBackend.decode_step`(plain=+1) 훅에
안 들어감(bench_serve_spec·serve_daemon_qwen_spec_online 별도 경로). serving_backend.h 주석의 [spec 확장점]:
"decode_step이 슬롯당 가변 accept를 out-배열로 반환하고 코어 slot-advance가 그걸 읽으면 spec-core도 같은 코어에
플러그인 가능"(이번 범위 밖·설계만). = **plain backend 훅은 안정 인터페이스, spec은 별도 추상화 미정**(M7+ 과제).

## STEP C — 무회귀 + 정립 노트
- **무회귀**: STEP A가 no-op(코드 변경 0)이라 4모델 게이트 자동 무회귀(정리가 없으니 계산 불변). 문서만 산출.
- **정립 노트 2C 갱신**: backend 경계 확정(트리거 ② 충족) — 아래 establishment-notes.md 반영.
