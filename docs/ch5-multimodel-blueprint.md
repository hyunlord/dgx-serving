# 챕터5 설계안 — 다중모델 서빙 엔진화 + 큰 모델 확장 STEP 0 (게이트)

> 0.8B 전용 연구코드 → 다중모델 서빙 엔진. 첫 확장 = 같은 Qwen3.5 패밀리 큰 dense 모델.
> 이 문서(진단 + 모델/메모리/구조 설계) 합의 후 STEP 1.

## 1. 코드 구조 진단 — 경계는 이미 깨끗 (좋은 소식)
- **커널**: `kernels/*.cuh`에 모델 #define/매직넘버 **0** (QBLK=32만, 헌법 허용). 전부 런타임 인자(IN/OUT/S/dims).
- **graph**: `models/*.cu`에 하드코딩 0.8B 값 **0** — 전부 `ModelConfig` 필드(H/FFN/L/NQ/NKV/HD/GDN dims/layer_types) 주도.
- **config**: `runtime/config.h`가 완전 런타임 ModelConfig. 헌법 원칙 3(#define 금지, config 주도) **이미 준수.**
- ⇒ **축 2(코드베이스화)는 우려보다 작음.** "커널/graph 경계 정리"는 거의 안 해도 됨. 남은 구조 작업:
  - **모델 레지스트리 부재**: tests가 `qwen35_decode`/`load_config_qwen` 등을 **직접 호출(32곳)**. 모델 디스패치 없음.
  - **공통 진입점 부재**: load/prefill/decode/batch가 qwen35 전용 시그니처. 다른 모델이 같은 API 못 씀.
  → 할 일 = (a) 모델 레지스트리(config→graph 함수 선택) (b) 공통 Engine 인터페이스.

## 2. 모델 가용성 — 27B 미캐시, 4B는 준비됨 (블로커/선택)
| 모델 | arch(model_type) | dims | weight 다운로드 | fp32 골든 | dense? |
|---|---|---|---|---|---|
| 0.8B(기준) | qwen3_5 | H1024 L24 FFN3584 NQ8 | ✓ | 3GB | dense+GDN |
| **Qwen3.5-4B** | **qwen3_5(동일)** | H2560 L32 FFN9216 NQ16 | **✓ 8.8GB** | **16GB(안전)** | dense+GDN |
| Qwen3.5-27B | (의뢰서 명명) | — | **미존재(캐시X)** | — | — |
| Qwen3.6-27B | qwen3_5(동일!) | H5120 L64 FFN17408 NQ24 | **✗ 22MB 메타만**(54GB 받아야) | 108GB(OOM위험) | dense+GDN |

- **Qwen3.5-4B = 0.8B와 동일 아키텍처(qwen3_5, GDN+full hybrid), 다운로드됨, 골든 안전.** 첫 확장 최적.
- 의뢰서의 "Qwen3.5-27B"는 **캐시에 없음**. 같은급은 **Qwen3.6-27B**(이름 3.6이나 arch는 동일 qwen3_5, dense, GDN) —
  단 **weight 미다운로드(54GB pull 필요)** + **fp32 골든 108GB(GB10 128GB서 활성화 포함 OOM 위험)**.

## 3. 메모리 설계 (큰 모델)
- 우리 엔진(q4): 4B 2GB, 27B 13.5GB — GB10 128GB서 여유.
- **HF 골든(검증 기준)**: 4B fp32 16GB(안전). **27B fp32 108GB = OOM 위험** → fp16 골든(54GB) 또는 레이어별 골든
  (레이어 출력만 덤프·비교, 전체 동시 미적재). 의뢰서가 예상한 "무거우면 레이어별/샘플".

## 4. 설계 — 모델 레지스트리 + 공통 인터페이스
- **레지스트리**: `model_type`(config) → graph 함수 테이블. 새 모델 = config 로더 + graph 엔트리(커널 불변).
  0.8B/4B/27B는 **같은 qwen3_5 graph 공유**(같은 패밀리 → 같은 graph, config 값만 다름). 검증으로 확인.
- **공통 Engine 인터페이스**: `engine_load(model_dir) → Engine`, `engine_decode_batch(Engine, tokens, B, ...) → logits`.
  qwen35_* 를 이 인터페이스 뒤로. 네 모델이 같은 API → 3-way 비교 하네스도 공용.
- **weight 추출**: `tools/extract_weights`를 config 형상 주도로(이미 범용?) — 27B q4 추출 스케일 확인.

## 5. 작업 순서 (정정 제안)
- **STEP 1**: 모델 레지스트리 + 공통 Engine 인터페이스 도입(리팩토링). **0.8B 회귀 0**(16/16 PASS, 오차0 불변).
- **STEP 2**: **Qwen3.5-4B**를 config+graph로 추가(같은 graph 공유 가정 검증). fp32 == HF 골든 오차0 + next-token.
  (4B 골든 16GB 안전 → 0.8B 검증방식 그대로.)
- **STEP 3**: 4B 전체 경로(decode/batch/q4/W4A8) + 처리량. 보편화(0.8B 경로 재사용) 검증.
- **STEP 4**: 27B — weight 다운로드(54GB) + fp16/레이어별 골든 + 큰 모델 메모리. (4B서 구조 증명 후 메모리 스케일.)
- **STEP 5**: 보편화 회고 → 다음(35B MoE).

## STEP 1 결과 — 모델 레지스트리 + 공통 Engine 인터페이스 [완료]

`runtime/engine.h`(additive 레이어, 기존 코드 무손상):
- **레지스트리**: `model_kind_from_manifest` — manifest "model_type" → `ModelKind`(qwen3_5 등록; 미등록은 MK_UNKNOWN=보편화 신호).
- **공통 인터페이스**: `engine_load(dir)`(config+kind+weights) / `engine_alloc_batch` / `engine_decode_batch`(kind로 graph 디스패치).
- 새 모델 = (같은 패밀리면) config만, (새 패밀리면) ModelKind+graph 엔트리. 커널/graph 불변.

### 검증
- **투명성**: `engine_decode_batch` == 직접 `qwen35_decode_batch` **max err 0.000e+00**(bit-identical 래퍼).
- **회귀 0**: 17/17 PASS(기존 16 + test_engine). 0.8B 동작 불변(additive라 기존 경로 미수정).
- 27B는 `engine_load(27B_dir)` 동일 경로(같은 qwen3_5 kind, config만 27B 값) — STEP 2서 검증.

## STEP 2 결과 — 27B 추출/골든 OK, but GDN 보편화 갭 발견 (멈추고 보고)

### 2.1 추출 (config 주도, 툴만 수정)
- `extract_weights_qwen35.py` config 주도 확인. 추가: `--dtype bf16`(108GB→54GB OOM회피) + `--q4-only`(big fp32 생략).
- **버그 수정**: 27B는 `tie_word_embeddings=False`(untied) — lm_head가 top-level `lm_head.weight`(prefix 밖)라 추출기가
  스킵했음. untied lm_head 처리 추가(prefix 무관 save_q4). 0.8B는 tied라 안 드러난 갭.
- 27B q4 추출 완료: 64층(16 full/48 GDN), q4 400 tensors, rel-err 0.0996(0.8B와 동일), ~19GB. 커널/graph 불변.

### 2.2 골든 (fp32-CPU가 fit!)
- make_reference fp32-CPU 로드: **108GB가 CPU RAM 119GB에 fit**(headroom 10GB) → **fp32 골든 가능**(bf16 폴백 불필요,
  err-0 품질). golden_fp32 + golden_q4 생성, argmax [2614,314,279,369,11751], fp32==q4 last-tok MATCH(11751).
- 메모리 물리학 재정정: 골든은 CPU(순차)라 fp32 가능. 우리 엔진 fp32(GPU 108GB)만 불가 → q4 엔진으로 검증.

### 2.3 정확성 검증 — **GDN 보편화 갭 발견 (FAIL, 예상된 신호)**
`test_27b`: engine_load(27B, q4) + qwen35_forward(use_q4=1) vs golden_q4. 결과: **L1(첫 GDN층)부터 hidden 발산
(worst 11.73), 최종 NaN**(logits/prenorm "err 0"은 mae가 fmax로 NaN 무시한 artifact, last-tok eng=0=NaN argmax).

**정체 = GDN GQA 그룹핑 미처리:**
| | 0.8B | 27B |
|---|---|---|
| gdn_num_k_heads (q/k) | 16 | **16** |
| gdn_num_v_heads (v) | 16 | **48** |
- 0.8B는 k=v=16(동일) → graph가 `launch_gdn_step(NH=nvh, ...)`로 q/k/v 모두 nvh heads 가정해도 우연히 맞음.
- 27B는 **k/q=16, v=48**(v-head 3개당 k-head 1개 공유 = GQA). graph의 qg2/kg2(kdim=16헤드) vs vg2(vdim=48헤드) +
  gdn_step 커널이 head h의 q/k/v를 같은 h로 페어링 → 27B서 v-head 17이 없는 k-head 17 참조 → 발산/NaN.
- 커널 주석도 "D<=128 가정(head_k=head_v)" — **k_heads==v_heads 암묵 가정이 0.8B 우연**.

### 결론 — 보편화 첫 갭 (멈추고 보고, work order대로)
**graph + GDN 커널이 num_k_heads==num_v_heads를 암묵 가정**(0.8B 우연). 27B의 GQA-스타일 GDN(16k/48v)이 이를 깸 →
**config로 안 빠지고 graph/kernel에 샌 것 = 보편화 실패 신호.** 헌법("커널 수정 필요하면 멈추고 보고")대로 자율 수정 안 하고 보고.
- 필요한 변경: gdn_step 커널 + graph 배선이 v-head→k-head 그룹핑(GQA in GDN) 지원. delta-rule에서 q/k는 k-head 인덱스,
  v/state는 v-head 인덱스로 분리. = 커널/graph 경계 재설계(STEP 3 후보).
- 추출/골든/엔진로드/레지스트리/인터페이스는 전부 OK(커널 불변) — 갭은 GDN 헤드 그룹핑 단 하나로 좁혀짐(좋은 신호).

## 6. 사용자 결정 (게이트)
1. **첫 큰 모델 = Qwen3.5-4B(다운로드됨, 동일 arch, 골든 안전)로 시작** 동의? (의뢰서 27B 전에 — 4B로 코드베이스화+dense
   확장을 싸게 증명, 27B는 STEP 4서 다운로드+메모리 스케일.) 아니면 바로 27B(3.6) 다운로드부터?
2. **27B 모델 = Qwen3.6-27B**(캐시 메타 존재, 동일 qwen3_5 arch·dense; "Qwen3.5-27B"는 미존재) — 이걸로 동의?
   아니면 Qwen3.5-27B가 실재하면 그걸 받을지?
3. **27B 골든 전략 = fp16 또는 레이어별**(fp32 108GB OOM 회피) — 동의?
4. **레지스트리/공통 인터페이스 구조**(model_type→graph 테이블 + Engine API) — 방향 동의?
