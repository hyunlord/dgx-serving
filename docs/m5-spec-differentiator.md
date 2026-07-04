# M5 — spec 곱 차별점 시연 (정립 노트 2A · 공정 벤치 아님)

> 3-way 26b 결과(dgx 0.28~0.56× 패 — 커널 성숙도, `docs/m5-3way-results.md`) 후속. 같은 게임(같은 모델·양자화·raw
> 속도)에서 llama 추월은 비현실(성숙도+대역폭 천장). 게임 바꾸기 경로 중 **(B) spec 곱 = 지금 있는 무기**를 정직 시연.
> ★프레이밍: **공정 벤치 아님 = 차별점 시연**. 아래 전제 정정을 반드시 먼저 읽을 것.

## 0. ★전제 정정 (overclaim 방지 — 이게 이 문서의 1순위)

작업 착수 전제였던 **"llama엔 spec 서빙 없음"은 사실이 아니다.** `llama-server --help` 검증:
```
----- speculative params -----
--spec-draft-hf, -hfd, --hf-repo-draft <user>/<model>   (draft 모델)
--spec-draft-type-k/v, --spec-draft-threads, ...          (draft KV·스레드 등 전체 세트)
```
llama.cpp는 **draft-model 기반 speculative decoding을 완비**했다(honest ⑦의 "llama-mtp 미측정, 켜면 갭↓" caveat가 실재).
**vLLM도 마찬가지**(ngram/draft-model/EAGLE/Medusa/MTP speculative — stage2-vllm-blueprint의 0.17.0 포함).
→ **spec은 두 주요 엔진 다 갖춘 일반 기법이다.** "우리만 있는 축"은 성립하지 않는다.

→ **정확한 차별점 재정의**: "spec 자체가 우리만"이 **아니라**, 우리는 **MTP-head 내장 spec**이다:
| | 우리(MTP-head) | llama(draft-model) |
|---|---|---|
| draft 소스 | 모델에 학습된 MTP head(내장) | **별도 draft 모델 필요** |
| 추가 메모리 | 없음(head만) | draft 모델 weight/KV 추가 |
| vocab 정합 | 자동(같은 모델) | draft-target vocab 매칭 필요 |
| 정확성 | **W4A8 bit-identical**(spec==greedy, verify 보증) | draft 무관(target이 보증) |
| 추가 | tree-spec + ragged 연속 배칭 곱 | (llama도 tree 일부 지원) |

**즉 우위는 "축이 우리만"이 아니라 "내장·무-draft-모델·bit-id"라는 구조적 편의**다. 과장 금지.

## 1. 왜 시연이고 공정 벤치가 아닌가

- **매칭 gguf 부재**: 우리 엔진 타깃(Qwen3.5-27B/35B)의 gguf가 없음(로컬 gguf=Qwen3-14B·gemma·Qwen2.5뿐). →
  같은 모델 우리-spec vs llama-spec 직접 대결 **불가**(공정 3-way 불성립).
- 그래서 이 문서는 **우리 엔진 내에서 spec의 곱을 측정**하고(측정 정직), llama와는 **구조 비교**(위 표)로 다룬다.
- "우리가 llama보다 빠르다"(X) → "**곱이 걸리는 워크로드에서, 내장 spec으로 우리가 우리 자신의 plain보다 빠르다**"(O).
  llama-plain(13.83) 대비 우리 spec(21.3)이 앞서는 건 사실이나 **spec-vs-nonspec = 불공정**(llama+draft 켜면 좁혀짐).

## 2. 측정 좌표 (기존 measured — ch9-engine-landscape·ch10-batch-spec-combine)

| 축 | 우리 | llama | 정직 주석 |
|---|---|---|---|
| 단일 decode non-spec | 11.9 t/s | **13.83** | llama 1.16× 빠름(raw 커널) |
| 단일 decode **spec** | **21.3** t/s (tree+MTP q4) | 13.83 (non-spec) | ours 1.54× — **단 spec-vs-nonspec(불공정)** |
| 우리 spec vs 우리 plain(단일) | 21.3 vs 11.9 = **1.79×** | — | 우리 엔진 내 곱(단일-스트림) |
| verify 곱 mechanism | **3.83×@B8** (measured) | — | 실증. 단 draft 순차가 e2e 희석 |
| spec 서빙 throughput | ~44 t/s (draft-batch 수정 후) | — | plain ~40.6 대비 ~1.08×(고배치=곱 작음) |

★**곱의 크기는 regime 의존**: **단일-스트림(latency)서 크고(1.79×), 고배치(throughput)선 작다(~1.1×)** — 배칭이 이미
대역폭을 amortize해서 spec의 read-amortization 여지가 준다. **즉 spec의 진짜 가치 = 단일/저동시 인터랙티브 latency.**

## 3. accept 워크로드 민감도 (양쪽 다 측정 — 곱-유리/불리)

accept(사이클당 promote 토큰)의 본질 = **"MTP head가 타깃 모델 자신의 greedy argmax를 얼마나 맞추나"**
(spec=lossless, verify가 출력 보증). 따라서 accept ∝ **모델 자신 continuation의 예측가능성(낮은 엔트로피)**:
- **높은 accept**(곱-유리): 반복적·정형(코드·리스트·템플릿) — 다음 토큰 near-certain.
- **낮은 accept**(곱-불리): 고엔트로피·다양·희귀(창의적 산문·주제 도약) — 매 스텝 top margin 작음(전례: n-gram 0 accept).

### 실측 (27b spec, single-stream Bmax=1 λ=0.8, K6W4 tree, warm·seed 고정)
같은 엔진·같은 config, 프롬프트만 예측가능성별로 교체(token_ids.bin) → accept·tok/s·tpot95 대조:

| 워크로드 | 성격 | accept(plen/cycle) | tok/s | tpot95(s) |
|---|---|---|---|---|
| repetitive | 반복 문장 ×8 (near-certain 연속) | **6.99** | 15.9 | 0.066 |
| baseline | 자연어 짧은 프롬프트(15tok) | 4.55 | 10.9 | 0.104 |
| natural | 사실 산문(mitochondrion) | 3.56 | 8.9 | 0.102 |
| diverse | 고엔트로피·주제도약·희귀어 | 3.50 | 8.9 | 0.128 |

- **곱-유리 확인**: repetitive(리터럴 반복) accept 6.99 vs 산문 ~3.5-4.5 = **~1.6-2× 스윙** → tok/s 15.9 vs ~9
  (+46~79%)·tpot 0.066 vs ~0.10(−35%). 반복/정형 워크로드일수록 곱↑(단일-스트림 latency 실이득).
- **★측정이 가설을 정정(measure-first)**: 가설 "diverse(창의/고엔트로피) ≪ natural(사실산문)"은 **틀림** —
  diverse 3.50 ≈ natural 3.56. 즉 accept의 진짜 lever는 "창의성/도메인"이 아니라 **리터럴 반복/정형**이다.
  일반 산문은 다양성·희귀어와 무관하게 ~3.5-4.5로 클러스터(그리디 국소구조[문법·기능어]를 MTP가 그럭저럭 예측).
  **repetitive만 도약**(6.99). → 곱-유리 = 코드/템플릿/반복 로그 등 **반복 구조**, 막연한 "예측가능 도메인" 아님.
- **곱-불리 = 일반 산문 전반**(accept ~3.5): 곱이 작음(반복 대비 tok/s ~55%). 단, 이 실측 tok/s(8.9)는 Bmax=1 λ0.8
  **서빙**값(draft 순차 오버헤드 포함)이지 순수 decode 아님 — plain 동일-config 미측정이라 "spec<plain 여부"는
  단정 안 함(정직). 확실한 건 **accept가 워크로드로 1.6-2× 변동 → 곱 크기도 그만큼 변동**.
- ★lossless: 모든 워크로드에서 출력은 spec==greedy bit-identical(verify 보증) — accept는 속도만 바꾼다(정확성 불변).

## 4. 정직 양면 · 캐비엇

- **곱-유리(단일·반복/정형·긴 생성)**: 내장 spec이 단일 latency를 우리 plain 대비 ~1.8×, llama-plain 대비 ~1.5×
  낮춤(measured 21.3 vs 11.9/13.83) = 인터랙티브 latency 실이득. ★measured: 반복/정형(코드·템플릿, accept 6.99).
- **곱-불리(고배치 OR 일반 산문)**: 곱 작음 — 고배치는 배칭이 amortize, 일반 산문은 accept ~3.5(반복 대비 절반).
  ★measured 정정: "창의적 텍스트"가 특별히 나쁜 게 아니라(diverse≈natural) **비-반복 산문 전반**이 accept ~3.5.
- **raw 커널**: 여전히 llama 우위(별개 축, 3-way 26b 0.28~0.56×·⑦ 27b 1.16~1.42×).
- **튜닝된 spec 계열**(DFlash 65~94, ⑦): 우리보다 빠름 — "spec이면 우리가 최고"가 아님.
- **llama도 spec 켜면**(draft 모델): 우리 단일 spec 우위(1.54×)는 좁혀지거나 사라짐 — 미측정 caveat 명시.

## 5. 결론

우리 MTP-head spec은 **단일-스트림 인터랙티브 latency**를, **예측가능 워크로드**에서, **별도 draft 모델 없이·bit-id
보존**으로 낮추는 **실재하는 구조적 차별점**이다. 그러나:
- "llama에 없는 유일한 축"이 **아니다**(llama=draft-model spec 완비).
- raw 속도 우위가 **아니다**(그건 llama).
- 전 워크로드 우위가 **아니다**(고배치·고엔트로피선 소멸).

정직 요약: **게임을 바꾸면(단일 latency×예측가능×내장 spec) 우리 강점이 드러나되, 그것도 "우리만의 축"이 아니라
"편의(내장·무-draft)+정확성(bit-id)"의 차별화**다. 다음 게임체인저 후보 = (C) 대역폭 돌파/q2 극한양자화(bytes↓).
