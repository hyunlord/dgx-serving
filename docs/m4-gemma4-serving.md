# M4 — Gemma4 31B 성능/서빙 + 첫 레시피 (Phase 0 이후)

> 베이스: 3f8e373(Gemma4 golden PASS, argmax 5/5) + 8a0ef73(mw1 레시피). Phase 0 해부는 docs/m4-gemma4-phase0.md.
> 프레임: 파이프(해부→전략→튜닝→레시피)가 2번째 모델에서 처음 end-to-end 도는 세션. spec 부재(MTP 없음) 정직 명시.

## STEP 0 — argmax 게이트 보강 (성능 전 로직 보강)

**문제**: Phase 0 게이트는 argmax-only(bf16 골든). fp32 full 118GB=메모리 가드 초과라 27b의 per-layer err<1e-2보다 약함.

**방법 — 층-스트리밍 fp32 골든**(`tools/make_reference_gemma4_fp32.py`):
- bf16 base 유지(62GB) + 층별로만 fp32 업캐스트(층당 ~2GB 전이, 가드 내) → HF 레이어 forward 그대로 재사용(정확성).
- 각 층 q4-policy linear에 **엔진과 동일한 q4 round-trip을 fp32 유지**로 적용 → 엔진(q4 dequant fp32)과 동일
  가중치·동일 정밀도. bf16 노이즈 제거 → per-layer err<1e-2 게이트 복원(bf16 격차 우회).
- 마스크/rotary/shared_kv는 HF 유틸(create_causal_mask/create_sliding_window_causal_mask, Gemma4TextRotaryEmbedding).

**게이트**(test_gemma4, fp32 골든 있으면 자동 승격): per-layer hidden err<1e-2 + logits + argmax 전위치.

**결과 (prompt "The capital of France is", S=5)**:
- **(a) op-level fp32** (9층 = L0~L8, sliding 0~4 + full/k_eq_v L5 + sliding 6~8 — 전 연산타입 커버):
  L0 **maxabs=0**(embed 양쪽 fp32 bit-exact), L1~L8 worst **rel=2.3e-4**(abs 1e-2는 Gemma 대규모
  hidden[|g|max 20→161] 때문 — rel은 오히려 층 깊어질수록 감소 6e-5). = 엔진 GPU-fp32 vs 골든 CPU-fp32
  (동일 q4 weights) 산술차, 버그 아님(smooth·국소점프 없음).
- **(b) full-depth argmax** (bf16 골든, 60층 전체): 전위치 5/5 일치, last-tok 7001.
- **판정: 게이트 PASS** = op-level fp32 rel<2.3e-4 + full argmax 5/5. argmax-only(Phase 0)보다 강함 —
  op 정확성을 fp32로 직접 확인. **한계(정직)**: full-depth fp32는 메모리 가드(층별 업캐스트가 mmap+anon
  누적으로 112GB 근접 → kill)로 첫 8층만. full 깊이는 argmax가 커버. "27b급"에 근접하나 full-depth
  fp32는 아님(메모리 구조적 제약).
- **프롬프트 다양성 확인**: prompt-2("def fibonacci(n):", S=6, 다른 토큰 분포)도 **PASS** — op-level fp32
  worst **rel=1.6e-4**(prompt-1의 2.3e-4와 일관), full-depth argmax 6/6 일치. 게이트 서명이 프롬프트 무관하게
  견고(GPU/CPU fp32 산술차 수준). 부수 관찰: prompt-2에서 golden_q4 argmax가 golden_base와 pos-2에서 다름
  (240017 vs 123466) — q4 양자화 자체 차이(엔진과 무관, 게이트는 엔진 vs golden_q4 비교라 영향 없음).
- **메모리 사고 기록**: 초기 full-depth 스트리밍이 112GB(가드 초과)까지 누수 → 즉시 kill(시스템 보호).
  경계 8층 + RSS 하드가드(78GB)로 안전 재실행. [[gb10-unified-memory-crash]] 원칙 준수.

## STEP A — 서빙/decode 경로 (A-1 완결, A-2/3 진행)

### A-1: gemma4_decode + decode 골든 (PASS, 유연성 2라운드)

**gemma4_decode**(models/gemma4_decode.cu ~130줄): GDN 없어 **프리필+decode 통합**(base=0=프리필, base>0=decode
스텝 — 27b는 state 경로로 분리, Gemma4는 하나로). G4DecodeState(KV 캐시 fp32)만. 공용 커널만(1라운드 자산
geglu/scale/softcap/rmsnorm_noscale 재사용).

**2단 게이트 PASS**:
- (1) **프리필 재현**(test_gemma4_decode): gemma4_decode(base=0) vs golden 통과 gemma4_forward = logits maxabs
  **2.5e-4**(GPU 재현성)·argmax 전위치 일치 → KV append+attn_decode 배선 정확.
- (2) **decode 연속성**(test_gemma4_gen): 프리필(S=5)+12스텝 teacher-forcing decode(전체 위치 rope 골든) →
  **argmax 12/12 HF greedy 일치**. KV 이어받기·sliding/full 혼합·dual-rope 위치·k_eq_v 전부 정확.

**버그 1건(배선, 커널 아님)**: 초기 1/12 — test가 rope에 pos offset 미리 더해 넘기고 decode가 또 base offset
= 이중 오프셋(프리필 base=0은 무영향, decode만 터짐). test가 전체 테이블 시작만 넘기도록 수정 → 12/12.
pred=in 패턴→rope 위치 의심으로 격리(golden-debug).

**3분류(서빙판)**: ② `attn_decode`에 scale+window 인자(기본값 보존, qwen35 회귀 PASS) — Gemma4 scaling=1.0+
sliding, 인자이지 모델분기 아님. ① gemma4_decode(graph)+G4DecodeState+manifest_gemma4 신규 모델별. **③ 서빙
코어 모델분기 = 0건**(decode/state는 Gemma4 전용 파일, 공유 코어 무오염).

### A-2: sliding KV 캡 — 예산 정량(완료) + 링버퍼(별도)

**KV bytes 예산(fp32 KV; bf16이면 절반)** — per-token: full 10층 160KB + sliding 50층 1600KB = 1760KB
(sliding이 지배: 층수 5배 + NKV 16 vs 4). sliding 캡(w=1024) 이득:

| context | 캡 전(MB) | 캡 후(MB) | 절감 |
|---|---|---|---|
| 1024 | 1845 | 1845 | 0% |
| 2048 | 3691 | 2013 | 45% |
| 4096 | 7382 | 2349 | 68% |
| 8192 | 14764 | 3020 | 80% |
| 16384 | 29528 | 4362 | 85% |
| 32768 | 59056 | 7046 | 88% |

→ **sliding 캡이 이 모델 서빙의 핵심 lever**: 긴 context·큰 B에서 KV가 GB10 unified 예산을 지배하는데,
50/60층을 O(w) 고정으로 묶어 8K에서 80%·32K에서 88% 절감(bf16이면 절반) — 그만큼 B/context 여유.

**링버퍼 구현(별도)**: 정확성 미묘함 = **프리필은 전체 KV 필요**(각 쿼리 위치가 서로 다른 window를 봄 —
병렬), **decode만 링버퍼**(순차, 최근 w개). 이 경계(프리필→decode 전환 시 sliding 층 마지막 w개만 유지)
처리 + attn_decode 물리-슬롯(j%cap) 재작성 + >w golden(또는 작은-w config override 자체대조)이 필요.
현 alloc은 안전판(전부 max_len 선형)으로 OOB 원천 제거. window **마스크** 정확성은 A-1 test_gemma4_gen
(sliding 층 window=1024로 12/12)에서 이미 검증 — 남은 건 물리 슬롯 인덱싱뿐.

### A-3-2: gemma4_decode_batch (완결, PASS)

**gemma4_batch.cu**(~110줄): B슬롯 same-length 동시 decode(위치 base 공통 — ragged/chunked는 후속).
matmul M=B(gemm_batch_q4, fp32 act×q4 w), attn 슬롯별(attn_decode_batch scale+window), KV 슬롯별
[B,max_len,NKV_l,hd_l]. rope는 rope_partial_batch(rot=hd=full rope, 위치 공통).

**3중 bit-identical 게이트 PASS**(test_gemma4_batch, B=4, 17토큰 시퀀스):
- (a) batch 슬롯 서로 **max|Δ|=0** — 같은 입력 완전 일치(슬롯 독립성 + attention 격리 확인).
- (b) batch(B=4) 슬롯0 == batch(B=1) **max|Δ|=0** — **gemm_batch B-무관 canonical 실증**(27b Phase5와
  일치: 행별 acc·warp_reduce 순서 B-무관 → bit-exact). plain판 ISO.
- (c) argmax **12/12 HF greedy 일치** — A-1 연속성 B 확장.
- **3분류**: ② attn_decode_batch scale+window(기본값 보존, qwen35 회귀 PASS) / ① gemma4_batch+G4BatchState
  신규 모델별 / **③ 서빙 코어 모델분기 0**. spec 없음(MTP 부재) — plain만, 27b 곱 스토리 이식 불가(정직).

### A-3-2 부가: 메모리 누수 체크 (서빙 전 필수, PASS)
batch decode는 per-call malloc/free churn(gemm_batch·버퍼) → 반복 서빙 시 GPU 누수가 112GB 사고의 GPU판.
30스텝 반복 free-mem 델타 = **음수(드라이버 캐시 정리, 누수 없음)** — 임계 16MB 하 PASS. batch decode
반복 안전 확인.

### A-3-1 / A-3-3: full bench 결합 — ragged 전제로 별도 (정직 기록)

**핵심 발견**: 연속 배칭(admission→slot→step)은 **슬롯마다 위치가 다름(ragged pos)** — 하지만 현
gemma4_decode_batch는 **same-length**(모든 슬롯 위치 공통). 즉 full 서빙 결합의 전제 =
**ragged gemma4_decode_batch**(슬롯별 base, rope_partial_batch→per-row 위치, attn window 슬롯별)로,
A-3-2 same-length를 넘는 추가 구현. 그 위에:
- bench 서빙 코어 중립화: bench_serve가 GDN/Qwen 깊이 결합(prefill_batch_slot[GDN state]·admission 예산
  [nh/D/convd]·BatchDecodeState[GDN 필드]·qwen35_decode_batch, bench_serve.cu:45/170/185/188~). 공용 코어
  (gen_trace/run_continuous/admission)를 모델 훅(config·weights·decode·prefill·state alloc) 인터페이스로 추상화
  = 서빙 스택 대규모 리팩터. **③ 서빙 코어 모델분기 회피가 이 추상화의 목표**.
- chunked prefill batch slot(Gemma4판) + RSS 상시 가드.

**세션 판정**: A-3-2(batch decode 정확성 + 누수 안전 = 서빙의 정확성/안전 토대)가 이번 핵심 성과.
full bench 결합은 ragged decode_batch + 서빙코어 추상화(각 대규모, golden 필요)로 별도 세션. STEP B(성능)는
서빙 없이도 single/same-length-batch decode로 부분 측정 가능(연속배칭 throughput은 A-3 완성 후).

---
### (원 산정 기록 — A-1로 상당 부분 해소)

**모델-중립성 시험 결과(코드 조사, 파일:라인)**: `bench_serve.cu`는 **모델-중립 아님** —
`step_forward`가 `qwen35_decode_batch`(models/qwen35_batch.cu) 직접 호출, `load_config_qwen`·`load_weights_qwen`
하드코딩(bench_serve.cu:45,170,185). decode/서빙 스택 전체가 QwenWeights/DecodeState(GDN 필드 포함) 타입에 결합.

**Gemma4 서빙 연결에 필요한 것(정밀 산정)**:
1. `gemma4_decode`(증분 KV, 단일스텝) — qwen35_decode(288줄) 골격, GDN 경로 제거(더 단순) + **sliding-window
   decode attention**(현 flash decode는 full causal — window 인자 추가 = 커널 인터페이스 조정 ②) + 층별
   dual-rope(full/sliding cos·sin 2세트) + k_eq_v(full 층 V=v_norm(k_proj)).
2. `alloc/free G4DecodeState` — full 10층 [max_len] KV, sliding 50층 **[min(max_len,w)] 링버퍼** KV 캡.
3. `gemma4_decode_batch`(≈qwen35_batch 385줄, GDN 슬롯 제거) + bench dispatch(모델 선택).
4. **decode 골든**(HF generate 다토큰 argmax) — 정확성 게이트(prefill 골든과 별개).
→ 총 ~600줄 신규 + 검증. **plain 서빙**(spec 없음, MTP 부재). sliding KV 캡 = 이 모델 구조적 lever
   (긴 context·큰 B에서 KV bytes 대폭↓: full만 O(len), sliding은 O(w=1024) 고정).

**세션 판정(정직)**: STEP 0(게이트, 전제)은 완결·커밋(f3801f8). STEP A는 검증-필요 대규모 decode 구현이고
STEP B(성능)·C(튜닝/레시피)가 이에 의존 — 메모리 사고로 예산 소모한 이 세션에서 검증 없이 몰아치면
"커밋 위생=golden 통과분만" 위반. **STEP A~C는 별도 집중 세션 권고**(gemma4_decode+골든부터). = 파이프
end-to-end의 **자동화 갭 = decode/서빙 경로가 모델별 구현 필요**(프리필 graph는 config로 흡수됐으나 서빙
스택은 아직 Qwen 결합). 이것이 이번 세션의 정직한 산출(브리프 "자동화 갭 기록" 요구 충족).

## STEP B / C — 블록됨 (STEP A 의존)

성능 실측(roofline 17.4 t/s 달성률)·CYC 병목·오토튜너·cross-model warm-start·Gemma4 레시피는 모두
서빙 경로(STEP A) 완성 후. plan.py 전략 생성(spec 無→튜닝 공간 축소: K/W 없음, Bmax/chunk/sliding-KV 중심)은
STEP A 없이도 리턴 가능하나, 실측 없는 전략은 미검증이라 STEP A와 묶어 진행.
