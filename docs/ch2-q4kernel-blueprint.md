# 챕터2 설계안 — q4 gemv 커널 최적화 (STEP 0, 게이트)

> 3a 결론: q4에서 llama.cpp에 1.5× 뒤짐, 격차는 전적으로 q4 gemv 커널. 이 문서 합의 후 STEP 1.
> 원칙: **정확성 불변(golden_q4 오차0) + in-engine 판정 + ncu로 규명 + 하나씩.**

## 1. 현 q4 gemv 진단 (ncu/nsys 실측, 추측 아님)

### 1a. 어디서 시간 새나 — per-kernel breakdown (q4 decode, N=96)
GPU 총 1132ms 중:
- **gemv_q4_kernel = 50.8%** ← 타깃 확정 (GPU 시간 절반).
- gemv_f32_mo 35.6%(이 run의 fp32 MTP draft), gdn_step 7.1%, rmsnorm/attn/기타 ~6%.

gemv_q4 내부(grid별 합): **작은 per-layer gemv(out 1024/3584 등, grid 128/448/640/768/2240)가
lm_head(grid 31040, 67ms decode)보다 합이 훨씬 큼.** → 비효율의 무게중심은 lm_head 아니라 **작은 gemv 다수**.

### 1b. 왜 느린가 — ncu 지표 (gemv_q4 전 shape 공통)
| 지표 | 값 | 해석 |
|---|---|---|
| **sectors/req (global ld)** | **2** | byte 로드라 64B fetch 중 ~16B만 사용 ≈ **25% sector 효율** |
| **long_scoreboard stall** | 10~20 inst | **메모리 latency-bound** (지배 stall) |
| memory throughput | 작은 15~27% / lm_head 60% | 작은 gemv가 대역폭 못 씀 |
| warps_active(occ) | 32~100% | occupancy는 주범 아님(혼재) |

**핵심 원인 = byte 단위 로드(uint8).** 32 lane이 1 byte씩 → 16 distinct byte만 커버, sector 낭비 +
load 명령 多 → latency 노출. **unpack은 이미 fused(inline shift)이라 주범 아님**(3a 가설 정정).

## 2. llama.cpp K-quant 기법 (정독, 복사 아님)

`vecdotq.cuh` vec_dot_q4_0_q8_1 (l.115-135, 720-736):
1. **weight를 int(32bit)로 로드** (`get_int_b2`, l.18) — 4 byte=**8 nibble/load**.
2. **SIMD unpack** — `(v>>0)&0x0F0F0F0F` / `(v>>4)&0x0F0F0F0F` (l.122-123): 4바이트 8 nibble을 2 op로.
3. **DP4A int8 dot** (`ggml_cuda_dp4a`, l.126-127): int8 4-MAC/instruction. 단 **activation을 int8(q8_1) 양자화** 필요.

## 3. 우리에게 적용 가능/불가 (정확성 제약 필터)

| 기법 | 적용? | 이유 |
|---|---|---|
| **32bit/128bit weight 로드 + inline unpack** | ✅ STEP 1 | 수치 동일(같은 fp32 MAC, 로드만 wide) → **golden_q4 오차0 보존**. sectors/req 2→4, load 8~32×↓ |
| **SIMD nibble unpack** (`&0x0F0F0F0F`) | ✅ STEP 1 | 동일 값, 마스크만. 수치 불변 |
| **DP4A int8 dot** | ❌ 제외 | activation int8 양자화 = **수치 변함 → golden_q4 오차0 위반**. work order "알고리즘 변경 금지"에도 위배 |

→ DP4A(연산 밀도)는 못 가져오지만, **GB10은 273 GB/s BW-bound** 머신 — binding 제약은 메모리.
메모리측(wide load)이 핵심이고 그게 정확성 보존하며 가능. 연산 밀도는 부차.

## 4. STEP 1 설계 — 벡터화 weight 로드 (수치 불변)

현 커널: 1 warp/output, lane이 i=lane,lane+32,… byte 1개씩.
**변경**: lane이 **uint4(128bit)=16 byte=32 nibble** 또는 uint32=8 nibble를 연속 로드, inline unpack, fp32 X 곱·누적.
- lane L이 byte [L·16 .. L·16+15] 로드 → warp가 512 nibble 커버(uint4), stride 512. IN=1024면 2 pass.
- coalesced: lane L이 연속 16 byte → warp 512 byte = 16 sector 완전 사용 (sectors/req → 4).
- scale: lane의 32 연속 nibble = 정확히 QBLK=32 한 블록 → scale 1개 broadcast(`sca[L]`). 깔끔.
- **레이아웃 변경 불필요**(현 byte-sequential packing 그대로 uint4 캐스팅). → **golden_q4 재생성 불요.**
- 수치: 같은 (q-8)·scale·x 누적 순서만 재배열(부동소수 합 순서 차이 가능 → err 0 아닌 ~1e-6, golden 1e-2 기준 통과 예상. 측정 확인).

## 5. 검증 계획 (각 STEP)
1. **정확성**: `make test`(test_qwen35 q4-prefill==golden_q4, test_decode q4, test_spec) 전부 PASS.
   q4 gemv 출력 golden_q4 대조 오차 <1e-2. spec on==off mismatch 0.
2. **in-engine t/s**: test_spec q4 plain & (c)both-q4 spec 전후 다회. microbench 금지.
3. **ncu 규명**: sectors/req 2→4? long_scoreboard stall↓? memory throughput↑? occupancy 안 깨졌나(2d MO=4 함정 감시).
4. **llama.cpp 재비교**: 같은 조건 q4 plain/spec 격차 얼마 좁혔나(176/191 → ? vs 266/314).

## 6. 작업 순서 (하나씩)
- **STEP 1**: uint4 벡터화 로드 + inline unpack (레이아웃 불변). 정확성+t/s+ncu. ← 가장 큰 레버.
- **STEP 2**: (STEP1 결과 보고) 추가 — 필요시 레이아웃 재packing(unpack 더 친화), 또는 multiple-output/warp(X 재사용,
  단 2d MO=4 occupancy 붕괴 함정 ncu 감시), 또는 on-GPU argmax(host DMA 제거).
- 각 STEP 게이트 보고.

## 7. 하지 않을 것 (범위)
- activation 양자화/DP4A(정확성 위반), q4 비트수·spec 알고리즘 변경, 배칭, vLLM, MoE.
- microbench만 보고 채택(in-engine 필수). occupancy 붕괴 강행(ncu로 잡고 후퇴).

## 8. 사용자 결정 (합의됨)
1. STEP 1 = uint4 벡터 로드 + inline unpack, fp32 MAC(DP4A 제외) — 동의.
2. 판정 = 수치 <1e-4 AND 토큰열 글자단위 일치(argmax 안 뒤집힘) + spec on==off. 토큰 바뀌면 멈추고 보고 — 동의.
3. 레이아웃 불변 시작, 재packing은 STEP 2 — 동의.

---

## STEP 1 결과 — uint4 벡터 로드 시도 → **기각(in-engine 회귀)** [완료]

### 정확성: 통과
uint4 contiguous-per-lane 로드 구현(레이아웃 불변). golden_q4 오차 logits 6.68e-6(<1e-4),
argmax ALL match, **토큰열 STEP0 baseline과 글자단위 동일**, spec mismatch 0, make test 12/12 PASS.
→ 수치는 완벽히 보존. 그러나:

### in-engine t/s: **회귀 −40%** (176→104 plain, 192→91 spec). 다회 안정적 회귀.

### ncu 원인 규명 (추측 아님)
| 지표 | baseline | uint4 시도 | 판정 |
|---|---|---|---|
| **sectors/req (global ld)** | 2 | **30.7** | X가 uncoalesce (정반대 악화) |
| L1 throughput | ~55% | 50~99%(saturate) | L1이 병목으로 |
| memory throughput | 작은 15~27% | 50~73% | weight는 나아졌으나 X가 망침 |

**원인**: contiguous-per-lane 파티션이 weight는 coalesce했지만 **X[s]를 uncoalesce**.
lane L이 x[L·32 … L·32+31]을 읽어 warp 내 lane간 X stride = 32 float = 128 byte → 요청당 ~30 sector,
L1 saturate. baseline은 lane이 x[lane],x[lane+32]…로 X 완전 coalesce(L1 89% hit)였음.

### 구조적 결론 (중요)
**weight wide-load(contiguous-per-lane) ⊥ X-coalescing(strided-per-lane)** — 양립 불가.
둘 다 만족하려면 **X를 shared로 1회 cooperative coalesced 로드** 후 per-lane이 shared에서 읽어야 함.
(블록 8 warp가 같은 s면 X 동일 → 블록당 1회 로드로 8 warp 재사용 = 큰 이득 가능.)
단 이는 **2d의 X-shared occupancy 함정 영역** + s-boundary(prefill S>1 시 블록이 s 경계 straddle) 처리 필요 = 큰 변경.

### 추가 전략 인식 (err-0 제약의 천장)
llama.cpp q4 우위의 일부는 **int8(q8_1) activation 경로**: 그들 X는 1 byte(우리 fp32 4byte의 1/4 트래픽)+
coalesced + DP4A. 우리는 err-0 위해 이 경로 제외. ⇒ **fp32 X + err-0 유지 시 X 트래픽이 구조적으로 4× 큼** —
메모리측 천장이 llama.cpp보다 낮음. err-0-safe 레버는 X-shared(재사용으로 X 트래픽↓)뿐.

### 다음 선택지 (게이트 — 사용자 방향 필요)
- **(A) X-shared**: X를 블록당 1회 coalesced 로드 → 8 warp 재사용 + contiguous-per-lane wide weight load.
  측정된 원인(X uncoalesce) 직격. 리스크: 2d occupancy 함정 + s-boundary 정확성. 큰 변경, ncu로 occupancy 감시하며.
- **(B) baseline 유지 + 전략 전환**: err-0 천장 인정. 진짜 레버(int8 activation=llama.cpp 경로)는
  별도 "q8 activation + 새 golden" 트랙(err-0을 q8 golden으로 재정의)으로 분리. 챕터2는 여기서 정리.
- **(C) 소폭 tweak**: lane이 1 byte로 2 nibble 처리(2-lane 로드 중복 제거). X 약간 declustering, 이득 불확실·작음.
