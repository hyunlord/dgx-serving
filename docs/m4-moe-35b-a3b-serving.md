# M4 MoE 35B-A3B — 서빙 경로 + 성능 (active-bytes 실측 vs 165.3)

> Phase 0(129b715) 후속. STEP A(서빙 batch 경로 활성화) + STEP B(성능 실측·달성률·그룹핑 판단).
> 정직 스코프: 정확성 먼저, 성능은 달성률 그대로(낮아도), 그룹핑은 판단만(구현 다음).

## STEP A — 서빙 batch 경로 (정확성) — 완료

- **batch MoE 분기 신규**(16852b0): qwen35_batch.cu MLP 스텝에 `if(num_experts>0) moe_mlp_block(c,L,n2,dbuf,B)`
  (prefill/single decode와 동형, moe_mlp_block=naive S=B×top_k 루프 공유). 기존 batch는 dense FFN만이라 MoE 서빙 불가였음.
- **게이트(test_batch_decode 35b, 시퀀스별 다른 토큰 = expert-index batch)**:
  - uq=1(q4): batch==solo **MEAN-rel 1.15e-3 + argmax ALL → PASS**(MoE 배선 bit-clean).
  - uq=2(W4A8): argmax ALL(라우팅 정확), MEAN-rel 2.64e-2 = **backbone W4A8 cross-graph**(sb vs gemv, honest-perf ⑥) —
    q4로 낮추면 23× 감소로 W4A8 backbone 원인·MoE 무결 확증. router는 fp32(선택 결정적).
- ③=0(config 분기, moe_mlp_block 공유 커널). bench_serve 회귀 컴파일 OK(27b는 num_experts=0 → dense else).

## STEP B — 성능 실측

### single decode vs roofline 165.3

| | t/s | 달성률 | TPOT | prefill |
|---|---|---|---|---|
| 35B-A3B MoE single(W4A8 backbone/q4 experts) | **40.0** | **24.2%** | 25.0ms | 131 t/s |

**대조: dense 27B 62.5%·Gemma4 53.3% ≫ MoE 24.2%** — 브리프의 "MoE 달성률 구조적 위험" 확증. 165.3은 활성-bytes
이상 상한(이동만 계상)이고, 실측 40은 그 24%.

### 왜 낮은가 — active-bytes 분해 + 구조(측정 40 + 구조 귀속)

활성 3.11B 구성: **GDN 40.5% · expert FFN 32.4% · lm_head 16.4% · shared 4.0% · router 0.7%**.
- **토큰당 ~640 expert-gemv launch**(L40 × top8 × 2[gate_up·down]) — 작은 matmul(H2048×512/2048→1024) 각각 별도 커널.
- 24% 달성률의 지배 원인(구조): **(a) launch 오버헤드**(토큰당 640+ 커널) **(b) tiny expert matmul 대역폭 저활용**
  (512폭 gemv는 이상 대역폭 못 씀) **(c) GDN 순차**(40% 바이트, 27B서도 병목). TPOT 25ms = 이상 6ms의 4.1×.
- ※ 정확한 커널별 시간 귀속은 ncu/nsys 필요(구조상 launch+tiny-matmul 유력) — 다음 세션 정밀화.

### batch expert scatter + 그룹핑 payoff (판단 — 구조 근거)

- **batch t/s 실측은 이번 미완(도구 갭, 정직 기록)**: bench.cu batch 블록이 dense-only+focus-gate, bench_serve는 MoE
  chunked-prefill 미구현(prefill_batch_slot에 MoE 분기 없음). batch 측정엔 MoE 서빙 backend(serving_core+MoE) 또는
  bench 확장 필요 — 별도(강행 안 함).
- **구조적 결론(naive 루프)**: expert matmul은 **토큰당 독립 B×top8 gemv, amortization 0** — batch가 expert weight
  적재를 나누지 못함(dense batch가 큰 FFN을 amortize하는 것과 반대). 즉 naive batch는 aggregate가 sublinear·expert부는
  B-불변 비효율. "곱은 커널이 만든다"의 MoE판.
- **그룹핑 lever 판단**: 토큰→expert 그룹핑(grouped GEMM, expert별 M=routed 토큰 1 GEMM)이 MoE 서빙의 핵심 lever.
  payoff = expert부(32% 바이트) × (현 tiny-matmul/launch 비효율 회수) × (batch tokens/expert amortization).
  - B=1(single): 그룹핑 무효(토큰 1개) → single lever는 **launch 감축**(CUDA graph 캡처 / fused expert 커널).
  - batch 서빙: top8×B가 256 expert에 분산 — B16이면 128 슬롯/256 expert(평균 0.5, 충돌 적음)라 초기 amortization 제한,
    B↑·집중 라우팅서 커짐. **권고: batch t/s 먼저 실측(도구 보강) → tokens/expert 통계로 그룹핑 payoff 확정 후 착수.**
    (draft GEMV 교훈: 배칭≠곱 — 측정 없이 그룹핑 강행 금지.)

## 결론 / 다음

- **MoE single 24.2%** = dense보다 낮은 구조적 달성률(launch+tiny-matmul+GDN). 정직 좌표.
- 서빙 batch 경로 정확성 완료(batch==solo argmax ALL). ③=0(4모델·서빙코어·MoE 전부).
- **다음**: (1) MoE 서빙 backend(serving_core+MoE) 또는 bench 확장으로 batch t/s+scatter 실측, (2) launch 감축
  (CUDA graph) single lever, (3) 측정 후 그룹핑 착수/보류. 이후 튜닝→MoE 첫 레시피, 26B-A4B(renorm 규약).

---

## batch 실측 + go/no-go (측정 세션 — 구현 전 판단)

### batch B-scaling (raw, naive 루프, W4A8, warm)

| B | TPOT(ms) | aggr t/s | /stream t/s |
|---|---|---|---|
| 1 | 23.1 | 43.3 | 43.3 |
| 4 | 48.1 | 83.2 | 20.8 |
| 8 | 81.8 | 97.8 | 12.2 |
| 16 | 150.1 | 106.6 | 6.7 |

**amortization 0 확증**: aggr B1→B16 = 43→107 (**2.46×**, dense Gemma4 sb 8.2× 대비 급락). TPOT가 B에 ~선형 증가
(expert 부분이 토큰당 독립 gemv라 안 amortize). aggr 천장 ~107@B16. (도구: bench.cu MoE batch 블록 추가.)

### scatter 통계 (M_e = expert당 routed 토큰, 실측 router logits)

골든 5토큰 실측: top-8이 거의 고유(36/40, max M_e=2, 평균 1.11 — 편중 약함). **M_e = B·top_k/E**:
| B | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 평균 M_e | 0.12 | 0.25 | 0.50 | 1.0 | 2.0 |

서빙 B≤16 → M_e≤0.5(대부분 expert 0-1토큰). 그룹핑(grouped GEMM)의 곱은 M_e≥2부터(B≥64, 실용 latency 초과).

### ★nsys 귀속 (측정 먼저 — 예상 못한 지배 비용)

- **GPU 커널: gemv_q4(expert matmul) 62.3%**(481,880 launches, avg 8µs tiny 2048×512) — expert 경로가 GPU 시간 지배
  (바이트 32%≪시간 62% = tiny matmul 저효율). backbone sb 7.3%, router 5.5%, gdn 5.1%, swiglu 4.2%.
- **★CUDA API: cudaFree 41.8%(4.26초, 53,091 calls, max 22ms)** — moe_mlp_block이 **호출마다 malloc/free**
  (11버퍼 × 30 MoE층 × 매 스텝). cudaFree=동기화 호출 → 층마다 직렬화. 다른 블록(qwen35_batch g_bxq)은 persistent scratch.

### go/no-go 권고 (measured)

1. **★#1 GO(cheap·high-impact): moe_mlp_block persistent scratch** — cudaFree 41.8% API 오버헤드 제거(호출당 malloc/free
   → static grow-on-demand, 코드베이스 기존 패턴). 저위험·큰 이득 예상(층별 동기화 직렬화 해소). nsys가 드러낸 예상 밖 lever.
2. **NO-GO: grouped GEMM** (서빙 B≤16) — M_e~1이라 weight amortization 0 + 활성 expert≈슬롯이라 launch 감축도 미미.
   B≥64에서만 이득(실용 서빙 latency 초과). "곱은 커널이 만든다" — MoE 그룹핑은 M_e가 곱을 만드는데 B≤16선 M_e<1.
3. **#2 GO(bigger·후순위): device-side fused MoE** — expert gemv 62% GPU 시간이 481k tiny launch. 온-디바이스 expert
   루프(라우팅 device-resident) → util↑ + launch↓ + **CUDA graph 가능**(현재 host round-trip[moe_block D2H+host loop]로 BLOCKED).
   #1 후 재측정하여 착수 판단.

### CUDA graph de-risk (STEP C)

**구조적 BLOCKED**: moe_mlp_block이 routing 인덱스를 D2H cudaMemcpy 후 **host 루프로 expert별 launch**(host-의존 launch
파라미터) → graph capture 불가(30 MoE층마다 host round-trip). graph 활성화 = device-side dispatch 필요(=#2와 동일 인프라).
동적 top-k 자체는 문제 아님(indirect dispatch로 device서 처리 가능) — 진짜 장벽은 현 구현의 host 왕복.

**종합**: MoE 24% 갭의 실 원인 = (a) moe_mlp_block malloc/free 직렬화(cudaFree 41.8%, cheap fix) (b) expert tiny-matmul
62% GPU(device fused 필요) (c) host round-trip(graph 차단). **그룹핑은 아니다**(M_e<1). 다음 세션: #1 구현→재측정→#2 판단.

---

## #1 persistent scratch 구현 + 재측정 (462f798) — measured가 판정

### 게이트 (수학 불변 measured, assumed 아님)
moe_mlp_block 11버퍼 호출당 malloc/free → static grow-on-demand(g_bxq 패턴). 게이트 전부 PASS·이전과 동일:
router idx 불일치=0·weight 1.49e-8, expert rel 6.99e-7, test_35b argmax ALL·last 11751, batch bit-id 1.148e-3
+argmax ALL. **버퍼 재사용 오염 없음**(다른 시퀀스 연속 실행 확인 — 각 버퍼는 매 호출 커널 출력으로 완전 덮어씀, 누산기 out은 memset).

### ★재측정 판정: e2e 이득 0 (환산 금지 실증)

| | single | B4 | B8 | B16 |
|---|---|---|---|---|
| 이전(malloc/free) | 42.8 | 83.2 | 97.8 | 106.6 |
| #1(persistent) | 42.8 | 83.2 | 97.0 | 105.7 |

**malloc 제거가 처리량 불변.** nsys: **cudaFree 41.8%→3.0%**(53k→1.6k calls, 제거 작동)인데 e2e 안 변함 →
cudaFree는 **GPU 실행(62% busy)과 오버랩된 비동기 API 시간, critical path 아니었음**. 브리프 경고("API%는 GPU와
겹칠 수 있음 — 환산 금지")가 정확히 실현. **교훈: nsys API-time % ≠ critical-path time — 재측정만이 판정.**

### 재귀속 (persistent scratch 후)
- GPU: **gemv_q4 62.4% 불변**(481k tiny 2048×512, avg 8µs=BW 23% 저활용) — 실 병목, malloc 무관.
- API: cudaMemcpy 60.2%(routing D2H **여전** — #1은 malloc만 제거, host 왕복 안 건드림), cudaLaunchKernel 22.9%(1.08M launch).

### #2 device-fused MoE go/no-go (새 지도 근거)

- **#1 가치(정직)**: perf lever 아님(e2e 0). 무회귀 + malloc/free 제거(CUDA graph capture 선결 — graph는
  cudaMalloc/Free 금지) + host API 부하↓. 유지(graph/#2 인프라 선결).
- **★#2 GO — 유일한 실 lever**: expert gemv 62.4% GPU(tiny matmul BW 23%) + routing D2H(cudaMemcpy 60% API) 둘 다
  device-side fused MoE로만 해결. 온-디바이스 expert 루프(routing device-resident → **D2H 제거**, 481k tiny launch
  → 층당 소수, occupancy↑로 BW util↑) + **CUDA graph 가능화**(host 왕복 제거).
  payoff 상한(정직): expert gemv가 BW-bound 도달(8µs→~1.8µs, 4.4×)이면 62%→~17% → single TPOT 23ms→~13ms →
  **~43→75 t/s(24%→~43%)**. 단 M_e~1이라 fusion은 per-expert 병렬 아닌 **다수 expert 동시 스케줄(occupancy)+launch 제거**로
  util 개선 — 실 이득은 커널 구현이 판정(상한이지 예측 아님). 상당한 커널 작업(다음 세션).

### 서빙 트랙 순서 권고 (measured 근거)
1. **#2 device-fused MoE**(최대 lever — GPU 62% + D2H 둘 다, graph 가능화). 상당 작업이나 24% 달성률의 유일 돌파구.
2. #2 후 재측정 → 서빙 backend(chunked prefill MoE — 현 bench_serve 미지원) → 튜닝(plan strategy MoE expert 축) → MoE 첫 레시피.
3. 그룹핑은 계속 NO-GO(M_e<1). 26B-A4B는 별도(router renorm 규약 확인).

---

## #2 device-fused MoE 커널 (831b4e6) — 실 WIN measured

### 설계
kernels/moe_fused.cuh: **grouped gemv 4커널**. warp per (slot, output), slot=b·K+k, expert e=didx[slot] **device 간접**
(★routing D2H 제거), weight=base+e·stride. per-output dot은 gemv_q4와 동일((q-8)·scale·lane-stride32·warp_reduce)
+ **k-order 고정 reduce**(naive s,k 루프 순서와 동일) → **naive와 bit-identical**·결정적(atomicAdd 회피).
launch 토큰당 ~640→층당 4(gu/swiglu/down/reduce), host loop 제거. moe_block MOE_FUSED 스위치(default 1, 0=naive 폴백).
모델독립(③=0): n_experts/K/H/MI 인자, Qwen 특수 없음(26B-A4B renorm은 router_topk config 분기 예정).

### 게이트 (bit-identical 확증)
- test_35b fused MEAN-rel **6.64e-2 = naive와 IDENTICAL**·argmax ALL·last 11751 → fused==naive bit-identical.
- test_batch_decode fused: batch==solo argmax ALL(재사용 오염 없음). default(env 없음)=fused 골든 PASS.

### 측정 (back-to-back warm, fused vs naive)
| | naive | fused | 개선 |
|---|---|---|---|
| single B=1 | 42.8 | **51.8** | +21% (달성률 24.2%→**29.8%** of 165.3) |
| batch B4 aggr | 82.9 | 126.8 | +53% |
| batch B8 aggr | 97.1 | 168.9 | +74% |
| batch B16 aggr | 105.3 | **195.3** | **+85%** |
| prefill(S8) | 94.7 | **183.1** | +93% |

★single은 예측 상한 75 미달(51.8) — "환산 금지" 실현(8-expert 병렬 한계, single은 fusion 이득 제한). **batch/prefill서
이득 훨씬 큼**(launch/D2H 제거가 B 비례 — B16 +85%). 전 축 실 WIN + bit-identical.

### nsys 재귀속 (fused)
- **cudaLaunchKernel 1.08M→161k(6.7×↓)** — grouped가 launch 붕괴. **cudaMemcpy 60.2%→6.6%**(routing D2H 제거).
- GPU: moe_fused_gu 33.9%+down 16.5%=50.4%(naive gemv_q4 62.3%↓, 같은 MAC 적은 큰 커널=occupancy↑).

### ★CUDA graph 장벽 해소 (기록 — capture는 다음 세션)
fused MoE 블록은 **완전 device-side**: routing D2H 제거(#2) + malloc/free 제거(#1) → host-의존 제어흐름 0.
→ **MoE decode 스텝 graph-capturable**(이전 BLOCK 원인 둘 다 해소). capture 시도·검증은 다음 세션(GDN 등 나머지 경로 포함 e2e).

**종합**: MoE 24% 갭의 실 lever = device-fused(measured: single 24→30%, batch B16 +85%, bit-id). #1(scratch)은 오버랩이라
e2e 0이었으나 graph 선결로 유효. 다음: graph capture 시도 → 서빙 backend(chunked prefill MoE) → 튜닝 → MoE 레시피.

---

## CUDA graph capture — stop-and-report (대수술, 브리프 명시 조건)

STEP A(graph) 실현가능성 실사 결과 **대수술 판정 → stop-report**(브리프: "위치 간접화가 대수술이면 stop, graph 없이 서빙 진행도 유효").

**장벽(구체)**: decode 스텝이 위치를 **host per-b 루프 + host 포인터 오프셋**으로 커널 launch에 박음:
- KV append: `for(b) launch_f32_to_bf16(kn, st.Kc[l] + b·kvstride + POS(b)·kvdim)` — POS(b) host 오프셋(qwen35_batch.cu:159).
- flash_attn: `for(b) launch_flash_attn(..., POS(b))` — attendlen host 스칼라(qwen35_batch.cu:165).
- rope: `d_cos + base·rot` — host 오프셋(base=st.n_past).

CUDA graph는 이 위치들을 **capture 시점에 고정** → 재생마다 같은 위치에 append/attend = decode 오답. 수정 = KV-append +
flash_attn을 **device-position 커널**로 재작성(host per-b 루프 제거, POS/attendlen을 device에서 읽기) = **검증된 27b/Gemma4
decode 경로 대수술**(회귀 위험). fused MoE(#2)가 이미 지배 launch(62%)를 붕괴했으므로, single launch-overhead lever를 위해
검증 경로를 대수술하는 건 정당화 안 됨. **graph는 보류**(decode 위치 인프라 device화가 선결 — 별도 대형 작업).

## 서빙 backend — MoE가 서빙 스택에 (코드 변경 0)

**발견**: 서빙 경로가 이미 MoE 처리 — prefill_batch_slot이 **qwen35_decode에 위임**(qwen35_batch.cu:71), qwen35_decode는
MoE 분기(fused default) 보유. decode=qwen35_decode_batch도 MoE 분기(line 187). QwenBackend(bench_serve)는 load_weights_qwen
(35b q4 ✓)·per_seq_bytes(GDN ✓)·양 경로 MoE ✓ = **qwen 계열 모델-중립이라 35b MoE 서빙에 코드 변경 불요**.

**서빙 스모크 PASS(bench_serve 35b, fused default, W4A8 backbone, Bmax8 nreq24 λ4)**:
- 완료: 24 요청 완주(crash/OOM 없음), c-ch λ4 = 56.2 tok/s·avgB 5.87·maxQ 9.
- 결정성: run1 56.2(e2e50 6.784) = run2 56.2(6.793), seed 고정 재현(wall 노이즈 <0.15%).
- 무누수: admission 예산 per-seq 71.1MB×Bmax8=0.57GB OK, 완주 = persistent scratch(#1) 서빙 안전.
※ 서빙 56.2(avgB 5.87<8, 큐 미포화+chunked prefill 비용) < raw batch B8 169 — 성능 주장은 튜닝 세션(스모크만).

## 종합 / 다음
- graph: 보류(decode 위치 device화 대수술 선결 — 별도). fused MoE(#2)가 이미 launch 6.7×↓로 주 이득 확보.
- 서빙: MoE가 서빙 스택에 진입(코드 0, fused). 결정성·완료·무누수 PASS. **튜닝→MoE 첫 레시피 unblock**.
- 다음: 튜닝(Bmax·regime — MoE 공간, plan strategy MoE expert 축 이미 존재) → MoE 첫 레시피(recipes/qwen35-35b-a3b-gb10.yaml) → 26B-A4B(renorm 규약).
