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
