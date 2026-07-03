# M4 MoE 35B-A3B — 튜닝 + 첫 레시피 (파이프 3호 완주)

> 해부→전략→튜닝→레시피 파이프가 **3번째 모델**에서 완주. 정직 스코프: plain 서빙(spec 無), fused MoE default,
> 서빙 throughput≪raw(정직 분해).

## STEP 1 — plan 전략 (MoE plain 공간)
plan.py strategy 35b: traits=[large MoE, hybrid(GDN+full), MoE, plain]. spec_capable=false → K/W/DRAFT 제거,
축=Bmax(주)+GDN state+expert sparsity+quant-mix. 지형(Bmax headroom)과 정합.

## STEP 2 — 튜닝 (regime→config)
bench_serve 35b(fused default, W4A8), Bmax{8,16}×λ{4,8,16,32}, warm:

| Bmax | λ4 | λ8 | λ16 | λ32 | tpot95 | avgB(포화) |
|---|---|---|---|---|---|---|
| 8 | 62.7 | 62.7 | 62.8 | 62.9 | 0.113 | 6.9 |
| 16 | 62.0 | 64.3 | 64.8 | 65.0 | 0.251 | 13.1 |

- **Bmax8 최적 운영점**: Bmax16이 throughput +3%(63→65)에 **tpot95 2×(0.113→0.251)** — latency 대비 무가치.
  Bmax8은 λ4서 포화(avgB 6.9, λ4~32 62.7~62.9 평탄). 메모리 필터(GDN 포함) 4 config OK.

## STEP 3 — cross-model warm-start 2차 관찰 (가설 검증)
Gemma4(plain) → 35b MoE(plain), 축 공유 케이스:
| | Gemma4 | 35b MoE |
|---|---|---|
| 저λ | B8 | B8 |
| 고λ | **B16** | **B8** |

- **축 SET transfer O**(둘 다 plain → Bmax 주 lever·quant-mix·λ×mix regime 동일 구조), **값 transfer X**(최적 Bmax
  model-specific — MoE는 B16 latency 무가치, dense Gemma4는 B16 실이득).
- ★1차(spec→plain: 축 SET도 transfer 안 됨) vs 2차(plain→plain: 축 SET transfer, 값 안 됨) = **transfer 곡선**:
  같은 paradigm이면 search 구조(어느 축)는 물려받되 winning config는 fresh 스윕. warm-start = 축 pruning(spec 스킵),
  답은 아님.

## STEP 4 — 첫 레시피 + verify
recipes/qwen35-35b-a3b-gb10.yaml(plain, config=bmax, scheme fused-moe[k-order bit-id]+w4a8+gdn). run_bench_plain
모델별 dispatch(serving_bench=bench_serve for qwen 계열). **verify --full PASS: 재현율 λ4 B8 99.2%·λ32 B8 100.5%,
메모리 4 config OK, bit-id(오프라인 골든 스위트).**

### ★서빙 throughput ≪ raw batch (정직 분해)
서빙 ~63 tok/s ≪ raw batch aggr 195(B16)/169(B8). tpot 113ms/token vs raw fused 5.9ms/token. 원인:
1. **chunked prefill이 decode 차단**(prefill-decode 간섭 — 단순 continuous 스케줄러가 admit 시 P토큰 prefill을 inline
   실행, 그 wall 동안 타 슬롯 decode 정지).
2. ragged decode(슬롯별 위치) + Poisson idle.
= **스케줄러 성숙도 여지(prefill 분리/continuous batching)**, 커널 아님(fused 커널은 raw서 이미 WIN). 다음 서빙 lever.

## 파이프 리포트 — 사람 개입 지점 감소 곡선 (M7 입력)

| 단계 | 2호 Gemma4(new family) | 3호 35B-A3B(qwen family) |
|---|---|---|
| Phase 0(해부·변환·golden) | 신규(대부분) | **0**(챕터5 기존) |
| plan 전략 | spec-reduction 신규 | **0**(MoE-aware 기존, STEP C서 확장) |
| 튜닝 도구 | bench_serve_gemma4 **신규** | **0**(bench_serve 재사용 — QwenBackend 모델-중립) |
| 서빙 backend | Gemma4Backend 신규 | **0**(QwenBackend가 MoE 서빙 — 코드 변경 0) |
| recipe verify | plain-tolerant 신규(per_slot_bytes_auto·run_bench_plain) | 소(run_bench_plain 모델 dispatch + serving_bench 필드) |
| recipe yaml | 수동 | 수동(스킴만 MoE로) |

**★measured 곡선**: 3호 개입이 2호 대비 **급감**(서빙 코드 0, 튜닝 도구 0). = 2호에서 만든 모델-중립 서빙 코어 +
plain-tolerant 파이프가 **같은 family(qwen)** 3호에 배당. 단 caveat: 3호가 싼 건 **qwen-family**라서(QwenBackend·GDN·
loader 재사용). 새 family(Gemma4가 그랬듯)는 여전히 backend 신규 비용. **M7 = family 경계 일반화**(새 family도 훅으로).

## 다음
- 서빙 lever: prefill-decode 분리(스케줄러) — 63→raw 접근. (커널 아님, #2 fused 이미 완료.)
- 4호 26B-A4B(Gemma MoE — router **renorm 규약** 확인 필수: router_topk.cuh:8 주석의 config 분기). 4모델 완주 마지막.
