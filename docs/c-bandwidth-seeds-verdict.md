# 대역폭 돌파 시드 지형도 완결 — 7시드 전부 measured 판정 (2번째 기둥 정직 결론)

> 잔여 3시드(L2·KV 압축·spec 극한) 값싼 진단으로 일괄 판정 → 7시드 전부 measured. 소거도 성과(환상 제거).

## STEP 0 — L2 캐시 명시적 활용

- **GB10 L2 = 24MB·persistence 지원**(l2_probe measured: persisting 최대 18MB·accessPolicyWindow 128MB·SetLimit OK).
  API는 살아있음(sm_121).
- ★비중 판정: decode 지배 read = weight **13.5GB/token ≫ L2 24MB**(562×) — persist 후보는 작은 반복 텐서뿐
  (norm 64층×2×5120×4B ≈ 2.6MB 등, 토큰당 read의 **~0.02%**). 게다가 HW L2가 작은 반복 텐서는 이미 자동 캐싱.
- **판정: 소거**(persist 이득 상한 ~0.1% 미만 — API 지원돼도 비중이 없음). frontier 전례(L2 102% 히트)는 working set이
  L2에 들어가는 특수 케이스 — 27B decode는 정반대(562× 초과).

## STEP A — KV 압축 극한 (시드 ④)

- **현 상태(측정/코드)**: qwen KV = **이미 bf16**(decode_state.h — 모델 dtype·메모리 절반). 27b는 **GDN 혼합 16/64층만
  full-attn** = KV 자체가 4×↓(GDN = 구조적 KV 압축 내장). gemma4만 fp32 KV(알려진 STEP B TODO) + sliding 1024 캡.
- **비중 산술(27b)**: KV read = 2×NKV4×HD256×2B×16층 = 64KB/ctx-tok → ctx 2048=128MB(weight의 0.9%)·8192=512MB(3.7%)·
  32K=2GB(13%).
- **실측(bench ctx 스윕)**: decode 11.9(ctx8)→11.2(512)→11.1(2048)→**9.9(8192, −17%)**. ★분해 정직: TPOT Δ12.3ms(ctx8→8K)
  중 KV read는 512MB÷273GB/s = **1.9ms(15%)뿐** — 나머지는 attn compute(커널 시간). → **q8 KV(read 절반)는 e2e ~1%**.
- **판정: 소거(27b)** — GDN+bf16이 이미 KV를 구조적으로 압축, 잔여 KV-read 비중이 작아 양자화 여지 무의미.
  조건부: gemma4 fp32→bf16 KV(2×, 긴 context 한정 modest)·32K+ 초장문 서빙이면 재고(13%→6.5%).

## STEP B — spec 극한 (시드 ⑥) — 기존 measured 인용(재측정 불요)

- **K/W 축은 이미 극한 탐색됨**: ch10 튜닝 하네스 120 trial — 전 B에서 **K6~7 최적**(B×K 상호작용 없음)·
  regime별 K7W2(long)/K7W3(short). **Nn>8 = MMQ 함정 e2e 실증**(≤8 소트리 필수) = K 확장 하드 캡.
- **tree도 측정됨**: tree K4W3 > chain(accept 2.27~2.86→ch10 K6W4 4.51)·width>depth·full-forking+MTP오버헤드감축이
  잔여 lever(modest). **accept 상한 = draft 품질**(단일 MTP head — 워크로드 민감 6.99/4.55/3.5 measured). 더 올리려면
  별도 draft 모델(DFlash류 65~94 — 커널 밖·모델 자산 필요).
- **판정: 현 K6~7 = measured sweet spot(소거)**. 잔여 = draft 모델 업그레이드(엔진 밖 축).

## STEP C — ★7시드 전부 판정표 (지형도 완결)

| 시드 | 판정 | 근거(measured) |
|---|---|---|
| ①⑤ unified 계층/weight 상주 | **소거** | 물리 계층 부재(memory-characterization)·상주할 "빠른 곳" 없음 |
| ②③ dense sparsity | **소거(modest)** | soft 31%+100% dynamic+예측비용 함정 — 상한 1.33×·MoE만 작동(내장 router) |
| ④ KV 압축 | **소거(27b)** | GDN 16/64+bf16 이미 구조적 압축·잔여 KV-read e2e ~2%·q8 KV ~1%. 조건부: gemma4 bf16화·32K+ |
| ⑥ spec 극한 | **소거(K/W축)** | 120 trial K6~7 sweet spot·Nn>8 MMQ 캡·tree measured. 잔여=draft 모델(엔진 밖) |
| ⑦ prefetch | **소거** | no-op measured |
| L2 persist (신규) | **소거** | API 지원(24MB/18MB)·but weight 13.5GB≫L2 = 비중 ~0.02% |
| q2 양자화 (bytes↓ 자매) | **소거(압살)** | naive 464× 붕괴·정교도 ~q4 bits 필요 — q3만 온건(+25%) |

### 2번째 기둥 정직 결론

**대역폭 돌파(2×급)는 이 장비·이 모델군·tractable 기법으로 measured 불가.** 273GB/s × 토큰당 bytes가 근본 벽 —
7시드 전부 부딪혀 확인(소거 = 환상 제거·헛수고 방지 자산).

**남는 실 lever 최종 목록(measured 전체 지도)**:
| lever | 이득 | 상태 |
|---|---|---|
| mmq 형상 256×64 | prefill +5%(bit-id) | 구현·in-engine 검증됨(opt-in) |
| q3 양자화 | decode ~+25%(품질 1.74× coherent) | measured·구현은 후속 |
| spec 곱 | 단일 latency 1.8×(반복 워크로드 6.99) | 구현·서빙 결합됨 |
| MoE(내장 sparsity) | active-bytes 8.7× roofline | 구현(moe_fused) |
| gemma4 KV bf16 | 긴 ctx 한정 modest | TODO(알려진) |
| draft 모델 업그레이드 | accept 4.5→? (DFlash 65~94 전례) | 엔진 밖(모델 자산) |
| SOTA 2-bit(QuIP#/AQLM) | decode ~2×(품질 유지 시) | 대형 연구(미착수) |
| FP4 텐서코어 | matmul 대폭 | toolkit-blocked(CUDA 13.2+ 대기) |

= **이 장비에서 가능한 것의 measured 전체 지도 완결.** 다음 = 현실 착지(제품화/M3) or q3 구현 or FP4 해금 대기.

## 산출물
- tools/l2_probe.cu(L2 진단)·docs/c-bandwidth-seeds-verdict.md(판정표). KV는 기존 bench+산술(신규 코드 0).
