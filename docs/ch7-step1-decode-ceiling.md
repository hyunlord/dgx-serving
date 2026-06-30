# 챕터7 STEP 1/2 — prefill matmul 천장 + decode 대역폭 천장 (정직 좌표)

> 2026-06-30 세션. 목표 "둘 다 추월"(prefill·decode 모두 llama 넘기). 결과: **둘 다 근접하나 추월 아님** —
> 측정으로 양쪽 천장 확정. 진짜 추월 경로 = MTP(구조적 바이트 감소)로 규명.

## 정직 좌표 (27B Q4_0/W4A8, B=1, GB10)

| | 세션 시작 | 세션 끝 | llama | 갭 |
|---|---|---|---|---|
| prefill 2K | 490 t/s | **671.8** | 894 | 1.82× → **1.33×** |
| prefill 8K | 442 | **584.7** | 873 | 1.98× → **1.49×** |
| decode 2K | 9.9 | 10.1 | ~13.7 | ~1.36× (무변) |

**추월 아님.** prefill은 GDN k-split로 1.82→1.33× 좁힘. decode는 천장 측정만(별도 경로 필요).

## STEP 1 — prefill

### GDN k-split (커밋 63ec59c) — 유일한 e2e 이득
- gdn_step이 grid=NH=48(value head)뿐 → occupancy 8.33%(1 block/SM) grid-starved.
- k-차원을 KSPLIT warp-group 분할(block=D×KSPLIT, scol[D/KSPLIT], 부분합 고정순서 결합) → 격리 **9.01×**.
- e2e prefill **490→671.8(1.371×)**. 게이트 전부 green: 오차0(rel 4.14e-6) / 결정성 30× bitwise 0/30 /
  Sq일관 청크==단발 0.000e+00(test_promote+test_chunk_prefill) / 직교성 q4·W4A8 argmax ALL match.
- 격리≠in-engine(X-shared) 안 물림 — de-risk dims(Sq=256)가 in-engine 청크와 정확 일치.

### matmul B-pre-unpack de-risk — 기각(평탄)
- per-MMA B unpack은 B가 wn에만 의존(wm 무관)이라 wm=0,1 워프가 같은 B 2× 중복 unpack.
  타일당 1회 pre-unpack(sBi int8 단일버퍼)으로 amortize 시도.
- **오차0**(probe==baseline bit-identical, 레이아웃 정합 `k_start=ks*32+(t>>1)*8+(t&1)*4`).
- **realistic dims(M512·N17408=grid544=full GPU): cudaEvent +1.0%·ncu +2.5% = 평탄.**
  - ★측정 교훈: 첫 측정 M512·N4096=128블록은 "compute under-utilized"(GPU 미충진)라
    cudaEvent 0.958× vs ncu 1.104× **방향 모순** = under-sized 워크로드 아티팩트. 격리벤치 dims가
    실 grid(544) 대표해야 함.
  - 기전: ALU(unpack) 절반 줄여도 sBi shared write+read 라운드트립이 LSU/L1TEX로 옮겨 상쇄
    (overhead 분산 LSU24>ALU18 재확인).
- **8번째 matmul micro-lever 기각**(BM·reg-block·ldmatrix·4B-unpack·KTILE·stream-K·occupancy·B-preunpack
  전부 평탄). **matmul = full-MMQ 천장(prefill 갭 1.33×).** 추가는 deeper restructure(미정).
- 도구: `tools/ncu_mmq_bpre.cu`.

## STEP 2.0 — decode 병목 분해

nsys ngen64−ngen0 차분(decode 격리, 27B W4A8 ctx2048):

| 커널 | decode % |
|---|---|
| **gemv_w4a8** (decode matmul) | **91.0%** |
| flash_attn2bs (FA) | 3.3% |
| rmsnorm / gemv_f32(in_proj) | 각 1.8% |
| gdn(ksplit+conv+qknorm) | 1.3% |
| 기타 | <0.5% |

- **브리프 잠정가설 "FA m16 1-query 낭비" 기각** — FA 3.3%뿐(측정이 또 가설 뒤집음).
- decode = memory-bound weight 적재(GPU ~96% busy, launch-gap 무시).

### gemv_w4a8 대역폭 — 이미 천장 근접
- **L2 캐싱 함정**(우리가 경계한 것): 작은 shape 200× 반복 = L2 재사용으로 207%(565GB/s) 비현실.
  실 decode는 27B>>L2 cold DRAM → 큰 FFN(>L2)만 유효 측정.
- FFN gate(K5120 N17408, 44.6MB) cudaEvent **72% ≈ llama 73%**. FFN down 65%.
- 실 aggregate(nsys 86.5ms/token, ~14-17GB) ~59-71%. ncu: gate/down occ 81/84%, 작은 N=1024 **occ 40%(grid-starve)**.
- **결론: decode gemv는 bulk(FFN=weight 다수)서 이미 llama 대역폭 천장 근접.** 95% memory-stall(memory-bound 당연).
- 헤드룸 = 작은 projection grid-starve뿐 → split-K 시 aggregate ~60→72% = decode ~1.18×(갭 1.36→~1.15×, **추월 아님**).
- 도구: `tools/ncu_gemv.cu`, `tools/bench.cu` focus 모드(argv ctx/ngen + batch skip).

## 전략 결론 — 진짜 추월 = MTP (2nd 기둥)

멘탈 모델 회귀(CLAUDE.md): **decode ≈ bandwidth ÷ bytes-per-token.**
- bandwidth(분자): 우리·llama 둘 다 273 GB/s를 ~72%로 사용 → **분자 천장**. 순수 대역폭 추월 불가
  (같은 효율이면 같은 속도).
- bytes-per-token(분모): 줄여야 추월. 후보:
  - **MTP(multi-token prediction)** — Qwen3.5 MTP head 보유. 1회 weight 적재로 K토큰 생성 →
    토큰당 바이트 1/K = 구조적 추월. **2nd 기둥(대역폭 한계 돌파).**
  - fp16 scale(fp32 scale ~20% 트래픽 → ~1.1×, 정확성 검증 필요) — 보조.

prefill matmul(instruction overhead 천장)·decode gemv(대역폭 천장) **둘 다 micro-lever 소진** =
측정으로 확정. 진짜 추월은 구조적 접근(MTP).

## 다음 세션 — MTP STEP 0
- MTP는 막힌 이력(과거 Python 재현 0% accept) → **de-risk부터**: MTP head weight 정확성 +
  draft/verify accept율 작은 단위 확인 후 엔진 통합.
- 게이트: 오차0(==golden) + 결정성 + 직교성 + Sq일관 + accept율 측정 + e2e t/s.
