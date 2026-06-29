# 챕터7 STEP 0 (FA 근본 재구조) — prefill occupancy 벽 넘어 llama 추월

> B로 prefill 24×→3.4×(240 t/s) / decode 6.7×→1.6×(7.4) 좁힘. 추월(llama prefill 806)엔 근본 재구조 필요.
> 측정이 "현 구조 천장 = occupancy 50%, 미세조정 다 기각" 확정 → 근본 재구조를 제대로 설계해서 추월(포기 아님).

## 0. 측정 입력 (FA.2b.2 — 추측 아닌 확정)

**occupancy 50% = Block Limit Shared(3) AND Reg(3) 동시 제한** (ncu, prefill Sq=256):
- Static Shared **26.3KB/block** (sQ[16][256]+sK[16][256]+sV[16][256] bf16 = **24KB 본질** + sP/sS/sm/sl).
  SM shared 102KB → 102/26.3 ≈ 3 CTA.
- **Registers 72/thread** × 256 = 18.4K/block. SM 64K reg → 3 CTA. (acc[4][4]=16 필수 + fragment/temp ~56.)
- Block Limit Warps 6. → occupancy = min(Shared 3, Reg 3, Warp 6)×8 warp = 24/48 = **50%**.

**미세조정 다 측정 기각 (단일 자원만 줄이면 다른 게 묶음):**
| 시도 | 결과 | 이유 |
|---|---|---|
| warp 협력 S k-분담(+8KB) | occupancy 49→32%, 2.60→4.63ms | shared↑ → Block Limit Shared 3→2 |
| cp.async | 스킵 | KV fp32→bf16 변환 위해 fp32 shared 2배 → occupancy↓ |
| NW 8→16 | occupancy 50→33%, dur 6.11ms | CTA당 thread 2배 → reg total 2배 → Block Limit 1 |
| NW 8→4 | 2.62 vs 2.60ms 미미 | warp0 단독 S가 NW 무관 sibling |

→ **shared(24KB) AND reg(72) 동시 감축**해야 occupancy↑. 단일 미세조정 불가 = 근본 재구조 확정.
(stall 73.7% = warp0 단독 S sibling 35.2% + shared 적재 38.5%도 occupancy 낮아 은닉 안 됨 → occupancy↑가 근본.)

## 1. 근본 재구조 후보 (architect 평가 + 측정 종합)

shared 24KB(sQ/sK/sV) + reg 72 동시 감축 방향:

| 후보 | shared 효과 | reg 효과 | 트레이드 | 비고 |
|---|---|---|---|---|
| **(a) KV bf16 글로벌 저장** | 동일(bf16) | — | 엔진 변경(decode_state KV fp32→bf16) | **cp.async 직접 가능**(변환 제거) → shared 적재 38.5% stall 은닉. 변환 reg/temp↓도. |
| (b) sQ register-resident | −8KB(18KB) | +8 reg | shared↓ ↔ reg↑(상충) | Q[16][256]/256thread=16 bf16=8 reg. occupancy min(shared,reg)라 효과 의문 |
| (c) 작은 타일(BR/BC↓ or HD 2-pass) | ↓ | ↓(acc↓) | tensor-core 효율↓ | shared+reg 둘 다↓ 가능, 단 MMA util |
| (d) warp-specialization(producer/consumer) | — | — | 복잡(CUTLASS-style) | 적재 warp / compute warp 분리 → stall 은닉 |
| (e) fragment/temp reg 정리 | — | reg 72→? | — | acc 16 외 56 temp 줄이면 Block Limit Reg↑ |

**우선순위 확정 (측정+산수+de-risk 기반):**
- **1순위 (a) KV bf16 글로벌 + cp.async** — bf16 cp.async de-risk OK(변환 없이 byte 복사 확인). occupancy 50% 유지하나 **shared 적재 38.5% stall 은닉** + **KV 메모리 절반**(긴 context bonus). 엔진 변경(decode_state KV fp32→bf16 + 저장 시 변환). **단 sibling 35.2%(warp0 S)는 안 풀림 → 부분 개선**(prefill 240→~300 추정, 추월 806엔 부족).
- **2순위 (c) 작은 타일/HD 2-pass** — occupancy↑(shared+reg 동시↓ → sibling 35.2% 은닉). 단 S=Q·Kᵀ가 HD-contract라 HD 분할이 부분합 복잡(k-분담 유사) + MMA 효율↓. occupancy 50→높이는 근본.
- **기각**: (b) sQ register화(reg가 묶어 occupancy 무변, 산수), (e) reg temp 정리(shared 3이 묶음).

**정직한 한계**: 추월(3.4×)은 cp.async(38.5%) + occupancy↑(35.2%) **둘 다** 필요 + α. 단일 후보론 부분. 단계적 큰 작업이고 **추월 보장 불확실**(occupancy↑가 MMA 효율↓ 상쇄할 수도). architect 결론 전달 실패(plain-text만) — 측정으로 진행.

## 2. 검증 (헌법 — 재구조라도 불변)
- **정확성 오차0**: CPU naive == rel<1e-2(bf16). 다중 split + GQA + 긴 KV(vLLM 함정).
- **결정성**: N회 bitwise(2-패스 race 없음 유지).
- **Sq·split 일관**: B 단일 커널 정신 유지(단발==청크, verify==decode, test_promote/chunk). 재구조가 이걸 깨면 안 됨.
- **직교성**: q4·W4A8 둘 다.
- **ncu**: occupancy 50%→? Block Limit Shared/Reg 동시 확인. prefill t/s(240→? vs llama 806).

## 2b. 1순위 구현 단계 (KV bf16 → cp.async)

**단계 1 — KV bf16 (정확성 베이스, cp.async 전):** decode_state Kc/Vc fp32→bf16, 저장 시 f32→bf16 변환커널,
flash_attn2bs Kc/Vc bf16 입력(sK/sV 적재 변환 제거). 효과: KV 메모리 절반(독립 가치) + cp.async 전제.
검증: make test + test_fa2bs rel<1e-2 + Sq·split 일관 + KV 메모리 확인. (진행 중.)

**단계 2 — cp.async 기각(산수 역효과 확정, 2026-06-29):** cp.async는 더블버퍼(2-stage) 필수 → sK/sV `[2][BC][HD]`
= 32KB(+16KB) → shared 42KB. **ncu 실측 SM shared 102KB ÷ 42 = 2 CTA → occupancy 50→33%**. warp 협력(+8KB→32%,
측정 4.63ms 역효과)과 같은 shared↑ 메커니즘에 더 큰 +16KB. KV bf16이 fp32 변환은 풀었으나 더블버퍼 shared 벽 동일.
→ 기각(occupancy 벽 공유, stall 은닉 < occupancy 33% 손실). KV bf16 단계1(메모리 절반)은 독립 가치로 유지(커밋 d202109).

**단계 2′ — HD 2-pass (occupancy↑ 유일 근본):** occupancy 50% = shared 24KB + reg 72 동시 제한. HD=256을 128씩
2-pass로 처리하면 sK/sV[BC][128]=4KB each(절반) + acc[BR][HD/2] 절반(reg↓) → **shared·reg 동시 감축**.
- 흐름(KV 타일별): S=Q·Kᵀ 2-pass 부분합(HD[0:128]+HD[128:]) → softmax(S 완성) → P·V 2-pass(acc[:,0:128]/acc[:,128:]).
- 산수(정정): shared 18KB(102/18≈5 CTA). 단 **reg = acc 16 + fragment/temp 56 = 72**. HD 2-pass로 acc만 절반(16→8)
  → reg ~64 → Block Limit Reg 4. occupancy = min(shared 5, reg 4, warp 6)×8 = **~67%**(50→67, 부분). **fragment/temp 56은
  못 줄임** — occupancy 50% 벽이 shared(24KB)뿐 아니라 **reg(MMA fragment 본질)로도 묶임**(둘 다 본질).
- 트레이드: occupancy 67%(부분) + **KV·Q HD 절반 2번 read(트래픽 2배)** → 순효과 불확실(occupancy↑ < KV 2배일 수). 측정 필요.
- **정직: HD 2-pass도 부분(67%)이고 KV 2배 트레이드라 추월 불확실.** occupancy 50%가 shared+reg(fragment) 둘 다 본질이라
  깊은 벽. 진짜 추월은 완전 다른 알고리즘(CUTLASS FA / FlashAttention-3 warp-spec+TMA)이나 GB10 sm_120+우리 범위서 매우 큰 작업.
- 정확성: S 2-pass 부분합 == 전체 S(자명, k-accumulate 분할). online softmax는 S 완성 후. CPU 오차0 + 결정성 + 직교성.
- 별도 커널(flash_attn2bs_hd2)로 de-risk → 측정 → 우월하면 교체(B 롤백 가능).

**참고 — cp.async 원본 단계(KV bf16 전제, 기각됨):** KV가 bf16라 변환 없이 cp.async 가능(de-risk OK).
- flash_attn2bs_p1의 split 구간 KV 타일(BC=16) 순회를 2-stage 더블버퍼: 다음 BC 타일 prefetch ↔ 현재 MMA.
- **★ canonical 순서(Phase C race 교훈 [[w4a8-chunk-prefill-divergence]]): wait_group→sync→prefetch→compute.**
  prefetch-먼저(Phase C 초기 버그)는 race — 정식 순서 재사용.
- bf16 cp.async = 16byte(8 bf16) 청크. bounds(gk≥maxlen) predicated zero-fill.
- 검증: 오차0 + **결정성 가드 필수**(cp.async race — 단발 테스트가 flaky race 못 잡음, N회 bitwise) + ncu(shared 적재
  stall 38.5%↓, occupancy 50% 유지하나 stall 은닉으로 prefill↑) + prefill t/s(240→?).
- 한계: sibling 35.2%(warp0 S)는 안 풀림 → 부분. occupancy↑(작은 타일, 2순위)가 그 다음.

## 3. 안전·단계
- 검증 베이스 = B(8972608). 재구조는 별도 커널/단계, B 롤백 가능.
- 작은 타일/Sq부터, occupancy 측정하며. GB10 메모리 가드.
- 큰 작업 단계적 — 후보 1개씩 측정(occupancy↑ 확인 후 다음).

---
*측정(occupancy 50% = shared+reg 동시) → architect 방향 → 후보 우선순위 → 단계 구현(각 occupancy/정확성/일관 검증) → prefill 추월. "다음 챕터" 미룸 아니라 지금 설계.*
