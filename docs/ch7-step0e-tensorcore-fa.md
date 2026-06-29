# 챕터7 STEP 0 (FA.2) — tensor-core FlashAttention 설계

> 측정이 준 작업목록: full-attn 16층 software FA가 긴-context decode 하락(9→3.2 vs llama flat 12) +
> prefill 32K 24× 격차의 공통 범인. llama는 거기 tensor-core FA. → full-attn 16층을 tensor-core로 추월.
> 선행: FA.1a(software, online softmax, 12K 벽 제거, 정확성 오차0) 위에 tensor-core 얹어 속도.

## 0. 측정 근거 (3-way, 27B Q4_0, B=1)

| | decode 2K→32K | prefill 2K→32K |
|---|---|---|
| 우리(software FA) | 9.0→3.2 (64%↓) | 223→33 |
| llama.cpp(tensor-core FA) | ~13.7→12.0 (flat) | 894→806 (flat) |

공통 범인 = full-attn 16층 software FA(O(Sq²), tensor-core 없음). GDN 48층은 동급(모델 덕, 공유).
→ full-attn 16층을 tensor-core FA로 → decode flat + prefill 격차 좁힘. **추월 가능 여부는 만들어 측정.**

## 1. 정밀도 결정 — fp16 (int8 아님)

FA의 두 matmul: ① S = Q·Kᵀ (scores) ② O = P·V (P=softmax(S)).
- **int8(Phase B 재사용) 기각**: P는 softmax 후 긴 context서 매우 작음(~1/attendlen). int8(고정소수)은
  작은 P 표현 못 해 **P·V 긴-context 부정확**. Q·Kᵀ는 int8 가능하나 P·V가 막힘.
- **fp16 채택**: floating이라 작은 P도 ~1e-3 정밀. sm_120 `mma.sync.m16n8k16.f32.f16.f16.f32` 컴파일 확인
  (A=4 B=2 C=4 regs). HF 레퍼런스도 bf16/fp16라 fp16 attention이 실모델에 더 가까움.
- **Phase B 재사용 = 패턴**(타일링/fragment 레이아웃/online softmax 구조/canonical cp.async 더블버퍼[race
  수정본 c55e3e8]), int8 operand가 아니라. 복붙 말고 공용 구조.

## 2. 구조 (FA.1 교체 — 추가 아님)

CTA = (쿼리 블록 Br, head h). KV를 Bc씩 타일 순회 + online softmax(running m/l/acc fp32, FA.1 검증된 것):
- **① S = Q·Kᵀ**: Q[Br,HD] fp16 · K[Bc,HD] fp16 → S[Br,Bc] fp32 acc. HD=256를 k=16씩 16스텝 MMA. /√HD.
- **online softmax**: m_new=max(m,rowmax S); α=exp(m−m_new); P=exp(S−m_new); l=l·α+rowsum P; acc=acc·α+...
- **② O = P·V**: P[Br,Bc] fp16(softmax 후 변환) · V[Bc,HD] fp16 → O[Br,HD] fp32 acc. Bc=16씩 contract.
- 종료 O/=l. causal: key>query 타일 스킵 + 대각 마스크. GQA: kv=h/(NQ/NKV)(데인 함정 — 커널·CPU 둘 다).
- **shared 상수**(context 무관, 벽 없음): Q[Br][HD]+K[Bc][HD]+V[Bc][HD] fp16 + S[Br][Bc] fp32.
  Bc=16,Br=32,HD=256 → 8+8+16KB(fp16)+2KB ≈ 34KB <48KB. 작은 타일부터, 큰 타일은 opt-in.
- Q/K/V fp32→fp16 변환은 shared 적재 시. cp.async(race 수정 canonical 순서)로 KV 타일 더블버퍼.

## 3. 검증 계획 (헌법 — 정확성 먼저, race/구조 교훈)

1. **CPU fp32 naive 레퍼런스 == FA.2** : fp16 MMA라 bitwise 0 아님, **rel ~1e-3 < 1e-2(오차0 기준)**.
   여러 타일(작은 Bc로 다수) online softmax running + GQA(NQ24/NKV4 등) 명시 검증. [[qwen35-head-dim-256]]대로 HD=256.
2. **== FA.1a(software)** 대조: 같은 입력 rel ~1e-3(fp16 vs fp32 차이만).
3. **결정성 가드**(race 교훈 [[w4a8-chunk-prefill-divergence]]): FA.2 N회 bitwise 동일(cp.async 더블버퍼라 필수).
4. **엔진 통합**: full-attn 16층 launch_flash_attn → FA.2 교체. 27B ≤8K 회귀(골든), >12K 정확, 0.8B/35B 회귀0.
   엔진 W4A8 청크 일관성(C 512~32K) 유지.
5. **ncu**: tensor-core(pipe) active 비율, mem/compute util. software FA 대비 속도.
6. **성능 재측정**: 3-way decode/prefill 곡선 — software 9→3.2를 얼마로, llama flat 12 추월하나.

## 4. 단계 (구현은 합의 후)
- **FA.2a**: tensor-core fp16 FA 커널(작은 Br/Bc, 정확성+결정성). CPU 오차0 + FA.1 대조.
- **FA.2b**: 타일 크기/opt-in shared/double-buffer 튜닝(ncu util). 엔진 통합.
- **FA.2c**: decode 특화(Sq=1 긴 KV) 최적 — decode 곡선 flat 목표.

## 5. 안전·구조
- 작은 Br/Bc/Sq부터 정확성·결정성 통과 후 상향. 메모리 가드([[gb10-unified-memory-crash]]).
- FA.2가 FA.1 **교체**(attn 커널 수 안 늘림). Phase B MMA 패턴 공용(복붙 금지).
- GDN 48층·matmul(Phase C memory-bound 천장)·decode q4 안 건드림.

---
*fp16 MMA(✅ 컴파일) → FA.2a STEP 0 합의 → tensor-core FA 구현 → CPU 오차0(<1e-2)+결정성+FA.1 대조 →
ncu tensor-core util → 3-way 재측정(llama flat 12 추월?). 단계마다 같이 본다.*
