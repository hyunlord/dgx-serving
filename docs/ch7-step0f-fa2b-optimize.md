# 챕터7 STEP 0 (FA.2b) — tensor-core FA 최적화 (occupancy 4% → llama 추월)

> FA.2 v1(정확성-우선, 1 warp + shared-왕복)은 정확하나 느림(prefill llama 6-8× 뒤짐, decode 하락).
> ncu가 병목 확정: **occupancy 4%**. FA.2.2서 "최적화는 FA.2b 분리" 결정의 그 단계. 정확 베이스(08f135d) 위에 최적화.

## 0. 측정 근거 (ncu, FA.2 커널, Sq=256 HD=256 base=2048)

| 지표 | 값 | 진단 |
|---|---|---|
| **Achieved Occupancy** | **4.16%** | 헤드라인 병목 |
| Block Size | 32 (1 warp/CTA) | 병렬성 부족 |
| **Block Limit Shared Mem** | **2** | shared 42KB가 SM당 CTA 2개로 제한 = occupancy 상한 |
| Compute(SM) / Memory | 10.6% / 39.8% | 둘 다 낮음 = occupancy-bound (tensor-core 거의 놂) |
| Block Limit Warps/Registers | 48 / 40 | 여유 — shared가 유일 제한 |
| Duration | 17.1 ms | (참고) |

→ 1 warp + 큰 shared(42KB) → occupancy 4% → tensor-core 10.6%만 활용. **멀티워프 + shared 축소가 처방.**
prefill GEMM Phase A→B→C 단계적 좁힘과 같은 다음 단계.

## 1. 처방 (측정에서 직접)

### (1) 멀티워프 CTA (핵심)
현 1 warp = 한 쿼리타일(BR16) 한 head. → CTA당 NW warp(4~8)가 협력:
- **옵션 A (쿼리 분할)**: 한 head의 여러 쿼리타일(BR×NW)을 warp들이 나눠. K/V 타일 shared 공유(재사용↑).
- **옵션 B (KV 분할 + 리덕션)**: 같은 쿼리타일을 warp들이 KV 구간 나눠 부분 online softmax → 마지막 결합(FlashAttention split-K식). 긴 context decode에 유리(KV 길어 분할 이득).
- prefill(Sq 큼)=A 자연, decode(Sq=1)=B(KV 분할)가 맞음 → decode 특화(FA.2c)와 연결.

### (2) shared 축소 (occupancy 상한 풀기)
현 42KB = Q[BR][HD] + K[BC][HD] + V[BC][HD] bf16 + S[BR][BC] + acc[BR][HD] fp32 + P bf16.
- acc[BR][HD] fp32(16KB)를 **레지스터 분산**(warp가 출력 일부 소유) → shared에서 제거.
- Block Limit Shared Mem 2→4+ 목표(occupancy 2×+). 멀티워프와 균형(warp 늘면 일 늘어 레지스터↑).

### (3) 레지스터-분산 softmax (FA.2.2서 미룬 것)
현 shared-왕복(S→shared→softmax→shared→P). 정확 베이스 있으니 이제 **warp shuffle 리덕션**으로 레지스터 softmax:
- row-max/row-sum을 `__shfl_xor` 워프 리덕션(MMA fragment 분산 레이아웃 맞춰). shared 왕복 제거 → 지연↓ + shared↓.
- 위험(분산 레이아웃 리덕션 에러)이라 FA.2.2서 미뤘음 — 정확성 가드(CPU 오차0)로 검증하며 도입.

### (4) decode 특화 (FA.2c)
decode Sq=1: BR16 타일 15/16 낭비가 본질. → (B) KV 분할 멀티워프로 단일 쿼리를 여러 warp가 긴 KV 나눠 처리.
현 하이브리드(decode=FA.1)를 FA.2c로 교체 시도 — flat decode 곡선 목표.

## 2. 검증 계획 (헌법 + race/직교 교훈 — 절대 준수)
1. **정확성 오차0**: 각 최적화 단계마다 CPU naive == rel<1e-2(bf16) — 다중타일 online softmax + GQA + 긴 context 누적(vLLM 함정). test_fa2 확장.
2. **결정성 가드**(Phase C race 교훈 [[w4a8-chunk-prefill-divergence]]): 멀티워프/리덕션은 race 위험↑. N회 bitwise 동일 필수. **"최적화가 정확성 깬" 사례 재발 금지.**
3. **직교성**(사용자 원칙): 엔진 통합서 **q4·W4A8 둘 다** 청크일관·결정. 한 쪽만 되면 멈추고 보고.
4. **ncu**: occupancy 4%→? tensor-core util 10.6%→? Block Limit Shared Mem 2→?
5. **성능**: 3-way 재측정 — prefill llama 806 추월? decode flat 13 추월? 단계마다.

## 3. 단계 (각 단계 정확성·결정성·직교성 통과 후 다음)
- **FA.2b.1**: 멀티워프(옵션 A, 쿼리 분할) + shared 축소(acc 레지스터). prefill 타깃. occupancy↑.
- **FA.2b.2**: 레지스터-분산 softmax(warp shuffle). shared 왕복 제거.
- **FA.2c**: decode 특화(옵션 B, KV 분할 멀티워프). decode 곡선 flat.
- 각 단계 ncu로 occupancy/util 확인 + 3-way 재측정.

## 4. 안전·구조
- 작은 타일/Sq부터, 정확성·결정성 통과 후 상향. 정확 베이스(08f135d) = 롤백 지점.
- FA.2b가 FA.2 커널 개선(새 커널 추가 아님 — 같은 flash_attn2_kernel 최적화 또는 명확히 교체).
- 메모리 가드([[gb10-unified-memory-crash]]). GDN/matmul/decode-q4 안 건드림.

---
*ncu가 occupancy 4% 확정 → FA.2b.1(멀티워프+shared축소) → 오차0·결정성·직교성 → ncu occupancy↑ →
3-way(추월?). 부족하면 b.2(레지스터 softmax)/c(decode 특화). 단계마다 같이 본다.*
