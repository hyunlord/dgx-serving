# 챕터7 STEP 0 (FlashAttention) — 긴 context full-attn 설계

> 측정이 정한 설계. 입력 = nsys 긴-context prefill 분해 + 12K shared 벽 직접 검증(이 세션).
> 헤드라인: **성능이 아니라 정확성** — 현 attention은 context >~12K에서 조용히 틀린 답.

## 0. 측정 근거 (추측 아님)

**긴 context prefill 분해 (nsys, 청크256, W4A8 Phase C, decode 격리):**
| context | full-attn | matmul | GDN | 비고 |
|---|---|---|---|---|
| 2048 | 15% | 74% | 7% | matmul 영역(Phase C 완료) |
| 8K | **60%** | 32% | 7% | full-attn 지배 (유효 마지막 점) |
| 16K/32K | (garbage) | — | — | 12K 벽 넘어 attn 미실행 |

**12K shared 벽 (직접 격리 검증):** 현 `attn_decode_kernel`은 full scores[attendlen]를 dynamic shared에
올림 → `shmem=(HD+base+Sq)*4`. base≈12K(shmem 49KB)에서 GB10 기본 48KB 초과 → **launch "invalid argument"
실패**. 치명적: `cudaDeviceSynchronize`가 "no error" 반환 → 엔진 CK가 못 잡음 → **>12K에서 full-attn 16층
조용히 스킵, 틀린 logits인데 정상 종료**. decode도 같은 커널(base=위치)이라 위치>12K 깨짐.
→ 현 엔진은 긴 context(>12K)를 **아예 못 한다(정확성 버그).** [[attn-12k-shared-wall]]

추가 비효율(유효 구간에서도): softmax를 단일 스레드(d==0)가 직렬 max+sum, 쿼리당 HD=128 스레드, 타일링 없음.

## 1. 임시 가드 (✅ 완료 — 이 세션)

FA 완성 전까지 조용한 오답 차단. `attn_smem_guard`(kernels/attention.cuh):
- shmem >48KB → `cudaFuncSetAttribute` opt-in(최대 99KB → base~25K까지 동작).
- shmem >opt-in 한계 → **명시적 stderr 에러 + abort**(조용한 garbage 대신 중단).
- 검증: base 12K/16K/24K launch OK(이전 실패), base 28K abort+메시지. ≤12K(48KB 미만)는 불변(가드 no-op).
이건 미봉책 — 우리 모델 256K 설계라 ~25K도 부족. 근본해결 = 아래 tiled FA.

## 2. tiled FlashAttention 설계 (full-attn 16층만)

핵심: KV를 타일로 순회하며 **online(streaming) softmax**로 누적 → **shared가 context 무관 상수**(48KB 벽 제거)
+ 직렬 softmax 제거 + 제대로 된 병렬성.

알고리즘 (FlashAttention-2 forward, causal, GQA):
- CTA = (쿼리 블록 Br개, head h 1개). KV를 Bc개씩 타일 순회.
- 쿼리행마다 running (m=max, l=denom, acc[HD]) 유지:
  - 각 KV 타일: K_tile[Bc][HD],V_tile[Bc][HD] → shared. S[Br][Bc]=Q·Kᵀ/√HD.
  - m_new=max(m, rowmax S); α=exp(m−m_new); P=exp(S−m_new);
  - l = l·α + rowsum(P); acc = acc·α + P·V_tile; m=m_new.
  - **causal**: key_pos>query_pos 타일 전체 스킵, 대각 타일만 마스크.
- 종료: O = acc / l.
- **shared = K_tile+V_tile+S = (2·Bc·HD + Br·Bc)·4** — context 길이 무관 상수. base=∞라도 안전.
- GQA: kv head = h/(NQ/NKV). Br·NQ·(KV타일수) 병렬.

타일 크기(참고, 구현서 ncu로 조정): Br=32~64(쿼리), Bc=64~128(키), HD=128. shared 예: Bc=128 →
2·128·128·4=128KB? → Bc=64로(2·64·128·4=64KB) 또는 K/V를 절반정밀/재사용. **상수라 길이 무관이 핵심.**

prefill(Sq 쿼리 청크, base 오프셋) + decode(Sq=1) 동일 커널로. 위치 = base+query_idx.

## 3. 검증 계획 (헌법 — 정확성 먼저)

1. **CPU fp32 naive full-attention 레퍼런스**(임의 길이, 작은 dim) == FA 오차0. 버그 커널과 독립한 ground truth.
2. **== 현 `attn_decode`(가드 opt-in)** for context ≤25K (엔진 dim) 오차0 — 기존 검증된 경로와 대조.
3. **>25K**: 현 커널 못 도는 영역 → CPU 레퍼런스로만(1번). prefill 청크 == 단발 == decode 누적 동등성.
4. **27B 엔진 통합**: full-attn 16층 FA 교체 후 ≤8K 골든 == 오차0(회귀), >12K 새로 정확(이전 garbage였음).
   GDN 48층 **안 건드림**. 0.8B/35B 회귀 0.
5. **ncu**: 단일-스레드 softmax 직렬 제거 확인, shared 상수(occupancy 길이 무관), util.
6. **성능**: 8K/16K/32K prefill t/s(현 8K 130 → ?), 긴 KV decode t/s 곡선. **그 뒤** GDN 이점 3-way.

## 4. 단계 (구현은 합의 후)
- **FA.1**: software fp32 tiled FA(online softmax, 상수 shared). 정확성+벽 제거+병렬성. 작은 Br/Bc부터.
- **FA.2(나중)**: tensor-core FA — Q·Kᵀ, P·V를 MMA로(Phase B/C MMA 인프라 재사용). 성능.
- **FA.3(나중)**: 긴 KV decode 최적 + 3-way(GDN 이점 실증).

## 5. 안전 (크래시 교훈)
- 작은 Br/Bc/Sq부터 정확성 통과 후 상향. 메모리 가드 + 청크 prefill 유지([[gb10-unified-memory-crash]]).
- FA는 상수 shared라 shared-벽/occupancy 붕괴 없음(현 버그의 근본 제거). KV 캐시 메모리는 여전히 O(context) — 예산 가드.

## 6. 하지 말 것
- GDN 48층 손대기(이미 긴 context 강함 — KV 안 자람).
- matmul 더 짜기(Phase C로 memory-bound 천장 — 끝).
- tensor-core FA 먼저(FA.1 software로 정확성+벽 제거 먼저, 위험 분리).
- 성급한 큰 Br/Bc/context — 정확성 통과 후. 성능 위해 정확성 양보.

---
*가드(✅) → FA.1 STEP 0 합의 → software fp32 tiled FA 구현 → CPU레퍼런스/현커널 오차0 + 회귀0 → ncu(직렬
softmax 제거·상수 shared) → 8K/16K/32K 성능. 그 뒤 FA.2 tensor-core + GDN 이점 3-way. 단계마다 같이 본다.*
