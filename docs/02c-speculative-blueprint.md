# 2c 설계안 — speculative 생성 루프 (수준1, STEP 0 게이트)

> 2a 본체(GDN+full, prefill) + 2b MTP(78.6% accept) 위에 chain speculative 생성 루프.
> 목표: 78.6% accept → 실제 t/s 가속 측정. 범위=단일시퀀스/greedy/chain. **합의 후 STEP 1.**
> GDN rollback = 의뢰서 확정 전략("임시 승격"). llama.cpp도 recurrent state를 checkpoint로 관리(확인).

## 0. 현 코드가 prefill 전용 — decode 인프라가 신규 (핵심)

| 커널(2a) | 현 상태 | decode/verify에 필요한 변형 |
|---|---|---|
| `attn_kernel` | threadIdx 이중역할 → **S≤256, S내부만 attend** | **신규** `attn_decode`: q(1~K+1개)가 **committed KV 캐시[0..n_past-1] + 신규토큰** attend |
| `gdn_delta_rule` | state 내부 0초기화, 미반환 | **신규** `gdn_step`: S_in→S_out (외부 버퍼), 1스텝(decode)/K+1스텝(verify) |
| `qwen35_forward` | prefill 일괄, KV/state 미지속 | **신규** decode/verify graph (캐시·state 사용) |
| gemv/rmsnorm_zc/rope/swiglu/gate | 그대로 | 재사용 |

→ 2c는 "커널 재배선"이 아니라 **상태 지속 인프라 + decode-attention/gdn-step 신규 커널**이 핵심.

## 1. 상태 3종 자료구조

- **full-attn KV** (6개 full 레이어): `Kc/Vc[layer][maxlen][NKV=2][HD=256]` fp32. append: 새 토큰 K/V를 pos에 기록.
  rollback = 길이 포인터(`kv_len`)를 accept 길이로 되돌리기(덮어쓰기 가능, 위치독립).
- **MTP 자체 KV** (MTP 1레이어): `mKc/mVc[maxlen][NKV][HD]`. catch-up으로 채움(2b). 동일하게 길이 포인터.
- **GDN recurrent state** (18 linear 레이어): committed `Sc[18][NH=16][128][128]` fp32 = **18MB**.
  순차누적이라 위치 포인터 rollback 불가 → 아래 "임시 승격".

## 2. autoregressive decode (단일토큰, STEP 1)

본체 1토큰 decode: 각 레이어 —
- full: rmsnorm_zc → q/k/v proj(1토큰) → q/k_norm → rope_partial(pos) → **append K/V to cache** →
  `attn_decode`(q 1개가 cache[0..pos] attend) → gate → o_proj → residual → MLP.
- GDN: in_proj → conv1d(**conv state** 필요 — k=4라 직전 3토큰; decode시 conv도 작은 상태 유지) → g/β →
  `gdn_step`(Sc[l] 1스텝 in-place 업데이트) → gated_norm → out_proj → residual → MLP.
- **conv1d state**: depthwise k=4 → 채널별 직전 3토큰 버퍼 `convstate[18][conv_dim][3]`. decode시 shift+append.
- **검증(STEP1)**: KV decode로 N토큰 생성 = 2a prefill로 같은 N토큰 forward 한 logits와 오차 0.

## 3. draft chain + batch verify (STEP 2)

- draft(2b MTP): t0 hidden→ d1; d1 embed + MTP 자체 hidden → d2; ... K개 (chain, 각 MTP KV append).
- verify: 본체가 [t0,d1..dK] (K+1토큰) 처리. full-attn = K+1 queries가 committed cache + 자기들끼리 causal
  (`attn_decode` 확장 or K+1 루프). **GDN = committed Sc에서 시작해 K+1토큰 순차** (`gdn_step` K+1회),
  **각 스텝 후 S 스냅샷** `Stmp[K+1][18][...]` (≈ 5×18=95MB, 재사용). conv state도 임시.
- **검증(STEP2)**: verify가 위치별 내는 토큰 = 그 위치를 일반 decode로 뽑은 토큰과 일치.

## 4. accept + 임시 승격 (STEP 3) — 의뢰서 확정 전략

- accept 길이 n = 앞에서부터 draft가 본체 verify argmax와 일치하는 수 (greedy longest-prefix).
- **GDN 승격**: `Stmp[n]` → `Sc` 복사(18MB). conv state도 step n 것 승격.
- **full-attn/MTP KV**: 길이 포인터를 (n_past + n)으로 확정 (rejected 위치 덮어쓰기 허용).
- **3종 동기화**: 같은 n으로 GDN Sc·full KV len·MTP KV len 일관 전진. 다음 토큰 = verify의 position n 토큰(본체 확정).
- **검증(STEP3)**: 한 사이클 후 3상태 = 같은 (n+1) 토큰을 일반 decode로 진행한 상태와 오차 0.

## 5. 전체 루프 + 측정 (STEP 4)
- 반복: decode 1 → draft K → verify K+1 → accept n → 전진(n+1 토큰). greedy.
- **정확성**: spec on 출력 == off(일반 greedy decode) 글자까지 동일 (deterministic).
- **성능**: t/s(on vs off), 평균 accept 길이, **ncu**(임시 state 버퍼 대역폭, verify batch 효율, 사이클 비용).
  헌법: 시간 단독 금지 — ncu 동반. 손익분기: accept가 verify batch 비용배수 넘어야 이득(78.6%면 안전 예상).

## 6. 비용 추정
- GDN committed Sc 18MB, Stmp(K+1 스냅샷) ~95MB(K=4). full KV(6레이어×maxlen×512×4), MTP KV 소량. → 총 ~수백MB, OK.
- 승격 복사 18MB/사이클(accept당 1회) — 대역폭 273GB/s 기준 ~0.07ms, 무시 가능.
- 임시 state 읽기/쓰기(verify K+1 GDN step)가 주 오버헤드 — ncu로 측정.

## 7. 검증 단계 요약 (헌법: 단계별 오차 0, 추측 금지)
1. incremental decode == prefill (오차 0)  2. verify 토큰 == decode 토큰
3. 사이클 후 상태 == 일반 decode 상태 (오차 0)  4. spec on==off 출력 + t/s + ncu

## 8. 사용자 결정 필요
1. **chain 길이 K** (draft 개수): K=3~4 권장(accept 78.6%면 평균 ~2-3 전진 기대). 시작값?
2. **GDN 승격 방식**: (a) Stmp K+1 스냅샷 후 accept점 복사(메모리↑ 단순) vs (b) accept 후 committed S에서 n토큰
   재실행(메모리↓ 약간 재계산). → (a) 권장(verify와 자연 통합).
3. **decode-attention 신규 커널** 범위: 1쿼리(decode) + K+1쿼리(verify)를 한 커널로 일반화 vs 분리. → 일반화 권장.
4. STEP 1 검증 프롬프트/생성 길이 (예: 16토큰 프롬프트 + 32토큰 생성).
