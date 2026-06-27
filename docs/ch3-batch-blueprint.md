# 챕터3 설계안 — batching (latency 벽 탈출) STEP 0 (게이트)

> 챕터2 확정: batch-1 decode gemv는 latency-bound(stall 지배, mem 4~10%), 폭·정밀도로 +2% 천장.
> 진짜 병목 = memory-level parallelism 부족(1-warp/output, weight 재사용 0). batch가 벽 자체를 벗어남.
> **이 문서(범위+gemm+batch 자료구조+검증) 합의 후 STEP 1.**

## 1. 핵심 진단 — 왜 현 S>1도 안 빨랐나 (gemv 구조)

gemv_*: `warp = s·gps + group`, 각 (s, output) warp가 weight row를 **매번 재로드**.
→ S행이면 weight를 **S번 읽음**(재사용 0). 그래서 verify(S=K+1)도 per-token 안 빨랐음(weight read가 S배).
**batch의 본질 = weight tile 1회 로드 → B행에 재사용(amortize).** 이게 latency 벽 탈출의 메커니즘:
warp가 weight 1로드당 B× 더 많은 compute(B 누적기) → 같은 DRAM latency를 더 많은 compute로 숨김 + weight read B×↓.

## 2. S 차원의 과부하 (설계 핵심 — 반드시 구분)

현 엔진의 "S행"은 **한 시퀀스의 연속 토큰**(prefill/verify): 공유 KV attend + GDN 순차누적.
multi-seq batch의 "B행"은 **B개 독립 시퀀스 각 1토큰**: 각자 KV attend + GDN 병렬(행 간 recurrence 없음).
→ gemv/norm/rope/swiglu/elementwise(행-독립)는 그대로 B행 동작. **attention/GDN만 batch 의미가 다름**(per-row 독립 KV/state).

## 3. batch 자료구조 (per-sequence 독립)

| 구조 | 현재(1-seq) | batch(B-seq) |
|---|---|---|
| full-attn KV Kc/Vc[l] | [max_len, NKV·HD] | [B, max_len, NKV·HD] (+ per-row n_past) |
| GDN state S[l] | [NH·D·D] | [B, NH·D·D] |
| conv ctx convctx[l] | [K-1, conv_dim] | [B, K-1, conv_dim] |
| n_past | 스칼라 | [B] (same-length 먼저면 동일값) |

→ `DecodeState`에 batch 차원 B. **same-length 먼저**(n_past 공통)로 ragged 복잡도 회피.

## 4. gemm 커널 설계 (STEP1 핵심 — weight 재사용)

gemv → batch gemm: **warp/block이 output(또는 output-tile)을 맡아 weight를 1회 로드, B행 X에 재사용.**
- 구조: warp가 output o 담당, IN 루프에서 weight[o] 1회 로드 → B개 누적기(레지스터)에 X[b][i]·w 누적.
- weight read = output당 1회(B행 공유) vs 현 B회 → **weight bytes/token B×↓**.
- X read = B행(작음). 누적기 B개 = 레지스터 B개/lane → **register pressure**(2d occupancy 함정! ncu 감시).
  큰 B는 smem tiling(Simon Boehm). 작은 B(2~16)는 레지스터로.
- q4/W4A8/fp32 각 경로: weight 언팩은 1회, B행에 재사용 → q4/W4A8 언팩 비용도 amortize(추가 이득).

## 5. 검증 계획 (정확성 먼저)
- **STEP1**: batch gemm(B행) == B개 독립 single gemv, **시퀀스별 오차0**(랜덤 X·weight, CPU 오라클). fp32 먼저.
- **STEP2**: batch decode(B-seq) == B개 독립 single decode, **시퀀스별 오차0**(같은 토큰열 B벌 → 동일 결과).
  batch가 결과 바꾸면 버그.
- spec/q4 일관성은 STEP3+.

## 6. 측정 계획 (처리량 vs latency 구분)
- **처리량**: total tokens/s across B (목적). batch-1 대비 몇 배.
- **per-sequence latency**: 한 시퀀스 토큰 간 시간(오히려↑ 가능 — 기록).
- **ncu (벽 탈출 확인)**: ① weight bytes/token B×↓ 실현 ② compute throughput↑(compute-bound 전환?) ③ stall↓
  ④ occupancy 안 깨짐(B 누적기 register pressure 감시 — 2d 재현 주의). **단순 t/s 아니라 지표로**(챕터2 교훈).
- S(=B) 스윕: 1,2,4,8,16. 어디서 compute-bound 전환·throughput 포화되나.

## 7. 작업 순서
- **STEP 1 (isolated gemm, fp32)**: batch gemm 커널 단독. 정확성(==B gemv 오차0) + ncu(weight 재사용·compute 전환) +
  B 스윕 처리량. **KV/GDN batch 없이 thesis 증명**(gemm이 latency 벽 벗어나나) — 최고가치·최저리스크.
- **STEP 2 (batch decode 통합)**: DecodeState×B + batch attention(per-row KV+base) + batch GDN(per-row state, B병렬).
  정확성(==B single decode 오차0) + 처리량/latency.
- **STEP 3 (q4/W4A8 × batch)**: batch gemm에 q4 → W4A8(use_q4==2 인프라 활성). **W4A8이 batch서 의미 갖나**
  (X가 B행→int8 트래픽 절감 실현 + 언팩/DP4A amortize). llama.cpp batch 재비교.
- STEP 4(후속, 별도 게이트): speculative×batch / ragged / continuous.

## 8. 하지 않을 것 (범위 고정)
- speculative×batch / ragged(다른 길이) / continuous batching (STEP4+, 토대 먼저).
- 정확성 변경. MoE.

## 9. 사용자 결정 필요 (게이트)
1. **범위 = multi-seq batch decode, plain, same-length 먼저** — 동의? (ragged/spec×batch/continuous는 후속.)
2. **STEP 1 = isolated batch gemm(fp32)부터** (KV/GDN batch 전에 thesis 증명) — 동의? 아니면 STEP2 통합부터?
3. **gemm = weight 1로드 → B 누적기(register) 재사용**, 큰 B는 smem tiling. register pressure는 ncu 감시 —
   이 방향 동의? (2d MO occupancy 붕괴 교훈 반영.)
4. B 스윕 범위(1,2,4,8,16) + 처리량/per-seq latency/ncu 3측정 — OK?
