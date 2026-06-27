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

## STEP 1 결과 — isolated batch gemm (fp32): latency 벽 탈출 실증 [완료]

`kernels/gemm_batch.cuh`(template<B> weight 1로드→B 누적기 재사용) + `tests/test_gemm_batch.cu`(==B gemv 오차0).

### 정확성: 통과
batch gemm == B개 독립 gemv, 시퀀스별 max err 2.3~7.3e-6 (B=1,2,4,8,16; lm_head/FFN 포함). 14/14 PASS.

### thesis 측정 (ncu B 스윕, q_proj IN=1024 OUT=2048)
| B | reg | occ% | long_sb stall | compute% | dur(us) | throughput(B/us) |
|---|---|---|---|---|---|---|
| 1 | 38 | 88 | **278** | 9.3 | 37.5 | 0.027 (1.0×) |
| 2 | 40 | 86 | 157 | 9.5 | 54.9 | 0.036 (1.4×) |
| 4 | 40 | 90 | 169 | 14 | 83.9 | 0.048 (1.8×) |
| 8 | 40 | 83 | 98 | **28** | 69.6 | 0.115 (**4.3×**) |
| 16 | **48** | — | 77 | 18 | 152 | 0.105 (3.9×) |

### 핵심 — 4가지 thesis 질문 다 YES
1. **weight bytes/token B×↓**: weight는 output당 1회 read(B 무관) = OUT·IN·4 고정 → per-token = OUT·IN·4/B. 구조적 B×↓. ✅
2. **compute-bound 전환**: stall 278→77 급감, compute% 9→28%(peak B=8). **챕터2 latency 벽(stall 지배)을 탈출.** ✅
   = 측정이 가리킨 길 확정: batch-1에서 폭·정밀도로 못 푼 걸 batch가 벽 자체를 벗어나 푼다.
3. **occupancy 깨지는 B**: B≤8 reg 40·occ 83~90% 안정. **B=16서 reg 48**(2d MO=4 함정 reg 46과 동일 영역) → compute% 28→18% 하락.
   → **최적 B≈8** (register pressure 천장). 사용자 예측("B 무작정 키우지 말고 occupancy 안 깨는 최적 B") 적중.
4. **throughput**: B=8서 4.3× (per-output time amortize). weight 재사용 실현.

### STEP 1 결론
**batch가 latency 벽을 벗어난다 — 측정으로 첫 실증.** B=8 sweet spot(compute 28%, throughput 4.3×, occ 83%).
= llama.cpp mmvq multiple-output과 같은 원리(weight 1로드당 compute↑)의 일반화. batch-1 천장(챕터2)과 batch 처리량을
같은 커널이 푼다. 단 isolated 수치(ncu) — 실 처리량은 STEP2(batch decode 통합)에서 end-to-end 판정.
다음 STEP 2: DecodeState×B + batch attention(per-row KV) + batch GDN(B병렬) 통합.

## STEP 2 결과 — batch decode 통합: latency 벽 in-engine 탈출 [완료]

### STEP 2a: batch 커널 (attn/gdn/conv) — bit-identical
attn_decode_batch(grid(B,NQ), per-row KV), gdn_step_batch(grid(NH,B), B 병렬 1스텝, in-place state),
gdn_conv1d_silu_decode_batch(per-row convctx). isolated 검증: 각 == B single 호출 **err 0.000e+00**.

### STEP 2b: 통합 (qwen35_batch.cu + BatchDecodeState×B + rope_partial_batch)
- 정확성: **batch decode == B개 single decode, max err 2.24e-05**(fp32 reduction noise), **argmax ALL match**
  (4 시퀀스 × 8 스텝, 서로 다른 토큰열 → position/KV/state/conv 배선 시퀀스별 검증). 16/16 PASS, 단일경로 무손상.

### in-engine 처리량 (fp32, B 스윕) — thesis 실현
| B | total tok/s | vs B=1 | per-seq tok/s |
|---|---|---|---|
| 1 | 70.5 | 1.0× | 70.5 |
| 2 | 136.1 | 1.93× | 68.1 |
| 4 | 254.5 | 3.61× | 63.6 |
| 8 | **462.1** | **6.55×** | 57.8 |
| 16 | 646.8 | 9.17× | 40.4 |

- **batch가 latency 벽을 in-engine서 탈출**: B=8 총 처리량 **6.55×**, per-seq는 **−18%만**(70→58) — 8 시퀀스가
  각자 거의 batch-1 속도로. isolated gemm 4.3×(STEP1)가 in-engine 전체선 6.5×로 **더 큼**(forward 전체 amortize).
- B=16: 9.17× total이나 per-seq −43%(40.4) — gemm B=16 register pressure(reg48) + saturation. 처리량/latency 균형은 B≈8.

### ncu — gemm 외(attention/GDN)가 병목 되나? (isolated≠in-engine 검증)
B=8 per-kernel: **gemm_batch 60.5% 지배**, gdn_step_batch 9%, attn_batch 0.9%, conv_batch 0.4%.
→ **attention/GDN은 batch서 병목 안 됨**(per-seq work라 B로 선형 증가하나 절대량 작음). 6.5× 처리량 scaling이
gemm 외 부분에 안 발목 잡힘을 확증. (작은 in_a/in_b는 아직 gemv_f32 — tiny, 비병목.)

### STEP 2 결론
**측정이 가리킨 길(batch) 끝까지 실현** — chapter2 latency 벽(batch-1 +2% 천장)을 batch가 벗어나 **6.5× 처리량**.
정확성 시퀀스별 오차0(배선 무결), 단일경로 무손상. 다음 STEP 3: q4/W4A8 × batch (W4A8이 batch서 비로소 의미 —
X가 B행이라 int8 트래픽 절감 실현 + 언팩/DP4A amortize) + llama.cpp batch 재비교.

## STEP 3 결과 — q4/W4A8 × batch: W4A8 부활 + llama.cpp 추월 (클라이맥스) [완료]

gemm_batch q4(weight unpack 1회→B행 재사용) + w4a8(int8 X B행 × int4 weight, DP4A, weight unpack amortize)
+ qwen35_decode_batch use_q4 배선(W4A8은 batch X 1 launch 양자화).

### 정확성 (구현)
batch q4 == B single q4 **err 0.000e+00**, batch W4A8 == B single W4A8 **err 0.000e+00** (bit-identical), argmax 전부 match. 16/16 PASS.

### 처리량 (total tok/s) — 정밀도 × B
| | B=1 | B=8 | B=16 |
|---|---|---|---|
| fp32 | 70.5 | 454 | 650 |
| q4 | 193.9 | 633 | 772 |
| **W4A8** | 215 | **980** | 1016 |

- **W4A8가 batch서 부활**: W4A8 vs q4 우위 +11%(B=1) → **+55%(B=8)** → +32%(B=16).
  챕터2b batch-1선 +2.5%(X=weight의 0.002~0.78%라 무의미)였던 "죽은 레버"가, batch서 X가 B행이 되며
  **int8-X 트래픽 절감 실현 + DP4A(4 int8 MAC/instr) + weight unpack amortize** → +55%. **챕터2b 약속의 검증.**
- W4A8 B=8 = 980 tok/s(fp32 batch 454의 2.16×, fp32 batch-1 70의 13.9×).

### llama.cpp batched 재비교 (llama-batched-bench, Q4_K_M, S_TG = 생성 처리량)
| B | 우리 W4A8 | llama.cpp Q4_K_M | |
|---|---|---|---|
| 1 | 215 | 277 | llama.cpp 앞 (single-stream 커널 성숙 — 챕터2 진단대로) |
| **8** | **980** | **912** | **우리 추월 (+7.5%)** |
| 16 | 1016 | 1266 | llama.cpp 앞 (대형 batch scaling — 우리 B=16 register pressure 천장) |

- **B=8 처리량 sweet spot서 llama.cpp 추월** — 세션 여정(q4→W4A8→batch)의 결실. per-seq도 우리 122 vs 그들 114.
- B=1은 여전히 뒤짐(single-stream 커널 성숙 = 챕터2 미해결 영역). B=16은 그들이 더 잘 scale(우리 reg48 천장).
- 공정성: 우리 W4A8(int4 sym/blk32 + int8 act) vs Q4_K_M(K-quant + q8_1) — 근사 비트, 방식 다름(기록). 품질은
  양쪽 int8 본질 한계(코드 flip) 공유(챕터2b). 둘 다 greedy plain 생성.

### STEP 3 결론 — 챕터3 완결
**측정-우선 여정의 결실**: 챕터2b에서 batch-1 무의미했던 W4A8이 batch서 +55%로 부활(약속 검증) →
B=8 처리량 980 tok/s로 **llama.cpp(912) 추월**. 진짜 서빙(batch>1)에서 우리 엔진이 경쟁력. 남은 격차:
B=1(single 커널), B=16+(대형 batch scaling, register pressure). 다음 후보: STEP4(spec×batch / ragged /
continuous) 또는 B=1·B=16 커널 다듬기(mmvq multiple-output 정독 — 챕터3 STEP1서 보류한 것).

## 9. 사용자 결정 필요 (게이트)
1. **범위 = multi-seq batch decode, plain, same-length 먼저** — 동의? (ragged/spec×batch/continuous는 후속.)
2. **STEP 1 = isolated batch gemm(fp32)부터** (KV/GDN batch 전에 thesis 증명) — 동의? 아니면 STEP2 통합부터?
3. **gemm = weight 1로드 → B 누적기(register) 재사용**, 큰 B는 smem tiling. register pressure는 ncu 감시 —
   이 방향 동의? (2d MO occupancy 붕괴 교훈 반영.)
4. B 스윕 범위(1,2,4,8,16) + 처리량/per-seq latency/ncu 3측정 — OK?
