# 챕터10 — 배칭+spec 결합 (Phase 0 아키텍처 설계)

> 목표: 커널 raw arc 수렴(batch 1.02× near-parity, prefill/single/spec measured 천장) 후 **유일한 역전 길**.
> **연속 배칭(throughput) × spec(accept 3.57) = 곱**. vLLM/llama도 어려운 영역, 우리는 조각(ragged-batch·tree-spec·scheduler) 다 보유.
> Phase 0 = 설계만(코드 X). 빌드는 Phase 1부터, Phase마다 mism=0 게이트.

## 왜 곱인가 (멘탈 모델)
- decode 속도 ≈ 대역폭 ÷ 토큰당-이동바이트. weight 1회 운반이 몇 토큰을 내느냐.
- **연속 배칭**: weight 1운반 → B요청 각 1토큰 = B토큰/운반.
- **spec**: weight 1운반(verify 1-forward) → accept 토큰(3.57)/운반.
- **결합**: weight 1운반(batched verify) → B요청 × accept토큰 = **B×3.57 토큰/운반**. 지금 둘이 따로 돌아 곱을 못 얻음.
- ★핵심 실현: **batched tree verify의 matmul이 M=Σ Nnodes(예 B16×7=112행)** → 128-타일 꽉 참(현 single verify는 M=7, 7/8 낭비). 1 weight-load가 B요청 트리 전부 서빙 = 곱.

## 단일 spec 사이클 (현 test_treespec_e2e, 결합의 단위)
1. **draft(MTP)**: `mtp_step` 체인으로 tree Nnodes(=K chain + W width) 구성 → tok[]/par[]/pos[]/anc[]. MtpKV + h_cur 사용. (spec 시간 ~22%)
2. **verify(1-forward)**: `qwen35_decode(Sq=Nnodes, commit=0, va, d_parent/d_anc/d_pos)` → vl(노드별 logits)·vpn(노드별 prenorm). tree attention mask(anc) + GDN tree-fork(`gdn_step_ksplit_tree`). (spec 시간 ~72% = 지배)
3. **accept(longest-greedy)**: host walk, verify argmax==draft인 최장 path → path[plen].
4. **promote**: `promote_tree_path(st,va,path,plen)` — GDN S=snap[path末], conv ctx, KV compaction, n_past+=plen.
5. **MTP catch-up**: accept 토큰들 mtp_step 재실행(MtpKV 갱신). cur/h_cur ← next.

## 연속 배칭 (현 bench_serve run_continuous)
- B slots, 각 `qwen35_decode_batch`(1토큰/step), per-seq position(`h_pos_b`), admission guard + slot 재사용 + 큐 + chunked prefill.
- **spec 0회 사용**(grep 확정) — batch와 spec 완전 분리.

## ═══ 결합 아키텍처 ═══

### ragged spec 데이터 구조 (per-slot)
슬롯 b마다 독립 spec 상태:
```
SpecSlot { int req;                     // 요청 id (-1=빈 슬롯)
           int base;                    // n_past (이 요청 committed 길이)
           int tok[NN], par[NN], pos[NN]; unsigned anc[NN];  // 이 슬롯의 draft tree
           int Nnodes;                  // 이 슬롯 트리 크기 (요청마다/사이클마다 다를 수)
           MtpKV mtpkv; float* h_cur;   // MTP draft 상태
           int plen; int path[NN]; }    // 직전 사이클 accept 결과
```
- 이미 있음: per-seq KV/GDN state(BatchDecodeState.S/Kc/Vc slot별), per-seq position(h_pos_b).
- 새로: 슬롯별 tree(tok/par/pos/anc) + MtpKV.

### 자산 엮기 (super-step = B 슬롯이 각 spec 사이클 1회)
1. **draft (per-slot MTP)**: 각 슬롯 자기 tree draft. MTP head 작음(22%) → Phase 초반 순차, 후반 배치.
2. **★batched ragged tree verify (핵심 신규)**: B 트리를 concat → 총 Sq=Σ Nnodes_b 한 forward.
   - **matmul**: M=Σ Nnodes 행(슬롯-무관, weight 1운반) → **128-타일 꽉 참 = 곱의 원천.** gemm_mmq(BM=128) 그대로 M=112 처리(현 자산, empty-row 없음).
   - **attention (ragged, per-slot)**: 슬롯 b의 노드들은 슬롯 b의 KV[0..base_b] + 자기 tree ancestor만 attend. **block-diagonal mask + tree anc.** flash_attn을 슬롯별 호출(현 batch attn = 슬롯별 루프 자산) + tree mask.
   - **GDN (ragged, per-slot tree-fork)**: 슬롯 b 노드들은 슬롯 b의 state S_b서 tree-fork. `gdn_step_ksplit_tree`를 슬롯별(현 batch GDN = 슬롯별 자산 + tree-fork 자산 결합).
   - **rmsnorm/rope/act**: 행-독립 → concat 행에 그대로(자산).
3. **accept (per-slot host)**: 슬롯별 longest-greedy → path_b/plen_b.
4. **ragged promote (per-slot)**: 슬롯별 `promote_tree_path`(각 plen_b만큼 commit, n_past_b += plen_b). 슬롯마다 다른 전진 = ragged. per-seq position 자산으로 다음 super-step 슬롯별 다른 base.
5. **MTP catch-up (per-slot)**: 슬롯별 accept 토큰 mtp_step.

### 복잡도 지도
| 조각 | 상태 | 난도 |
|---|---|---|
| matmul M=ΣNnodes | **자산(gemm_mmq)** | 재사용 — concat 행만 |
| rmsnorm/rope/act/swiglu | **자산** | 행-독립, 재사용 |
| per-slot position/KV/state | **자산(h_pos_b, BatchDecodeState)** | 재사용 |
| tree verify machinery(par/anc/pos, promote_tree_path) | **자산(single)** | 슬롯별 offset 확장 |
| MTP draft(mtp_step) | **자산(single)** | 슬롯별 반복 |
| scheduler(admission/slot/큐) | **자산(bench_serve)** | spec 상태 얹기 |
| **★ragged tree attention(block-diag + anc)** | **신규** | vLLM급 — 슬롯별 mask 조립 |
| **★ragged GDN tree-fork batched** | **신규** | vLLM급 — 슬롯별 state서 fork, concat |
| **★ragged accept/commit(슬롯별 plen)** | **신규** | 슬롯별 전진, 다음 step 배치 구성 변동 |
→ **신규는 3개(ragged attention mask·ragged GDN fork·ragged commit). 나머지 다 자산.** vLLM이 어려운 게 이 3개지만 우리는 single-tree 버전을 이미 가짐(슬롯 차원 추가가 작업).

## Phase 로드맵 + 게이트 (Phase마다 신선한 세션, de-risk+mism=0)
- **Phase 1 — 2-요청 결합(최소)**: 슬롯 2개, 같은 Nnodes, batched verify(Sq=2×Nnodes), per-slot attention/GDN/accept/promote. **게이트: 각 요청 out == 단독 spec(test_treespec_e2e) bit-identical(mism=0).** 고정 2-요청 시나리오. 신규 3개 중 attention mask·GDN fork의 2-슬롯 버전.
- **Phase 2 — ragged accept(핵심 난제)**: 요청마다 다른 accept 수 + 슬롯별 다른 Nnodes + 다른 base 진행. **게이트: 발산 accept서도 각 요청 mism=0.** ragged commit(슬롯별 plen) 완성.
- **Phase 3 — 연속 배칭 통합**: admission/slot/큐 + spec, 요청 도착·완료하며 spec batch 변동(B 가변, 슬롯 재사용 시 spec 상태 clear). **게이트: 동적 도착서 mism=0 + 서빙 throughput 측정.**
- **Phase 4 — e2e + 역전 측정**: Poisson 부하 + spec 결합. throughput×accept, TPOT/p95. **게이트: in-engine controlled A/B, near-parity(1.02×) 대비 역전? overclaim 금지(공정 조건).**

## Phase 1 결과 (완료 8c8a03d — mism=0 토대 확립)
- `qwen35_verify_batch`(2슬롯 tree concat verify) + tests/test_batchspec2 빌드. row-독립(norm/rope/act/matmul) concat 행, attn/GDN 슬롯별 루프(block-diagonal 격리).
- **★게이트 통과(직접 검증)**: fp32 mism 0/24·W4A8 mism 0/24, 3 시나리오(A/B 다른 프롬프트) 모두 PASS. A가 B와 batch돼도 각 == 단독 spec bit-identical.
- **★정직(STEP C 속도)**: W4A8은 **슬롯별 matmul(M=Nnodes)**로 bit-identity 보장 — concat M이 sb→mmq(S≤8) 경계 넘어 borderline flip하는 트랩 회피. **∴ W4A8 concat-matmul 속도 이득 미실현**(각 슬롯 weight 재적재). fp32만 concat. **곱 premise가 W4A8(서빙)엔 아직 X** — 실현엔 **단독 spec을 batched와 동일 large-M 커널로 정렬(Phase 2+ 필수 refinement).** Phase 1=정확성 토대만.

## Phase 1 상세 스코프 (참고 — 위 결과로 완료)
- **하네스**: `tests/test_batchspec2.cu` — 요청 2개(각 프롬프트), 각 자기 tree(고정 K,W). batched verify 1콜(Sq=2×Nnodes) → 각 요청 accept/logits. vs 단독 spec(test_treespec_e2e 로직) 슬롯별 대조.
- **신규 함수**: `qwen35_verify_batch(...)` — qwen35_decode(tree verify)를 슬롯 차원 추가. 입력: 슬롯별 tok/par/pos/anc(offset 배열) + 슬롯별 base. matmul concat, attention/GDN 슬롯별 루프(+tree mask/fork).
- **자산 재사용**: qwen35_decode의 tree verify 본체 + qwen35_decode_batch의 슬롯별 attention/GDN 루프. **결합만 신규**(슬롯별 tree offset + concat 행).
- **게이트**: 2 요청 각 == 단독 spec argmax/accept bit-identical(mism=0). 격리 아님 in-engine.
- **스코프 밖(Phase 2+)**: ragged accept(Phase 1은 같은 Nnodes), 연속 배칭(Phase 3), 역전 측정(Phase 4).

## 리스크 / 정직
- **vLLM급**: ragged attention mask + GDN fork + commit이 진짜 난제(vLLM도 어려움). 단 우리는 single-tree 자산 보유 → 슬롯 차원 추가.
- **payoff 불확실성**: 곱의 상한은 크나(B×accept), verify의 attention/GDN이 ragged라 matmul만큼 안 배칭될 수(attention/GDN 슬롯별 = 배칭 이득 부분적). Phase 1서 batched-verify가 실제 얼마 빠른지 measured(격리 오도 주의).
- **mism=0 불변**: 전 Phase 핵심 게이트. 깨지면 그 Phase서 정지·보고.
- **단계적**: 세션급 아님. Phase마다 de-risk+mism=0, overclaim 금지(honest-perf ⑧⑨⑩).

## Phase 2 결과 (완결 61e1e62 — W4A8 concat 곱 실현+측정)
> Phase 1은 mism=0 토대만(W4A8은 per-slot, 곱 미실현). Phase 2 = W4A8 concat 곱 실현.

### golden-debug 여정 (추측 금지, 헌법)
- Phase 2 첫 시도(SPEC_MMQ concat)서 **slot-B 발산(A=0/24, B=14~16/24)** — reasoning 4회 막힘.
- **격리 probe(tools/probe_mmq_rowdep)**: mmq(M=14)[slot-B rows] == mmq(M=7) solo = max|Δ|=0.0 → **mmq row-독립 증명**(곱 achievable, 벽=verify_batch 배선?).
- **in-context golden dump(실제 verify_batch xqn 버퍼, MMrow 첫 호출)**: concat-mmq[slot-B] == solo-mmq[slot-B] **max|d|=0.0**(입력 sum|xqs[A]-xqs[B]|≠0로 실제 다른 입력 확인). → **배선 3후보(quantize 커버리지/Y offset/solo-ref 전파) 전부 제거.**
- **진짜 원인**: matmul은 in-context서도 row-독립(정확). Phase2 발산은 **Nnodes≤8서 단독 spec=sb인데 concat=mmq를 강제한 커널 불일치(W4A8 borderline flip)** — 배선 버그가 아니었음. (초기 "fixable plumbing" 진단은 부정확.)

### fix (재설계 아님, 올바른 게이팅)
- `MMrow`/`MMrow_q`(models/qwen35_batch.cu): **Nnodes>8이면 concat mmq(M=B*Nnodes) 1콜.** Nnodes>8이면 단독 spec(S=Nnodes)도 자동 mmq → mmq-vs-mmq row-독립 bit-identical. Nnodes≤8은 per-slot sb 유지(Phase1 불변).

### 측정 (in-engine controlled A/B, 2×single vs batched 1콜, warmup 스킵)
| 모델 | N=K+W | single(M=N) | batched(M=2N) | **2×single/batched** |
|---|---|---|---|---|
| 0.8b W4A8 | 10 | 13.46 ms | 17.96 ms | **1.498×** |
| **27b W4A8** | 10 | 195.29 ms | 224.27 ms | **1.742×**(서빙 타깃, DRAM-bound라 큼) |
- **mism=0**: 0.8b·27b, W4A8+fp32, A·B 0/24 전 시나리오. **곱 mechanism 증명+측정 성공**(weight 1운반이 2슬롯 서빙).

### 정직 스코프
- **Nnodes>8서만 곱**(둘 다 mmq). 1.74×는 **verify-forward matmul 배칭(B=2)**, full e2e serving throughput 아님.
- **실용 서빙(작은 트리 ≤8 + 큰 B로 M=B*Nnodes>8)**은 단독 spec을 **canonical mmq로 정렬**해야 mism=0 — **Phase 3 핵심 결정**(sb vs mmq 중 spec의 정본 W4A8; 둘 다 fp32 golden 아닌 유효 argmax).
- probe/GDUMP 계측 제거, fix만 커밋(미커밋 원칙).
- **다음 = Phase 3**: 연속 배칭 통합(admission/slot+spec, B 가변) + canonical-kernel 결정으로 ≤8 트리 곱 확장.

## Phase 3 축1 결과 (완결 50c8d1c — concat-sb로 ≤8 트리 곱, canonical 결정)
> STEP C는 Nnodes>8만 곱. 실용 서빙 작은 트리(≤8) 곱 확장 = Phase 3 축1.

### 핵심 통찰 — sb도 weight 1적재 (곱)
- `gemv_w4a8_sb`: **grid=OUT warps(S 무관), 각 warp가 weight 행 1회 적재 후 S 컬럼 DP4A**(L2 상주 activation). 컬럼-독립.
- ∴ **concat-sb(M=B*Nnodes)** 1콜 = weight 1적재로 M 컬럼 서빙 vs per-slot sb(B콜) = weight B적재. **weight B→1 = 곱**(mmq와 같은 원천).
- **컬럼-독립 → concat-sb == per-slot sb bit-identical**(K-누적 순서 동일, SMAX만 다름). 단독 spec(2≤Nnodes≤8)도 sb → **canonical 자동 일치, mism=0, empty-row 낭비·flip 없음, 딜레마 없음.** sb SMAX=16 한계라 M≤16.

### fix + 측정
- `MMrow`/`MMrow_q`: 2≤Nnodes≤8 && M≤16 → concat-sb(M) 1콜. Nnodes==1(단독 mlp)·M>16 → per-slot.
- **측정(in-engine controlled): 27b W4A8 N=7(M=14) 곱 1.27×, 0.8b 1.24×.** mism=0: 0.8b N=7·N=4(SMAX8)·N=10(mmq 회귀)+fp32 A·B 0/24 PASS.

### canonical-kernel 결정 (de-risk 완료)
| M 범위 | 커널 | mism=0 | 곱 | 딜레마 |
|---|---|---|---|---|
| M≤16 (트리≤8, 저-중 B) | **concat-sb** | 단독 sb와 일치 | 1.24~1.27× | 없음 |
| M>16 (트리>8, or 큰 B) | **concat-mmq** | 단독 mmq와 일치 | 1.74×(M=20) | 트리>8이면 없음(단독도 mmq) |
- **B=2: 경계 정렬**(Nnodes≤8⟺M≤16→sb, Nnodes>8⟺M>16→mmq) → **모든 트리 크기 곱 완결, 딜레마 없음.**
- mmq 곱 > sb 곱(MMA가 batched compute amortize↑; sb는 S 컬럼 DP4A라 compute S 비례 → batched M=14가 single M=7의 1.57배 시간, 곱 희석). 단 sb는 단독 정렬 불필요.
- **가변 B의 진짜 딜레마**: 트리≤8인데 큰 B로 M>16 → sb 한계 초과 mmq 강제, 단독 spec은 sb → flip. 해법 (a)**adaptive**(sb≤16/mmq>16, 경계서 load-dependent-but-valid W4A8 argmax) or (b)**uniform mmq**(단독 mm+MMs 둘 다 mmq 정렬 + B=1 empty-row 페널티). MMQ_FORCE 실험(mm만 게이팅=mism 31)이 "모든 matmul 경로 정렬 필요" 확인. = **축2 설계 결정.**

### 축2/3 = major 빌드 (축1 세션 시점 미착수 → 아래서 완결)
- **축2 연속 배칭+spec**: ragged verify_batch + per-slot MTP draft/accept/promote + 가변 B admission = vLLM급, dedicated 세션으로 미룸 → 아래 "Phase 3 축2 결과"서 완결.

## Phase 3 축2 결과 — ragged spec 연속 배칭 (STEP A/B 완결 + C 측정)

### STEP A — ragged verify_batch (9699c8e)
- `qwen35_verify_batch`: uniform `int Nnodes` → **슬롯별 `const int *Nnodes_b`**. concat 행 = `off[b]+i`(off=prefix 누적합), M=ΣNnodes_b. per-slot 루프(KV copy/flash_attn/conv tree/GDN ksplit tree/va snapshot) 전부 off[b]·Nnodes_b[b] 인덱싱.
- **canonical 게이팅의 ragged 일반화**: 전 슬롯 같은 커널 family여야 단독 spec과 bit-identical — 전 슬롯 >8 → concat-mmq(M) / 전 슬롯 2~8 && M≤16 → concat-sb(M) / **혼합·M>16 소트리 → per-slot**(각 슬롯 자기 커널, mism=0 유지, 곱 부분 손실).
- test_batchspec2 슬롯별 트리 형상(K2/W2 argv). **게이트 전부 mism=0 PASS**: 0.8b 회귀 uniform 7(sb)·10(mmq) + ragged 7/5(concat-sb M=12)·10/12(concat-mmq M=22)·7/10(mixed per-slot)·fp32 7/5; **27b ragged 7/5·10/12도 PASS**.

### STEP B — spec-serving (tools/bench_serve_spec, 76e7413)
- bench_serve 프로토콜(Poisson/admit/slot 재사용/seed trace/warm-up 제외) 위에 decode 스텝 → **spec 사이클**: 슬롯별 MTP draft(tree K,W) → verify_batch(ragged concat, 활성 B_act 가변) → 슬롯별 accept/promote(**ragged accept**: plen 슬롯별 commit). 슬롯 상태 = per-slot DecodeState+MtpKV+VerifyArtifacts(수명=요청). admission 예산 가드(KV+MTP+VA).
- **★ISO 게이트(서빙 각 요청 출력 == 단독 spec bit-identical) 12/12 PASS × 3종**: W4A8 K6W4(concat-mmq)/fp32/W4A8 K4W3(소트리 per-slot 폴백). **스케줄링(B_act 요동·슬롯 재사용·도착 순서)이 출력에 영향 없음** — 서빙 spec 정확성 완성.

### STEP C — B-scaling 곱 + 가변 B 결정 + 서빙 A/B (measured)
- **verify 곱 B-scaling (0.8b W4A8, K6W4 Nn=10, 포화 λ, TIME 버킷, in-engine)**: 곱(B)=B·t(1)/t(B):
  B=2 **1.48×** / B=3 1.70× / B=4 **1.85×** / B=8 **2.25×** — **곱이 B로 커짐(원천 실증)**, 단 saturating.
- **가변 B 딜레마의 실전 답(데이터)**: 서빙 기본 트리 **Nn=10(>8) → 모든 B_act서 all_gt8=concat-mmq — flip 자체가 없음**(단독도 mmq, empty-row 없음). accept도 K6W4가 5.2~5.7/사이클로 우수. 소트리(≤8)는 per-slot 폴백이 자동 mism=0(곱 부분). ∴ ⒜adaptive/⒝uniform-mmq 선택 불필요 — **⒞ "트리>8 기본 + 소트리 자동 폴백"**이 flip 원천 제거+곱+mism=0 동시 달성.
- **0.8b e2e 정직(A/B, 같은 seed/Bmax=8)**: plain c-ch 503.4 tok/s(λ8) vs c-spec 112.4 — **plain 4.5× 우세**. 원인: 0.8b는 body가 싸서 **per-slot 순차 MTP draft(mtp_step마다 호스트 sync+V-logits argmax)가 사이클 지배**(tpot95 0.013 vs 0.093). "draft Phase 초반 순차"가 실측 병목 확정 — **draft 배칭/디바이스화 = Phase 4 lever**.
- **★draft lm_head 함정(27b accept 붕괴→fix)**: 첫 27b 측정 accept=1.00(퇴화, 1.8 tok/s) — draft logits를 E(embed)-as-lmhead로 계산(0.8b tied=정상, 27b untied=draft 붕괴). fix: mtp_step에 lmq/dq4 복원, uq!=0 기본 dq4=1(진짜 lm_head q4). 재게이트 0.8b ISO 12/12 PASS. **교훈: draft 품질 파라미터는 속도 도구서 필수 켬.**
- **27b B-scaling (dq4=1, K6W4, 포화 λ, verify TIME 버킷, in-engine)** — 곱(B)=B·t(1)/t(B), t(1)=195.7ms:
  | B | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|
  | 곱 | 1.73× | 2.29× | 2.73× | 3.07× | 3.41× | 3.61× | **3.83×** |
  **곱이 B로 크게 스케일**(0.8b B=8 2.25×보다 훨씬 큼 — 27b DRAM-bound라 weight 재사용 이득↑). accept 4.51(K6W4 > 챕터9 K4W3 2.27~2.86). e2e: 9.1(B1)→18.4(B4)→19.8 tok/s(Bmax=8, avgB 5.04).
- **★27b 서빙 e2e A/B (Bmax=8, 같은 seed/trace/λ=1,2,4, controlled)**:
  | λ | plain c-ch | c-spec | 판정 |
  |---|---|---|---|
  | 1 | 38.3 | 20.2 | plain 1.9× |
  | 2 | 40.3 | 20.3 | plain 2.0× |
  | 4 | **40.7** | **20.4** | **plain 2.0× 우세** |
  **정직: e2e 서빙 throughput은 spec이 plain에 2.0× 짐.** 병목 분해(측정): 사이클 = verify(M~76, ~0.41s) + **per-slot 순차 MTP draft(~7.6슬롯×~11 mtp_step, 호스트 sync 포함 ≈ 사이클의 ~75-80%)**. verify 곱 3.83×는 실증됐으나 **draft 순차가 e2e서 완전 희석**(tpot95 0.50s vs plain 0.18s). Amdahl: draft 배칭/디바이스화 시 사이클 ~2.3s→~0.65s → **~50 tok/s = plain(40.7) 추월권** — **Phase 4 확정 lever(추측 아님, 사이클 분해 측정)**.
- 27b 서빙 ISO(dq4=1) 미실행(0.8b ISO 12/12만) — 27b bit-identity는 test_batchspec2 27b 게이트(concat==solo)로 검증, 서빙 iso 27b는 Phase 4 시작 게이트로.

### 축2 종합 (정직)
- **완성**: ragged spec 연속 배칭 인프라(vLLM급 신규 3개 중 ragged verify+ragged accept 완성, 정확성 mism=0/iso 12/12) + 곱 mechanism B-scaling 실증(verify 3.83×@B8).
- **미달**: e2e 서빙 throughput(plain 2.0× 우세) — 원인은 batch+spec 결합이 아니라 **draft 파이프라인**(순차+호스트 sync+fp32 MTP body). **Phase 4 = draft 배칭**(슬롯 병렬 mtp_step 배치 forward + top-k 디바이스화)이 곱을 e2e로 옮기는 열쇠.

## Phase 4 — draft 배칭 (곱을 e2e로)

### STEP 0 — 시작 게이트
- **27b 서빙 ISO(dq4=1, 순차 draft) 8/8 bit-identical PASS** — 축2가 남긴 실 lm_head 경로 게이트 충족.

### STEP A+B — mtp_step_batch (슬롯 배치 draft)
- `mtp_step_batch`(bench_serve_spec): B슬롯 draft를 depth별 1 forward(M=B행)로. depth 루프는 순차 유지(본질), 슬롯 루프를 배치.
- **bit-identity 근거(설계로 확보)**: MTP body matmul은 전부 `gemv_f32`/`gemv_q4` — **단일 커널, S-무관, warp당 (s,o) row-독립**(launch_gemv_f32 dispatch는 IN 기준이라 S로 커널이 안 바뀜) → **sb/mmq flip 자체가 없음**(verify와 다른 지형, canonical 게이팅 불필요). rmsnorm/embedding/swiglu/add/gate row-독립. rope는 `rope_partial_pos`(per-row pos, verify_batch서 검증된 자산). attn/KV append는 per-slot 루프(순차와 동일 커널·kv.n·순서). top-k는 host 동일 코드(M×V logits 1회 D2H) → **sync B×K→K**.
- **catchup도 배치**: promote 후 depth j별 active 부분집합({b: a_b≥j})을 mtp_step_batch — ragged depth를 부분집합으로 처리.
- 순차 경로는 DRAFT_SEQ=1로 보존(A/B·게이트). STEP B(top-k 디바이스화)는 depth당 sync 1회가 남지만 tree 구성이 host라 현 구조에선 host top-k 유지(배칭 D2H 1회로 이미 sync 최소화) — 완전 디바이스 상주는 후속.
- **게이트: 0.8b 서빙 ISO(배칭 draft) 12/12 bit-identical PASS**(solo=순차 draft이므로 ISO가 곧 draft mism=0 게이트) + DRAFT_SEQ=1 회귀 12/12. 버그 1건(워크스페이스 8H를 6H로 산정 → OOB memcpy) 수정.
- 0.8b 속도: λ4 258.4 / λ8 240.9 tok/s(accept 5.55) — 순차 대비 대폭 개선(조건 상이라 참고).

### STEP C — 27b 판정 (measured)
- **27b 서빙 ISO(배칭 draft, dq4=1) 8/8 bit-identical PASS** — solo=순차 draft이므로 이것이 곧 27b draft mism=0 게이트. accept 4.50 불변.
- **★27b e2e A/B (Bmax=8, nreq=60, 같은 seed/trace, λ=1,2,4, controlled back-to-back)**:
  | λ | plain c-ch | c-spec(배칭 draft) | (축2 순차 draft) | 판정 |
  |---|---|---|---|---|
  | 1 | 38.3 | 20.6 | 20.2 | plain 1.9× |
  | 2 | 40.3 | 20.7 | 20.3 | plain 1.9× |
  | 4 | **40.7** | **20.8** | 20.4 | **plain 2.0× 우세** |
  **정직: 미달.** draft 배칭이 27b e2e를 거의 못 움직임(20.4→20.8, **+2%**) — Amdahl 예측(~50 tok/s)이 크게 빗나감. plain 재측정은 축2와 소수점까지 일치(재현성 OK). verify 곱은 3.79×@B8(t(1)=196.3ms) 재확인 — 곱 mechanism은 그대로.
- **사이클 재분해(CYC=1, 5구간 경계 deviceSync, λ4 Bmax=8 nreq=30, 77스텝 동일 trace, avgB 6.06)**:
  | 구간 | 배칭 draft | 순차 draft(DRAFT_SEQ=1) |
  |---|---|---|
  | draft | **37.0s (44.8%)** | **38.1s (45.2%)** |
  | verify | 27.3s (33.0%) | 27.2s (32.3%) |
  | accept·host | 3.0s (3.6%) | 19.0s (22.5%, catchup 포함 통짜) |
  | catchup(배치) | 15.4s (18.6%) | — |
  | 합 / tok/s | 82.6s / 21.1 | 84.3s / 20.7 |
- **★근본 원인(수치 정합으로 확정): draft matmul은 GEMV라 배칭 곱이 원천적으로 없다.**
  - 27b MTP 1 forward 운반 = fp32 body 1.70GB(fc 210+q 252+k/v 42+o 126+gate/up/down 1070MB) + lm_head q4 0.79GB ≈ **2.5GB → 하한 9.2ms@273GB/s**. 순차 draft 슬롯당 81ms(6 forward×13.6ms) — 대역폭 천장 부근.
  - 배칭 draft 스텝당 480ms ≈ **순차 슬롯당 81ms × avgB 6.06 = 494ms (오차 3%)** — M=B행인데 시간이 M배 그대로. `gemv_f32`/`gemv_q4`는 warp당 (s,o) **row-독립 = weight 재사용 0** → M행이어도 weight를 M번 운반. **배칭이 없앤 건 host sync뿐(-3%)**. verify서 곱이 난 건 mmq(MMA 타일)/sb(컬럼-독립 grid)가 **weight 1적재를 M행에 재사용**해서 — draft엔 그 커널이 안 붙어 있었던 것.
  - 축2의 "draft 75-80%(host sync 지배)" 분해도 과대였음 — 실측 draft 45%(그것도 대역폭), verify 33%, catchup 19%.
- **판정: Phase 4의 가설("draft 배칭=사이클 재균형")은 기각 — 배칭 인프라(mtp_step_batch, ISO PASS)는 완성됐으나 곱은 커널이 만든다.** 다음 lever(Phase 5 후보, Amdahl):
  1. **draft matmul을 M-재사용 커널로**(verify의 concat-mmq/sb 원리를 mtp_step_batch matmul에) — draft 37→~10s, catchup 15→~4s ⇒ 합 ~44s ≈ **40 tok/s(plain 동급)**.
  2. **MTP body fp32→W4A8**(1.70→~0.43GB, 운반 4×↓) — 1과 결합 시 합 ~35s ⇒ **~50 tok/s(추월권)**. 27b body가 이미 W4A8이므로 MTP만 fp32로 남은 상태(추출 파이프라인 확장).
  3. top-k 디바이스화·D2H 축소는 잔여(~10-15%) — 1·2 이후에.

### Phase 4 종합 (정직)
- **완성**: draft 슬롯 배치 인프라(mtp_step_batch + 배치 catchup + DRAFT_SEQ 폴백) — 정확성 완전(0.8b ISO 12/12 배칭·순차 회귀, **27b ISO 8/8**, accept 4.50 불변). 0.8b는 대폭 개선(λ4 258.4 tok/s — 작은 모델은 sync가 실제 병목이었음).
- **미달**: 27b e2e 20.4→20.8(+2%) — plain 40.7의 2.0× 열세 유지. **교훈: "배칭=곱"이 아니다 — 곱은 weight-재사용 커널(mmq/sb)이 만들고, GEMV 경로는 M행이어도 bytes가 M배.** 사이클 분해를 구간 계측(CYC) 없이 추정으로 말한 축2의 75-80%도 부정확했음(실측 draft 45%).
- **다음(챕터10 Phase 5)**: draft matmul M-재사용 커널화 + MTP body W4A8 — 둘 다 "토큰당 bytes ↓"라는 프로젝트 제1원리로의 회귀. Amdahl 상 ~50 tok/s(추월권), 단 이번처럼 실측 전까진 예측일 뿐.

## Phase 5 — draft 곱 실현 (① M-재사용 커널 + ② MTP W4)

### STEP A — ① draft matmul GEMV→gemm_batch (M-재사용)
- draft matmul(fc/qkvo/SwiGLU 8개 + lm_head)을 `gemm_batch_f32`/`gemm_batch_q4`(챕터3 자산 — weight row 1로드→B행 재사용)로 교체. `draft_mm_f32/q4` 디스패처, `DRAFT_GEMV=1` 폴백(P4 gemv 경로 재현).
- **canonical 설계**: gemm_batch는 행별 acc가 B-template과 무관하게 같은 i 순서로 누적 → **같은 행은 B=1(순차 mtp_step)과 B=Mr(배칭) bit-identical** — 순차·배칭 둘 다 gemm_batch로 통일하면 ISO가 커널 구조로 성립. 기존 gemv(float4 mo)와는 합산 순서가 달라 draft 토큰 flip 가능(출력은 verify가 보장 — accept로 관찰).
- **Mr round-up 패딩**: 템플릿 B∈{1,2,4,8,16} → M=6은 Mr=8로 gemm만 패딩 호출(나머지 커널·D2H는 M — row-독립이라 패딩 행 무해). lm_head가 caller 버퍼(Down)를 Mr행 읽으므로 g_bown은 Mr 할당.
- **게이트**: 0.8b ISO 12/12(새 canonical, 배칭==순차) + DRAFT_SEQ 회귀 12/12 + DRAFT_GEMV(P4 경로) 회귀 12/12, **27b ISO 8/8(dq4=1)**. 0.8b accept 5.52(P4 5.55 근접 — flip 영향 미미).
- 측정 결과 아래 STEP C 표.

### STEP A 측정 (27b, controlled back-to-back, Bmax=8 nreq=60 같은 seed)
- **e2e**: plain 38.2/40.2/40.6 (P4와 ±0.1 재현) vs **c-spec ① 36.8/37.6/37.9 tok/s**(λ=1/2/4) — P4 20.8에서 **+82%**, 예측(~40)과 부합. **plain 대비 0.93×(아직 미달)**. accept **4.50 불변**(커널 교체에도 — flip이 accept에 무영향). verify 곱 3.80×@B8 유지.
- **CYC 재분해(λ4, 77스텝 동일 trace)**: draft 37.0→**8.9s(4.2×↓)**, catchup 15.4→**4.5s(3.4×↓)**, verify 27.2s(이제 62.5%로 지배), 합 82.6→**43.5s(1.9×↓)** — **"커널이 곱을 만든다" 실측 입증**(gemm_batch=weight 1로드→M행 상각).
- 잔여 draft+catchup 13.4s의 지배분 = fp32 MTP body 운반(forward당 하한 9.2ms) → ②의 몫.

### STEP B — ② MTP body q4 (bytes 4×↓)
- `MTP_Q4=1`: load_mtp에서 host 양자화(본체 추출과 동일 스킴 QBLK=32/scale=absmax/7/2per-byte) → big 8개 q4만 상주(fp32 1.70GB 미탑재→~0.43GB). matmul은 `gemm_batch_q4`(fp32 act × q4 weight — act 양자화 오차 없음, accept 위험 최소). lm_head는 기존 q4(dq4).
- **accept 게이트: 4.50 → 4.48(-0.4%, 사실상 유지 — GO).** q4 draft의 추측 품질 손실이 accept에 거의 무영향(fp32 act 유지 덕).
- **ISO(①+②)**: 0.8b 12/12 + **27b 8/8 bit-identical PASS**(순차·배칭 같은 q4 canonical).

### STEP C — 결합 e2e 판정 (measured, controlled back-to-back)
- **★27b e2e 서빙 A/B (Bmax=8, nreq=60, 같은 seed/trace, λ=1,2,4)**:
  | λ | plain c-ch | c-spec ①만 | c-spec ①+② | ①+② vs plain |
  |---|---|---|---|---|
  | 1 | 38.2 | 36.8 | **40.0** | **1.05×** |
  | 2 | 40.2 | 37.6 | **43.6** | **1.08×** |
  | 4 | 40.6 | 37.9 | **44.1** | **1.09×** |
  **판정: 27b 연속배칭 서빙에서 spec(배칭 draft + MTP q4)이 plain을 전 λ에서 추월 — 챕터10의 곱(B×accept) 역전이 e2e에 처음 실현.** 조건 명시(overclaim 방지): Qwen3.5-27B W4A8 본체, K6W4 트리(Nn=10, concat-mmq), MTP_Q4=1(accept 4.48), in-engine plain 대비(llama.cpp 서빙과의 비교 아님 — 챕터9 apples-to-oranges). ~50 예측에는 미달(44.1) — verify가 지배구간이 되며 Amdahl 상한 도달. per-request tpot는 여전히 plain 우세(0.239 vs 0.179s — spec은 사이클이 커서 개별 지연↑, 총량↑).
- **CYC 재분해(λ4, 77스텝)**: draft 6.6s(16.9%) / verify 27.5s(**70.7% — 지배**) / accept·host 3.0s / catchup 1.8s — 사이클합 **38.8s**(P4 82.6 → ① 43.5 → ①+② 38.8). drafting 경로(draft+catchup)는 P4 52.4s → **8.4s(6.2×↓)**.
- verify 곱 3.71×@B8 유지(t(1)=192.2ms).

### Phase 5 종합
- **예측 vs 실측**: ① ~40 예측 → 37.9 실측(부합), ①+② ~50 예측 → **44.1 실측(방향 맞음, 크기 미달 — verify 70.7% 지배로 Amdahl 상한)**. Phase 4 교훈(단계 실측) 준수 — 예측 연쇄 없이 각 lever 기여 분리 측정.
- **lever 기여**: ① M-재사용 커널(gemm_batch) = 20.8→37.9(+82%), ② MTP q4(bytes 4×↓) = 37.9→44.1(+16%). 곱은 커널이 만들고, bytes가 남은 갭을 채웠다 — 프로젝트 제1원리(bandwidth ÷ bytes) 그대로.
- **다음 lever(후속 후보)**: verify가 70.7% — verify 자체는 이미 concat-mmq 곱이므로 남은 건 (a)트리 형상 최적화(K/W vs accept 트레이드), (b)verify attention/GDN의 per-slot 루프 배칭, (c)top-k 디바이스화(accept·host 7.8%). 모두 예측일 뿐 — 실측이 판정.

## Phase 6 — 트리 형상 K/W 재스윕 + verify 내부 분해

### 재스윕 정당성 (전제 변화)
챕터9 "accept W-flat 3.57(draft-limited)"은 옛 draft(품질≈3.57, GEMV 비용) 전제. 지금은 draft 품질↑(lm_head q4 → 4.48)+비용 4.2×↓(gemm_batch) — 균형점 이동 가능해 재측정(honest-perf ⑩ 예외). 형상은 argv config(커널 무변경).

### STEP A — K/W 스크리닝 (27b MTP_Q4=1, λ4, nreq=40, 같은 seed=42, controlled 형상 비교)
| K/W | Nn | tok/s | tpot(p95) | accept |
|---|---|---|---|---|
| 6/4(현행) | 10 | 44.6 | 0.236 | 4.44 |
| **6/3** | 9 | **45.6** | 0.231 | 4.37 |
| 6/5 | 11 | 42.7 | 0.246 | 4.43 |
| 5/4 | 9 | 44.9 | 0.224 | 4.20 |
| 5/5 | 10 | 43.5 | 0.232 | 4.22 |
| 4/5 | 9 | 42.8 | 0.229 | 3.78 |
- **패턴**: **K(depth)가 accept 드라이버**(K4→5→6: 3.78→4.20→4.37, Nn=9 고정 비교), **W(width)는 accept flat**(W3→5: 4.37→4.43·4.44 — 챕터9 W-flat이 새 전제에서도 재확인) — W↑는 verify 비용(Nn)만 늘림. ∴ 최적 = 최대 K + 최소 W(단 Nn>8 canonical 유지 → W=3). Nn≤8은 sb/mmq 가변 B 딜레마 복귀(per-slot 폴백=곱 손실)라 미시험.
- **K6W3 full 확정(nreq=60, controlled)**: ISO 8/8 PASS(형상도 mism=0). e2e **47.2 / 44.5 / 44.8 tok/s**(λ=1/2/4), accept 4.41, tpot(p95) 0.231(K6W4 0.239서 소폭 개선, plain 0.179 대비 여전히 열세).
  | λ | plain | K6W4(P5) | **K6W3(P6)** | vs plain |
  |---|---|---|---|---|
  | 1 | 38.2 | 40.0 | **47.2** | **1.24×** |
  | 2 | 40.2 | 43.6 | **44.5** | 1.11× |
  | 4 | 40.6 | 44.1 | **44.8** | **1.10×** |
  **44.1 → 44.8(λ4, +1.6%), λ1은 +18%(40.0→47.2 — 저부하에서 작은 트리의 사이클 단축 효과가 큼).** CYC(λ4): draft 17.7% / **verify 70.2%(26.6s)** / accept·host 7.4% / catchup 4.8% — 사이클 38.8→37.9s, verify 지배 유지.

### STEP B — verify 내부 분해 (nsys cuda_gpu_kern_sum, K6W4 λ4 nreq=12 — draft·verify·prefill 혼입 주의, avgB 3.98로 작음)
| 커널군 | GPU 시간 비중 | 성격 |
|---|---|---|
| **gemm_mmq(verify concat matmul+prefill)** | **59.2%** | 이미 곱 실현(concat-mmq) |
| gemm_batch_q4(draft+catchup, B1~8 합) | 25.8% | P5 ①② 완료 경로(CYC 22.5%와 정합) |
| **GDN(gdn_step_ksplit_tree+conv1d tree)** | **9.4%** | per-slot 루프 — (b) 대상 |
| **attention(flash_attn2bs p1+p2)** | **1.1%** | per-slot 루프 — (b) 대상 |
| 기타(rmsnorm/quantize/elementwise 등) | ~4.5% | row-독립 |
- **★(b) payoff 상한 = per-slot attn+GDN 합 ~10.5%** — "verify 70%의 대부분이 per-slot 루프"라는 가설은 **기각**: verify의 대부분은 이미 곱이 실현된 concat-mmq matmul(59.2%). GDN tree 커널은 평균 167μs/launch(16k회) — launch 오버헤드(~5μs)가 아니라 **슬롯별 bf16 state 운반이 지배로 추정**(state는 슬롯 고유라 배칭해도 운반 총량 불변 → 배칭 이득은 병렬성/launch 일부만, 예측).

### STEP C — 판정
- **최적 형상 = K6W3(Nn=9)로 확정** — 서빙 기본값 변경 권고(config, 커널 무변경). 27b e2e 최고점 **44.8 tok/s(λ4, plain 1.10×) / 47.2(λ1, 1.24×)**. tpot 트레이드: 0.231 vs plain 0.179 — throughput 우위·지연 열세 유지.
- **재스윕의 수확**: 챕터9 "W-flat"은 새 전제(accept 4.4대)에서도 유효 — 단 결론이 뒤집힘: W는 accept를 못 늘리므로 **W를 줄이는 게 이득**(verify ∝ Nn). K는 accept 드라이버(K4→6: 3.78→4.37) — depth 유지가 필수.
- **(b) verify attn/GDN 배칭: 낮은 우선순위 권고(사실상 기각)** — payoff 상한 ~10%(GPU 시간), 그 지배분(GDN 9%)은 state 대역폭 bound로 배칭 이득 제한적(예측). 남은 큰 벽은 **concat-mmq 커널 자체(59.2%)** — 이는 챕터7에서 micro-lever 천장 판정된 영역(FP4 MMA가 toolkit-blocked 해제 시 최대 lever).
- Phase 6은 config 변경뿐(커널 무변경) — 스윕·계측 결과만 doc/메모리 기록.

## Phase 7 — 극한 config 스윕: Bmax↑ + K연장 + λ포화 + 교차 격자

### STEP 0 — 메모리 가드 정확화 (코드 변경은 이것 하나)
- B12/B16 진입 시 admission 가드가 오차단: 예산 18.66/24.88GB vs "0.7·free 17.5GB". 원인 = **GB10 unified에서 `cudaMemGetInfo` free가 리눅스 페이지캐시(회수 가능 ~70GB)를 가용으로 안 쳐 과소보고**(실제 MemAvailable 113GB).
- 수정: `/proc/meminfo MemAvailable`을 병용해 **max(cudaFree, MemAvailable)** 사용 — 가드 완화가 아니라 **정확화**(회수 가능 메모리는 실제로 할당 가능, oversubscribe 판정 기준은 동일 0.7 마진 유지). 수정 후 B16 예산 24.88~27.32GB vs 0.7·free ~69GB — OK, 여유 큼.

### STEP 1 — 축별 스윕 (27b MTP_Q4=1, controlled 같은 seed=42)
**①Bmax B-scale (K6W3): spec은 B↑로 계속 오르고 plain은 B8에서 포화** — 곱(concat-mmq M=B×Nn, draft gemm_batch M=B)이 B를 계속 흡수:
| Bmax | plain λ2/λ4 | spec λ2/λ4 | spec/plain(λ4, 같은 Bmax) | spec tpot(λ4) |
|---|---|---|---|---|
| 8 (P6) | 40.2 / 40.6 | 44.5 / **44.8** | 1.10× | 0.231 |
| 12 | 39.1 / 39.8 | 47.3 / **48.1** | **1.21×** | 0.321 |
| 16 | 39.2 / 40.4 | 48.2 / **48.7** | **1.21×** | 0.424 |
- plain은 B12/16이 B8보다 오히려 낮음(39.8/40.4 vs 40.6 — 포화+스케줄 오버헤드). spec은 B↑가 e2e로 계속 나옴(44.8→48.1→48.7). **단 B12→16은 +1.2%로 체감이고 tpot 0.32→0.42로 악화 — 실용 스윗스팟 = B12.**
- **②K 연장(B8, λ4, nreq40): K7 44.7(accept 4.59) / K8 43.5(accept 4.83)** — accept는 계속 오르나(4.41→4.83) verify Nn(10/11) 비용이 이겨서 **throughput 무득**(K6 44.8이 최고). MTP 연쇄 깊이의 accept 증분(K6→8: +0.42)이 노드당 verify 비용 증분을 못 이기는 구조.
- CYC 사이클 분해(λ4): B8 draft 17.7%/verify 70.2% → **B16 draft 17.0%/verify 69.9%** — **B↑에도 비중 구조 불변**(draft·verify 모두 곱 커널이라 함께 스케일). verify(그 중 concat-mmq)가 여전히 지배 → 다음 lever = FP4 MMA(toolkit-blocked) 유지.

### STEP 2 — 교차 격자 (Bmax×K, W3, λ4): 상호작용 없음
| | K6 | K7 | K8 |
|---|---|---|---|
| B8 | **44.8** | 44.7 | 43.5 |
| B12 | **48.1** | 47.7 | — |
| B16 | **48.7** | 48.2 | — |
- 가설("B↑ → verify 상각 → 최적 K 연장")은 **기각**: 모든 B에서 K6 우세, 순위 이동 없음. accept 증분(4.41→4.59)이 Nn 9→10 verify 비용을 못 이기는 관계가 B와 독립 — verify가 이미 곱(concat-mmq)이라 B가 노드당 비용을 바꾸지 않기 때문. (K8은 B8에서 -1.3 커서 B12/16 교차 생략.)

### STEP 3 — λ 포화 + 정직 좌표
- **λ 포화 곡선(B12 K6W3, nreq60, tpot 병기)** — spec·plain 모두 λ4~8에서 포화, **용량: spec ≈48.6 vs plain ≈40.1 = 1.21×**(spec이 곱으로 용량 자체가 큼):
  | λ | plain B12 (tpot) | spec B12 K6W3 (tpot) |
  |---|---|---|
  | 2 | 39.1 (0.245) | 47.3 (0.301) |
  | 4 | 39.8 (0.248) | 48.1 (0.321) |
  | 8 | 40.0 (0.251) | 48.5 (0.322) |
  | 12 | **40.1** (0.251) | **48.6** (0.321) |
- **챕터10 최종 좌표**: 27b spec **48.1(B12 λ4) / 48.6(포화 용량)** vs 같은 Bmax plain 39.8/40.1 → **1.21×**. B16은 48.7로 최고점이나 tpot 0.42로 지연 악화 커서 참고점(B12 권고). tpot는 spec 0.32 vs plain 0.25 — **throughput·용량 우위, 지연 열세**(정직 병기).
- 무득 축(정직): plain Bmax↑(B8 포화, B12/16 오히려 -1~2%), K7/8(모든 B에서 무득 — accept↑가 verify Nn 비용에 짐), B12→16(+1.2% 체감·tpot 0.32→0.42 악화).
- ISO 스팟체크(B12 K6W3, nreq12 λ4): **12/12 요청 bit-identical PASS** — Bmax·형상은 config지만 게이트 재확인 완료(스케줄링 출력 영향 0 유지).
- Phase 7로 **config 축(B/K/W/λ)은 소진** — 남은 lever는 커널 축(FP4 MMA toolkit 해제 대기, GDN state 운반 등)뿐.
