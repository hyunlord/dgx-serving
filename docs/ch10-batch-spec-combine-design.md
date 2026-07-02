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

### 축2/3 = major 빌드 (미착수, STOP-report)
- **축2 연속 배칭+spec**: ragged verify_batch(슬롯별 다른 Nnodes — 현재 uniform Nnodes 가정) + per-slot MTP draft/accept/promote + 가변 B admission. **vLLM급 ragged speculative continuous batching = dedicated 세션.**
- **축3 곱→e2e throughput**: 축2 의존(연속 배칭 없이 서빙 throughput 측정 불가).
- 긴 세션 강행 = mism=0 미검증 위험 → 보고(헌법). **다음 세션: ragged verify_batch(Nnodes_b[]+prefix-offset, per-slot 루프 기존 자산) → bench_serve spec 사이클 → 곱 B-scaling(Nnodes>8로 mism=0 유지 B=2,4,8 — 곱이 B로 커지나) → e2e Poisson(Phase 4).**
