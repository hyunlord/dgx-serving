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

## Phase 1 상세 스코프 (다음 세션 첫 빌드)
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
