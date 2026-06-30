# 챕터9 tree-spec STEP 4 — 풀 accept 루프 + e2e (payoff)

> 베이스 11805f1(STEP 1~3: GDN tree-fork + tree attn + 1-forward 통합, bit-identical + amortize 안 깨짐).
> STEP 4 = tree draft + longest-greedy accept + promote + e2e. **남은 변수 = 실현 accept**(draft 분포가 천장 당기나).

## 정확성 — tree-spec 생성 == greedy bit-identical (mism=0), 최종 게이트 ✅

tree가 출력을 바꾸지 않음(accept는 verify가 body-greedy로 확인한 토큰만 수용). 0.8B·27B, fp32·W4A8 전부 **mism=0**.
- `promote_tree_path`(qwen35_decode.cu): accept된 scattered path 승급 = KV slot compaction + GDN snap[path末] + conv ctx along path.
  격리 게이트 `test_treepromote`: tree verify[R,C0,C1,GC]+promote([0,1,3]) == linear[R,C0,GC] committed state **0.000e+00**(fp32·W4A8).
- e2e `test_treespec_e2e`: tree-spec 생성 == plain greedy **mism=0**(0.8B·27B, fp32·W4A8) + 결정성 3회 동일.

## 하이브리드 tree (추가 MTP forward 0)

root(cur) + MTP chain(d1..dK, top-1) + **각/첫 위치 top-W 대안(leaf)**. 대안은 그 step logits의 2nd~W번째 argmax →
**추가 MTP forward 없음**(공짜). chain을 strictly dominate: body가 +1서 #2/#3 고르면 chain은 0 accept, tree는 그 alt 복구.
W=1이면 chain spec과 동일(내장 sanity, accept 일치 확인). longest-greedy accept: root부터 verify-argmax 일치 child 최장 path.

## 측정 (27B W4A8, N=48, 실현 accept — 정직)

| K(chain) | W(width) | nodes | accept | alt_hits | t/s | mism |
|----------|----------|-------|--------|----------|-----|------|
| plain greedy | | | — | | 10.1 | — |
| 4 | 1 (=chain spec) | 5 | 1.88 | 0 | 14.9 | 0 |
| **4** | **3** | 7 | 2.27 | 3 | **15.9** ★t/s최고 | 0 |
| 4 | 4 | 8 | 2.27 | 3 | 15.3 | 0 |
| 5 | 1 (순수 깊이) | 6 | 2.31 | 0 | 14.2 | 0 |
| 5 | 3 | 8 | 2.79 | 2 | 15.1 | 0 |
| 6 | 2 | 8 | **2.86** ★accept최고 | 2 | 14.3 | 0 |
| 7 | 1 (순수 깊이) | 8 | 2.67 | 0 | 12.8 | 0 |
| 3 | 5 (wide-shallow) | 8 | 2.19 | 2 | 15.3 | 0 |
| 8 | 1 (>8 MMQ) | **9** | 2.73 | 0 | **9.8** ⚠<plain | 0 |

(t/s는 이 하베스트의 host 오버헤드[per-cycle malloc·clock] 포함 — plain 10.1은 최적 루프 12.1보다 낮음. 상대 비교가 신호.)

### 발견
1. **tree가 chain 추월**: K=4 W=3 = t/s 15.9·accept 2.27 vs chain K=4 W=1 = 14.9·1.88 (**+7% t/s, +21% accept, mism=0**).
2. **width가 depth보다 t/s-효율적** (같은 nodes=8): K=6 W=2(accept 2.86) > K=7 W=1(2.67). **alts는 공짜**(기존 logits top-k)지만
   chain 깊이는 step마다 MTP forward 1회 비용(K=7 draft 37% vs K=4 27%). 순수 깊이는 t/s도 낮음(K=5 W=1=14.2 < K=4 W=1=14.9,
   accept↑여도 draft↑). → 얕은 chain + width가 t/s 최적(K=4 W=3).
2b. **>8 MMQ 함정 e2e 실증**: K=8 W=1(nodes=9)는 accept 2.73여도 **t/s 9.8 < plain 10.1** — verify Sq=9가 MMQ 반대역폭(STEP 3.3
   예측)으로 붕괴. **tree ≤8 절대 제약 확인**(>8 가려면 gemv_w4a8_sb 확장 선행). STEP 3.3 amortize de-risk가 실제 e2e서 정확했음.
3. **accept 1.88→2.86(+52%)**, 단 **de-risk 천장(per-position top-4 0.949 → 누적 ~4-5)엔 미달**. 이유(정직):
   하이브리드는 leaf-alt로 *한 번의* miss만 복구(더 깊이 fork하려면 alt 확장=MTP KV forking 필요), + 단일 greedy-path,
   + draft 품질. 천장까지 당기려면 multi-level forking tree(다음 정교화).
4. **t/s gain이 accept만큼 크지 않음**(14.9→15.9, +7%): MTP draft+catch-up 오버헤드가 cycle의 ~44%로 커짐(verify는
   amortize되나 MTP head 반복이 새 병목). accept→t/s 직결엔 **MTP 오버헤드 감축**(q4 MTP weight, catch-up 축소)이 다음 lever.

## 정직 좌표 ([[honest-perf-coordinates-not-overtake]])
- tree-spec **정확성 완성**(mism=0, 27B). chain 대비 accept +52%·t/s +7%(이 하베스트). llama 13.7 추월 유지(상대).
- 실현 accept 2.86 < 천장 ~4-5 — 하이브리드 한계(정직). full forking tree + MTP 오버헤드 감축이 추가 lever.
- **DFlash식 큰 speculation 이득(accept 5+)은 multi-level forking tree 필요** — 이번 STEP 4는 단일-head 하이브리드의 payoff.

## 자산 / 다음
- `promote_tree_path`(scattered path commit), `test_treespec_e2e`(하이브리드 tree e2e), `test_treepromote`(promote 격리).
- 다음 lever: ① multi-level forking tree(alt 확장, MTP KV fork — accept↑) ② MTP head q4/오버헤드 감축(accept→t/s 직결)
  ③ verify Sq>8 SB확장(더 큰 tree).
