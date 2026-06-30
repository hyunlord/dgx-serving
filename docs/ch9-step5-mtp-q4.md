# 챕터9 tree-spec STEP 5 — MTP 오버헤드 감축 (accept→t/s 전환)

> 베이스 098cb58(STEP 4: tree-spec 완성, mism=0, tree accept 2.86/t/s 15.9). 문제: accept +52%인데 t/s +7%만 —
> MTP draft+catch-up(~39%)이 accept를 t/s로 못 전환. STEP 5 = MTP 오버헤드 감축으로 **이미 올린 accept를 t/s로**.
> (accept 천장 올리는 게 아님 — full forking 별도. 이번은 전환 효율.)

## STEP 5.1 — 오버헤드 분해 (측정 먼저)

STEP 4 harness가 draft/catchup 분리 측정: **draft 26~37% >> catchup 12~14%** (draft = K MTP forward에 비례). 44%의
대부분이 draft. draft 내부 = GPU(matmul 대역폭) vs host(mtp_step의 호출당 ~48 cudaMalloc/free + argmax/sort).

**A/B 측정으로 분해**: mtp_step을 persistent scratch(malloc 제거, body 패턴)로 → **27B draft 27%→26%, t/s 15.9 불변**.
→ **malloc은 27B draft 병목 아님. draft는 GPU-bound(matmul 대역폭).** (0.8b는 malloc 비중 큼 — 모델 작아.)
27B MTP body fp32 = **1.72GB/forward**(lm_head q4 636MB보다 큼) = draft 지배 비용. **처방 = MTP body q4.** (추측 아닌 측정.)

## STEP 5.2 — 처방: MTP head q4

MTP body 8개 matmul(fc, q/k/v/o_proj, gate/up/down_proj)을 런타임 q4 양자화(`q4up` = extract_weights.quantize_int4
복제: QBLK=32, scale=maxabs/7, q=clip(round(w/s)+8,0,15), pack 2/byte — body q4와 동일 스킴, gemv_q4 포맷 일치).
norms/lm_head는 그대로(lm_head는 이미 dq4 q4). 1.72GB→215MB/forward.

**정확성:** q4 MTP는 draft *품질*만 약간 영향(후보 제안). **mism=0 불변**(verify가 body-greedy로 보장 — draft가 뭐든).

## STEP 5.3 — e2e 측정 (27B W4A8, N=48)

| 설정 | mtp_q4 | accept | t/s | draft% | catchup% | mism |
|------|--------|--------|-----|--------|----------|------|
| K=4 W=3 | 0 | 2.27 | 15.9 | 26 | 13 | 0 |
| **K=4 W=3** | **1** | 2.25 | **18.8** (+18%) | 19 | 6 | 0 |
| K=5 W=3 | 0 | 2.79 | 15.1 | 30 | 14 | 0 |
| K=5 W=3 | 1 | 2.53 | 18.3 | 22 | 6 | 0 |

**결과:**
1. **MTP q4 = t/s +18%** (15.9→18.8, K=4 W=3). draft 26%→19% + **catchup 13%→6%**(catch-up도 mtp_step=q4로 싸짐).
   MTP 오버헤드 합 39%→25%. verify(amortize)가 75%로 지배 = 올바른 구조(병목이 amortize된 verify로).
2. **accept 거의 불변** (2.27→2.25, K=4 W=3) — 27B MTP head는 q4에 robust(모델 큼). (0.8b는 3.67→3.10로 더 떨어짐 — 소형 모델.)
   K=5는 2.79→2.53(약간↓)이나 t/s는 여전히↑(15.1→18.3). **draft 품질 약간↓를 속도가 상쇄 = 순이득.**
3. **mism=0 불변** — q4 draft여도 verify가 정확성 보장.

부가: persistent scratch(malloc 제거)는 27B엔 중립이나 0.8b draft↓·메모리↓, 무해 — 유지.

## 정직 좌표 ([[honest-perf-coordinates-not-overtake]])
- STEP 5 = **accept→t/s 전환 효율** 개선(MTP 오버헤드 39%→25%): tree t/s 15.9→18.8 (+18%, 같은 accept).
- t/s는 이 e2e 하베스트 host 오버헤드 포함(plain 10.1, 최적 루프 12.1 대비 낮음) — **상대 비교가 신호**(mq4 +18%).
- **accept 천장(~4-5)은 여전히 별개**(full multi-level forking tree — 미구현). 이번은 전환 효율, accept↑ 아님(정직 구분).
- 누적: chain spec(STEP D 16.9) → tree(STEP 4) → +MTP q4(STEP 5). decode 추월 폭 확대, mism=0 유지.

## 자산 / 다음 lever
- `q4up`(런타임 q4 양자화, test_treespec_e2e.cu), MTP body q4 경로(mq4 플래그), persistent MTP scratch.
- 남은 lever: ① full multi-level forking tree(accept 천장↑) ② verify Sq>8 SB확장(더 큰 tree) ③ MTP catch-up 추가 축소.
