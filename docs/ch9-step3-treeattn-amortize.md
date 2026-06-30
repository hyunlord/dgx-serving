# 챕터9 tree-spec STEP 3 — tree attn mask + 1-forward verify 통합 + amortize 속도 de-risk

> 커밋 11805f1. 베이스 e0d0931(GDN tree-fork STEP1·2 완성). STEP 1·2가 GDN의 두 과거-의존 상태
> (recurrence S, conv1d window)를 tree-fork했고, STEP 3은 full-attn 16층에 tree mask를 얹어 **1-forward
> tree verify를 완성**하고 — 사용자 지적대로 — **STEP 4 풀빌드 전에 amortize 속도를 먼저 측정**한다.

## 왜 속도를 STEP 4 전에 보나 (사용자 지적, 챕터8 교훈)

STEP 1~3은 전부 정확성(bit-identical)만 게이트했다. payoff(accept↑)는 STEP 4(accept 루프) 완성 시 드러나지만,
**STEP 4에서 한 번에 e2e를 재면, 안 나올 때 "STEP 1~3 다 했는데 느리네"** 가 된다 — 챕터8 MTP가 통합 후
negative(verify Sq=K가 MMQ 128-tile로 79 GB/s 반토막)였던 것과 동형. 그래서 STEP 3 끝에 **작은 tree로
1-forward verify가 실제 weight 1적재 amortize를 주는지** 먼저 확인한다.

## STEP 3.1 — tree attention mask

`kernels/attention_flash.cuh` flash_attn2bs_p1_kernel에 `const unsigned *anc`(null=기존 causal):

```
bool att = anc ? ((gk<base) || ((gk-base)<=gq && ((anc[gq]>>(gk-base))&1u))) : (gk<=qpos);
```

- `anc[gq]` = tree노드 gq의 **조상 비트마스크**(비트 i = tree노드 i가 gq의 조상-or-self).
- prefix(`gk<base`)는 항상 attend, tree노드(`gk>=base`)는 anc 비트만 → 형제/비조상 mask.
- chain이면 anc[gq]=(1<<(gq+1))-1 = causal과 동일(bit-identical). B 커널(MMA+KV split) 그대로, mask만 얹음.

**rope-by-depth**: `kernels/rope.cuh` `rope_partial_pos`(노드별 위치 `pos[s]=base+depth`). tree 형제는 같은
depth(=같은 위치)지 layout slot이 아니다 — branch 노드 정확성의 전제(chain은 depth==slot이라 기존과 동일).

## STEP 3.2 — 1-forward 통합 배선 + 정확성 게이트

`qwen35_decode`에 `d_parent`(GDN/conv fork), `d_anc`(attn mask), `d_pos`(rope) 추가(모두 기본 null=기존 linear).
tree 모드면 4곳 분기: conv→tree(STEP2), gdn_step→tree(STEP1), attn→anc, rope→pos. (D≠128 tree는 abort — 조용한 오답 금지.)

`tests/test_treeverify`(0.8B, prefill 후 commit=0 verify, st 불변 재사용):
- **① chain tree([-1,0,1], anc=causal, pos=base+slot) == linear forward [R,C0,GC0]** → 전체 배선이 chain서 선형과 동일.
- **② branch tree([-1,0,0]: C0,C1=R의 두 자식) node2(C1, 조상 R→C1) == linear[R,C1] node1** → 분기 노드가
  조상-path만 fork/attend/rope.
- 결과: **fp32·W4A8 둘 다 0.000e+00**(W4A8도 같은 양자화 weight·커널이라 bit-identical) + **결정성 30회 고유1**.

`tests/test_treeattn`(커널 단위): ① chain anc==null causal 0.000e+00 ② branch node 조상-only attend 0.000e+00.

## STEP 3.3 — amortize 속도 de-risk (27B W4A8, decode 대역폭-bound 실모델)

`tests/test_treeamort`: greedy 1-tok vs tree verify Sq=K(SB) vs linear verify Sq=K. (GB10 ncu dram__bytes n/a →
시간 비율 + 구조 + bit-exact로 amortize 확정.)

| Sq | greedy 1-tok 대비 | amortize (K·t₁/t_tree) | tree/linear | 경로 |
|----|------------------|------------------------|-------------|------|
| 1  | 96.47 ms (기준)  | —                      | —           | gemv |
| 2  | 96.71 ms         | **2.00×**              | 1.012       | SB(2~8) |
| 3  | 99.60 ms         | **2.91×**              | 1.016       | SB |
| 5  | 103.44 ms        | **4.66×**              | 1.032       | SB |
| 7  | 107.31 ms        | **6.29×**              | 1.035       | SB |
| 16 | 180.04 ms        | 8.57×                  | 1.075       | MMQ(>8) ← 챕터8 함정 |

**판정 (PASS):**
1. **weight 1적재 amortize 확정** — Sq=2 tree(96.7ms)가 greedy 1-tok(96.5ms)와 동일 = 2토큰을 1토큰 시간에.
   Sq=7은 6.29× amortize(7토큰을 1.11× 시간 107ms에). amortize factor가 K에 비례 → 1-forward가 sequential 아님.
2. **tree fork 오버헤드 무시가능** — tree/linear = 1.01~1.04(≤8). GDN snapshot/restore + conv ancestor-walk +
   attn mask가 linear 대비 +1~4%뿐. **사용자가 우려한 "노드별 snapshot/restore 오버헤드가 amortize 먹지 않나"
   위험 제거.**
3. **챕터8 함정은 >8(MMQ)서 재현** — Sq=16=180ms(SB 추세 대비 ~50%↓, per-token MMQ ~4.5× SB). →
   **tree는 ≤8 노드 유지**(SB 경로) **or tree>8이면 gemv_w4a8_sb 확장**(STEP4 기지 항목).

→ **1-forward amortize 안 깨짐 = STEP 4 풀빌드 정당.** accept 천장(top-4 0.949)이 실현되면 7노드 트리 verify가
1.11× 시간이므로 accept↑가 거의 그대로 t/s 이득. (챕터8처럼 "통합 후 negative" 위험을 작은 단위로 미리 차단.)

## STEP 4 (다음)
tree draft(top-k, ≤8노드) → 1-forward verify(STEP1~3) → longest-greedy accept → e2e accept(chain 1.73→?)·t/s.
amortize 확인됐으니 남은 변수는 **실현 accept**(draft 분포가 top-k 천장 0.949를 얼마나 당기나)뿐.

## 회귀
test_decode/test_verify/test_spec(0.8B) linear 경로 불변, **spec mism=0**(tree 인자 null이면 기존과 bit-identical).
