# 챕터5 STEP 0 — Qwen3.5-35B-A3B MoE 설계 (보편화 3번째 시험)

> 게이트 문서. 사용자 합의 후 코딩. 핵심 질문: **MoE가 config+graph로 빠지나, 아니면 새 커널이 필요한가** —
> GDN 갭(27B)에서 배운 "특수성이 graph/kernel에 새는가" 시험의 3번째.

## 확정 사실 (HF 검증)
- repo: **`Qwen/Qwen3.5-35B-A3B`** (실재 확인, 사용자 지정 그대로). model_type **`qwen3_5_moe`**(텍스트 `qwen3_5_moe_text`).
- config: 40층, hidden 2048, vocab 248320, head_dim 256, full_attention_interval 4.
- **MoE**: num_experts **256**, top-**8**/tok, moe_intermediate **512**, shared_expert(512) + `shared_expert_gate`(H→1).
  mlp_only_layers=[] (전층 MoE). num_shared_experts 개념 = 1 shared.
- attn NQ=16/NKV=2(group 8). **GDN linear_num_key_heads=16, value_heads=32 (group 2)** — 또 다른 GQA 비율.
- 메모리: bf16 ~70GB(골든 CPU 적재 가능, 119GB), q4 엔진 ~16~18GB GPU.

## HF MoE 구조 (modeling_qwen3_5_moe.py 검증)
```
# Router (Qwen3_5MoeTopKRouter)
router_logits = Linear(hidden, [H→256])
probs = softmax(router_logits, dim=-1, fp32)        # ★ 256 전체 softmax 먼저
top_w, idx = topk(probs, 8)                          # ★ 그 다음 top-8
top_w /= top_w.sum(-1)                               # ★ renormalize
# Experts: 선택된 8개 각각 SwiGLU(gate/up [H→512], down [512→H]) → top_w 가중합
# Shared expert (항상): MLP(inter=512) → sigmoid(Linear[H→1]) 게이트로 스케일
out = experts_out + sigmoid(shared_gate(hidden)) * shared_mlp(hidden)
```

## 보편화 평가 — 무엇이 재사용 / 새것인가
**재사용 (커널 불변, config 주도)** — 백본 전체:
- GDN(16k/32v): STEP 3 일반화 커널 `g=h*NHk/NH`로 그대로(group 2). full-attn(16/2): `kv=h/GRP` 그대로.
- rmsnorm, gemv(q4/f32/w4a8), swiglu, rope, attention, add, embedding, lm_head — 전부 기존.
- expert FFN = SwiGLU(기존 커널), shared expert = MLP+sigmoid(기존).

**새것 (graph + 커널 1개)**:
1. **새 graph**: MLP 섹션이 dense SwiGLU → MoE 블록(router → expert 루프 → shared). `models/qwen35_moe.cu`.
2. **새 커널 1개: `router_topk`** = softmax(256) + top-8 + renorm. GDN처럼 "정말 새 연산"이라 커널 1개 추가 정당
   (헌법: 새 모델 = 새 graph + 정말 새 연산일 때만 커널 1~2개). 작고 격리 가능.
3. **expert dispatch 패턴**(커널 아님): 정확성-우선 = naive 루프(토큰별 선택 8 expert GEMV+swiglu 가중합).
   효율화(토큰을 expert별 group → batched GEMM)는 **나중**(정확성 먼저, 헌법). decode B=1은 8 GEMV.
4. **레지스트리**: `MK_QWEN3_5_MOE` 추가, model_type→graph 디스패치. 백본 헬퍼는 27B와 공유.

→ **평가: MoE는 graph + 커널 1개(router_topk)로 빠진다.** expert/shared는 기존 SwiGLU 재사용 =
보편화 유지(특수성이 백본에 새지 않음). router top-k는 GDN처럼 합법적 신규 연산. **보편화 실패 아님**(예상).

## 메모리 / 추출 / 골든 전략
- **추출**: 256×40 expert FFN = 30720 텐서 + router/shared. extract_weights를 MoE 레이아웃(experts.{e}.gate/up/down
  또는 fused [num_experts,...])에 맞게 확장. q4 정책 동일(큰 matmul q4, router/gate/norm fp32). q4 ~16GB.
- **골든(메모리)**: bf16 로드(70GB, 119GB RAM 안전). HF forward fp32 가능(35B fp32=140GB → OOM, **bf16 골든** 사용).
  → 27B서 확립한 q4 골든(bf16 소스) 방식 재사용. test 게이트 = MEAN-rel + argmax + next-token(STEP 3 교정 척도).
- **격리 골든(신규)**: router_logits, selected_experts(8 idx), routing_weights, expert별 출력, shared 출력,
  MoE 블록 출력 — make_reference에 MoE 훅 추가(GDN/attn 훅 패턴 재사용).

## 검증 계획 (단계별, 헌법)
1. **router_topk 격리**: softmax256+top8+renorm vs HF router (idx 집합 일치 + weight 오차0).
2. **단일 expert FFN 격리**: SwiGLU(H→512→H) vs HF expert (기존 swiglu 테스트 패턴 재사용).
3. **MoE 블록 격리**: 8 expert 가중합 + shared(sigmoid gate) vs HF SparseMoeBlock 출력 (golden 입력).
4. **GDN 16/32 격리**: STEP 3 test_gdn_27b 패턴으로 group2 확인(일반화 커널 재검증 — 공짜 보너스).
5. **전체 forward**: argmax ALL + next-token + MEAN-rel<2e-3. 0.8B/27B 회귀 0.

## STEP 0 게이트 — 사용자 결정 필요
1. **MoE = graph + router_topk 커널 1개** 방향 동의? (expert/shared는 SwiGLU 재사용 = 보편화 유지)
2. **expert dispatch = naive 루프(정확성 우선), 효율화 나중** 동의?
3. **골든 = bf16(35B fp32 140GB OOM 회피), q4 검증** 동의? (27B 방식 재사용)
4. **레지스트리 MK_QWEN3_5_MOE 추가 + 백본 헬퍼 27B 공유** 방향 동의?
5. 다운로드 완료 후 STEP 1(추출/레지스트리) → STEP 2(router/expert/MoE블록 격리) → STEP 3(전체+회귀) 순서 동의?
