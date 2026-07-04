# 26B-A4B B/C/D 구현 spec (modeling 확정 — 추측 0)

> Phase 0(5f59ea8) 후속. 이 문서 = transformers 5.12.1 `modeling_gemma4.py`에서 **정확히** 뜬 구현 명세.
> ★대형·메모리-위험 빌드(변환 + 새 graph dense∥MoE + 26B MoE 골든[31B 118GB 전례]) — 단계별 안전 실행 근거.

## 디코더 층 위상 (modeling_gemma4.py Gemma4TextDecoderLayer.forward — verbatim)
```
residual = h
h = input_layernorm(h); h = self_attn(h); h = post_attention_layernorm(h); h = residual + h   # Gemma sandwich(31B 동일: sliding/full·k_eq_v·dual-rope)
residual = h
h = pre_feedforward_layernorm(h); h = mlp(h)                # dense GeGLU MLP (intermediate 2112)
if enable_moe_block:
    h1 = post_feedforward_layernorm_1(h)                    # dense 경로 post-norm
    rin = residual.reshape(-1, H)                           # ★MoE 입력 = residual(pre_ff_norm 출력 아님!)
    _, top_w, top_i = router(rin)                           # 아래 router 규약
    h2 = pre_feedforward_layernorm_2(rin)                   # MoE 입력 norm(별도)
    h2 = experts(h2, top_i, top_w)                          # 128e top8
    h2 = post_feedforward_layernorm_2(h2)                   # MoE 경로 post-norm
    h = h1 + h2                                             # dense ∥ MoE 합산
h = post_feedforward_layernorm(h); h = residual + h         # 최종 FFN norm + residual
# layer_scalar = register_buffer ones(1) (31B처럼 스칼라 — 값 확인 필요, 대개 1.0)
# hidden_size_per_layer_input=0(26B) → per-layer-input 분기 스킵
```

## router (Gemma4TextRouter.forward — verbatim)
```
h = norm(rin)                       # Gemma4RMSNorm(H, with_scale=False) — no weight
h = h * self.scale * H^-0.5         # ★self.scale = Parameter[H] (weight), scalar_root_size = H^-0.5
scores = self.proj(h)               # Linear H→128, no bias (weight: router.proj.weight)
probs = softmax(scores, -1)
top_w, top_i = topk(probs, 8)
top_w /= top_w.sum(-1, keepdim=True)         # renorm (router_topk.cuh 가정 성립)
top_w = top_w * self.per_expert_scale[top_i] # ★per_expert_scale = Parameter[128] (weight)
```
= router_topk.cuh(softmax→top8→renorm) 재사용 + **전처리**(rmsnorm_noscale + scale[H] + H^-0.5, launch_scale 재사용) + **후처리**(top_w에 per_expert_scale[idx] 곱 — moe_fused down에서 dw로 흡수).

## experts (Gemma4TextExperts — verbatim)
gate_up_proj=Parameter[E=128, 2·MI=1408, H=2816], down_proj=Parameter[E, H, MI=704]. gate,up=chunk(2). act(gate)*up(GeGLU). down. ×top_w.
= **moe_fused.cuh 그대로**(shared expert 없음 → moe_fused의 shared 경로 조건화[config gate] 필요).

## STEP A — extract_weights 확장 (변환)
- experts.gate_up_proj/down_proj → q4(이미 is_q4 처리 ✓, 3D [E,out,in]). dense mlp gate/up/down → q4(백본).
- **신규 fp32**: router.proj.weight(H→128), router.scale[H], router.per_expert_scale[128]. norms 7종(input/post_attn/
  pre_ff/post_ff/post_ff_1/pre_ff_2/post_ff_2)+router 없음(no-scale). layer_scalar. embed(tied, ×√H).
- prefix 자동감지=model.language_model(31B와 동형 예상). 예산 q4 ~13GB. 중간물 즉시 정리.

## STEP B — 새 graph gemma4_moe (신규 커널 0)
- 31B gemma4 graph 복제 + MLP 스텝을 위 위상으로: dense mlp(GeGLU) ∥ [router→moe_fused(no-shared)→post_ff_2] → h1+h2.
- config: enable_moe_block, moe_parallel_dense, router_own_norm, per_expert_scale(flag), shared_expert=false.
- moe_fused 확장: shared 경로 조건화(SI==0이면 스킵), dw에 per_expert_scale 곱(router 후처리를 dw에 융합 or 별도 kernel).
- decode/batch/ragged 대칭. ③=0(전부 config/graph/기존 커널).

## STEP C — golden (★메모리 위험 — 31B 방식 준수)
- 26B fp32 full = 가드 초과 → **bf16 골든 + 층-스트리밍 fp32**(make_reference_gemma4_fp32.py 방식). **RSS 하드가드 78GB
  + 경계 8층** 필수(31B가 112GB 근접·kill 전례 — [[gb10-unified-memory-crash]]). MemAvailable 118GB지만 fp32 전개+mmap 누적 위험.
- 게이트: op-level rel<임계 + full-depth argmax(다양 프롬프트) + **expert-index 확장판**(top-k idx==HF + per_expert_scale·
  router-norm 반영 weight 대조 — 신규 규약 2종 여기서 검증) + batch bit-id + decode 연속성.

## STEP D — 서빙: Gemma4Backend decode_step이 gemma4_moe 호출(훅 뒤 교체), per_seq_bytes=KV only(GDN 無)+MoE scratch. Poisson 스모크.

## ★실행 순서 안전 원칙
다운로드 → STEP A 변환(안전·committable) → STEP B graph → **STEP C 골든은 메모리 스테이징 신중**(bf16+층-스트리밍+RSS가드,
동시 상주 최소 — 다운로드/변환 중간물 먼저 정리). 골든이 메모리 위험하면 stop-report(시스템 보호 우선).
