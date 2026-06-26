# 2a 설계안 — Qwen3.5-0.8B 본체 (STEP 0 산출물, 사용자 합의 대기)

> 출처: 실제 config/state_dict 덤프 + transformers `modeling_qwen3_5.py` 직접 정독(line 인용).
> GDN/llama.cpp 교차검증은 별도 진행. **이 문서 합의 후 STEP 1 착수.** MTP는 2b (여기서 제외).

## 1. 확정된 본체 구조 (실측)

멀티모달 모델(`Qwen3_5`)의 **text body**(`qwen3_5_text`)만 구현. 텐서 prefix = `model.language_model.*`.

| H | FFN | L | NQ | NKV | head_dim | V | rms_eps | tie | act |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 3584 | 24 | 8 | 2 | 256 | 248320 | 1e-6 | True | silu |

- weight dtype = **BF16** (extract 시 fp32 upcast). lm_head 텐서 없음 → embed와 tied.
- **레이어 타입 (config `layer_types` 명시, 추측 아님)**: full attention = 인덱스 **3,7,11,15,19,23** (i%4==3, 6개),
  나머지 18개 = linear/GDN. `full_attention_interval=4`.
- rope: `mrope_interleaved=True, mrope_section=[11,11,10], rope_theta=1e7, partial_rotary_factor=0.25`
  → rotary_dim = 256*0.25 = **64**.

### 텐서 인벤토리
- 공통: `embed_tokens.weight [248320,1024]`, `norm.weight [1024]` (lm_head=tied).
- **FULL attn 레이어** (6개): `q_proj [4096,1024]`(=8*256*2, Q+gate), `k_proj/v_proj [512,1024]`(=2*256),
  `q_norm/k_norm [256]`, `o_proj [1024,2048]`, `input_layernorm/post_attention_layernorm [1024]`,
  `mlp.gate/up [3584,1024]`, `mlp.down [1024,3584]`.
- **GDN 레이어** (18개): `in_proj_qkv [6144,1024]`(=key_dim*2+value_dim=2048*2+2048),
  `in_proj_z [2048,1024]`, `in_proj_a [16,1024]`, `in_proj_b [16,1024]`,
  `conv1d.weight [6144,1,4]`(depthwise k=4), `A_log [16]`(F32), `dt_bias [16]`,
  `norm [128]`(F32, gated), `out_proj [1024,2048]`, + 같은 input/post LN + 같은 MLP.
- GDN config: num_k_heads=num_v_heads=16, head_k_dim=head_v_dim=128, conv_kernel=4, conv_dim=6144.

## 2. SmolLM3 대비 새로 필요한 것 (핵심 차이)

### (A) RMSNorm 컨벤션이 다름 — silent killer, 2종 공존
- `Qwen3_5RMSNorm` (modeling:736-749): `out = _norm(x) * (1.0 + weight.float())` — **zero-centered (1+w)**, fp32.
  사용처: input_layernorm, post_attention_layernorm, final norm, **q_norm, k_norm**.
  → 1단계 rmsnorm 커널(`y=x*inv*W`)을 **`y=x*inv*(1+W)`** 변형 필요. (별도 커널 or flag)
- `Qwen3_5RMSNormGated` (modeling:187-202): `_norm(x) * weight`(표준) 후 `* silu(gate)`. GDN 출력 전용.

### (B) Partial RoPE (앞 64dim만), theta=1e7
`apply_rotary_pos_emb`(modeling:570-605)는 **표준 rotate_half**를 q[...,:64]에만 적용, q[...,64:]는 통과.
rotate_half: 64를 두 절반(32+32)으로, `[-x2,x1]`. → pair (i, i+32), i<32, cos/sin[i] 사용.
mrope interleave[11,11,10]은 cos/sin **생성**에만 관여하며 text-only는 모든 position stream 동일 → 표준 1D로 degenerate
(modeling Task-B trap6). **make_reference가 HF `rotary_emb`로 cos/sin[seq,64] 덤프** → mrope 재구현 회피, 골든이 보장.

### (C) Full attention: q/k_norm + output_gate (modeling:645-717)
순서: q_proj→view[*,8,512]→chunk(2)→(query[*,8,256], gate[*,8,256]); gate→[*,2048].
query=**q_norm**(query.view[*,8,256]) [RoPE 전, head_dim RMSNorm]; key=**k_norm**(k_proj.view[*,2,256]); value=v_proj.
→ partial rope(q,k) → GQA attention(scale=256^-0.5, group=4, causal softmax) → reshape[*,2048]
→ **`*= sigmoid(gate)`** → o_proj. (gate는 sigmoid, silu 아님)

### (D) GDN / Gated Delta Net (modeling:371-559) — 최대 작업
forward (prefill, cache 없음 경로):
```
mixed_qkv = in_proj_qkv(h)            # [B,seq,6144]
mixed_qkv = silu(conv1d(mixed_qkv^T)[..., :seq])   # depthwise k=4 causal(pad=3) → silu
q,k,v = split(mixed_qkv, [2048,2048,2048]) → reshape [B,seq,16,128]
z = in_proj_z(h) → [B,seq,16,128]
beta = sigmoid(in_proj_b(h))          # [B,seq,16]
g = -exp(A_log) * softplus(in_proj_a(h) + dt_bias)   # [B,seq,16]  (fp32)
# (num_v//num_k=1 → repeat 없음)
# delta rule (recurrent form, modeling:327-368; q,k L2norm; q*=1/sqrt(128)):
per head h, S[128k,128v]=0:
  for t in seq:
    S *= exp(g[t])                              # scalar decay
    kv[v] = Σ_k S[k,v]*k_t[k]
    Δ[v]  = (v_t[v]-kv[v]) * beta[t]
    S[k,v] += k_t[k]*Δ[v]                        # outer product
    out_t[v] = Σ_k S[k,v]*q_t[k]
core_attn_out → norm_gated(out, z) = (_norm(out)*norm.weight) * silu(z)   # per head_v_dim=128
out = out_proj(core_attn_out.reshape[B,seq,2048])     # → H
```
주의: q,k는 **L2 정규화**(RMSNorm 아님, modeling:240-243, eps=1e-6), 그 뒤 q에만 1/sqrt(128) scale.
HF 골든은 prefill에서 `torch_chunk_gated_delta_rule`(modeling:246-324)을 쓰지만 recurrent와 수학적 동치 →
엔진은 검증 쉬운 **recurrent** 구현, HF chunk 골든과 fp32 오차(~1e-4)로 대조.

## 3. 커널 계획

재사용(1단계 그대로): gemv_q4, swiglu, embedding, elementwise add, reduce.
변형/신규:
| 커널 | 상태 | 비고 |
|---|---|---|
| `rmsnorm` | **변형** | `(1+w)` zero-centered 버전 추가 (flag 또는 신규 `rmsnorm_zc`). q/k_norm도 이걸로 |
| `rope_partial` | **신규** | rotary_dim 인자, 앞 64만 rotate_half, 나머지 통과. cos/sin[seq,64] |
| `attention` | **변형** | scale 인자화(256^-0.5), q/k_norm은 graph에서 별도 호출, output_gate는 graph에서 sigmoid 곱 |
| `gdn_conv1d` | **신규** | depthwise causal conv k=4 + silu, 6144 채널 |
| `gdn_delta_rule` | **신규(핵심)** | head별 recurrent state[128,128], L2norm+scale, decay/beta/delta/out |
| `rms_norm_gated` | **신규** | `_norm(x)*w*silu(z)`, head_v_dim=128 |
| `softplus`/elementwise | 소규모 신규 | g 계산: `-exp(A_log)*softplus(a+dt_bias)` |

## 4. GDN 분해 검증 (헌법: 단계별 격리 — GDN 틀리면 본체 다 틀어짐)
작게 쪼개 각각 HF 중간값과 단독 대조:
1. `in_proj_qkv/z/a/b` 출력 (gemv) 대조.
2. `gdn_conv1d`(+silu) 출력 대조.
3. `g`, `beta` 계산 대조.
4. `gdn_delta_rule` core_attn_out 대조 (q/k l2norm·scale 포함).
5. `rms_norm_gated`(z) 출력 대조.
6. out_proj → GDN 레이어 전체 출력 대조.
HF에서 GDN 레이어 하나의 입력 hidden + 각 하위단계 출력을 forward_hook으로 골든 덤프.

## 5. graph 통합 (`models/qwen35.cu`)
config에 `layer_types[]`(0=linear,1=full) 추가. 레이어 루프에서 분기:
- full: input_ln → [q/k/v proj, q/k_norm, partial rope, GQA attn, output_gate, o_proj] → +res → post_ln → MLP → +res
- linear: input_ln → GDN블록 → +res → post_ln → MLP → +res
final: norm → lm_head(tied embed, q4). embed fp32.

## 6. 검증 플랜 (`tests/test_qwen35.cu`, 오차 0 기준)
- (a) 레이어별 입력 hidden 단계 대조 (full·linear 모두) vs golden_q4.
- (b) **pre-norm hidden**(final norm 입력 = MTP가 받을 값) 단독 대조 — forward_hook으로 model.norm 입력 덤프.
- (c) 최종 logits vs golden_q4 < 1e-2 + next-token = HF fp32 일치.
- pos>0 다토큰 prefill (partial rope/mrope/attn 실제 exercise — 1단계 NoPE 교훈).
- 1단계와 동일 fp32/q4 2-경로(golden_q4 = 가중치 round-trip)로 커널버그 격리.

## 7. 도구 변경
- `extract_weights.py`: (1) **key prefix 일반화** (`model.language_model.` 지원), (2) GDN 텐서 추출
  (conv1d/A_log/dt_bias/in_proj_* — 대부분 작아 fp32 유지, 큰 in_proj_qkv/z·out_proj·mlp·q/k/v/o만 q4),
  (3) layer_types를 manifest에 기록, (4) rope rotary_dim/theta 기록.
- `make_reference.py`: full+linear 레이어별 hidden, GDN 하위단계 골든, pre-norm hidden(hook),
  cos/sin[seq,64](HF rotary_emb), partial rope 반영.

## 7.5 GDN delta rule — 못박은 3개 (modeling_qwen3_5.py line 인용, 변종마다 달라 가장 틀리기 쉬움)

`torch_recurrent_gated_delta_rule` (327-368) + `l2norm` (240-243) 기준. state `S[k_dim=128, v_dim=128]`.

(a) **q/k L2norm + scale 순서·대상** (331-341): `l2norm(q)` **AND** `l2norm(k)` 둘 다 (dim=-1, eps=1e-6,
    `x*rsqrt(Σx²+eps)`) → **그 다음** `q *= 1/sqrt(k_head_dim=128)` (q만; **k는 스케일 없음**).
(b) **decay g** (356,359): `g[:,:,i]`는 head당 **스칼라**([B,H]). `g_t=exp(g[i])`; `S *= g_t` (k·v 양축 broadcast).
    채널 벡터 아님. (g = `-exp(A_log)*softplus(a+dt_bias)`, line 518; per-head.)
(c) **Δ / outer / contract 방향** (347,360-363): `kv[v]=Σ_k S[k,v]·k[k]`; `Δ[v]=(v[v]-kv[v])·β`(β=sigmoid(b), head 스칼라);
    `S[k,v] += k[k]·Δ[v]` (k⊗Δ outer); `out[v]=Σ_k S[k,v]·q[k]`. (S 첫 축=k, 둘째 축=v.)

검증 추가 요구(사용자): 전체 시퀀스 ~1e-4 대조 **전에**, 단일 head·t=0,1,2 몇 스텝만 작게 먼저 대조해
state S 첫 업데이트가 HF와 맞는지 확인 (수식 방향 오류를 가장 먼저 잡기).

## 8. 사용자 결정 (합의 완료)
- STEP 1 승인·완료. 큰 matmul만 q4 / gating파라미터(A_log,dt_bias,conv1d,in_proj_a,b) fp32. delta rule recurrent naive 우선.

## 9. 2a 결과 (STEP 1–4 완료, PASS)

프롬프트 "The capital of France is" (Qwen 토큰 5개), full 6 / GDN 18 레이어.

| 검증 | 결과 |
|---|---|
| STEP1 커널 단위 | rmsnorm_zc/qk_norm/rope_partial/gate_sigmoid 전부 <1e-4 (rope 0) |
| STEP2 GDN 격리 | conv1d 1.2e-7, delta_rule 1.7e-8 (head0 t=0,1,2 ~1e-11), gated_norm 2.4e-7 |
| STEP4 레이어별 hidden (full+GDN) vs golden_q4 | hs[0..24] 전부 **< 1e-5** (worst 8e-5 @ final-norm) |
| **pre-norm hidden (MTP 입력)** vs golden_q4 | **1.3e-5** ✓ |
| 최종 logits vs golden_q4 | **6.7e-5** (graph 정확성) |
| argmax vs golden_q4 | 전 위치 일치 [198,314,279,369,279] |

→ 엔진이 Qwen3.5 본체(GDN 18 + full 6, partial RoPE, q/k_norm, output_gate, gated delta net)를
  HF-q4와 단계별 <1e-5 로 정확히 구현. MTP가 받을 pre-norm hidden도 정확.

### ⚠️ 미해결 보고 사항 — q4가 fp32 예측을 바꿈
- golden_fp32 last-token = **11751** vs golden_q4 = **279** (DIFF). 엔진(q4) = golden_q4 = 279.
- 즉 엔진은 "HF를 q4 가중치로 돌린 것"과 완전히 일치하나, **fp32 HF와는 top-1이 다름**.
- 원인: 0.8B 소형 모델 + GDN 누적이 int4(mean rel-err 0.10)에 민감. (1단계 SmolLM3-3B는 q4가 예측 보존했음.)
- chunk vs recurrent delta rule 동치 확인(7e-9)이라 GDN 구현 문제 아님 — 순수 양자화 손실.
- **결정 필요**: (A) q4 본체 유지(프로젝트의 bandwidth 취지에 부합, golden_q4로 정확성 증명 완료) +
  2b MTP도 q4 hidden 위에 구축, vs (B) 2a/MTP는 fp32 경로로 HF-fp32 일치 확인 후 q4는 별도 최적화.
  → 설계 대화로 가져갈 항목.

### (원래) 결정 필요했던 포인트
1. **GDN 양자화 범위**: in_proj_qkv[6144,1024]/out_proj/in_proj_z만 q4, 작은 것(A_log/dt_bias/conv/norm/in_proj_a,b)은 fp32 — 동의?
2. **delta rule는 recurrent로 구현**(HF chunk와 동치, 검증 용이) — 동의? (성능은 2a 범위 밖)
3. **rope는 HF cos/sin 덤프 재사용**(mrope 재구현 안 함, 1단계와 동일 방침) — 동의?
4. 검증 프롬프트/길이: 1단계처럼 짧은 다토큰(예: 5~8토큰) — 특정 프롬프트 선호 있는지?
