#!/usr/bin/env python3
"""Gemma4 assistant HF 골든 덤프 (draft-spec 2관문 — 엔진 assistant graph 게이트용).
candidate generator(SinglePositionMultiToken)와 동일 의미로 K스텝 draft를 재현·전 중간값 덤프:
  target(12B bf16) 프롬프트 forward → shared KV(마지막 full/sliding 층·[len,NKV,hd]로 전치)·h0(hidden_states[-1] 마지막 위치)·
  anchor 토큰(argmax) → assistant(0.4B ★fp32 — 골든 타이트) K스텝: emb(target ScaledWordEmbedding=√H 스케일 내장)+h concat
  → pre_proj → 4층 cross-attn(고정 position=P) → post_proj 재귀. 스텝별 emb/h_in/logits/h_out/argmax 덤프.
rope는 assistant rotary_emb가 실제 낸 cos/sin 행(고정 위치)을 직접 덤프 — 엔진 게이트에서 공식 재현 불요(리스크 제거).
★hidden_states[-1]의 final-norm 여부 실측 판정(③ 통합 de-risk)도 meta에 기록.
사용: python tools/make_reference_assistant.py --out artifacts/gemma4-12b-assistant/golden [--k 8]
"""
import argparse, json, os, sys
import numpy as np

TARGET = "google/gemma-4-12B-it"
ASSIST = "google/gemma-4-12B-it-assistant"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--prompt", default="Explain why the sky is blue in one paragraph.")
    ap.add_argument("--q4", action="store_true",
                    help="assistant 가중치 q4 라운드트립(dequant(quant(W)) — extract 스킴 동일) 후 forward = 엔진 W4A8 게이트용 골든(target golden_q4 전례)")
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.makedirs(args.out, exist_ok=True)
    def w(name, t): t.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.out, name + ".bin"))

    tok = AutoTokenizer.from_pretrained(TARGET)
    print("[golden-ast] loading target(bf16)+assistant(fp32) → cuda ...", flush=True)
    tgt = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16).to("cuda").eval()
    ast = AutoModelForCausalLM.from_pretrained(ASSIST, dtype=torch.float32).to("cuda").eval()
    if args.q4:   # 큰 matmul(2D — extract is_q4 동일 집합·tied lm_head는 embed 스토리지 공유로 함께) q4 라운드트립
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from extract_weights import QBLK
        with torch.no_grad():
            nq = 0
            for k, v in ast.state_dict().items():
                if k == "lm_head.weight" or v.ndim != 2: continue
                wnp = v.detach().float().cpu().numpy()          # ★이름 주의: w는 덤프 함수(shadow 금지)
                wr = wnp.reshape(-1, QBLK)
                s = np.abs(wr).max(axis=1) / 7.0; s[s == 0] = 1e-8
                q = np.clip(np.round(wr / s[:, None]) + 8, 0, 15)
                v.copy_(torch.from_numpy(((q - 8) * s[:, None]).reshape(wnp.shape).astype(np.float32)).to(v.device))
                nq += 1
        print(f"[golden-ast] q4 라운드트립 적용: {nq} tensors (QBLK={QBLK})", flush=True)

    msgs = [{"role": "user", "content": args.prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "input_ids"): ids = ids["input_ids"]
    ids = ids.to("cuda"); P = ids.shape[1]

    with torch.inference_mode():
        out = tgt(ids, output_hidden_states=True, return_shared_kv_states=True, use_cache=False)
    hs = out.hidden_states
    anchor = int(out.logits[0, -1].argmax().item())          # prefill 다음 토큰(= draft 사이클 anchor)
    # ★hidden_states[-1] final-norm 여부 실측: 텍스트모델 final norm을 직접 적용해 비교
    base = None
    for cand in ("model.language_model", "model"):
        try: base = tgt.get_submodule(cand); break
        except AttributeError: pass
    hn = base.norm(hs[-1].float())
    pre_vs_post = (hs[-1].float() - hn).abs().max().item()
    is_postnorm = pre_vs_post < 1e-5
    print(f"[golden-ast] len(hidden_states)={len(hs)} | hidden_states[-1] vs norm적용: max|Δ|={pre_vs_post:.3e} "
          f"→ hidden_states[-1]은 final-norm {'후(post)' if is_postnorm else '전(pre — 층 출력)'}", flush=True)

    skv = out.shared_kv_states
    lens = {}
    for lt, tag in (("full_attention", "f"), ("sliding_attention", "s")):
        k, v = skv[lt]                                        # (1, heads, len, hd)
        kk = k[0].permute(1, 0, 2).contiguous()               # → [len, NKV, hd] (엔진 캐시 레이아웃)
        vv = v[0].permute(1, 0, 2).contiguous()
        w(f"ast_k{tag}", kk); w(f"ast_v{tag}", vv)
        lens[tag] = list(kk.shape)
        print(f"[golden-ast] shared KV {lt}: {tuple(k.shape)} → dump {tuple(kk.shape)}", flush=True)

    h0 = hs[-1][:, -1:].float()                               # candidate generator가 쓰는 h (마지막 위치)
    w("ast_h0", h0[0, 0])
    pos = P                                                   # input_ids(프롬프트+anchor).shape[1]-1 = P (고정)
    # assistant rotary가 실제 내는 cos/sin(고정 위치·층타입별) — 엔진은 이 행을 그대로 사용
    rot = ast.model.rotary_emb
    pid = torch.tensor([[pos]], device="cuda")
    dummy = torch.zeros(1, 1, 1, device="cuda", dtype=torch.float32)
    for lt, tag in (("full_attention", "f"), ("sliding_attention", "s")):
        cos, sin = rot(dummy, pid, lt)
        w(f"ast_rope_cos_{tag}", cos[0, 0]); w(f"ast_rope_sin_{tag}", sin[0, 0])
        print(f"[golden-ast] rope {lt}: cos dim={cos.shape[-1]}", flush=True)

    emb_mod = tgt.get_input_embeddings()                      # ScaledWordEmbedding(√H 스케일 내장)
    # ★fp32 assistant + bf16 target KV = SDPA dtype 거부 → fp32 업캐스트(무손실 — 엔진 fp32 덤프와 동일 값)
    skv_dev = {k: (a.to("cuda", torch.float32), b.to("cuda", torch.float32)) for k, (a, b) in skv.items()}
    steps = []
    cur = anchor; h = h0
    with torch.inference_mode():
        for j in range(args.k):
            emb = emb_mod(torch.tensor([[cur]], device="cuda")).float()
            w(f"step{j}_emb", emb[0, 0]); w(f"step{j}_hin", h[0, 0])
            ie = torch.cat([emb, h], dim=-1)
            o = ast(inputs_embeds=ie, position_ids=pid, shared_kv_states=skv_dev, attention_mask=None)
            lg = o.logits[0, -1].float(); hn2 = o.last_hidden_state[0, -1].float()
            w(f"step{j}_logits", lg); w(f"step{j}_hout", hn2)
            cur = int(lg.argmax().item()); h = o.last_hidden_state[:, -1:].float()
            steps.append(cur)
            print(f"[golden-ast] step{j}: draft tok={cur} ({tok.decode([cur])!r})", flush=True)

    np.array(steps, dtype=np.int32).tofile(os.path.join(args.out, "ast_drafts.bin"))   # 엔진 테스트용(int32 [K])
    json.dump({"target": TARGET, "assistant": ASSIST, "prompt": args.prompt, "P": P, "pos": pos,
               "anchor": anchor, "K": args.k, "draft_tokens": steps,
               "kv_full_shape": lens["f"], "kv_sliding_shape": lens["s"],
               "hidden_states_last_is_postnorm": bool(is_postnorm),
               "prompt_ids": ids[0].tolist()},
              open(os.path.join(args.out, "ast_meta.json"), "w"), indent=2)
    print(f"[golden-ast] done → {args.out} | anchor={anchor} drafts={steps}", flush=True)

if __name__ == "__main__":
    main()
