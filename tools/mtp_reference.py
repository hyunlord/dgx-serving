#!/usr/bin/env python3
"""Qwen3.5-0.8B MTP head — torch 레퍼런스 (golden 빌더). HF엔 MTP 모듈이 없어 손수 구현.

목적: 과거 Python 0% accept를 깬다. 진범 4개를 정확히 반영하고, h(pre/post-norm)·pos 오프셋을
스윕해 어느 조합이 non-zero accept를 내는지 표로 보고. non-zero 조합이 곧 CUDA의 검증 골든.

진범(과거 실패 원인) 반영:
  1. attn = prefix causal attend (self-only 아님): MTP 자체 q/k/v를 전 위치에 대해 causal attention.
  2. partial RoPE: 손수짜기 금지 — transformers apply_rotary_pos_emb + 모델 rotary_emb 차용.
  3. 페어링: 위치 P 입력쌍 (e=embed(token[P+1]), h=body_hidden[P]), draft 타깃 = token[P+2].
  4. h 시점: pre-norm(model.norm 입력) vs post-norm(model.norm 출력=h_nextn) — 둘 다 실측.

accept = mean_P( argmax(MTP_logits[P]) == argmax(body_logits[P+1]) ), P=0..S-2.
  (MTP가 (token[P+1],h[P])로 예측한 다음토큰 == 본체가 prefix 0..P+1에서 greedy로 낼 토큰)

사용:
  python tools/mtp_reference.py --model Qwen/Qwen3.5-0.8B --prompt "The capital of France is ..."
"""
import argparse, glob, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is Tokyo.")
    ap.add_argument("--dump", default=None, help="CUDA 대조용 골든 디렉토리 (post-norm/P1 기준)")
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    from safetensors import safe_open

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    H = tcfg.hidden_size; NQ = tcfg.num_attention_heads; NKV = tcfg.num_key_value_heads
    HD = tcfg.head_dim; eps = float(tcfg.rms_norm_eps); GRP = NQ // NKV
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    base = model.get_submodule("model") if not hasattr(model, "language_model") else model.get_submodule("model.language_model")
    rotary = base.rotary_emb
    E = model.get_input_embeddings().weight.detach()    # [V,H], lm_head tied = E

    # --- MTP weights (safetensors 직접 로드; HF는 무시) ---
    sf_path = glob.glob(f"/home/hyunlord/.cache/huggingface/hub/models--{args.model.replace('/','--')}/snapshots/*/model.safetensors*.safetensors")[0]
    W = {}
    with safe_open(sf_path, framework="pt") as sf:
        for k in sf.keys():
            if k.startswith("mtp"):
                W[k] = sf.get_tensor(k).float()
    def w(n): return W["mtp." + n]

    ids = tok(args.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    print(f"[mtp-ref] prompt={args.prompt!r}\n  tokens={ids[0].tolist()} S={S}")

    # --- body forward: per-position pre/post-norm hidden + body argmax ---
    pre_box = {}
    hk = base.norm.register_forward_pre_hook(lambda m, inp: pre_box.__setitem__("h", inp[0].detach()))
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hk.remove()
    h_pre = pre_box["h"][0]                       # [S,H] model.norm 입력 (pre-norm)
    h_post = out.hidden_states[-1][0].detach()    # [S,H] model.norm 출력 (post-norm = h_nextn)
    body_argmax = out.logits[0].argmax(-1)        # [S]  body greedy next at each pos

    def rmsnorm_zc(x, weight):  # Qwen (1+w) zero-centered
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        return v * (1.0 + weight.float())

    def mtp_forward(h_src, pos_mode, inter=None):
        """h_src: [S,H] body hidden. pos_mode: 'P1'(token pos=P+1) or 'P0'(=P).
        returns draft argmax per P (length S-1). inter(dict) 주면 중간값 채움(CUDA 골든)."""
        M = S - 1
        e = E[ids[0, 1:S]]                         # [M,H] = embed(token[P+1]), P=0..S-2
        h = h_src[0:M]                              # [M,H] = body hidden[P]
        e_n = rmsnorm_zc(e, w("pre_fc_norm_embedding.weight"))
        h_n = rmsnorm_zc(h, w("pre_fc_norm_hidden.weight"))
        x = torch.cat([e_n, h_n], dim=-1) @ w("fc.weight").T   # [M,H], e first
        cur = rmsnorm_zc(x, w("layers.0.input_layernorm.weight"))
        # q_proj → [M,NQ,2HD] → Q | gate
        qg = (cur @ w("layers.0.self_attn.q_proj.weight").T).view(M, NQ, 2 * HD)
        q, gate = qg[..., :HD], qg[..., HD:]                    # [M,NQ,HD]
        k = (cur @ w("layers.0.self_attn.k_proj.weight").T).view(M, NKV, HD)
        v = (cur @ w("layers.0.self_attn.v_proj.weight").T).view(M, NKV, HD)
        q = rmsnorm_zc(q, w("layers.0.self_attn.q_norm.weight"))
        k = rmsnorm_zc(k, w("layers.0.self_attn.k_norm.weight"))
        # rope (transformers util, partial+mrope). positions:
        pos = torch.arange(1, S) if pos_mode == "P1" else torch.arange(0, M)   # [M]
        cos, sin = rotary(cur.unsqueeze(0), pos.unsqueeze(0))   # [1,M,rot]
        if inter is not None: inter["cos"] = cos[0]; inter["sin"] = sin[0]
        qh = q.transpose(0, 1).unsqueeze(0)                     # [1,NQ,M,HD]
        kh = k.transpose(0, 1).unsqueeze(0)                     # [1,NKV,M,HD]
        qh, kh = apply_rotary_pos_emb(qh, kh, cos, sin)
        # GQA causal attention over prefix
        kh = kh.repeat_interleave(GRP, dim=1); vh = v.transpose(0, 1).unsqueeze(0).repeat_interleave(GRP, dim=1)
        scale = 1.0 / (HD ** 0.5)
        attn = (qh @ kh.transpose(-1, -2)) * scale              # [1,NQ,M,M]
        mask = torch.triu(torch.full((M, M), float("-inf")), diagonal=1)
        attn = (attn + mask).softmax(-1)
        o = (attn @ vh)[0].transpose(0, 1).reshape(M, NQ * HD)  # [M,NQ*HD]
        o = o * torch.sigmoid(gate.reshape(M, NQ * HD))         # output gate
        cur2 = o @ w("layers.0.self_attn.o_proj.weight").T + x
        ff = rmsnorm_zc(cur2, w("layers.0.post_attention_layernorm.weight"))
        g = ff @ w("layers.0.mlp.gate_proj.weight").T; u = ff @ w("layers.0.mlp.up_proj.weight").T
        cur3 = (F.silu(g) * u) @ w("layers.0.mlp.down_proj.weight").T + cur2
        logits = rmsnorm_zc(cur3, w("norm.weight")) @ E.T       # lm_head tie
        if inter is not None:
            inter["x"] = x; inter["cur2"] = cur2; inter["logits"] = logits
        return logits.argmax(-1)                                # [M]

    print(f"\n=== accept 표: draft[P]==body_argmax[P+1], P=0..{S-2} ===")
    print(f"{'h시점':10s}{'pos':6s}{'accept':>10s}")
    results = {}
    for hname, hsrc in [("pre", h_pre), ("post", h_post)]:
        for pos_mode in ["P1", "P0"]:
            with torch.no_grad():
                draft = mtp_forward(hsrc, pos_mode)
            acc = (draft == body_argmax[1:S]).float().mean().item()
            results[(hname, pos_mode)] = acc
            print(f"{hname:10s}{pos_mode:6s}{acc*100:9.1f}%")

    # --- 페어링 민감도: 같은 h, token[P+1]만 바꿔 draft[P] 가 변하나? (과거 0%는 e가 죽어 둔감했음) ---
    print("\n=== 페어링 민감도 (P=0, h 고정, token[1] 변경 → draft[0] 변해야 정상) ===")
    best = max(results, key=results.get)
    hsrc = h_pre if best[0] == "pre" else h_post
    ids_bak = ids.clone()
    drafts = []
    for tk in [ids_bak[0, 1].item(), 100, 5000, 50000]:
        ids[0, 1] = tk                       # mtp_forward 가 클로저로 ids 참조 → token[P+1]=token[1] 변경
        with torch.no_grad():
            d = mtp_forward(hsrc, best[1])[0].item()
        drafts.append((tk, d))
        ids.copy_(ids_bak)
    uniq = len(set(d for _, d in drafts))
    for tk, d in drafts:
        print(f"  token[1]={tk:6d} → draft[0]={d}")
    print(f"  → draft 종류 {uniq}개 ({'민감(정상)' if uniq > 1 else '둔감 — eh_proj/concat 의심!'})")

    print(f"\n[mtp-ref] best={best} accept={results[best]*100:.1f}%  "
          f"{'NON-ZERO ✓ (과거 0% 돌파)' if results[best] > 0.01 else '0% — 단계 격리 디버그 필요'}")

    # --- CUDA 대조 골든 덤프 (post-norm / P1 기준) ---
    if args.dump:
        import os, json
        os.makedirs(args.dump, exist_ok=True)
        d = {}
        with torch.no_grad():
            draft = mtp_forward(h_post, "P1", inter=d)
        def W32(name, t): t.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.dump, name + ".bin"))
        W32("body_h_post", h_post)                  # [S,H] MTP 입력 h (post-norm)
        ids[0].cpu().numpy().astype(np.int32).tofile(os.path.join(args.dump, "token_ids.bin"))  # [S]
        body_argmax.cpu().numpy().astype(np.int32).tofile(os.path.join(args.dump, "body_argmax.bin"))  # [S]
        W32("rope_cos", d["cos"]); W32("rope_sin", d["sin"])     # [M,rot]
        W32("mtp_x", d["x"]); W32("mtp_cur2", d["cur2"]); W32("mtp_logits", d["logits"])  # [M,H]/[M,H]/[M,V]
        json.dump({"S": S, "M": S - 1, "rot": d["cos"].shape[-1],
                   "accept_post_P1": results[("post", "P1")],
                   "draft": draft.cpu().tolist(), "body_next": body_argmax[1:S].cpu().tolist()},
                  open(os.path.join(args.dump, "meta.json"), "w"), indent=2)
        # MTP weights (fp32; 2b는 fp32 기준선, q4 MTP는 이후). matmul→_f32.bin, norm→.bin
        wd = args.dump
        def wf32(name, t): t.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(wd, name + ".bin"))
        wf32("mtp_fc_f32", w("fc.weight"))
        wf32("mtp_enorm", w("pre_fc_norm_embedding.weight")); wf32("mtp_hnorm", w("pre_fc_norm_hidden.weight"))
        wf32("mtp_norm", w("norm.weight"))
        wf32("mtp_input_layernorm", w("layers.0.input_layernorm.weight"))
        wf32("mtp_post_attention_layernorm", w("layers.0.post_attention_layernorm.weight"))
        wf32("mtp_q_norm", w("layers.0.self_attn.q_norm.weight")); wf32("mtp_k_norm", w("layers.0.self_attn.k_norm.weight"))
        for nm in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            wf32("mtp_" + nm + "_f32", w("layers.0.self_attn." + nm + ".weight"))
        for nm in ["gate_proj", "up_proj", "down_proj"]:
            wf32("mtp_" + nm + "_f32", w("layers.0.mlp." + nm + ".weight"))
        print(f"[mtp-ref] golden+weights(fp32) dumped → {args.dump} (post-norm/P1, accept={results[('post','P1')]*100:.1f}%)")


if __name__ == "__main__":
    main()
