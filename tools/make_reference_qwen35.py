#!/usr/bin/env python3
"""Qwen3.5-0.8B 본체 HF 골든 덤프 (full + GDN). config 기반.

골든 2종 (docs/02a 검증전략, 1단계와 동일 격리):
  golden_fp32 : 원본 fp32.
  golden_q4   : 큰 matmul만 dequant(quant(W)) round-trip (gating/norm/embed fp32 유지) — 엔진 q4와 동일 가중치.

덤프:
  공통        : token_ids[S], rope_cos/sin [S, rotary_dim] (full-attn 레이어에서 캡처), meta.json
  {tag}_hs_{l}: 레이어 l 입력 hidden [S,H] (l=0..L) — 마지막은 final-norm 적용 후 (HF output_hidden_states)
  {tag}_prenorm: final-norm '입력' hidden [S,H] (model.norm forward_pre_hook) — MTP가 받을 값
  {tag}_logits : [S,V]
  GDN 격리(layer 0): in_proj_qkv 출력, post-conv+silu mixed_qkv, g, beta, z, delta-rule core_attn_out,
                     recurrent 레퍼런스 core_attn_out(동일입력), gated-norm 출력, out_proj 출력
사용:
  python tools/make_reference_qwen35.py --model Qwen/Qwen3.5-0.8B --out artifacts/qwen35-0.8b/golden --prompt "The capital of France is"
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_weights import quantize_int4, QBLK
from extract_weights_qwen35 import is_q4


def dequant_roundtrip(w):
    shape = w.shape
    packed, scale = quantize_int4(w)
    n = w.size; nblk = n // QBLK
    lo = (packed & 0x0F).astype(np.float32); hi = (packed >> 4).astype(np.float32)
    qv = np.empty(n, dtype=np.float32); qv[0::2] = lo; qv[1::2] = hi
    return (((qv.reshape(nblk, QBLK) - 8) * scale[:, None]).reshape(shape)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gdn-layer", type=int, default=0)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    os.makedirs(args.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    try:
        base = model.get_submodule("model.language_model")  # 멀티모달 래퍼 로드 시
    except AttributeError:
        base = model.get_submodule("model")                 # text-only CausalLM 로드 시
    L = tcfg.num_hidden_layers
    layer_types = list(tcfg.layer_types)
    gl = args.gdn_layer
    assert layer_types[gl] == "linear_attention", f"layer {gl} is not linear"

    ids = tok(args.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    ids.cpu().numpy().astype(np.int32).tofile(os.path.join(args.out, "token_ids.bin"))
    print(f"[ref] prompt={args.prompt!r} tokens={ids.tolist()} S={S}")

    def w(name, arr):
        arr.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.out, name + ".bin"))

    # ---- 캡처용 hooks/wrappers ----
    cap = {}
    def dump_forward(tag):
        cap.clear()
        # pre-norm hidden: model.norm 입력
        h_pre = []
        def norm_pre(mod, inp): h_pre.append(inp[0].detach())
        h1 = base.norm.register_forward_pre_hook(norm_pre)
        # cos/sin: 첫 full-attn 레이어 self_attn 의 position_embeddings
        fa = layer_types.index("full_attention")
        def attn_pre(mod, a, kw):
            if "cos_sin" not in cap:
                pe = kw.get("position_embeddings") or (a[1] if len(a) > 1 else None)
                if pe is not None: cap["cos_sin"] = (pe[0].detach(), pe[1].detach())
        h2 = base.layers[fa].self_attn.register_forward_pre_hook(attn_pre, with_kwargs=True)
        # GDN layer gl 격리 캡처
        la = base.layers[gl].linear_attn
        hooks = [h1, h2]
        hooks.append(la.in_proj_qkv.register_forward_hook(lambda m, i, o: cap.__setitem__("in_proj_qkv", o.detach())))
        hooks.append(la.in_proj_z.register_forward_hook(lambda m, i, o: cap.__setitem__("in_proj_z", o.detach())))
        hooks.append(la.conv1d.register_forward_hook(lambda m, i, o: cap.__setitem__("conv_raw", o.detach())))
        hooks.append(la.norm.register_forward_pre_hook(
            lambda m, a, kw: cap.__setitem__("norm_in", (a[0].detach(), (kw.get("gate") if kw.get("gate") is not None else a[1]).detach())), with_kwargs=True))
        hooks.append(la.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("norm_out", o.detach())))
        hooks.append(la.out_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("out_proj", o.detach())))
        # delta-rule 함수 래핑 (HF가 prefill에서 쓰는 chunk + 동일입력 recurrent 레퍼런스)
        orig_chunk = la.chunk_gated_delta_rule
        def wrap(q, k, v, g, beta, **kw):
            cap["dr_in"] = {x: torch.as_tensor(val).detach() for x, val in
                            dict(q=q, k=k, v=v, g=g, beta=beta).items()}
            out = orig_chunk(q, k, v, g, beta, **kw)
            cap["dr_chunk_out"] = out[0].detach()
            # 동일 입력으로 recurrent 레퍼런스 (state 0, t순 누적) — 엔진 naive 커널의 tight 골든
            rec, _ = M.torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta,
                        initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
            cap["dr_recurrent_out"] = rec.detach()
            return out
        la.chunk_gated_delta_rule = wrap

        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        for h in hooks: h.remove()
        la.chunk_gated_delta_rule = orig_chunk

        hs = out.hidden_states
        for l, h in enumerate(hs): w(f"{tag}_hs_{l}", h[0])
        w(f"{tag}_logits", out.logits[0])
        w(f"{tag}_prenorm", h_pre[-1][0])
        # GDN 격리 골든
        post_conv = F.silu(cap["conv_raw"][:, :, :S])           # [B, conv_dim, S]
        w(f"{tag}_gdn_in_proj_qkv", cap["in_proj_qkv"][0])       # [S, 6144]
        w(f"{tag}_gdn_post_conv",   post_conv[0].transpose(0, 1))# [S, 6144]
        w(f"{tag}_gdn_dr_q", cap["dr_in"]["q"][0]); w(f"{tag}_gdn_dr_k", cap["dr_in"]["k"][0])
        w(f"{tag}_gdn_dr_v", cap["dr_in"]["v"][0]); w(f"{tag}_gdn_dr_g", cap["dr_in"]["g"][0])
        w(f"{tag}_gdn_dr_beta", cap["dr_in"]["beta"][0])
        w(f"{tag}_gdn_dr_chunk_out", cap["dr_chunk_out"][0])
        w(f"{tag}_gdn_dr_recurrent_out", cap["dr_recurrent_out"][0])
        w(f"{tag}_gdn_z", cap["in_proj_z"][0])                  # [S, 2048] (gate z, reshape→[S*16,128])
        w(f"{tag}_gdn_norm_out", cap["norm_out"])               # [S*16, 128] gated-norm 출력
        w(f"{tag}_gdn_out_proj", cap["out_proj"][0])            # [S, H]
        am = out.logits[0].argmax(-1).tolist()
        dr_diff = (cap["dr_chunk_out"] - cap["dr_recurrent_out"]).abs().max().item()
        print(f"[ref:{tag}] hs={len(hs)} logits={tuple(out.logits[0].shape)} argmax={am} "
              f"chunk-vs-recurrent maxabs={dr_diff:.2e}")
        if "cos_sin" in cap:
            w("rope_cos", cap["cos_sin"][0][0]); w("rope_sin", cap["cos_sin"][1][0])
            print(f"[ref] rope cos/sin shape={tuple(cap['cos_sin'][0].shape)}")
        return am

    print("[ref] golden_fp32 ...")
    am32 = dump_forward("fp32")

    # ---- q4 round-trip (큰 matmul만; extract 정책과 동일) ----
    print("[ref] int4 round-trip on q4-policy linear weights ...")
    with torch.no_grad():
        for name, p in base.named_parameters():  # 상대명: layers.., embed_tokens.weight, norm.weight
            if name.startswith("visual") or name.startswith("mtp"): continue
            if name == "embed_tokens.weight": continue  # embed fp32 유지
            if not is_q4(name, p.dim()): continue
            p.data.copy_(torch.from_numpy(dequant_roundtrip(p.detach().cpu().float().numpy())))
        # tied lm_head: embed 와 공유 → untie 후 q4 (embed 는 위에서 skip 해 fp32 유지)
        if bool(getattr(cfg, "tie_word_embeddings", False)):
            ew = base.embed_tokens.weight.detach().cpu().float().numpy()
            model.lm_head.weight = torch.nn.Parameter(torch.from_numpy(dequant_roundtrip(ew)), requires_grad=False)
    print("[ref] golden_q4 ...")
    amq = dump_forward("q4")

    meta = {"model": args.model, "prompt": args.prompt, "S": S, "token_ids": ids[0].tolist(),
            "num_layers": L, "layer_types": layer_types, "gdn_layer": gl,
            "argmax_fp32": am32, "argmax_q4": amq}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"[ref] done → {args.out} (fp32 vs q4 last-token: "
          f"{am32[-1]} vs {amq[-1]} {'MATCH' if am32[-1]==amq[-1] else 'DIFF'})")


if __name__ == "__main__":
    main()
