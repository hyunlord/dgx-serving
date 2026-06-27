#!/usr/bin/env python3
"""Qwen3.5 full-attention 레이어 격리 골든 덤프 (GQA 그룹핑 보편화 검증용).

목적: 엔진 full-attn 경로(q_norm/k_norm + partial rope + GQA attn + output_gate)를 HF 와 격리 대조.
      proj 가중치는 q4-only 추출이라 fp32 미보유 → proj '출력'을 골든으로 떠서 엔진 커널만 격리.

덤프(레이어 = 첫 full_attention, config 주도):
  token_ids[S], rope_cos/sin[S, rot]
  attn_qproj[S, NQ*HD*2]  : q_proj 출력 (per-head Q|gate chunk 전)
  attn_kproj[S, NKV*HD]   : k_proj 출력 (k_norm 전)
  attn_vproj[S, NKV*HD]   : v_proj 출력
  attn_o_in [S, NQ*HD]    : o_proj 입력 = attn_output * sigmoid(gate) (엔진 비교 타깃)
메모리: bf16 로드(27B 54GB, fp32 108GB OOM 회피). grouping 버그=O(0.1~1)이라 bf16 노이즈(~1e-2) 무관.
사용:
  python tools/make_attn_reference_qwen35.py --model Qwen/Qwen3.5-27B --out artifacts/qwen35-27b/golden-attn
"""
import argparse, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--attn-layer", type=int, default=-1, help="-1 = 첫 full_attention 레이어")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    os.makedirs(args.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    tok = AutoTokenizer.from_pretrained(args.model)
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[attn-ref] loading {args.model} ({args.dtype})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, low_cpu_mem_usage=True)
    model.eval()
    try:
        base = model.get_submodule("model.language_model")
    except AttributeError:
        base = model.get_submodule("model")
    layer_types = list(tcfg.layer_types)
    al = args.attn_layer if args.attn_layer >= 0 else layer_types.index("full_attention")
    assert layer_types[al] == "full_attention", f"layer {al} is not full_attention"
    print(f"[attn-ref] attn layer = {al} (NQ={tcfg.num_attention_heads} NKV={tcfg.num_key_value_heads} HD={tcfg.head_dim})")

    ids = tok(args.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    ids.cpu().numpy().astype(np.int32).tofile(os.path.join(args.out, "token_ids.bin"))

    def w(name, arr):
        arr.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.out, name + ".bin"))

    cap = {}
    sa = base.layers[al].self_attn
    hooks = []
    hooks.append(sa.q_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("qproj", o.detach())))
    hooks.append(sa.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("kproj", o.detach())))
    hooks.append(sa.v_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("vproj", o.detach())))
    hooks.append(sa.o_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("o_in", i[0].detach())))
    def attn_pre(mod, a, kw):
        pe = kw.get("position_embeddings") or (a[1] if len(a) > 1 else None)
        if pe is not None: cap["cos_sin"] = (pe[0].detach(), pe[1].detach())
    hooks.append(sa.register_forward_pre_hook(attn_pre, with_kwargs=True))

    with torch.no_grad():
        model(ids)
    for h in hooks: h.remove()

    w("attn_qproj", cap["qproj"][0]); w("attn_kproj", cap["kproj"][0])
    w("attn_vproj", cap["vproj"][0]); w("attn_o_in", cap["o_in"][0])
    cos, sin = cap["cos_sin"]
    w("rope_cos", cos[0]); w("rope_sin", sin[0])
    print(f"[attn-ref] dumped S={S} qproj{tuple(cap['qproj'][0].shape)} kproj{tuple(cap['kproj'][0].shape)} "
          f"o_in{tuple(cap['o_in'][0].shape)} rope{tuple(cos[0].shape)} → {args.out}")

if __name__ == "__main__":
    main()
