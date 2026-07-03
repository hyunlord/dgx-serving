#!/usr/bin/env python3
"""Gemma4 dense HF 골든 덤프 (M4 Phase 0). config 기반 — make_reference_qwen35 패턴.

골든 2종:
  golden_fp32 : 로드 dtype 그대로 forward (31B는 --dtype bfloat16 — fp32 full 로드 118GB는 메모리 가드 초과.
                태그는 테스트 재사용 위해 fp32 유지, meta.json에 실제 dtype 정직 기록).
  golden_q4   : 큰 matmul q4 round-trip (extract is_q4 정책과 동일 가중치) — 엔진 q4 게이트.

덤프: token_ids, rope cos/sin **2세트**(full/sliding 층 각각 — Gemma4 이원 RoPE), {tag}_hs_{l}, {tag}_logits.
사용:
  python tools/make_reference_gemma4.py --model google/gemma-4-31B-it --out artifacts/gemma4-31b/golden \
      --dtype bfloat16 --prompt "The capital of France is"
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_weights import quantize_int4, QBLK
from extract_weights_qwen35 import is_q4
from make_reference_qwen35 import dequant_roundtrip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    os.makedirs(args.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    tok = AutoTokenizer.from_pretrained(args.model)
    _dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[ref] loading {args.model} ({args.dtype})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=_dt, low_cpu_mem_usage=True)
    model.eval()
    try:
        base = model.get_submodule("model.language_model")   # 멀티모달 래퍼
    except AttributeError:
        base = model.get_submodule("model")                  # text-only
    layer_types = list(tcfg.layer_types)

    ids = tok(args.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    ids.cpu().numpy().astype(np.int32).tofile(os.path.join(args.out, "token_ids.bin"))
    print(f"[ref] prompt={args.prompt!r} tokens={ids.tolist()} S={S}")

    def w(name, arr):
        arr.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.out, name + ".bin"))

    def dump_forward(tag):
        cap = {}
        # cos/sin 2세트: 첫 full 층·첫 sliding 층 self_attn 의 position_embeddings 캡처
        hooks = []
        for lt, key in (("full_attention", "cs_full"), ("sliding_attention", "cs_sliding")):
            li = layer_types.index(lt)
            def mk(key):
                def pre(mod, a, kw):
                    if key not in cap:
                        pe = kw.get("position_embeddings") or (a[1] if len(a) > 1 else None)
                        if pe is not None: cap[key] = (pe[0].detach(), pe[1].detach())
                return pre
            hooks.append(base.layers[li].self_attn.register_forward_pre_hook(mk(key), with_kwargs=True))
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        for h in hooks: h.remove()

        hs = out.hidden_states
        for l, h in enumerate(hs): w(f"{tag}_hs_{l}", h[0])
        w(f"{tag}_logits", out.logits[0])
        am = out.logits[0].argmax(-1).tolist()
        print(f"[ref:{tag}] hs={len(hs)} logits={tuple(out.logits[0].shape)} argmax={am}")
        for key, fn in (("cs_full", "rope_full"), ("cs_sliding", "rope_sliding")):
            if key in cap and not os.path.exists(os.path.join(args.out, fn + "_cos.bin")):
                w(fn + "_cos", cap[key][0][0]); w(fn + "_sin", cap[key][1][0])
                print(f"[ref] {fn} cos/sin shape={tuple(cap[key][0].shape)}")
        return am

    print("[ref] golden_fp32 (실제 dtype=%s — meta 기록) ..." % args.dtype)
    am_base = dump_forward("fp32")

    print("[ref] int4 round-trip on q4-policy linear weights ...")
    with torch.no_grad():
        for name, p in base.named_parameters():
            if name == "embed_tokens.weight": continue      # embed fp32 유지
            if not is_q4(name, p.dim()): continue
            p.data.copy_(torch.from_numpy(dequant_roundtrip(p.detach().cpu().float().numpy())).to(p.dtype))
        if bool(getattr(cfg, "tie_word_embeddings", False) or getattr(tcfg, "tie_word_embeddings", False)):
            ew = base.embed_tokens.weight.detach().cpu().float().numpy()
            model.lm_head.weight = torch.nn.Parameter(
                torch.from_numpy(dequant_roundtrip(ew)).to(_dt), requires_grad=False)

    print("[ref] golden_q4 ...")
    am_q4 = dump_forward("q4")

    meta = {"model": args.model, "prompt": args.prompt, "S": S,
            "token_ids": ids[0].tolist(), "dtype": args.dtype,
            "note": "31B급은 bfloat16 로드(fp32 full 118GB=메모리 가드 초과) — fp32 태그는 하네스 관례, 실제 연산 dtype은 여기 기록",
            "argmax_base": am_base, "argmax_q4": am_q4}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[ref] done → {args.out}")


if __name__ == "__main__":
    main()
