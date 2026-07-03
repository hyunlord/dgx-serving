#!/usr/bin/env python3
"""Gemma4 decode 연속성 골든 (M4 STEP A-1): HF greedy generate N토큰 + 전체 위치 rope 캡처.

프리필(S)+decode(N스텝) 연속성 게이트용. 엔진: 프리필 base=0 후 base=S..S+N-1 1토큰씩 →
각 스텝 argmax가 HF greedy 다음 토큰과 일치(KV 이어받기 정확성). bf16 로드(31B, argmax 기준).
출력: gen_tokens.bin(프롬프트+생성 전체 [S+N]), gen_argmax.json(스텝별 다음토큰),
      rope_full_cos/sin·rope_sliding_cos/sin [S+N, hd] (전체 위치 — decode rope offset).
사용:
  python tools/make_reference_gemma4_gen.py --model google/gemma-4-31B-it --out artifacts/gemma4-31b/golden \
      --prompt "The capital of France is" --n 12
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    os.makedirs(args.out, exist_ok=True)
    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[gen] loading {args.model} (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    try:
        base = model.get_submodule("model.language_model")
    except AttributeError:
        base = model.get_submodule("model")

    ids = tok(args.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    # greedy generate N — 각 스텝 다음 토큰(argmax). use_cache로 실제 decode 경로.
    steps = []
    pkv = None
    with torch.no_grad():
        for t in range(args.n):
            inp = ids if t == 0 else nxt
            out = model(inp, use_cache=True, past_key_values=pkv)
            pkv = out.past_key_values
            tokv = int(out.logits[0, -1].argmax().item())
            steps.append({"pos": S + t, "next_token": tokv})
            nxt = torch.tensor([[tokv]])
    full_ids = ids[0].tolist() + [s["next_token"] for s in steps]
    total = S + args.n
    np.array(full_ids, dtype=np.int32).tofile(os.path.join(args.out, "gen_tokens.bin"))
    print(f"[gen] prompt S={S} + gen {args.n} = {total}. steps: {[s['next_token'] for s in steps]}", flush=True)

    # 전체 위치 rope 캡처: full/sliding 각 첫 층 self_attn 에 position_ids=arange(total) forward hook.
    def w(name, arr): arr.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.out, name + ".bin"))
    layer_types = list(tcfg.layer_types)
    cap = {}
    hooks = []
    for lt, key in (("full_attention", "rope_full"), ("sliding_attention", "rope_sliding")):
        li = layer_types.index(lt)
        def mk(key):
            def pre(mod, a, kw):
                if key not in cap:
                    pe = kw.get("position_embeddings") or (a[1] if len(a) > 1 else None)
                    if pe is not None: cap[key] = (pe[0].detach(), pe[1].detach())
            return pre
        hooks.append(base.layers[li].self_attn.register_forward_pre_hook(mk(key), with_kwargs=True))
    with torch.no_grad():
        model(torch.tensor([full_ids]), use_cache=False)   # 전체 위치 한 번에 → rope[total]
    for h in hooks: h.remove()
    for key, fn in (("rope_full", "gen_rope_full"), ("rope_sliding", "gen_rope_sliding")):
        w(fn + "_cos", cap[key][0][0]); w(fn + "_sin", cap[key][1][0])
        print(f"[gen] {fn} shape={tuple(cap[key][0].shape)}")
    json.dump({"S": S, "n": args.n, "total": total, "prompt": args.prompt,
               "prompt_tokens": ids[0].tolist(), "gen_steps": steps},
              open(os.path.join(args.out, "gen_meta.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[gen] done → {args.out}")


if __name__ == "__main__":
    main()
