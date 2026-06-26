#!/usr/bin/env python3
"""HF fp32 forward로 골든(단계별 중간값 + logits) 덤프. config 기반.

골든 2종 (docs/01-stage1-blueprint.md 검증전략):
  golden_fp32 : 원본 fp32 가중치 forward.
  golden_q4   : 각 선형 가중치를 dequant(quant(W))로 round-trip 후 forward
                (= 엔진이 쓰는 q4 가중치와 동일 가중치에서의 fp32 수학 → 커널 버그 격리 기준).

각 골든: per-layer 입력 hidden state [S,H] (hs_{l}.bin, l=0..L), 최종 logits [S,V] (logits.bin).
공통: token_ids.bin (int32 [S]), rope_cos.bin/rope_sin.bin (fp32 [S,HD]), meta.json.

사용:
  python tools/make_reference.py --model HuggingFaceTB/SmolLM3-3B \
      --out artifacts/smollm3-3b/golden --prompt "The capital of France is"
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_weights import quantize_int4, QBLK


def dequant_roundtrip(w_np: np.ndarray) -> np.ndarray:
    """w_np [OUT,IN] fp32 → dequant(quant(w)) (동일 shape)."""
    shape = w_np.shape
    packed, scale = quantize_int4(w_np)
    n = w_np.size
    nblk = n // QBLK
    lo = (packed & 0x0F).astype(np.float32)
    hi = (packed >> 4).astype(np.float32)
    qv = np.empty(n, dtype=np.float32)
    qv[0::2] = lo; qv[1::2] = hi
    deq = ((qv.reshape(nblk, QBLK) - 8) * scale[:, None]).reshape(shape)
    return deq.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    cfg = AutoConfig.from_pretrained(args.model)
    H, L = cfg.hidden_size, cfg.num_hidden_layers
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    ids.cpu().numpy().astype(np.int32).tofile(os.path.join(args.out, "token_ids.bin"))
    print(f"[ref] prompt={args.prompt!r} tokens={ids.tolist()} S={S}")

    # rope cos/sin (가중치 무관 — 한 번만)
    with torch.no_grad():
        pos = torch.arange(S).unsqueeze(0)
        h0 = model.model.embed_tokens(ids)
        cos, sin = model.model.rotary_emb(h0, pos)
    cos[0].detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.out, "rope_cos.bin"))
    sin[0].detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.out, "rope_sin.bin"))

    def dump_forward(tag):
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple len L+1, 각 [1,S,H]; hs[0]=embed출력, hs[l]=레이어 l 입력
        for l, h in enumerate(hs):
            h[0].detach().cpu().float().numpy().astype(np.float32).tofile(
                os.path.join(args.out, f"{tag}_hs_{l}.bin"))
        logits = out.logits[0].detach().cpu().float().numpy().astype(np.float32)  # [S,V]
        logits.tofile(os.path.join(args.out, f"{tag}_logits.bin"))
        am = logits.argmax(-1).tolist()
        print(f"[ref:{tag}] dumped {len(hs)} hidden states + logits[S,V]={logits.shape}; argmax/pos={am}")
        return am

    print("[ref] golden_fp32 forward...")
    am_fp32 = dump_forward("fp32")

    # ---- q4 round-trip: 선형 가중치만 dequant(quant(W)) 로 교체 (embed fp32 유지) ----
    print("[ref] applying int4 round-trip to linear weights...")
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    with torch.no_grad():
        # lm_head: tied면 embed와 공유 → untie 후 lm_head만 q4 (embed는 fp32 유지)
        lmw = model.lm_head.weight.detach().cpu().float().numpy()
        new_lm = torch.from_numpy(dequant_roundtrip(lmw))
        if tied:
            model.lm_head.weight = torch.nn.Parameter(new_lm, requires_grad=False)
        else:
            model.lm_head.weight.data.copy_(new_lm)
        for i in range(L):
            ly = model.model.layers[i]
            for mod in [ly.self_attn.q_proj, ly.self_attn.k_proj, ly.self_attn.v_proj,
                        ly.self_attn.o_proj, ly.mlp.gate_proj, ly.mlp.up_proj, ly.mlp.down_proj]:
                w = mod.weight.detach().cpu().float().numpy()
                mod.weight.data.copy_(torch.from_numpy(dequant_roundtrip(w)))

    print("[ref] golden_q4 forward...")
    am_q4 = dump_forward("q4")

    meta = {"model": args.model, "prompt": args.prompt, "S": S,
            "token_ids": ids[0].tolist(), "hidden_size": H, "num_layers": L,
            "head_dim": hd, "vocab_size": cfg.vocab_size,
            "argmax_fp32": am_fp32, "argmax_q4": am_q4}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ref] done → {args.out} (fp32 vs q4 argmax match: {am_fp32 == am_q4})")


if __name__ == "__main__":
    main()
