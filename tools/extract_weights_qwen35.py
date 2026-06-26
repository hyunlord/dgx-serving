#!/usr/bin/env python3
"""Qwen3.5-0.8B 본체(text) → 엔진 포맷 추출. config 기반.

양자화 정책 (사용자 합의): 큰 matmul만 int4 q4, gating/norm 파라미터는 fp32.
  q4  : q/k/v/o_proj, in_proj_qkv, in_proj_z, out_proj, mlp gate/up/down, (tied) lm_head
  fp32: 모든 norm(input/post/final/q_norm/k_norm/gdn norm), conv1d, A_log, dt_bias, in_proj_a, in_proj_b, embed
파일명 = state_dict 상대키 (layers.{i}. → L{i}_, .weight strip). 범용 스킴.
q4 양자화 스킴은 extract_weights.quantize_int4 와 공유 (docs/01-stage1-blueprint.md).

사용:
  python tools/extract_weights_qwen35.py --model Qwen/Qwen3.5-0.8B --out artifacts/qwen35-0.8b/weights
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_weights import quantize_int4, dequant_relerr, QBLK


def is_q4(name, ndim):
    if ndim != 2:                      # norms(1D), conv1d(3D), A_log/dt_bias(1D) → fp32
        return False
    if "in_proj_a" in name or "in_proj_b" in name:  # 작은 gating proj [16,1024] → fp32
        return False
    if name.endswith("embed_tokens.weight"):        # embed 는 fp32 (룩업)
        return False
    return True


def rope_theta(cfg):
    for attr in ("rope_parameters", "rope_scaling"):
        rp = getattr(cfg, attr, None)
        if isinstance(rp, dict) and rp.get("rope_theta") is not None:
            return float(rp["rope_theta"])
    return float(getattr(cfg, "rope_theta", 10000.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    os.makedirs(args.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    print(f"[extract] loading {args.model} (fp32)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    sd = model.state_dict()

    # weight key prefix 자동 감지
    prefix = None
    for p in ("model.language_model.", "model."):
        if (p + "embed_tokens.weight") in sd:
            prefix = p; break
    if prefix is None:
        raise RuntimeError("embed_tokens prefix 못 찾음")
    print(f"[extract] prefix = {prefix!r}")

    g = lambda k, d=None: getattr(tcfg, k, d)
    L = g("num_hidden_layers")
    layer_types = list(g("layer_types"))
    hd = g("head_dim") or (g("hidden_size") // g("num_attention_heads"))
    prf = (getattr(tcfg, "rope_parameters", {}) or {}).get("partial_rotary_factor",
           getattr(tcfg, "partial_rotary_factor", 1.0))
    manifest = {
        "model": args.model,
        "hidden_size": g("hidden_size"), "intermediate_size": g("intermediate_size"),
        "num_layers": L, "num_q_heads": g("num_attention_heads"),
        "num_kv_heads": g("num_key_value_heads"), "head_dim": hd,
        "vocab_size": g("vocab_size"), "rms_norm_eps": float(g("rms_norm_eps")),
        "rope_theta": rope_theta(cfg), "partial_rotary_factor": float(prf),
        "rotary_dim": int(round(hd * float(prf))),
        "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "qblk": QBLK, "hidden_act": g("hidden_act"),
        "layer_types": layer_types,
        # GDN dims
        "linear_num_key_heads": g("linear_num_key_heads"), "linear_num_value_heads": g("linear_num_value_heads"),
        "linear_key_head_dim": g("linear_key_head_dim"), "linear_value_head_dim": g("linear_value_head_dim"),
        "linear_conv_kernel": g("linear_conv_kernel_dim"),
        "attn_output_gate": bool(g("attn_output_gate", False)),
        "full_attention_interval": g("full_attention_interval"),
    }

    def save_f32(fname, t):
        a = t.detach().cpu().float().contiguous().numpy().astype(np.float32)
        a.tofile(os.path.join(args.out, fname + ".bin"))
    def save_q4(fname, t):
        w = t.detach().cpu().float().contiguous().numpy().astype(np.float32)
        packed, scale = quantize_int4(w)
        packed.tofile(os.path.join(args.out, fname + "_q4.bin"))
        scale.tofile(os.path.join(args.out, fname + "_scale.bin"))
        return dequant_relerr(w, packed, scale)

    def fname_of(rel):  # 상대키 → 파일 basename
        m = re.match(r"layers\.(\d+)\.(.*)", rel)
        if m:
            rest = m.group(2)
            if rest.endswith(".weight"): rest = rest[:-7]
            return f"L{m.group(1)}_{rest}"
        return None

    errs = []
    n_q4 = n_f32 = 0
    for key, t in sd.items():
        if not key.startswith(prefix):
            continue
        rel = key[len(prefix):]
        if rel.startswith("visual") or rel.startswith("mtp") or ".mtp" in rel:
            continue
        # top-level
        if rel == "embed_tokens.weight":
            save_f32("w_embed", t)
            if manifest["tie_word_embeddings"]:
                errs.append(save_q4("w_lm_head", t))  # tied lm_head
            continue
        if rel == "norm.weight":
            save_f32("w_final_norm", t); continue
        if rel == "lm_head.weight":  # untied 모델 대비
            errs.append(save_q4("w_lm_head", t)); continue
        fn = fname_of(rel)
        if fn is None:
            continue  # 본체 외 (rotary_emb buffers 등)
        if is_q4(key, t.dim()):
            errs.append(save_q4(fn, t)); n_q4 += 1
        else:
            save_f32(fn, t); n_f32 += 1

    manifest["quant_mean_relerr"] = float(np.mean(errs)) if errs else 0.0
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[extract] done. layers={L} ({layer_types.count('full_attention')} full / "
          f"{layer_types.count('linear_attention')} linear). q4 tensors={n_q4}, fp32={n_f32}. "
          f"mean q4 rel-err={manifest['quant_mean_relerr']:.4f} → {args.out}")


if __name__ == "__main__":
    main()
