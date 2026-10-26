#!/usr/bin/env python3
"""HF 모델 → 엔진 포맷(int4 q4.bin + scale.bin) 추출. config 기반 범용 (모델 하드코딩 금지).

선형(q/k/v/o/gate/up/down/lm_head)은 int4 양자화, embedding/norm은 fp32 유지.
양자화 스킴은 docs/01-stage1-blueprint.md 와 정확히 일치해야 한다 (커널 dequant 전제).

사용:
  python tools/extract_weights.py --model HuggingFaceTB/SmolLM3-3B --out artifacts/smollm3-3b/weights
"""
import argparse, json, os, re, sys
import numpy as np
from typing import TypedDict

QBLK = 32  # docs/blueprint와 일치. 컴파일 타임 상수와 동일.
QUANT_MODES = ("rtn", "qat-ident", "qat-q40", "orig-rtnfull", "orig-clip")


class ReconstructionStats(TypedDict):
    elements: int
    blocks: int
    squared_error: float
    squared_input: float
    relative_error: float
    minus_eight: int
    plus_seven: int


def variant_projection(name: str) -> bool:
    """Experimental policies affect only the seven decoder linear projections."""
    return bool(re.fullmatch(r"L\d+_(?:(?:self_attn\.)?[qkvo](?:_proj)?|(?:mlp\.)?(?:gate|up|down)(?:_proj)?)", name))


def add_quant_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quant", choices=QUANT_MODES, default="rtn")
    parser.add_argument("--quant-rule", choices=("signed-max", "positive-max"))
    parser.add_argument("--quant-c", type=float)


def validate_quant_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.quant == "qat-ident":
        if args.quant_rule is None:
            parser.error("qat-ident requires the B' selected --quant-rule")
        if args.quant_rule == "positive-max" and (args.quant_c is None or not np.isfinite(args.quant_c) or args.quant_c <= 0):
            parser.error("positive-max requires finite positive --quant-c")
    elif args.quant_rule is not None or args.quant_c is not None:
        parser.error("--quant-rule/--quant-c apply only to qat-ident")
    if args.quant_rule != "positive-max" and args.quant_c is not None:
        parser.error("--quant-c applies only to positive-max")
    if args.quant != "rtn" and getattr(args, "scale_dtype", "fp32") != "fp32":
        parser.error("experimental quantization requires fp32 scales")


def quantize_variant(w_2d: np.ndarray, mode: str = "rtn", rule: str | None = None,
                     c: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Round onto an explicit FP32 scale grid; no BF16 product rounding."""
    if mode == "rtn":
        return quantize_int4(w_2d)
    if mode not in QUANT_MODES or (mode == "qat-ident" and rule not in ("signed-max", "positive-max")):
        raise ValueError("unknown mode or missing qat-ident rule")
    if rule == "positive-max" and (c is None or not np.isfinite(c) or c <= 0):
        raise ValueError("positive-max requires a finite positive divisor")
    w = np.ascontiguousarray(w_2d, dtype=np.float32).reshape(-1, QBLK)
    packed = np.empty(w.size // 2, dtype=np.uint8)
    scales = np.empty(w.shape[0], dtype=np.float32)
    for start in range(0, len(w), 65536):
        block = w[start:start + 65536]
        if not np.isfinite(block).all():
            raise ValueError("nonfinite weight")
        peak = np.argmax(np.abs(block), axis=1)
        scale = block[np.arange(len(block)), peak] / np.float32(-8)
        if mode == "qat-ident" and rule == "positive-max":
            scale = np.max(np.abs(block), axis=1) / np.float32(c)
        scale[scale == 0] = np.float32(1e-8)
        q = np.clip(np.rint(block / scale[:, None]), -8, 7)
        if mode == "orig-clip":
            best = np.square((q * scale[:, None]).astype(np.float64) - block).sum(axis=1)
            selected = scale.copy()
            for step in range(1, 11):
                candidate = scale * np.float32((100 - 3 * step) / 100)
                candidate_q = np.clip(np.rint(block / candidate[:, None]), -8, 7)
                error = np.square((candidate_q * candidate[:, None]).astype(np.float64) - block).sum(axis=1)
                improve = error < best
                selected[improve], q[improve], best[improve] = candidate[improve], candidate_q[improve], error[improve]
            scale = selected
        nibble = (q.astype(np.int8) + 8).astype(np.uint8).reshape(-1)
        packed[start * 16:(start + len(block)) * 16] = nibble[::2] | (nibble[1::2] << 4)
        scales[start:start + len(block)] = scale
    return packed, scales


def reconstruction_stats(w_2d: np.ndarray, packed: np.ndarray, scale: np.ndarray) -> ReconstructionStats:
    w = np.ascontiguousarray(w_2d, dtype=np.float32).reshape(-1, QBLK)
    squared_error = squared_input = 0.0
    low_count = high_count = 0
    for start in range(0, len(w), 65536):
        block = w[start:start + 65536]
        data = packed[start * 16:(start + len(block)) * 16]
        q = np.empty(data.size * 2, dtype=np.float32)
        q[::2], q[1::2] = (data & 15).astype(np.float32) - 8, (data >> 4).astype(np.float32) - 8
        reconstructed = q.reshape(-1, 32) * scale[start:start + len(block), None]
        squared_error += float(np.square(reconstructed.astype(np.float64) - block).sum())
        squared_input += float(np.square(block.astype(np.float64)).sum())
        low_count += int(np.count_nonzero(q == -8))
        high_count += int(np.count_nonzero(q == 7))
    return {"elements": w.size, "blocks": len(w), "squared_error": squared_error,
            "squared_input": squared_input, "relative_error": float(np.sqrt(squared_error / squared_input)) if squared_input else 0.0,
            "minus_eight": low_count, "plus_seven": high_count}


def quantize_int4(w_2d: np.ndarray):
    """w_2d: float32 [OUT, IN], IN은 QBLK 배수. 반환 (packed uint8 [OUT*IN/2], scale fp32 [OUT*IN/QBLK]).

    blueprint 스킴: scale=maxabs/7, q=clip(round(w/scale)+8,0,15), pack 2/byte (low=even, high=odd).
    블록은 flatten(row-major) 후 32단위 — IN이 32배수이므로 각 블록은 한 OUT row 내부에 머묾.
    """
    w = np.ascontiguousarray(w_2d, dtype=np.float32).reshape(-1)
    n = w.size
    assert n % QBLK == 0, f"size {n} not multiple of QBLK={QBLK}"
    nblk = n // QBLK
    wr = w.reshape(nblk, QBLK)
    scale = np.abs(wr).max(axis=1) / 7.0
    scale[scale == 0] = 1e-8
    q = np.round(wr / scale[:, None]) + 8
    q = np.clip(q, 0, 15).astype(np.uint8).reshape(-1)
    packed = (q[0::2] | (q[1::2] << 4)).astype(np.uint8)
    return packed, scale.astype(np.float32)


def dequant_relerr(w_2d, packed, scale):
    w = np.ascontiguousarray(w_2d, dtype=np.float32).reshape(-1)
    nblk = w.size // QBLK
    lo = (packed & 0x0F).astype(np.float32)
    hi = (packed >> 4).astype(np.float32)
    qv = np.empty(w.size, dtype=np.float32)
    qv[0::2] = lo; qv[1::2] = hi
    deq = ((qv.reshape(nblk, QBLK) - 8) * scale[:, None]).reshape(-1)
    return float(np.sqrt(((deq - w) ** 2).sum() / ((w ** 2).sum() + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    add_quant_arguments(ap)
    args = ap.parse_args()
    validate_quant_arguments(args, ap)
    os.makedirs(args.out, exist_ok=(args.quant == "rtn"))

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model)
    print(f"[extract] loading {args.model} (fp32)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    sd = model.state_dict()

    H  = cfg.hidden_size
    L  = cfg.num_hidden_layers
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    # no_rope_layers: 1=apply rope, 0=NoPE. 없으면 전부 rope.
    use_rope = list(getattr(cfg, "no_rope_layers", None) or [1] * L)

    # rope_theta: transformers 5.x SmolLM3는 rope_parameters/rope_scaling dict 에 넣음 (top-level 아님).
    def _rope_theta(cfg):
        for attr in ("rope_parameters", "rope_scaling"):
            rp = getattr(cfg, attr, None)
            if isinstance(rp, dict) and rp.get("rope_theta") is not None:
                return float(rp["rope_theta"])
        return float(getattr(cfg, "rope_theta", 10000.0))

    manifest = {
        "model": args.model,
        "hidden_size": H,
        "intermediate_size": cfg.intermediate_size,
        "num_layers": L,
        "num_q_heads": cfg.num_attention_heads,
        "num_kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": hd,
        "vocab_size": cfg.vocab_size,
        "rms_norm_eps": float(cfg.rms_norm_eps),
        "rope_theta": _rope_theta(cfg),
        "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "use_rope": use_rope,
        "qblk": QBLK,
        "hidden_act": getattr(cfg, "hidden_act", "silu"),
    }

    def save_f32(name, t):
        a = t.detach().cpu().float().contiguous().numpy().astype(np.float32)
        a.tofile(os.path.join(args.out, name + ".bin"))
        return a.shape

    quantization_report = []
    def save_q4(name, t):
        w = t.detach().cpu().float().contiguous().numpy().astype(np.float32)  # [OUT, IN]
        policy = args.quant if variant_projection(name) else "rtn"
        packed, scale = quantize_variant(w, policy, args.quant_rule, args.quant_c)
        packed.tofile(os.path.join(args.out, name + "_q4.bin"))
        scale.tofile(os.path.join(args.out, name + "_scale.bin"))
        if args.quant != "rtn":
            baseline_packed, baseline_scale = quantize_int4(w)
            quantization_report.append({"tensor": name, "shape": list(w.shape), "policy": policy,
                                        "variant": reconstruction_stats(w, packed, scale),
                                        "baseline_rtn": reconstruction_stats(w, baseline_packed, baseline_scale)})
        return w.shape, dequant_relerr(w, packed, scale)

    # 키 패턴 매칭 (Llama/SmolLM3 계열 범용). 새 아키텍처는 여기 매핑만 확장.
    def k(name):
        if name in sd: return sd[name]
        raise KeyError(f"missing weight key: {name}")

    print("[extract] embed/final_norm (fp32) + lm_head (q4)...", flush=True)
    save_f32("w_embed", k("model.embed_tokens.weight"))
    save_f32("w_final_norm", k("model.norm.weight"))
    # tied면 lm_head.weight == embed.weight; 키가 없을 수 있으니 fallback
    lm = sd.get("lm_head.weight", sd.get("model.embed_tokens.weight"))
    _, lm_err = save_q4("w_lm_head", lm)
    print(f"  lm_head rel-err {lm_err:.4f}")

    errs = []
    for i in range(L):
        pfx = f"model.layers.{i}."
        save_f32(f"L{i}_input_ln", k(pfx + "input_layernorm.weight"))
        save_f32(f"L{i}_post_ln",  k(pfx + "post_attention_layernorm.weight"))
        for tag, key in [("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
                         ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
                         ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"),
                         ("down", "mlp.down_proj.weight")]:
            _, e = save_q4(f"L{i}_{tag}", k(pfx + key))
            errs.append(e)
        if (i + 1) % 9 == 0:
            print(f"  ... {i+1}/{L} (mean rel-err {np.mean(errs):.4f})", flush=True)

    manifest["quant_mean_relerr"] = float(np.mean(errs))
    if args.quant != "rtn":
        manifest.update({"quant_method": args.quant, "quant_rule": args.quant_rule,
                         "quant_c": args.quant_c, "scale_dtype": 0, "lm_head_quant": "rtn"})
        with open(os.path.join(args.out, "quantization_report.json"), "x") as f:
            json.dump(quantization_report, f, indent=2)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[extract] done. mean rel-err {np.mean(errs):.4f}. manifest + weights → {args.out}")


if __name__ == "__main__":
    main()
