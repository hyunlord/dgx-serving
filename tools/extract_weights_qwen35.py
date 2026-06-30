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
    # MoE fused experts: 3D batched [E, out, in] = 본체 bulk → q4 (per-block-32, expert offset 인덱싱).
    if "experts.gate_up_proj" in name or "experts.down_proj" in name:
        return True
    if ndim != 2:                      # norms(1D), conv1d(3D), A_log/dt_bias(1D) → fp32
        return False
    if "in_proj_a" in name or "in_proj_b" in name:  # 작은 gating proj [16,1024] → fp32
        return False
    if name.endswith("mlp.gate.weight"):            # MoE router (H→E) → fp32 (라우팅 정밀, gate_proj와 다름)
        return False
    if "shared_expert_gate" in name:                # shared expert sigmoid gate (H→1) → fp32
        return False
    if name.endswith("embed_tokens.weight"):        # embed 는 fp32 (룩업)
        return False
    return True


def rope_theta(cfg, tcfg=None):
    # rope_parameters/rope_scaling 는 text config 에 있음(멀티모달은 top cfg 에 없음) → 둘 다 확인.
    for c in (tcfg, cfg):
        if c is None: continue
        for attr in ("rope_parameters", "rope_scaling"):
            rp = getattr(c, attr, None)
            if isinstance(rp, dict) and rp.get("rope_theta") is not None:
                return float(rp["rope_theta"])
        if getattr(c, "rope_theta", None) is not None:
            return float(c.rope_theta)
    return 10000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="로드 dtype. 큰 모델(27B+)은 bfloat16로 OOM 회피 (q4 양자화는 source dtype에 거의 무관).")
    ap.add_argument("--q4-only", action="store_true",
                    help="big matmul의 fp32 복사본 생략(27B+ fp32 108GB는 엔진 OOM이라 불필요). q4+small-fp32만.")
    ap.add_argument("--scale-dtype", default="fp32", choices=["fp32", "fp16"],
                    help="weight scale dtype. fp16: scale fp32→fp16 변환 저장 (트래픽↓, rel 9.3e-4 안전).")
    args = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    os.makedirs(args.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    _dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    print(f"[extract] loading {args.model} ({args.dtype})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=_dt, low_cpu_mem_usage=True)
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
        "model_type": getattr(cfg, "model_type", None),   # qwen3_5 / qwen3_5_moe → 레지스트리 디스패치
        "hidden_size": g("hidden_size"), "intermediate_size": g("intermediate_size"),
        "num_layers": L, "num_q_heads": g("num_attention_heads"),
        "num_kv_heads": g("num_key_value_heads"), "head_dim": hd,
        "vocab_size": g("vocab_size"), "rms_norm_eps": float(g("rms_norm_eps")),
        "rope_theta": rope_theta(cfg, tcfg), "partial_rotary_factor": float(prf),
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
    # MoE dims (num_experts 있을 때만 — dense 모델엔 미추가, config 주도)
    if g("num_experts"):
        manifest.update({
            "num_experts": g("num_experts"),
            "num_experts_per_tok": g("num_experts_per_tok"),
            "moe_intermediate_size": g("moe_intermediate_size"),
            "shared_expert_intermediate_size": g("shared_expert_intermediate_size"),
            "norm_topk_prob": int(bool(g("norm_topk_prob", True))),  # int (C mf_int_opt 파싱)
        })

    scale_fp16 = (args.scale_dtype == "fp16")
    def save_f32(fname, t):
        a = t.detach().cpu().float().contiguous().numpy().astype(np.float32)
        a.tofile(os.path.join(args.out, fname + ".bin"))
    def save_q4(fname, t):
        w = t.detach().cpu().float().contiguous().numpy().astype(np.float32)
        packed, scale = quantize_int4(w)
        packed.tofile(os.path.join(args.out, fname + "_q4.bin"))
        if scale_fp16:
            scale.astype(np.float16).tofile(os.path.join(args.out, fname + "_scale.bin"))
        else:
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
        # untied lm_head: top-level "lm_head.weight" (prefix 밖) — tie=False 모델(27B 등) 대비
        if key == "lm_head.weight" or key.endswith(".lm_head.weight"):
            if "w_lm_head_q4.bin" not in os.listdir(args.out):
                errs.append(save_q4("w_lm_head", t)); n_q4 += 1
            continue
        if not key.startswith(prefix):
            continue
        rel = key[len(prefix):]
        if rel.startswith("visual") or rel.startswith("mtp") or ".mtp" in rel:
            continue
        # top-level
        if rel == "embed_tokens.weight":
            save_f32("w_embed", t)
            if manifest["tie_word_embeddings"]:
                errs.append(save_q4("w_lm_head", t))  # tied lm_head (q4)
                if not args.q4_only:
                    save_f32("w_lm_head_f32", t)       # + fp32 (검증/MTP 기준선 경로)
            continue
        if rel == "norm.weight":
            save_f32("w_final_norm", t); continue
        if rel == "lm_head.weight":  # untied 모델 대비
            errs.append(save_q4("w_lm_head", t)); continue
        fn = fname_of(rel)
        if fn is None:
            continue  # 본체 외 (rotary_emb buffers 등)
        if is_q4(key, t.dim()):
            errs.append(save_q4(fn, t))      # q4 경로
            if not args.q4_only:
                save_f32(fn + "_f32", t)     # fp32 경로 (검증/MTP 기준선)
            n_q4 += 1
        else:
            save_f32(fn, t); n_f32 += 1

    manifest["quant_mean_relerr"] = float(np.mean(errs)) if errs else 0.0
    manifest["scale_dtype"] = 1 if scale_fp16 else 0
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[extract] done. layers={L} ({layer_types.count('full_attention')} full / "
          f"{layer_types.count('linear_attention')} linear). q4 tensors={n_q4}, fp32={n_f32}. "
          f"mean q4 rel-err={manifest['quant_mean_relerr']:.4f} → {args.out}")


if __name__ == "__main__":
    main()
