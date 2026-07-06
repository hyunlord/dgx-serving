#!/usr/bin/env python3
"""Gemma4 assistant(0.4B MTP drafter, EAGLE류) → 엔진 포맷 추출 (draft-spec 2관문).
구조(코드 실사·docs/gemma-assistant-spec-measured.md): pre_projection[1024,7680]·4층 트렁크(q_proj/q_norm/o_proj만 —
k/v는 target KV 직독이라 없음·sandwich 4-norm·layer_scalar·GeGLU 8192)·model.norm·post_projection[3840,1024]·
tied lm_head(=embed_tokens [262144,1024])·centroid 비활성(텐서 자체 없음 확인).
★fp32 전량 + 큰 matmul q4(_q4/_scale, target blueprint 스킴 동일 — Lever1: draft 비용 bytes 8×↓).
q4 대상: q/o/gate/up/down·pre/post_projection·embed(assistant embed는 룩업이 아니라 tied lm_head 매트멀 전용
— target embed[룩업]와 달리 q4 안전). norm/layer_scalar/q_norm은 fp32 유지.
파일명 = state_dict 키(layers.{i}. → L{i}_, .weight strip) — weights_qwen/gemma4 관례.
사용: python tools/extract_weights_assistant.py --model google/gemma-4-12B-it-assistant --out artifacts/gemma4-12b-assistant/weights
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_weights import quantize_int4, QBLK

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    os.makedirs(args.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    t = cfg.get_text_config()
    print(f"[extract-ast] loading {args.model} (fp32 업캐스트 — bf16 체크포인트)...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    m.eval()
    sd = m.state_dict()

    def is_q4(k, v):   # 큰 matmul 전부(embed 포함 — lm_head 전용) q4. 1D(norm/scalar)는 fp32.
        return v.ndim == 2
    n = nq = 0
    for k, v in sd.items():
        if k == "lm_head.weight":  # tied(embed_tokens와 동일 스토리지) — embed만 저장
            continue
        name = k.replace("model.layers.", "L").replace(".weight", "").replace(".", "_")
        w = v.detach().float().numpy()
        w.tofile(os.path.join(args.out, name + ".bin"))
        if is_q4(k, v):
            packed, scale = quantize_int4(w)
            packed.tofile(os.path.join(args.out, name + "_q4.bin"))
            scale.tofile(os.path.join(args.out, name + "_scale.bin"))
            nq += 1
        n += 1
    print(f"[extract-ast] {n} tensors ({nq} q4+scale) → {args.out}", flush=True)

    mani = {
        "model": args.model,
        "num_layers": t.num_hidden_layers,
        "hidden_size": t.hidden_size,
        "intermediate_size": t.intermediate_size,
        "num_q_heads": t.num_attention_heads,
        "head_dim": t.head_dim,                     # sliding hd(256)
        "global_head_dim": t.global_head_dim,       # full hd(512)
        "num_kv_heads": t.num_key_value_heads,      # target sliding KV 헤드수(8) — cross-attn GQA용
        "num_global_kv_heads": t.num_global_key_value_heads,  # target full KV 헤드수(1)
        "layer_is_full": [1 if lt == "full_attention" else 0 for lt in t.layer_types],
        "sliding_window": t.sliding_window,
        "rms_norm_eps": t.rms_norm_eps,
        "vocab_size": t.vocab_size,
        "backbone_hidden_size": cfg.backbone_hidden_size,   # 3840 (target hidden)
        "attn_scaling": 1.0,                        # Gemma4UnifiedTextAttention.scaling=1.0 (코드 실사)
        "tie_word_embeddings": True,                # lm_head = embed_tokens
        "final_logit_softcapping": 0.0,             # null → softcap 없음(코드 실사)
        "use_ordered_embeddings": bool(getattr(cfg, "use_ordered_embeddings", False)),  # False=centroid OFF
        "qblk": QBLK, "scale_dtype": 0,             # q4 스킴(blueprint 동일·scale fp32)
    }
    json.dump(mani, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    print(f"[extract-ast] manifest: L={mani['num_layers']} H={mani['hidden_size']} FFN={mani['intermediate_size']} "
          f"hd_s/f={mani['head_dim']}/{mani['global_head_dim']} V={mani['vocab_size']} backbone={mani['backbone_hidden_size']}")

if __name__ == "__main__":
    main()
