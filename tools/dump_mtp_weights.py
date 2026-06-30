#!/usr/bin/env python3
"""MTP head weight 덤프 (fp32 .bin) — body forward 없음(fla 크래시 회피).

safetensors의 mtp.* 텐서만 직접 읽어 test_spec.cu/load_mtp 명명 규칙으로 fp32 .bin 저장.
rope/token_ids는 0.8B와 동일(theta·tokenizer 공유)하므로 별도 복사로 충분 — 여기선 weight만.

사용: python tools/dump_mtp_weights.py --model Qwen/Qwen3.5-27B --out artifacts/qwen35-27b/mtp_golden
"""
import argparse, glob, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from safetensors import safe_open

    pat = f"/home/hyunlord/.cache/huggingface/hub/models--{args.model.replace('/','--')}/snapshots/*/*.safetensors"
    shards = sorted(s for s in glob.glob(pat) if "index" not in s)
    assert shards, f"no shards: {pat}"
    print(f"[dump_mtp] {len(shards)} shards, model={args.model}")

    # mtp.* 텐서 수집 (여러 shard에 흩어질 수 있음)
    W = {}
    for s in shards:
        with safe_open(s, framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("mtp"):
                    W[k] = sf.get_tensor(k).float().cpu().numpy().astype(np.float32)
    assert W, "mtp.* 키 없음"
    print(f"[dump_mtp] mtp tensors: {len(W)}")

    os.makedirs(args.out, exist_ok=True)

    def save(fname, key):
        t = W[key]
        t.tofile(os.path.join(args.out, fname))
        print(f"  {fname:42s} <- {key:48s} {list(t.shape)}")

    # test_spec.cu/load_mtp 명명 규칙 (PyTorch Linear weight [out,in] = 엔진 gemv [OUT,IN] 직접)
    save("mtp_fc_f32.bin",                    "mtp.fc.weight")
    save("mtp_enorm.bin",                     "mtp.pre_fc_norm_embedding.weight")
    save("mtp_hnorm.bin",                     "mtp.pre_fc_norm_hidden.weight")
    save("mtp_norm.bin",                      "mtp.norm.weight")
    save("mtp_input_layernorm.bin",           "mtp.layers.0.input_layernorm.weight")
    save("mtp_post_attention_layernorm.bin",  "mtp.layers.0.post_attention_layernorm.weight")
    save("mtp_q_norm.bin",                    "mtp.layers.0.self_attn.q_norm.weight")
    save("mtp_k_norm.bin",                    "mtp.layers.0.self_attn.k_norm.weight")
    save("mtp_q_proj_f32.bin",                "mtp.layers.0.self_attn.q_proj.weight")
    save("mtp_k_proj_f32.bin",                "mtp.layers.0.self_attn.k_proj.weight")
    save("mtp_v_proj_f32.bin",                "mtp.layers.0.self_attn.v_proj.weight")
    save("mtp_o_proj_f32.bin",                "mtp.layers.0.self_attn.o_proj.weight")
    save("mtp_gate_proj_f32.bin",             "mtp.layers.0.mlp.gate_proj.weight")
    save("mtp_up_proj_f32.bin",               "mtp.layers.0.mlp.up_proj.weight")
    save("mtp_down_proj_f32.bin",             "mtp.layers.0.mlp.down_proj.weight")
    print(f"[dump_mtp] done -> {args.out}")


if __name__ == "__main__":
    main()
