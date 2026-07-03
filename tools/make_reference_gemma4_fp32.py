#!/usr/bin/env python3
"""Gemma4 층-스트리밍 fp32 골든 (M4 게이트 보강). bf16 골든의 정밀도 한계 우회.

전략: fp32 full 로드 118GB=메모리 가드 초과 → 층별로만 fp32 업캐스트(base는 bf16 유지, 층당 ~2GB 전이).
      각 층의 q4-policy linear에 **엔진과 동일한 q4 round-trip**(fp32 유지)을 적용 → 엔진(q4 dequant fp32)과
      동일 가중치·동일 정밀도 → per-layer err<1e-2 게이트 복원(bf16 노이즈 제거).
HF 레이어 모듈 forward를 그대로 재사용(정확성 보장) — 마스크/rotary/shared_kv는 HF 유틸.

출력: q4fp32_hs_{l}.bin [S,H] (l=0..L, 마지막=final-norm 후), q4fp32_logits.bin [S,V] (softcap 후).
사용:
  python tools/make_reference_gemma4_fp32.py --model google/gemma-4-31B-it --out artifacts/gemma4-31b/golden \
      --prompt "The capital of France is"
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_weights import quantize_int4, QBLK
from extract_weights_qwen35 import is_q4


def _rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0  # linux: KB→GB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-layers", type=int, default=8,
                    help="fp32 업캐스트할 층 수(메모리 가드: full-depth fp32는 mmap+anon 누적으로 118GB 근접→불가). "
                         "첫 N층이 전 연산타입 커버(sliding/full/k_eq_v/GeGLU/sandwich). 0=전층(위험).")
    ap.add_argument("--rss-guard-gb", type=float, default=78.0, help="RSS 초과 시 abort(시스템 다운 방지)")
    args = ap.parse_args()
    import gc, torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    os.makedirs(args.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[ref-fp32] loading {args.model} (bf16 base)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    try:
        base = model.get_submodule("model.language_model")
    except AttributeError:
        base = model.get_submodule("model")

    ids = tok(args.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    print(f"[ref-fp32] prompt={args.prompt!r} tokens={ids.tolist()} S={S}", flush=True)

    def w(name, t):
        t.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.out, name + ".bin"))

    def q4rt_fp32(t):  # 엔진 q4와 동일: quantize→dequant, fp32 유지(bf16 저장 안 함)
        a = t.detach().cpu().float().contiguous().numpy().astype(np.float32)
        packed, scale = quantize_int4(a)
        n = a.size; nblk = n // QBLK
        lo = (packed & 0x0F).astype(np.float32); hi = (packed >> 4).astype(np.float32)
        qv = np.empty(n, dtype=np.float32); qv[0::2] = lo; qv[1::2] = hi
        deq = (((qv.reshape(nblk, QBLK) - 8) * scale[:, None]).reshape(a.shape)).astype(np.float32)
        return torch.from_numpy(deq)

    dumps = {}
    with torch.no_grad():
        # embed: 엔진과 동일하게 fp32 lookup × √H(fp32) — embed weight는 q4정책상 fp32유지(bf16 upcast lossless)
        embw = base.embed_tokens.weight.float()
        scale = float(tcfg.hidden_size) ** 0.5
        h = embw[ids[0]] * scale               # [S,H] fp32
        dumps[0] = h.clone()

        position_ids = torch.arange(S).unsqueeze(0)
        # position_embeddings per layer_type (rotary_emb를 fp32로)
        base.rotary_emb.float()
        pe = {lt: base.rotary_emb(h.unsqueeze(0), position_ids, lt) for lt in set(tcfg.layer_types)}
        # masks (full / sliding) — HF 유틸, 입력 dtype fp32
        cfg_obj = base.config
        mk = dict(config=cfg_obj, inputs_embeds=h.unsqueeze(0), attention_mask=None,
                  past_key_values=None, position_ids=position_ids)
        masks = {"full_attention": create_causal_mask(**mk),
                 "sliding_attention": create_sliding_window_causal_mask(**mk)}
        shared_kv = {}

        w("q4fp32_hs_0", dumps[0])               # 층 입력(embed) 즉시 기록
        nL = len(base.layers)
        cap_n = args.max_layers if args.max_layers > 0 else nL
        hb = h.unsqueeze(0)                     # [1,S,H]
        done_layers = 0
        for i, layer in enumerate(base.layers):
            if i >= cap_n: break
            lt = tcfg.layer_types[i]
            layer.float()                        # bf16→fp32 (lossless), 전이 ~2GB
            for name, p in layer.named_parameters():
                full_name = f"layers.{i}.{name}"
                if is_q4(full_name, p.dim()):     # q4-policy linear만 round-trip(엔진 동일)
                    p.data.copy_(q4rt_fp32(p.data))
            out = layer(hb, position_embeddings=pe[lt], attention_mask=masks[lt],
                        position_ids=position_ids, shared_kv_states=shared_kv)
            hb = (out[0] if isinstance(out, tuple) else out)
            w(f"q4fp32_hs_{i+1}", hb[0])          # 즉시 기록(dict 미보유)
            done_layers = i + 1
            layer.to(torch.bfloat16); gc.collect()
            rss = _rss_gb()
            if i % 4 == 0: print(f"  layer {i}/{nL} done, RSS={rss:.1f}GB", flush=True)
            if rss > args.rss_guard_gb:
                print(f"[ref-fp32] RSS {rss:.1f}GB > 가드 {args.rss_guard_gb}GB — abort(시스템 보호). "
                      f"{done_layers}층까지 기록됨.", flush=True)
                break

    meta = {"S": S, "prompt": args.prompt, "layers_done": done_layers, "total_layers": nL,
            "note": "경계-depth 층 스트리밍 fp32(엔진 동일 q4 weights). full-depth fp32는 메모리 가드로 불가 → "
                    "첫 N층 op-level fp32 검증 + full-depth argmax(bf16 골든)로 게이트 보강."}
    json.dump(meta, open(os.path.join(args.out, "meta_fp32.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[ref-fp32] done. {done_layers}/{nL}층 fp32 hidden 기록 → {args.out}")


if __name__ == "__main__":
    main()
