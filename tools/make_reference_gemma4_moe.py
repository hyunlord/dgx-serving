#!/usr/bin/env python3
"""Gemma4 26B-A4B (dense∥MoE) HF 골든 — M4 세션2 게이트. make_reference_gemma4.py + router 캡처 + RSS 가드.

크래시 방어 first-class (전례 2회 118GB·112GB kill):
  - **bf16 로드 full-depth**(fp32 full 전개 금지 — 26B×4B=104GB 크래시 영역). model(ids) 한 번 = 활성분만 이동.
  - **RSS 78GB hard-abort**(레퍼런스 생성기 자체 방어 — kill 전 중단). q4 round-trip 루프·forward 전후 체크.
  - 이 도구=생성만(엔진 대조는 별도 test 바이너리 — 프로세스/시점 분리).

세션1 격리점 최종 판정 = **router**: 각 MoE 층 router 출력(top_i, top_w)을 forward 훅으로 캡처.
  - expert-idx(top_i) == 엔진 didx → 라우팅 로직(router-norm scale **raw** 전제: +1 fold면 scores 발산→top_i 불일치).
  - top_w(renorm × per_expert_scale) == 엔진 dw → router-norm + per_expert_scale gather-multiply 정확성.

덤프(artifacts/gemma4-26b-a4b/golden): token_ids, {tag}_hs_{l}(전층), {tag}_logits(softcap 후),
  rope_full/sliding cos/sin, router_topi_{l}/router_topw_{l}(MoE 층별). tag = bf16 / q4(엔진 게이트).
사용:
  python tools/make_reference_gemma4_moe.py --model google/gemma-4-26B-A4B-it \
      --out artifacts/gemma4-26b-a4b/golden --prompt "The capital of France is"
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_weights import quantize_int4, QBLK
from extract_weights_qwen35 import is_q4
from make_reference_qwen35 import dequant_roundtrip


def _rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0  # linux: KB→GB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--rss-guard-gb", type=float, default=78.0,
                    help="RSS 초과 시 abort(시스템 다운 방지 — 118GB·112GB kill 전례)")
    args = ap.parse_args()
    import gc, torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    os.makedirs(args.out, exist_ok=True)

    def guard(where):
        rss = _rss_gb()
        print(f"  [rss] {where}: {rss:.1f}GB", flush=True)
        if rss > args.rss_guard_gb:
            print(f"[ref-moe] RSS {rss:.1f}GB > 가드 {args.rss_guard_gb}GB @ {where} — abort(시스템 보호).", flush=True)
            sys.exit(2)

    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    tok = AutoTokenizer.from_pretrained(args.model)
    layer_types = list(tcfg.layer_types)
    K = int(getattr(tcfg, "num_experts_per_tok", 8))
    print(f"[ref-moe] loading {args.model} (bf16, low_cpu_mem)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    guard("after load")
    try:
        base = model.get_submodule("model.language_model")   # 멀티모달 래퍼
    except AttributeError:
        base = model.get_submodule("model")                  # text-only

    ids = tok(args.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    ids.cpu().numpy().astype(np.int32).tofile(os.path.join(args.out, "token_ids.bin"))
    print(f"[ref-moe] prompt={args.prompt!r} tokens={ids.tolist()} S={S} K(top)={K}", flush=True)

    # --- router 모듈 탐지: layer i 내부의 클래스명에 'Router' 포함 서브모듈 ---
    router_of_layer = {}   # layer_idx -> router module
    for i, layer in enumerate(base.layers):
        for _, mod in layer.named_modules():
            if "Router" in type(mod).__name__:
                router_of_layer[i] = mod
                break
    print(f"[ref-moe] router modules found: {len(router_of_layer)} layers "
          f"(idx {sorted(router_of_layer)[:3]}...{sorted(router_of_layer)[-1:] if router_of_layer else []})", flush=True)

    def w(name, arr):
        arr.detach().cpu().float().numpy().astype(np.float32).tofile(os.path.join(args.out, name + ".bin"))

    def save_router(tag, cap):
        # cap[i] = router forward 출력. int 텐서→top_i, float [.,K] 텐서→top_w. 존재하는 것만 저장.
        for i, out in cap.items():
            tensors = [t for t in (out if isinstance(out, (tuple, list)) else [out]) if torch.is_tensor(t)]
            topi = topw = None
            for t in tensors:
                tf = t.detach().cpu()
                if not torch.is_floating_point(tf) and tf.shape and tf.shape[-1] == K and topi is None:
                    topi = tf.reshape(-1, K).to(torch.int32).numpy()
                elif torch.is_floating_point(tf) and tf.shape and tf.shape[-1] == K and topw is None:
                    topw = tf.reshape(-1, K).float().numpy()
            if topi is not None: topi.tofile(os.path.join(args.out, f"{tag}_router_topi_{i}.bin"))
            if topw is not None: topw.tofile(os.path.join(args.out, f"{tag}_router_topw_{i}.bin"))
        # 형상 기록(gate 소비용): 첫 MoE 층 기준
        if cap:
            i0 = sorted(cap)[0]
            ex = [tuple(t.shape) for t in (cap[i0] if isinstance(cap[i0], (tuple, list)) else [cap[i0]]) if torch.is_tensor(t)]
            print(f"[ref-moe:{tag}] router L{i0} out tensor shapes={ex} (K={K})", flush=True)

    def dump_forward(tag):
        cap_rope = {}
        cap_router = {}
        hooks = []
        # rope cos/sin 2세트(full/sliding 첫 층 self_attn)
        for lt, key in (("full_attention", "cs_full"), ("sliding_attention", "cs_sliding")):
            if lt not in layer_types: continue
            li = layer_types.index(lt)
            def mk(key):
                def pre(mod, a, kw):
                    if key not in cap_rope:
                        pe = kw.get("position_embeddings") or (a[1] if len(a) > 1 else None)
                        if pe is not None: cap_rope[key] = (pe[0].detach(), pe[1].detach())
                return pre
            hooks.append(base.layers[li].self_attn.register_forward_pre_hook(mk(key), with_kwargs=True))
        # router 출력 훅 + 입력(rin) 훅 (isolated 검증용 — 동일입력 엔진 router 대조)
        cap_rin = {}
        for i, rmod in router_of_layer.items():
            def mk_r(i):
                def fh(mod, inp, out): cap_router[i] = out
                return fh
            def mk_in(i):
                def pre(mod, inp):
                    if inp and torch.is_tensor(inp[0]): cap_rin[i] = inp[0].detach()
                return pre
            hooks.append(rmod.register_forward_hook(mk_r(i)))
            hooks.append(rmod.register_forward_pre_hook(mk_in(i)))

        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        for h in hooks: h.remove()

        hs = out.hidden_states
        for l, h in enumerate(hs): w(f"{tag}_hs_{l}", h[0])
        w(f"{tag}_logits", out.logits[0])
        save_router(tag, cap_router)
        for i, rin in cap_rin.items():   # router 입력 rin [.,H] (isolated 검증)
            rin.detach().cpu().float().reshape(-1, rin.shape[-1]).numpy().astype(np.float32).tofile(
                os.path.join(args.out, f"{tag}_router_rin_{i}.bin"))
        am = out.logits[0].argmax(-1).tolist()
        print(f"[ref-moe:{tag}] hs={len(hs)} logits={tuple(out.logits[0].shape)} argmax={am}", flush=True)
        for key, fn in (("cs_full", "rope_full"), ("cs_sliding", "rope_sliding")):
            if key in cap_rope and not os.path.exists(os.path.join(args.out, fn + "_cos.bin")):
                w(fn + "_cos", cap_rope[key][0][0]); w(fn + "_sin", cap_rope[key][1][0])
        guard(f"after forward {tag}")
        return am

    print("[ref-moe] golden bf16 (full-depth) ...", flush=True)
    am_bf16 = dump_forward("bf16")

    print("[ref-moe] int4 round-trip on q4-policy linears (experts 포함, RSS 가드) ...", flush=True)
    with torch.no_grad():
        n = 0
        for name, p in base.named_parameters():
            if name == "embed_tokens.weight": continue
            if not is_q4(name, p.dim()): continue
            p.data.copy_(torch.from_numpy(dequant_roundtrip(p.detach().cpu().float().numpy())).to(p.dtype))
            n += 1
            if n % 16 == 0:
                gc.collect(); guard(f"q4rt {n} params")
        if bool(getattr(cfg, "tie_word_embeddings", False) or getattr(tcfg, "tie_word_embeddings", False)):
            ew = base.embed_tokens.weight.detach().cpu().float().numpy()
            model.lm_head.weight = torch.nn.Parameter(
                torch.from_numpy(dequant_roundtrip(ew)).to(model.lm_head.weight.dtype), requires_grad=False)
        gc.collect(); guard(f"q4rt done ({n} params)")

    print("[ref-moe] golden q4 (엔진 게이트) ...", flush=True)
    am_q4 = dump_forward("q4")

    meta = {"model": args.model, "prompt": args.prompt, "S": S, "K": K,
            "token_ids": ids[0].tolist(), "dtype": "bfloat16",
            "moe_layers": sorted(router_of_layer), "total_layers": len(base.layers),
            "note": "26B-A4B dense∥MoE. bf16 full-depth(fp32 full 104GB=크래시 회피). router top_i/top_w 훅 캡처="
                    "세션1 격리점(router.scale raw) 최종 판정. op-level fp32는 make_reference_gemma4_fp32.py 별도.",
            "argmax_bf16": am_bf16, "argmax_q4": am_q4}
    json.dump(meta, open(os.path.join(args.out, "meta_moe.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[ref-moe] done. argmax bf16=={am_bf16==am_q4 and 'q4' or 'q4(diff)'} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
