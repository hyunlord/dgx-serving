#!/usr/bin/env python3
"""Per-token decode kernel accounting from an Nsight Systems SQLite export (R14 PART C).

Usage: python tools/r14_nsys_decode.py TRACE.sqlite engine|llama OUT.json [--tokens 32]

Token boundaries: engine = end of each g4g_argmax launch (Graph v2 records one per step);
llama = end of each lm_head launch (the linear launch with the most output rows per step,
identified as the largest-grid matrix-vector launch; prefill GEMMs are excluded). The window is the last N complete
tokens: from the boundary N+1 from the end to the last boundary. Accounting follows R08/R13:
per-category kernel-duration sums, kernel-interval union, gap = window wall - union,
overlap = sum - union. Categories are name based and conservative; unmatched names stay in
"other" and are listed. Attention is split into sliding/full by head dim (engine: block
size; llama: flash_attn_ext template D). No GPU work is executed.
"""
import json, re, sqlite3, sys
from collections import defaultdict

ENGINE_LINEAR = ("gemv_w4a8", "gemm_mmq", "gemm_tiled", "gemm_batch", "gemv_f32")
LLAMA_LINEAR = ("mul_mat_vec_q", "mul_mat_q", "mul_mat_vec_f", "mul_mat_f")


def union(iv):
    total, end = 0, -1
    for a, b in sorted(iv):
        total += max(0, b - max(a, end)); end = max(end, b)
    return total


def classify(kind, name, short, block, grid, lm_head):
    if lm_head:
        return "lm_head"
    if short.startswith(ENGINE_LINEAR + LLAMA_LINEAR):
        return "linear"
    if "quantize" in short:
        return "activation_quantize"
    if "attn" in short or "soft_max" in short:
        if kind == "engine" and short.startswith("attn_split"):
            m = re.search(r"attn_split_\w+<\(int\)(\d+)|attn_split_\w+<(\d+)", name)
            hd = int(m.group(1) or m.group(2)) if m else 0
            sub = "full" if hd >= 512 else "sliding"
        elif kind == "engine":
            sub = "full" if block[0] >= 512 else "sliding"
        else:
            m = re.search(r"flash_attn_ext_f16<\(int\)(\d+)", name) or re.search(r"<\(int\)(\d+)", name)
            sub = ("full" if int(m.group(1)) >= 512 else "sliding") if m else "unknown"
        return f"attention_{sub}"
    if "norm" in short:
        return "norm"
    if "rope" in short:
        return "rope"
    if any(x in short for x in ("add", "scale", "geglu", "softcap", "k_bin_", "unary", "gelu", "silu",
                                "tanh", "embedding", "get_rows", "mul")):
        return "elementwise"
    if kind == "engine" and short.startswith(("g4g_argmax", "g4g_snapshot")):
        return "sampling_snapshot"
    if "cpy" in short or "copy" in short or "set_rows" in short:
        return "kv_copy"
    return "other"


def main():
    path, kind, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ntok = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 32
    db = sqlite3.connect(path); db.row_factory = sqlite3.Row
    rows = db.execute("""SELECT k.start,k.end,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
        n.value AS name, s.value AS short FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds n ON n.id=k.demangledName JOIN StringIds s ON s.id=k.shortName
        ORDER BY k.start,k.end""").fetchall()
    ks = [dict(r) for r in rows]
    for k in ks:
        k["grid"] = (k["gridX"], k["gridY"], k["gridZ"]); k["block"] = (k["blockX"], k["blockY"], k["blockZ"])
        k["threads"] = k["gridX"] * k["gridY"] * k["gridZ"] * k["blockX"] * k["blockY"] * k["blockZ"]
    # lm_head candidates: matrix-vector kernels only (prefill GEMMs can have larger grids).
    if kind == "engine":
        marks = [i for i, k in enumerate(ks) if k["short"].startswith("g4g_argmax")]
        lin = [k for k in ks if k["short"].startswith(("gemv_w4a8", "gemv_f32"))]
    else:
        lin = [k for k in ks if k["short"].startswith(("mul_mat_vec_q", "mul_mat_vec_f"))]
    big = max(k["threads"] for k in lin)
    lm_ids = {id(k) for k in lin if k["threads"] == big}
    if kind == "llama":
        marks = [i for i, k in enumerate(ks) if id(k) in lm_ids]
    if len(marks) < ntok + 1:
        sys.exit(f"only {len(marks)} token boundaries")
    t0, t1 = ks[marks[-ntok - 1]]["end"], ks[marks[-1]]["end"]
    win = [k for k in ks if k["start"] >= t0 and k["end"] <= t1]
    sums, counts, names = defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for k in win:
        cat = classify(kind, k["name"], k["short"], k["block"], k["grid"], id(k) in lm_ids)
        d = k["end"] - k["start"]
        sums[cat] += d; counts[cat] += 1
        e = names[cat][f'{k["short"]} grid={list(k["grid"])} block={list(k["block"])}']; e[0] += 1; e[1] += d
    wall, uni, tot = t1 - t0, union([(k["start"], k["end"]) for k in win]), sum(sums.values())
    res = {"trace": path, "kind": kind, "tokens": ntok, "window_ns": [t0, t1],
           "per_token_ms": {"wall": wall / ntok / 1e6, "kernel_union": uni / ntok / 1e6, "kernel_sum": tot / ntok / 1e6,
                            "gap_wall_minus_union": (wall - uni) / ntok / 1e6, "overlap_sum_minus_union": (tot - uni) / ntok / 1e6},
           "kernels_per_token": len(win) / ntok,
           "category_ms_per_token": {c: v / ntok / 1e6 for c, v in sorted(sums.items())},
           "category_launches_per_token": {c: v / ntok for c, v in sorted(counts.items())},
           "kernels": {c: sorted(([n, v[0] / ntok, v[1] / ntok / 1e6] for n, v in d.items()), key=lambda x: -x[2])
                       for c, d in names.items()},
           "lm_head_rule": f"largest-grid linear launch ({big} threads)"}
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("per_token_ms", "kernels_per_token", "category_ms_per_token")}))


if __name__ == "__main__":
    main()
