#!/usr/bin/env python3
"""Read-only dense Gemma logical weight-byte inventory; no tensor values or CUDA.

Usage: r08_weight_inventory.py GGUF ENGINE_WEIGHT_DIR OUTPUT.json
       r08_weight_inventory.py --self-test
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypedDict


class TensorJSON(TypedDict):
    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    storage_bytes: int
    logical_bytes_per_token: int
    note: str


class Inventory(TypedDict):
    storage_bytes: int
    logical_weight_bytes_per_token: int
    roofline_228_GBps_tps: float
    roofline_273_GBps_tps: float
    logical_bytes_by_kind: dict[str, int]
    tensors: list[TensorJSON]


@dataclass(frozen=True, slots=True)
class Tensor:
    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    storage_bytes: int
    logical_bytes_per_token: int
    note: str = "one unique payload read per token; cache reuse not modeled"


LINEARS: Final = {
    "attn_q": "q", "attn_k": "k", "attn_v": "v", "attn_output": "o",
    "ffn_gate": "gate", "ffn_up": "up", "ffn_down": "down",
}


def tensor_kind(name: str) -> str:
    """Classify standard llama text tensors without inferring unobserved fusion."""
    stem = name.removesuffix(".weight")
    if stem == "token_embd":
        return "embedding"
    if stem == "output":
        return "lm_head"
    suffix = stem.rsplit(".", 1)[-1]
    if suffix in LINEARS:
        return LINEARS[suffix]
    if "norm" in suffix:
        return "norm"
    if "scale" in suffix:
        return "scalar"
    return "other"


def gguf_rows(path: Path) -> list[Tensor]:
    """Read GGUF descriptors with mmap; never materialize tensor payloads."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path), "r")
    has_head = any(t.name == "output.weight" for t in reader.tensors)
    rows: list[Tensor] = []
    for tensor in reader.tensors:
        shape = tuple(int(v) for v in tensor.shape)
        size = int(tensor.n_bytes)
        kind = tensor_kind(tensor.name)
        logical = size
        note = "one unique payload read per token; cache reuse not modeled"
        if kind == "embedding":
            if len(shape) != 2 or size % shape[1]:
                raise RuntimeError(f"Invalid token embedding descriptor: {tensor.name}")
            logical = size // shape[1]
            note = "one embedding row only"
            if not has_head:
                rows.append(Tensor(
                    "output.weight (alias token_embd.weight)", "lm_head",
                    tensor.tensor_type.name, shape, 0, size,
                    "tied projection reads whole embedding once; zero extra storage",
                ))
        rows.append(Tensor(tensor.name, kind, tensor.tensor_type.name,
                           shape, size, logical, note))
    if not any(r.kind == "lm_head" for r in rows):
        raise RuntimeError("GGUF lacks both output and token embedding weights")
    return rows


def engine_rows(path: Path) -> list[Tensor]:
    """Mirror dense weights_gemma4 loader and decode's consumed weight tensors."""
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("num_experts", 0):
        raise RuntimeError("Dense Gemma only; MoE active bytes need explicit routing")
    h, f, v = (int(manifest[k]) for k in ("hidden_size", "intermediate_size", "vocab_size"))
    qblk = int(manifest["qblk"])
    scale_dtype = int(manifest.get("scale_dtype", 0))
    if scale_dtype not in (0, 1) or qblk != 32:
        raise RuntimeError("Only qblk=32 with fp32/fp16 scales is supported")
    rows: list[Tensor] = []
    consumed: set[str] = set()
    if len(manifest["layer_types"]) != int(manifest["num_layers"]):
        raise RuntimeError("Layer-type count differs from manifest num_layers")

    def append(name: str, kind: str, shape: tuple[int, ...], dtype: str,
               expected: int, logical: int, note: str = "unique payload read") -> None:
        file = path / f"{name}.bin"
        size = file.stat().st_size
        if size != expected:
            raise RuntimeError(f"Size mismatch {file}: {size} != {expected}")
        consumed.add(file.name)
        rows.append(Tensor(file.name, kind, dtype, shape, size, logical, note))

    def fp32(name: str, kind: str, shape: tuple[int, ...], logical: int | None = None,
             note: str = "one unique payload; repeated head reuse not counted") -> None:
        size = math.prod(shape) * 4
        append(name, kind, shape, "F32", size, size if logical is None else logical, note)

    def matrix(name: str, kind: str, out_dim: int, in_dim: int) -> None:
        count = out_dim * in_dim
        if in_dim % qblk:
            raise RuntimeError(f"Unaligned q4 row: {name}")
        append(f"{name}_q4", kind, (out_dim, in_dim), "INT4_OFFSET8", count // 2, count // 2)
        size = count // qblk * (2 if scale_dtype else 4)
        append(f"{name}_scale", f"{kind}_scale", (out_dim, in_dim // qblk),
               "F16" if scale_dtype else "F32", size, size)

    fp32("w_embed", "embedding", (v, h), h * 4, "one FP32 embedding row only")
    fp32("w_final_norm", "norm", (h,))
    matrix("w_lm_head", "lm_head", v, h)
    for layer, layer_type in enumerate(manifest["layer_types"]):
        full = layer_type == "full_attention"
        hd = int(manifest.get("global_head_dim") or manifest["head_dim"]) if full else int(manifest["head_dim"])
        nkv = int(manifest.get("num_global_kv_heads") or manifest["num_kv_heads"]) if full else int(manifest["num_kv_heads"])
        qdim, kvdim = int(manifest["num_q_heads"]) * hd, nkv * hd
        prefix = f"L{layer}_"
        for name in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            fp32(prefix + name, "norm", (h,))
        fp32(prefix + "layer_scalar", "host_scalar", (1,), 0,
             "host-loaded scalar passed as kernel argument; not GPU weight storage read")
        for name, kind, out_dim, in_dim in (
            ("self_attn.q_proj", "q", qdim, h), ("self_attn.k_proj", "k", kvdim, h),
            ("self_attn.o_proj", "o", h, qdim), ("mlp.gate_proj", "gate", f, h),
            ("mlp.up_proj", "up", f, h), ("mlp.down_proj", "down", h, f),
        ):
            matrix(prefix + name, kind, out_dim, in_dim)
        if not (full and manifest.get("attention_k_eq_v")):
            matrix(prefix + "self_attn.v_proj", "v", kvdim, h)
        for name in ("self_attn.q_norm", "self_attn.k_norm"):
            fp32(prefix + name, "norm", (hd,))
    for file in sorted(path.glob("*.bin")):
        if file.name not in consumed:
            rows.append(Tensor(file.name, "not_loaded", "UNKNOWN", (),
                               file.stat().st_size, 0, "not consumed by dense q4 loader"))
    return rows


def report(rows: list[Tensor]) -> Inventory:
    """Summarize logical payload rather than claim physical DRAM traffic."""
    grouped: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        grouped[row.kind] += row.logical_bytes_per_token
    logical = sum(grouped.values())
    if logical <= 0:
        raise RuntimeError("Empty logical byte inventory")
    return {
        "storage_bytes": sum(r.storage_bytes for r in rows),
        "logical_weight_bytes_per_token": logical,
        "roofline_228_GBps_tps": 228e9 / logical,
        "roofline_273_GBps_tps": 273e9 / logical,
        "logical_bytes_by_kind": dict(sorted(grouped.items())),
        "tensors": [asdict(r) for r in rows],
    }


def main() -> None:
    """Write a new evidence file; existing evidence is never overwritten."""
    if sys.argv[1:] == ["--self-test"]:
        rows = [Tensor("a", "q", "Q4_0", (32, 2), 36, 36),
                Tensor("e", "embedding", "F32", (32, 10), 1280, 128)]
        result = report(rows)
        assert result["logical_weight_bytes_per_token"] == 164
        assert result["storage_bytes"] == 1316
        assert tensor_kind("blk.1.attn_q.weight") == "q"
        assert tensor_kind("blk.1.attn_q_norm.weight") == "norm"
        print("PASS logical bytes, embedding-row accounting, tensor classification")
        return
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    gguf, weights, output = map(Path, sys.argv[1:])
    result = {
        "gguf": str(gguf), "engine_weights": str(weights),
        "definition": "Unique weight payload used per generated token with full-vocabulary logits. "
                      "Excludes KV, activations, launch overhead, repeated cache hits, metadata, alignment. "
                      "A bandwidth-only roofline, not predicted throughput or measured DRAM bytes.",
        "shape_convention": "GGUF native fastest-dimension-first; engine matrices OUT,IN",
        "llama": report(gguf_rows(gguf)), "engine": report(engine_rows(weights)),
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
