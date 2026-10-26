#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "safetensors"]
# ///
# Run with the existing reference environment, CPU only:
# python tools/analyze_qat_grid.py --model CHECKPOINT --out NEW_DIRECTORY
"""Measure registered QAT grid hypotheses exactly, in bounded CPU chunks."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Final, TypedDict

import torch
from safetensors import safe_open

SIZES: Final = (32, 64, 128)
REQUIRED: Final = frozenset({'q', 'k', 'v', 'o', 'gate', 'up', 'down', 'gate_up'})


class Evidence(TypedDict):
    blocks: int
    exact_blocks: int
    mismatch_elements: int
    exact_fp32_product_blocks: int
    signed_zero_mismatch_elements: int
    zero_blocks: int
    out_of_range_blocks: int
    examples: list[int]
    scale_min: float | None
    scale_max: float | None
    scale_nonzero_below_fp16_normal: int
    scale_abs_above_fp16_max: int
    scale_fp16_exact: int
    scale_bf16_fp16_bf16_exact: int


def empty_evidence() -> Evidence:
    """Create one independent accumulator; scale statistics cover exact blocks only."""
    return {'blocks': 0, 'exact_blocks': 0, 'mismatch_elements': 0, 'examples': [],
            'exact_fp32_product_blocks': 0, 'signed_zero_mismatch_elements': 0, 'zero_blocks': 0, 'out_of_range_blocks': 0,
            'scale_min': None, 'scale_max': None, 'scale_nonzero_below_fp16_normal': 0,
            'scale_abs_above_fp16_max': 0, 'scale_fp16_exact': 0,
            'scale_bf16_fp16_bf16_exact': 0}


def analyze_blocks(weight: torch.Tensor) -> dict[str, Evidence]:
    """Compare all six registered reconstructions with original bf16 bit patterns."""
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError('Expected a two-dimensional BF16 block matrix')
    w = weight.float()
    if not torch.isfinite(w).all():
        raise ValueError('Nonfinite checkpoint weights cannot define a quantization grid')
    maximum, index = w.abs().max(dim=1)
    signed = w.gather(1, index[:, None]).squeeze(1)
    candidates = {'H1': signed / -8.0, 'H2': maximum / 7.0, 'H3': maximum / 8.0}
    result: dict[str, Evidence] = {}
    for hypothesis, initial in candidates.items():
        for precision in ('fp32', 'bf16'):
            scale = initial if precision == 'fp32' else initial.bfloat16().float()
            scale = torch.where(maximum == 0, torch.zeros_like(scale), scale)
            divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
            lower = -7 if hypothesis == 'H2' else -8
            rounded = torch.round(w / divisor[:, None])
            in_range = ((rounded >= lower) & (rounded <= 7)).all(dim=1)
            q = rounded.clamp(lower, 7).to(torch.int8).float()
            product = scale[:, None] * q
            restored = product.bfloat16()
            mismatch = restored.view(torch.int16) != weight.contiguous().view(torch.int16)
            exact = ~mismatch.any(dim=1) & in_range
            valid = scale[exact]
            row = empty_evidence()
            row.update(blocks=w.shape[0], exact_blocks=int(exact.sum()),
                       mismatch_elements=int(mismatch.sum()),
                       exact_fp32_product_blocks=int(((product.view(torch.int32) == w.view(torch.int32)).all(dim=1) & in_range).sum()),
                       signed_zero_mismatch_elements=int((mismatch & (restored == 0) & (weight == 0)).sum()),
                       zero_blocks=int((maximum == 0).sum()),
                       out_of_range_blocks=int((~in_range).sum()),
                       examples=torch.where(~exact)[0][:8].tolist())
            if valid.numel():
                row.update(scale_min=float(valid.min()), scale_max=float(valid.max()),
                           scale_nonzero_below_fp16_normal=int(((valid.abs() < 2**-14) & (valid != 0)).sum()),
                           scale_abs_above_fp16_max=int((valid.abs() > 65504).sum()),
                           scale_fp16_exact=int((valid.half().float().view(torch.int32) == valid.view(torch.int32)).sum()),
                           scale_bf16_fp16_bf16_exact=int((valid.bfloat16().half().bfloat16().view(torch.int16) == valid.bfloat16().view(torch.int16)).sum()))
            result[f'{hypothesis}_{precision}'] = row
    return result


def merge(target: Evidence, source: Evidence) -> None:
    """Accumulate counts and bounded flattened block-index examples."""
    offset = target['blocks']
    target['examples'] = (target['examples'] + [offset + n for n in source['examples']])[:8]
    for key in ('blocks', 'exact_blocks', 'mismatch_elements', 'exact_fp32_product_blocks',
                'signed_zero_mismatch_elements', 'zero_blocks', 'out_of_range_blocks', 'scale_nonzero_below_fp16_normal',
                'scale_abs_above_fp16_max', 'scale_fp16_exact', 'scale_bf16_fp16_bf16_exact'):
        target[key] += source[key]
    for key, operation in (('scale_min', min), ('scale_max', max)):
        value = source[key]
        if value is not None:
            previous = target[key]
            target[key] = value if previous is None else operation(previous, value)


def category(name: str) -> str:
    """Classify tensor names without classifying norms as projection matrices."""
    if 'norm' in name:
        return 'norm'
    for token, kind in (('embed', 'embed'), ('lm_head', 'lm_head'), ('q_proj', 'q'),
                        ('k_proj', 'k'), ('v_proj', 'v'), ('o_proj', 'o'),
                        ('gate_up_proj', 'gate_up'), ('gate_proj', 'gate'),
                        ('up_proj', 'up'), ('down_proj', 'down')):
        if token in name:
            return kind
    return 'other'


def checkpoint_files(model: Path) -> list[Path]:
    """Use the index shard set when present, avoiding duplicate alternate checkpoints."""
    index = model / 'model.safetensors.index.json'
    if index.exists():
        data = json.loads(index.read_text())
        return [model / name for name in sorted(set(data['weight_map'].values()))]
    files = sorted(model.glob('*.safetensors'))
    if not files:
        raise FileNotFoundError(f'No safetensors files in {model}')
    return files


def analyze(model: Path, output: Path, chunk_elements: int) -> bool:
    """Stream each tensor, persist per-tensor evidence, and aggregate the fixed gate."""
    output.mkdir(parents=True, exist_ok=False)
    grouped: dict[str, dict[str, Evidence]] = {}
    inventory: list[dict[str, str | list[int] | int | bool]] = []
    seen: set[str] = set()
    unsupported_required: list[str] = []
    required_elements = 0
    with (output / 'tensors.jsonl').open('x') as records:
        for file in checkpoint_files(model):
            with safe_open(file, framework='pt', device='cpu') as handle:
                for name in sorted(handle.keys()):
                    if name in seen:
                        raise ValueError(f'Duplicate tensor across shards: {name}')
                    seen.add(name)
                    view = handle.get_slice(name)
                    shape = view.get_shape()
                    dtype = view.get_dtype()
                    kind = category(name)
                    if kind in REQUIRED and name.endswith('.weight'):
                        required_elements += math.prod(shape)
                    sizes = [size for size in SIZES if shape and shape[-1] % size == 0]
                    eligible = len(shape) >= 2 and name.endswith('.weight') and dtype == 'BF16' and bool(sizes)
                    if kind in REQUIRED and name.endswith('.weight') and not eligible:
                        unsupported_required.append(name)
                    inventory.append({'name': name, 'shape': shape, 'dtype': dtype,
                                      'category': kind, 'elements': math.prod(shape), 'analyzed': eligible})
                    if not eligible:
                        continue
                    totals: dict[str, Evidence] = {}
                    rows = max(1, chunk_elements // shape[-1])
                    for prefix in itertools.product(*(range(n) for n in shape[:-2])):
                        for start in range(0, shape[-2], rows):
                            slab = view[prefix + (slice(start, min(start + rows, shape[-2])), slice(None))].contiguous()
                            for size in sizes:
                                for hypothesis, evidence in analyze_blocks(slab.reshape(-1, size)).items():
                                    key = f'b{size}_{hypothesis}'
                                    merge(totals.setdefault(key, empty_evidence()), evidence)
                    group = grouped.setdefault(kind, {})
                    for key, evidence in totals.items():
                        merge(group.setdefault(key, empty_evidence()), evidence)
                    records.write(json.dumps({'name': name, 'category': kind, 'shape': shape, 'results': totals}) + '\n')
                    records.flush()
                    print(json.dumps({'tensor_done': name, 'elements': math.prod(shape)}), flush=True)
    required: dict[str, Evidence] = {}
    for kind, variants in grouped.items():
        if kind in REQUIRED:
            for key, evidence in variants.items():
                merge(required.setdefault(key, empty_evidence()), evidence)
    coverage = {key: value['blocks'] * int(key.split('_')[0][1:]) == required_elements for key, value in required.items()}
    covered = [key for key in required if coverage[key]]
    best = max(covered, key=lambda key: required[key]['exact_blocks'] / required[key]['blocks'], default=None)
    fraction = required[best]['exact_blocks'] / required[best]['blocks'] if best else 0.0
    missing = sorted(REQUIRED - grouped.keys())
    # k_eq_v checkpoints legitimately omit v_proj; report absence, do not synthesize it.
    passed = best is not None and fraction >= 0.999 and not unsupported_required
    per_kind_best = {kind: max(values, key=lambda k: values[k]['exact_blocks'] / values[k]['blocks']) for kind, values in grouped.items() if values}
    summary = {'model': str(model.resolve()), 'tensor_count': len(inventory), 'block_sizes': SIZES,
               'rounding': 'torch round ties-to-even; out-of-range rounded q fails block; canonical integer zero; exact BF16 bits including signed zero',
               'layout': 'row-major contiguous blocks along last dimension; first absmax wins ties',
               'scale_statistics_scope': 'only exact blocks, all-zero scale canonicalized to +0',
               'mismatch_examples': 'first 8 flattened block indices per tensor and candidate; not exhaustive',
               'grouped': grouped,
               'grouped_exact_fraction': {kind: {key: value['exact_blocks'] / value['blocks'] for key, value in variants.items()} for kind, variants in grouped.items()},
               'unsupported_required': unsupported_required, 'candidate_full_required_coverage': coverage,
               'best_candidate_per_kind': per_kind_best,
               'required_linear': required, 'missing_linear_categories': missing,
               'best_candidate': best, 'best_exact_fraction': fraction, 'gate_threshold': 0.999, 'gate_pass': passed}
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'gate_pass': passed, 'best_candidate': best, 'best_exact_fraction': fraction}))
    return passed


def main() -> int:
    """Refuse output reuse and restrict execution to CPU with bounded thread count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk-elements', type=int, default=1_048_576)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.chunk_elements < 128 or not 1 <= args.threads <= 8:
        parser.error('chunk-elements must be >=128 and threads must be 1..8')
    torch.set_num_threads(args.threads)
    with torch.inference_mode():
        passed = analyze(args.model, args.out, args.chunk_elements)
    return 0 if passed else 3


if __name__ == '__main__':
    raise SystemExit(main())
