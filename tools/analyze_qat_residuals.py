#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "safetensors"]
# ///
# Run: python tools/analyze_qat_residuals.py --qat QAT --original HF --out NEW_DIR
"""Paired CPU residual distributions for the pre-registered QAT scale candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypedDict

import torch
from safetensors import safe_open

from analyze_qat_grid import checkpoint_files


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    denominator: float
    signed: bool


CANDIDATES: Final = (Candidate('signed_absmax_div_minus8', -8.0, True),) + tuple(
    Candidate(f'positive_absmax_div_{c / 100:.2f}', c / 100, False) for c in range(650, 851, 5)
)
PROJECTION: Final = re.compile(
    r'^(?:model\.language_model|language_model\.model|model)\.'
    r'(layers\.\d+\.(?:self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj\.weight)$'
)
COUNT_KEYS: Final = ('blocks', 'zero_blocks', 'elements', 'lt_005', 'lt_01', 'lt_02', 'ge_04')


class Counts(TypedDict):
    blocks: int
    zero_blocks: int
    elements: int
    lt_005: int
    lt_01: int
    lt_02: int
    ge_04: int


@dataclass(frozen=True, slots=True)
class TensorInfo:
    file: Path
    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str


def empty_counts() -> Counts:
    """Allocate a fresh integer accumulator; undefined zero-scale residuals stay excluded."""
    return {'blocks': 0, 'zero_blocks': 0, 'elements': 0, 'lt_005': 0,
            'lt_01': 0, 'lt_02': 0, 'ge_04': 0}


def distribution(blocks: torch.Tensor, candidate: Candidate) -> Counts:
    """Use float64 scale/residual arithmetic on original checkpoint values, without clipping."""
    values = blocks.to(dtype=torch.float64, device='cpu')
    if values.ndim != 2 or values.shape[1] != 32 or not torch.isfinite(values).all():
        raise ValueError('Expected finite CPU blocks with exactly 32 elements')
    maxima, positions = values.abs().max(dim=1)
    nonzero = maxima != 0
    selected = values.gather(1, positions[:, None]).squeeze(1) if candidate.signed else maxima
    scale = selected[nonzero] / candidate.denominator
    normalized = values[nonzero] / scale[:, None]
    residual = (normalized - normalized.round()).abs()
    return {'blocks': values.shape[0], 'zero_blocks': int((~nonzero).sum()),
            'elements': residual.numel(), 'lt_005': int((residual < 0.05).sum()),
            'lt_01': int((residual < 0.1).sum()), 'lt_02': int((residual < 0.2).sum()),
            'ge_04': int((residual >= 0.4).sum())}


def sample_indices(total: int, name: str, seed: int) -> tuple[list[int], str]:
    """Sample at most 20,000 unique blocks, independently of traversal and shard order."""
    if total < 1:
        raise ValueError(f'Empty block population for {name}')
    local_seed = int.from_bytes(hashlib.sha256(f'{seed}\n{name}'.encode()).digest(), 'big')
    indices = sorted(random.Random(local_seed).sample(range(total), min(20_000, total)))
    raw = struct.pack(f'<{len(indices)}Q', *indices)
    return indices, hashlib.sha256(raw).hexdigest()


def inventory(model: Path) -> dict[str, TensorInfo]:
    """Map known HF text-only projection prefixes; reject ambiguous canonical names."""
    result: dict[str, TensorInfo] = {}
    for file in checkpoint_files(model):
        with safe_open(file, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                match = PROJECTION.fullmatch(name)
                if not match:
                    continue
                canonical, kind = match.groups()
                view = handle.get_slice(name)
                shape = tuple(view.get_shape())
                if canonical in result or len(shape) != 2 or shape[1] % 32:
                    raise ValueError(f'Ambiguous or unsupported projection {name}: {shape}')
                result[canonical] = TensorInfo(file, name, shape, view.get_dtype(), kind)
    if not result:
        raise ValueError(f'No supported text projection tensors in {model}')
    return result


def load_blocks(info: TensorInfo, indices: list[int]) -> torch.Tensor:
    """Memory-map one tensor and copy only its selected blocks into a bounded sample."""
    with safe_open(info.file, framework='pt', device='cpu') as handle:
        tensor = handle.get_tensor(info.name)
        if not tensor.is_floating_point():
            raise ValueError(f'Non-floating tensor: {info.name}')
        return tensor.reshape(-1, 32).index_select(0, torch.tensor(indices, dtype=torch.int64))


def accumulate(total: Counts, values: Counts) -> None:
    """Sum exact counts before calculating any cross-tensor fraction."""
    for key in COUNT_KEYS:
        total[key] += values[key]


def fractions(values: Counts) -> dict[str, float | None]:
    """Return null for a wholly undefined distribution, never a synthetic zero score."""
    count = values['elements']
    return {key: values[key] / count if count else None for key in ('lt_005', 'lt_01', 'lt_02', 'ge_04')}


def winner(totals: dict[str, Counts]) -> str | None:
    """Break exact count-ratio ties by the pre-registered candidate order."""
    best: str | None = None
    numerator, denominator = -1, 1
    for candidate in CANDIDATES:
        value = totals[candidate.name]
        if value['elements'] and value['lt_01'] * denominator > numerator * value['elements']:
            best = candidate.name
            numerator, denominator = value['lt_01'], value['elements']
    return best


def analyze(qat: Path, original: Path, output: Path, seed: int) -> None:
    """Evaluate identical block indices and persist integer evidence plus derived fractions."""
    paired = {'qat': inventory(qat), 'original': inventory(original)}
    if paired['qat'].keys() != paired['original'].keys():
        raise ValueError(f'Projection name sets differ: {sorted(paired["qat"].keys() ^ paired["original"].keys())}')
    for name, info in paired['qat'].items():
        if info.shape != paired['original'][name].shape:
            raise ValueError(f'Projection shapes differ: {name}')
    output.mkdir(parents=True, exist_ok=False)
    global_counts = {source: {c.name: empty_counts() for c in CANDIDATES} for source in paired}
    grouped: dict[str, dict[str, dict[str, Counts]]] = {}
    samples: list[dict[str, str | int | list[int]]] = []
    with (output / 'tensors.jsonl').open('x') as records:
        for name in sorted(paired['qat']):
            info = paired['qat'][name]
            indices, digest = sample_indices(info.shape[0] * info.shape[1] // 32, name, seed)
            sample = {'canonical_name': name, 'qat_name': info.name, 'original_name': paired['original'][name].name,
                      'shape': list(info.shape), 'qat_dtype': info.dtype, 'original_dtype': paired['original'][name].dtype, 'seed': seed, 'population_blocks': info.shape[0] * info.shape[1] // 32,
                      'sample_blocks': len(indices), 'indices_sha256_le_uint64': digest}
            samples.append(sample)
            kinds = grouped.setdefault(info.kind, {source: {c.name: empty_counts() for c in CANDIDATES} for source in paired})
            for source in paired:
                blocks = load_blocks(paired[source][name], indices)
                measured = {c.name: distribution(blocks, c) for c in CANDIDATES}
                for candidate, counts in measured.items():
                    accumulate(kinds[source][candidate], counts)
                    accumulate(global_counts[source][candidate], counts)
                records.write(json.dumps({'name': name, 'kind': info.kind, 'source': source, 'counts': measured}) + '\n')
            records.flush()
            print(json.dumps({'tensor_done': name, 'sample_blocks': len(indices)}), flush=True)
    summary = {'qat': str(qat.resolve()), 'original': str(original.resolve()), 'seed': seed,
               'sample_policy': 'per tensor min(20000, population), without replacement; name-derived SHA256 seed; sorted indices reused for both checkpoints',
               'python_version': sys.version, 'torch_version': torch.__version__,
               'arithmetic': 'CPU float64 scales and residuals from original checkpoint values; round ties-to-even; no integer clipping',
               'zero_policy': 'zero-scale blocks reported and excluded from residual fractions; zero entries inside nonzero blocks remain included',
               'candidates': [asdict(c) for c in CANDIDATES], 'samples': samples,
               'global': global_counts, 'grouped': grouped,
               'global_fractions': {s: {c: fractions(v) for c, v in counts.items()} for s, counts in global_counts.items()},
               'grouped_fractions': {k: {s: {c: fractions(v) for c, v in counts.items()} for s, counts in sources.items()} for k, sources in grouped.items()},
               'winner_global': winner(global_counts['qat']),
               'winner_per_kind': {k: winner(sources['qat']) for k, sources in grouped.items()}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'winner_global': summary['winner_global'], 'tensor_count': len(samples)}))


def main() -> None:
    """Run without GPU initialization, new dependencies, or output overwrites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qat', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    torch.set_num_threads(8)
    with torch.inference_mode():
        analyze(args.qat, args.original, args.out, args.seed)


if __name__ == '__main__':
    main()
