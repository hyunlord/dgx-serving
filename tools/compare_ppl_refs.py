#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# How to run: python tools/compare_ppl_refs.py --reference HF_REF --candidate ENGINE_REF --out NEW.json
"""CPU-only KL diagnostics from normalized fp16 PPL_REF_V1 distributions.

Both saved log-softmax distributions are renormalized in float64. These are
approximate stored-distribution diagnostics, not comparisons to current raw
logits. Top-1 uses argmax.i32 saved from the original logits before fp16 storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final, TypedDict

import numpy as np
from numpy.typing import NDArray

BATCH_ROWS: Final = 128
INPUT_FILES: Final = ('tokens.bin', 'lengths.bin', 'mask.bin')


@dataclass(frozen=True, slots=True)
class ReferenceError(RuntimeError):
    """Invalid or incompatible saved probability artifact."""
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class Header:
    """Validated dimensions of a completed reference."""
    vocab: int
    rows: int


class Comparison(TypedDict):
    reference: str
    candidate: str
    diagnostic: str
    ntok: int
    vocab: int
    mean_kl: float
    kl_p99: float
    p99_method: str
    top1_matches: int
    top1_match_rate: float
    max_batch_rows: int
    input_sha256: dict[str, str]


def read_header(directory: Path) -> Header:
    """Check completion, dimensions and exact binary payload lengths."""
    if not (directory / 'complete').is_file():
        raise ReferenceError(f'Incomplete reference: {directory}')
    text = (directory / 'format.txt').read_text()
    fields = text.splitlines()
    if len(fields) != 3 or fields[0] != 'PPL_REF_V1' or not all(s.isdecimal() for s in fields[1:]):
        raise ReferenceError(f'Invalid header: {directory}')
    vocab, rows = map(int, fields[1:])
    if vocab < 1 or rows < 1 or text != f'PPL_REF_V1\n{vocab}\n{rows}\n':
        raise ReferenceError(f'Noncanonical header: {directory}')
    for name, size in [('logp.fp16', rows * vocab * 2), ('argmax.i32', rows * 4)]:
        if (directory / name).stat().st_size != size:
            raise ReferenceError(f'Wrong payload size: {directory / name}')
    return Header(vocab, rows)


def validate_inputs(reference: Path, candidate: Path, header: Header) -> dict[str, str]:
    """Require identical input bytes and valid independent-window score masks."""
    payloads = {}
    for name in INPUT_FILES:
        payload = (reference / name).read_bytes()
        if payload != (candidate / name).read_bytes():
            raise ReferenceError(f'Input mismatch: {name}')
        payloads[name] = payload
    if len(payloads['tokens.bin']) % 4 or len(payloads['lengths.bin']) % 4:
        raise ReferenceError('Unaligned int32 input')
    tokens = np.frombuffer(payloads['tokens.bin'], dtype='<i4')
    lengths = np.frombuffer(payloads['lengths.bin'], dtype='<i4')
    mask = np.frombuffer(payloads['mask.bin'], dtype='u1')
    if (not len(lengths) or np.any(lengths < 2) or int(lengths.astype(np.int64).sum()) != len(tokens)
            or len(mask) != len(tokens) or np.any(mask > 1) or int(mask.sum()) != header.rows
            or np.any(tokens < 0) or np.any(tokens >= header.vocab)):
        raise ReferenceError('Invalid input or scoring count accounting')
    starts = np.r_[0, np.cumsum(lengths[:-1], dtype=np.int64)]
    if np.any(mask[starts] != 0):
        raise ReferenceError('First window token cannot be scored')
    return {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}


def read_values(stream: BinaryIO, dtype: str, count: int) -> NDArray:
    """Read exactly one bounded batch and advise release of read-only cache."""
    payload = stream.read(np.dtype(dtype).itemsize * count)
    values = np.frombuffer(payload, dtype=dtype)
    if values.size != count:
        raise ReferenceError(f'Truncated stream: {stream.name}')
    if hasattr(os, 'posix_fadvise'):
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return values


def normalized_logp(values: NDArray) -> NDArray:
    """Recover a unit-mass distribution from finite rounded log probabilities."""
    result = values.astype(np.float64)
    if not np.isfinite(result).all():
        raise ReferenceError('Nonfinite stored log probability')
    result -= result.max(axis=1, keepdims=True)
    result -= np.log(np.exp(result).sum(axis=1, keepdims=True))
    return result


def compare_references(reference: Path, candidate: Path) -> Comparison:
    """Compare completed references without materializing full probability files."""
    header = read_header(reference)
    if read_header(candidate) != header:
        raise ReferenceError('Vocabulary/row count mismatch')
    hashes = validate_inputs(reference, candidate, header)
    divergences = np.empty(header.rows, dtype=np.float64)
    matches = 0
    with ((reference / 'logp.fp16').open('rb') as ref_logp,
          (candidate / 'logp.fp16').open('rb') as cur_logp,
          (reference / 'argmax.i32').open('rb') as ref_top,
          (candidate / 'argmax.i32').open('rb') as cur_top):
        for begin in range(0, header.rows, BATCH_ROWS):
            count = min(BATCH_ROWS, header.rows - begin)
            a = normalized_logp(read_values(ref_logp, '<f2', count * header.vocab).reshape(count, header.vocab))
            b = normalized_logp(read_values(cur_logp, '<f2', count * header.vocab).reshape(count, header.vocab))
            divergences[begin:begin + count] = np.sum(np.exp(a) * (a - b), axis=1)
            top_a = read_values(ref_top, '<i4', count)
            top_b = read_values(cur_top, '<i4', count)
            if np.any((top_a < 0) | (top_a >= header.vocab)) or np.any((top_b < 0) | (top_b >= header.vocab)):
                raise ReferenceError('Invalid saved argmax index')
            matches += int(np.count_nonzero(top_a == top_b))
    return {'reference': str(reference), 'candidate': str(candidate),
            'diagnostic': 'renormalized_fp16_stored_distributions_not_raw_logits',
            'ntok': header.rows, 'vocab': header.vocab,
            'mean_kl': math.fsum(divergences) / header.rows,
            'kl_p99': float(np.sort(divergences)[(header.rows * 99 + 99) // 100 - 1]),
            'p99_method': 'nearest_rank', 'top1_matches': matches,
            'top1_match_rate': matches / header.rows, 'max_batch_rows': BATCH_ROWS,
            'input_sha256': hashes}


def main() -> None:
    """Write one new diagnostic artifact; never overwrite prior results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ReferenceError(f'Output already exists: {args.out}')
    result = compare_references(args.reference, args.candidate)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with args.out.open('x') as stream:
        stream.write(text)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
