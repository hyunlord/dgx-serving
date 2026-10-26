#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "transformers", "accelerate"]
# ///
# How to run: existing offline environment: python tools/ppl_hf_qat.py --hf LOCAL --data CHAT --out NEW --save-ref NEW_REF
"""HF FP32 QAT evaluation on the unchanged 36-dialogue chat corpus, offline only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ppl_hf_qat_loader import load_streamed, original_tensors

CHECKS: Final = {
    'ids.bin': '0a9f6d447c15c1f80fd1cdacd87660dda4dca9facabd45b4080facfe8b40161e',
    'lengths.bin': 'b05ce369fae7c7d76e5920a03838badd7f39399a5f3cf4831ed629c93b2c68cf',
    'mask.bin': '8c2743f2bc4ee0771691836022f295e4003714e9a98e3f5589da30d2c63c2934',
}
POSITION_BINS: Final = ((1, 15), (16, 63), (64, 127), (128, 255), (256, 511), (512, 1023))


def validate_dialogs(ids: NDArray, lengths: NDArray, mask: NDArray) -> None:
    """Check independent window boundaries and scored target positions."""
    if len(ids) != len(mask) or int(lengths.sum()) != len(ids) or np.any((lengths < 2) | (lengths > 1024)):
        raise RuntimeError('Invalid dialogue length accounting')
    if np.any((mask != 0) & (mask != 1)):
        raise RuntimeError('Mask must be binary')
    offset = 0
    for length in lengths:
        end = offset + int(length)
        if ids[offset] != 2 or np.count_nonzero(ids[offset:end] == 2) != 1 or mask[offset] != 0:
            raise RuntimeError('BOS must occur once and never be scored')
        if not mask[offset:end].sum():
            raise RuntimeError('Dialogue has no scored targets')
        offset = end


def score_logits(logits: NDArray, targets: NDArray) -> tuple[NDArray, NDArray]:
    """Compute stable double log-softmax/NLL; input is already softcapped HF output."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or len(targets) != len(values) or not np.isfinite(values).all():
        raise RuntimeError('Invalid logits shape or nonfinite values')
    if np.any((targets < 0) | (targets >= values.shape[1])):
        raise RuntimeError('Target outside vocabulary')
    peak = values.max(axis=1)
    normalizer = peak + np.log(np.exp(values - peak[:, None]).sum(axis=1))
    logp = values - normalizer[:, None]
    return -logp[np.arange(len(targets)), targets], logp


def durable_write(path: Path, payload: bytes, mode: str = 'xb') -> None:
    """Sync each written batch before advising eviction; no system cache changes."""
    with path.open(mode) as stream:
        stream.write(payload)
        stream.flush()
        if hasattr(os, 'fdatasync'):
            os.fdatasync(stream.fileno())
        else:
            os.fsync(stream.fileno())
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


class ReferenceWriter:
    """Mutable row counter for engine PPL_REF_V1; completion is explicitly sealed."""

    def __init__(self, directory: Path, vocab: int, ids: NDArray, lengths: NDArray, mask: NDArray) -> None:
        validate_dialogs(ids, lengths, mask)
        self.directory, self.vocab, self.expected = directory, vocab, int(mask.sum())
        self.written = 0
        self.payload_bytes = self.expected * (vocab * 2 + 4) + ids.size * 4 + lengths.size * 4 + mask.size
        print(json.dumps({'event': 'reference_disk', 'payload_bytes': self.payload_bytes, 'metadata_reserve': 4096}), flush=True)
        if shutil.disk_usage(directory.parent).free < self.payload_bytes + 4096:
            raise RuntimeError('Insufficient reference disk space')
        directory.mkdir(exist_ok=False)
        for name, payload in [('tokens.bin', ids.astype('<i4').tobytes()), ('lengths.bin', lengths.astype('<i4').tobytes()),
                              ('mask.bin', mask.astype('u1').tobytes()), ('format.txt', f'PPL_REF_V1\n{vocab}\n{self.expected}\n'.encode()),
                              ('logp.fp16', b''), ('argmax.i32', b'')]:
            durable_write(directory / name, payload)

    def __enter__(self) -> ReferenceWriter:
        return self

    def __exit__(self, *_exc) -> None:
        # append owns each file descriptor; exceptions deliberately leave no complete marker.
        return None

    def append(self, logp: NDArray, argmax: NDArray) -> None:
        if logp.shape != (len(argmax), self.vocab) or not 1 <= len(argmax) <= 128 or self.written + len(argmax) > self.expected:
            raise RuntimeError('Reference batch shape/row accounting mismatch')
        half = logp.astype('<f4').astype('<f2')  # Same double -> float -> __half rounding as engine.
        if not np.isfinite(half).all() or np.any((argmax < 0) | (argmax >= self.vocab)):
            raise RuntimeError('Nonfinite reference or invalid original argmax')
        durable_write(self.directory / 'logp.fp16', half.tobytes(), 'ab')
        durable_write(self.directory / 'argmax.i32', argmax.astype('<i4').tobytes(), 'ab')
        self.written += len(argmax)

    def finish(self) -> None:
        if self.written != self.expected:
            raise RuntimeError('Reference row accounting mismatch')
        if (self.directory / 'logp.fp16').stat().st_size != self.expected * self.vocab * 2 or (self.directory / 'argmax.i32').stat().st_size != self.expected * 4:
            raise RuntimeError('Reference byte accounting mismatch')
        durable_write(self.directory / 'complete', b'complete\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--save-ref', type=Path)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import transformers

    for name, digest in CHECKS.items():
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Chat corpus SHA256 mismatch: {name}')
    ids = np.fromfile(args.data / 'ids.bin', dtype='<i4')
    lengths = np.fromfile(args.data / 'lengths.bin', dtype='<i4')
    mask = np.fromfile(args.data / 'mask.bin', dtype='u1')
    validate_dialogs(ids, lengths, mask)
    if len(lengths) != 36 or int(mask.sum()) != 16201:
        raise RuntimeError('Expected unchanged 36 dialogues / 16201 scored targets')
    config = json.loads((args.hf / 'config.json').read_text())
    vocab = int(config['text_config']['vocab_size'])
    if np.any((ids < 0) | (ids >= vocab)):
        raise RuntimeError('Input token outside model vocabulary')
    reference_bytes = int(mask.sum()) * (vocab * 2 + 4) + ids.nbytes + lengths.nbytes + mask.nbytes
    print(json.dumps({'event': 'reference_preflight', 'save_ref': str(args.save_ref) if args.save_ref else None,
                      'payload_bytes': reference_bytes, 'metadata_reserve': 4096}), flush=True)
    if args.save_ref and (args.save_ref.exists() or shutil.disk_usage(args.save_ref.parent).free < reference_bytes + 4096):
        raise RuntimeError('Reference destination exists or insufficient free disk space')
    source_bytes = sum(math.prod(t.shape) * 4 for t in original_tensors(args.hf).values())
    reserve = 16_000_000_000  # Streaming, eager attention, full logits, <=128 host double rows, allocator.
    budget = max(64 * 1024**3, source_bytes + reserve)
    mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'MemFree:', 'Cached:'))}
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({'event': 'memory_gate', 'expected_bytes': budget, 'source_fp32_bytes': source_bytes,
                      'workspace_reserve_bytes': budget - source_bytes, 'cuda_free': free, 'cuda_total': total, **mem}), flush=True)
    if budget > free * .8:
        raise RuntimeError('Expected memory exceeds CUDA free x 0.8')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    print(json.dumps({'event': 'load_start', 'epoch': time.time(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'tf32': False}), flush=True)
    model = load_streamed(args.hf)
    if model.config.text_config.vocab_size != vocab:
        raise RuntimeError('Config vocabulary changed during loading')
    reference = ReferenceWriter(args.save_ref, vocab, ids, lengths, mask) if args.save_ref else None
    if reference:
        durable_write(args.save_ref / 'metadata.json', (json.dumps({'model': str(args.hf), 'dtype': 'float32',
                      'path': 'HF_FP32_QAT_EAGER', 'hashes': CHECKS, 'nwin': 36, 'ntok': 16201}) + '\n').encode())
    rows = []; offset = 0; bins: list[list[float]] = [[] for _ in POSITION_BINS]
    with (args.out / 'positions.jsonl').open('x') as output, torch.inference_mode():
        for index, length_value in enumerate(lengths):
            length = int(length_value); window = ids[offset:offset + length]
            positions = np.flatnonzero(mask[offset:offset + length])
            tokens = torch.tensor(window.astype(np.int64), device='cuda').unsqueeze(0)
            torch.cuda.synchronize(); start = time.time()
            result = model(input_ids=tokens, use_cache=False, return_dict=True)
            torch.cuda.synchronize(); end = time.time()
            print(f'HF_INTERVAL window={index} start={start:.17g} end={end:.17g} scope=independent_forward', file=sys.stderr, flush=True)
            if result.past_key_values is not None:
                raise RuntimeError('Independent forward unexpectedly returned cache')
            nlls = []
            for begin in range(0, len(positions), 128):
                selected = positions[begin:begin + 128]
                logits = result.logits[0].index_select(0, torch.tensor(selected - 1, device='cuda')).cpu().numpy()
                nll, logp = score_logits(logits, window[selected])
                if reference:
                    reference.append(logp, logits.argmax(axis=1))
                for position, value in zip(selected, nll, strict=True):
                    output.write(json.dumps({'window': index, 'position': int(position), 'target': int(window[position]), 'nll': float(value)}) + '\n')
                    for bucket, (lower, upper) in enumerate(POSITION_BINS):
                        if lower <= position <= upper:
                            bins[bucket].append(float(value))
                nlls.extend(float(value) for value in nll)
                del logits, logp, nll
            row = {'window': index, 'length': length, 'ntok': len(nlls), 'nll_sum': math.fsum(nlls),
                   'mean_nll': math.fsum(nlls) / len(nlls), 'forward_start': start, 'forward_end': end}
            rows.append(row); print(json.dumps(row), flush=True); output.flush()
            del result, tokens
            offset += length
    ntok = sum(row['ntok'] for row in rows)
    if ntok != int(mask.sum()):
        raise RuntimeError('Final scoring count mismatch')
    if reference:
        reference.finish()
    mean = math.fsum(row['nll_sum'] for row in rows) / ntok
    summary = {'model': str(args.hf), 'weights': str(args.hf), 'path': 'HF_FP32_QAT_EAGER', 'scale_dtype': None,
               'nwin': len(rows), 'ntok': ntok, 'mean_nll': mean, 'ppl': math.exp(mean),
               'per_window_nll': [row['mean_nll'] for row in rows], 'windows': rows, 'hashes': CHECKS,
               'position_bins': [{'begin': lo, 'end': hi, 'ntok': len(values), 'mean_nll': math.fsum(values) / len(values) if values else None}
                                 for (lo, hi), values in zip(POSITION_BINS, bins, strict=True)],
               'tf32': False, 'reference_rows': reference.written if reference else 0}
    durable_write(args.out / 'summary.json', (json.dumps(summary, indent=2) + '\n').encode())
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
