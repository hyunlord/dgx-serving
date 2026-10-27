#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "transformers", "accelerate"]
# ///
# How to run: existing offline environment: python tools/make_reference_gemma4_qat.py --hf QAT --engine QAT_ENGINE --template OLD_GOLDEN --out NEW_GOLDEN --device cpu
"""New QAT golden files from restored engine q4 weights, never overwriting a golden.

The template supplies the exact five tokens and op-level depth coverage only.
Every emitted numerical reference is from the FP32 QAT/restored-weight forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ppl_hf_qat_loader import original_tensors
from qat_golden_loader import load_qat_restored

WORKSPACE_BYTES: Final = 4_000_000_000


def template_inputs(template: Path) -> tuple[NDArray[np.int32], list[int]]:
    """Preserve exact token bytes and the existing op-level test coverage."""
    ids = np.fromfile(template / 'token_ids.bin', dtype='<i4')
    if ids.size != 5 or np.any(ids < 0):
        raise RuntimeError('The golden template must contain exactly five token IDs')
    depths = sorted(int(path.stem.removeprefix('q4fp32_hs_')) for path in template.glob('q4fp32_hs_*.bin'))
    if not depths or depths != list(range(max(depths) + 1)):
        raise RuntimeError('Template op-level depths must be nonempty and contiguous from zero')
    return ids, depths


def write_tensor(path: Path, tensor) -> None:
    """Write FP32 little-endian values with exclusive creation and bounded data."""
    with path.open('xb') as stream:
        tensor.detach().cpu().float().numpy().astype('<f4').tofile(stream)


def main() -> None:
    """Generate one FP32 reference forward in a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve() == args.template.resolve():
        parser.error('Output must be a new directory, distinct from the original golden')
    ids, depths = template_inputs(args.template)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.engine / 'manifest.json').read_text())
    source_bytes = sum(int(np.prod(tensor.shape)) * 4 for tensor in original_tensors(args.hf).values())
    separate_head_bytes = int(manifest['hidden_size']) * int(manifest['vocab_size']) * 4
    expected = source_bytes + separate_head_bytes + WORKSPACE_BYTES
    mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
           if line.startswith(('MemAvailable:', 'MemFree:', 'Cached:'))}
    evidence = {'event': 'memory_gate', 'device': args.device, 'source_fp32_bytes': source_bytes,
                'separate_head_bytes': separate_head_bytes, 'workspace_bytes': WORKSPACE_BYTES,
                'expected_bytes': expected, **mem}
    if args.device == 'cuda':
        free, total = torch.cuda.mem_get_info()
        evidence.update(cuda_free=free, cuda_total=total)
        print(json.dumps(evidence), flush=True)
        if expected > free * 0.8:
            raise RuntimeError('CUDA memory guard failed; rerun explicitly with --device cpu')
    else:
        print(json.dumps(evidence), flush=True)
    if expected > mem['MemAvailable'] * 0.7:
        raise RuntimeError('Host unified-memory 70% guard failed')
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / 'token_ids.bin').open('xb') as output:
        output.write(ids.tobytes())
    start = time.time()
    model, mapping = load_qat_restored(args.hf, args.engine, args.device)
    base = model.model.language_model
    tokens = torch.tensor(ids.astype(np.int64), device=args.device).unsqueeze(0)
    positions = torch.arange(len(ids), device=args.device).unsqueeze(0)
    with torch.inference_mode():
        dummy = torch.zeros((1, len(ids), int(manifest['hidden_size'])), device=args.device, dtype=torch.float32)
        for layer_type, tag in (('full_attention', 'full'), ('sliding_attention', 'sliding')):
            cos, sin = base.rotary_emb(dummy, positions, layer_type)
            write_tensor(args.out / f'rope_{tag}_cos.bin', cos[0])
            write_tensor(args.out / f'rope_{tag}_sin.bin', sin[0])
        if args.device == 'cuda':
            torch.cuda.synchronize()
        forward_start = time.time()
        output = model(input_ids=tokens, use_cache=False, output_hidden_states=True, return_dict=True)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        forward_end = time.time()
    if output.past_key_values is not None or len(output.hidden_states) != int(manifest['num_layers']) + 1:
        raise RuntimeError('Unexpected cache or hidden-state count')
    if max(depths) >= len(output.hidden_states):
        raise RuntimeError('Template depth outside QAT model')
    for depth, hidden in enumerate(output.hidden_states):
        write_tensor(args.out / f'q4_hs_{depth}.bin', hidden[0])
        if depth in depths:
            write_tensor(args.out / f'q4fp32_hs_{depth}.bin', hidden[0])
    write_tensor(args.out / 'q4_logits.bin', output.logits[0])
    write_tensor(args.out / 'q4fp32_logits.bin', output.logits[0])
    # Preserve legacy fp32-tag file layout if present, with honest restored-weight provenance.
    for path in sorted(args.template.glob('fp32_*.bin')):
        match = re.fullmatch(r'fp32_hs_(\d+)\.bin', path.name)
        if match:
            write_tensor(args.out / path.name, output.hidden_states[int(match.group(1))][0])
        elif path.name == 'fp32_logits.bin':
            write_tensor(args.out / path.name, output.logits[0])
        else:
            raise RuntimeError(f'Unknown template numerical file: {path.name}')
    metadata = {'model': str(args.hf), 'engine_weights': str(args.engine), 'template': str(args.template),
                'dtype': 'float32', 'device': args.device, 'S': len(ids), 'token_ids': ids.tolist(),
                'token_sha256': hashlib.sha256(ids.tobytes()).hexdigest(), 'op_level_depths': depths,
                'argmax_q4': output.logits[0].argmax(-1).tolist(), 'mapping': mapping,
                'attention': 'eager', 'tf32': False, 'threads': args.threads,
                'load_start': start, 'forward_start': forward_start, 'forward_end': forward_end,
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'note': 'All numerical tags contain HF FP32 with exact engine q4 restoration; remaining tensors are QAT. No RTN re-quantization. Legacy fp32 aliases are restored-weight data, not unquantized QAT.'}
    (args.out / 'meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.out / 'meta_fp32.json').write_text(json.dumps({'layers_done': max(depths), 'total_layers': manifest['num_layers'], 'dtype': 'float32'}) + '\n')
    (args.out / 'complete').write_text('complete\n')
    print(json.dumps({'event': 'complete', 'argmax': metadata['argmax_q4'], 'mapping_count': len(mapping),
                      'forward_start': forward_start, 'forward_end': forward_end}), flush=True)


if __name__ == '__main__':
    main()
