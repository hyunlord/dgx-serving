"""Offline FP32 QAT loader with exact engine q4 restoration and bounded staging."""
from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from ppl_hf_qat_loader import ELEMENT_CHUNK, original_tensors

PROJECTIONS: Final = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def engine_key(stem: str) -> str:
    """Mirror extract_weights_qwen35 filename mapping without touching embedding."""
    if stem == 'w_lm_head':
        return 'lm_head.weight'
    layer, separator, suffix = stem.partition('_')
    if not separator or not layer.startswith('L') or not layer[1:].isdigit():
        raise RuntimeError(f'Unsupported engine tensor: {stem}')
    return f'model.language_model.layers.{int(layer[1:])}.{suffix}.weight'


def dequant_chunks(qpath: Path, spath: Path, scale_dtype: int) -> Iterator[tuple[int, NDArray[np.float32]]]:
    """Restore low nibble first, q=nibble-8, per32 scale, in <=16MiB chunks."""
    dtype = {0: np.dtype('<f4'), 1: np.dtype('<f2')}[scale_dtype]
    count = qpath.stat().st_size * 2
    if count % 32 or spath.stat().st_size != count // 32 * dtype.itemsize:
        raise RuntimeError(f'Invalid q4/scale size: {qpath}')
    with qpath.open('rb') as qfile, spath.open('rb') as sfile:
        for begin in range(0, count, ELEMENT_CHUNK):
            size = min(ELEMENT_CHUNK, count - begin)
            packed = np.frombuffer(qfile.read(size // 2), dtype=np.uint8)
            scale = np.frombuffer(sfile.read(size // 32 * dtype.itemsize), dtype=dtype).astype(np.float32)
            if packed.size != size // 2 or scale.size != size // 32 or not np.isfinite(scale).all():
                raise RuntimeError(f'Invalid or short tensor read: {qpath}')
            values = np.empty(size, dtype=np.float32)
            values[0::2] = packed & 15
            values[1::2] = packed >> 4
            values -= np.float32(8)
            values.reshape(-1, 32)[:] *= scale[:, None]
            yield begin, values
        for stream in (qfile, sfile):
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def load_qat_restored(hf: Path, engine: Path, device: Literal['cpu', 'cuda']):
    """Stream directly into final parameters; only small source chunks coexist."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Gemma4UnifiedForConditionalGeneration
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightRenaming, rename_source_key

    manifest = json.loads((engine / 'manifest.json').read_text())
    if manifest.get('num_experts', 0) or manifest['qblk'] != 32 or manifest.get('scale_dtype', 0) not in (0, 1):
        raise RuntimeError('Only dense per32 q4 manifests are supported')
    config = AutoConfig.from_pretrained(hf, local_files_only=True)
    config._attn_implementation = 'eager'
    config.text_config._attn_implementation = 'eager'
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with init_empty_weights(include_buffers=False):
            model = Gemma4UnifiedForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(previous)
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    if any(value.is_meta for value in buffers.values()):
        raise RuntimeError('Official initialized buffers must be real tensors')
    state = model.state_dict()
    transforms = get_model_conversion_mapping(model)
    if not all(isinstance(item, WeightRenaming) for item in transforms):
        raise RuntimeError('Non-renaming checkpoint conversion requires review')
    source = {}
    for source_key, payload in original_tensors(hf).items():
        key, converter = rename_source_key(source_key, transforms, [], model.base_model_prefix, state)
        if converter is not None or key in source or key not in state or payload.shape != tuple(state[key].shape):
            raise RuntimeError(f'Unsupported checkpoint mapping: {source_key} -> {key}')
        source[key] = payload
    missing = set(state) - set(source)
    if missing - {'lm_head.weight'} or (missing and not (config.tie_word_embeddings or config.text_config.tie_word_embeddings)):
        raise RuntimeError(f'Missing checkpoint tensors: {sorted(missing)}')
    qpaths = {engine_key(path.name.removesuffix('_q4.bin')): path for path in sorted(engine.glob('*_q4.bin'))}
    expected = {key for key in params if key.startswith('model.language_model.layers.')
                and any(key.endswith('.' + projection + '.weight') for projection in PROJECTIONS)} | {'lm_head.weight'}
    if set(qpaths) != expected:
        raise RuntimeError(f'Engine projection coverage mismatch: {sorted(set(qpaths) ^ expected)}')
    mapping = []
    for key in sorted(set(source) | set(qpaths)):
        shape = tuple(state[key].shape)
        count = math.prod(shape)
        value = torch.empty(shape, device=device, dtype=torch.float32)
        if key in qpaths:
            qpath = qpaths[key]
            if len(shape) != 2 or shape[1] % 32 or qpath.stat().st_size * 2 != count:
                raise RuntimeError(f'Engine tensor shape mismatch: {key}')
            spath = qpath.with_name(qpath.name.removesuffix('_q4.bin') + '_scale.bin')
            for begin, chunk in dequant_chunks(qpath, spath, manifest.get('scale_dtype', 0)):
                value.view(-1)[begin:begin + len(chunk)].copy_(torch.from_numpy(chunk))
            mapping.append({'engine': qpath.name, 'hf': key, 'shape': list(shape)})
        else:
            item = source[key]
            source_dtype, width = {'BF16': (torch.bfloat16, 2), 'F16': (torch.float16, 2), 'F32': (torch.float32, 4)}[item.dtype]
            if item.end - item.offset != count * width:
                raise RuntimeError(f'Checkpoint tensor byte size mismatch: {key}')
            with item.path.open('rb') as stream:
                stream.seek(item.offset)
                for begin in range(0, count, ELEMENT_CHUNK):
                    size = min(ELEMENT_CHUNK, count - begin)
                    raw = bytearray(stream.read(size * width))
                    if len(raw) != size * width:
                        raise RuntimeError(f'Short checkpoint read: {key}')
                    value.view(-1)[begin:begin + size].copy_(torch.frombuffer(raw, dtype=source_dtype))
                    if hasattr(os, 'posix_fadvise'):
                        os.posix_fadvise(stream.fileno(), stream.tell() - len(raw), len(raw), os.POSIX_FADV_DONTNEED)
        parent, attr = key.rsplit('.', 1)
        if key in params:
            setattr(model.get_submodule(parent), attr, torch.nn.Parameter(value, requires_grad=False))
        else:
            if key not in buffers:
                raise RuntimeError(f'Unexpected state entry: {key}')
            setattr(model.get_submodule(parent), attr, value)
    if any(parameter.is_meta or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError('Incomplete FP32 parameter load')
    if model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr():
        raise RuntimeError('Q4 lm_head must be untied from original embedding')
    model.eval().to(device)
    model.config.use_cache = False
    return model, mapping
