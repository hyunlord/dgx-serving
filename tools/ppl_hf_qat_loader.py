"""Offline HF FP32 loader, preserving the supplement-7 conversion and real buffers.

Parameters stream from safetensors to CUDA in at most 16 MiB host chunks.
Unlike the diagnostic predecessor, no engine q4 replacement is performed.
"""
from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ELEMENT_CHUNK: Final = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TensorSource:
    path: Path
    offset: int
    end: int
    shape: tuple[int, ...]
    dtype: str


def original_tensors(hf: Path) -> dict[str, TensorSource]:
    """Read headers only, with checked shard-relative tensor bounds."""
    index = hf / 'model.safetensors.index.json'
    names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    tensors: dict[str, TensorSource] = {}
    for name in names:
        path = (hf / name).resolve()
        if not path.is_relative_to(hf.resolve()):
            raise RuntimeError(f'Checkpoint shard escapes directory: {name}')
        with path.open('rb') as source:
            length = struct.unpack('<Q', source.read(8))[0]
            header = json.loads(source.read(length))
        for key, item in header.items():
            if key == '__metadata__':
                continue
            if key in tensors:
                raise RuntimeError(f'Duplicate tensor: {key}')
            begin, end = item['data_offsets']
            if not 0 <= begin <= end <= path.stat().st_size - 8 - length:
                raise RuntimeError(f'Tensor bounds invalid: {key}')
            tensors[key] = TensorSource(path, 8 + length + begin, 8 + length + end,
                                       tuple(item['shape']), item['dtype'])
    return tensors


def load_streamed(hf: Path):
    """Use official key conversion and initialization, never from_pretrained weights."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Gemma4UnifiedForConditionalGeneration
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightRenaming, rename_source_key

    config = AutoConfig.from_pretrained(hf, local_files_only=True)
    config._attn_implementation = 'eager'
    config.text_config._attn_implementation = 'eager'
    with init_empty_weights(include_buffers=False):
        model = Gemma4UnifiedForConditionalGeneration(config)
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    if any(item.is_meta for item in buffers.values()):
        raise RuntimeError('Meta buffer: cannot preserve official initialization')
    state = model.state_dict()
    transforms = get_model_conversion_mapping(model)
    if not all(isinstance(item, WeightRenaming) for item in transforms):
        raise RuntimeError('Non-renaming conversion requires explicit review')
    source: dict[str, TensorSource] = {}
    renames = []
    for source_key, payload in original_tensors(hf).items():
        key, converter = rename_source_key(source_key, transforms, [], model.base_model_prefix, state)
        if converter is not None or key in source or key not in state:
            raise RuntimeError(f'Unsupported source mapping: {source_key} -> {key}')
        if payload.shape != tuple(state[key].shape) or payload.dtype not in ('BF16', 'F16', 'F32'):
            raise RuntimeError(f'Tensor shape/dtype mismatch: {key}')
        source[key] = payload
        if source_key != key:
            renames.append({'source': source_key, 'target': key})
    missing = set(state) - set(source)
    if missing - {'lm_head.weight'} or (missing and not (config.tie_word_embeddings or config.text_config.tie_word_embeddings)):
        raise RuntimeError(f'Missing checkpoint tensors: {sorted(missing)}')
    print(json.dumps({'event': 'checkpoint_coverage', 'source_tensors': len(source),
                      'model_state_tensors': len(state), 'missing_tied_aliases': sorted(missing),
                      'buffer_names_loaded': sorted(set(source) & set(buffers)), 'renames': renames}), flush=True)
    for key, item in source.items():
        source_dtype, source_bytes = {'BF16': (torch.bfloat16, 2), 'F16': (torch.float16, 2), 'F32': (torch.float32, 4)}[item.dtype]
        count = math.prod(item.shape)
        if item.end - item.offset != count * source_bytes:
            raise RuntimeError(f'Tensor byte length mismatch: {key}')
        value = torch.empty(item.shape, device='cuda', dtype=torch.float32)
        with item.path.open('rb') as stream:
            stream.seek(item.offset)
            for begin in range(0, count, ELEMENT_CHUNK):
                size = min(ELEMENT_CHUNK, count - begin)
                raw = bytearray(stream.read(size * source_bytes))
                if len(raw) != size * source_bytes:
                    raise RuntimeError(f'Short tensor read: {key}')
                value.view(-1)[begin:begin + size].copy_(torch.frombuffer(raw, dtype=source_dtype))
                os.posix_fadvise(stream.fileno(), stream.tell() - len(raw), len(raw), os.POSIX_FADV_DONTNEED)
        parent, attr = key.rsplit('.', 1)
        if key in params:
            setattr(model.get_submodule(parent), attr, torch.nn.Parameter(value, requires_grad=False))
        else:
            if key not in buffers:
                raise RuntimeError(f'Checkpoint key is neither parameter nor buffer: {key}')
            setattr(model.get_submodule(parent), attr, value)
    if missing:
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
    if any(p.is_meta or p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('Incomplete FP32 parameter load')
    model = model.eval().to('cuda')
    model.config.use_cache = False
    torch.cuda.empty_cache()
    return model
