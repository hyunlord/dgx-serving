"""CPU-only scoring and engine-reference interoperability checks."""
import math
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ppl_hf_qat import ReferenceWriter, score_logits, validate_dialogs
from ppl_hf_qat_loader import original_tensors


def test_nll_matches_independent_scalar_logsumexp() -> None:
    logits = np.array([[1000, 999, 998], [-4, 0, 3]], dtype=np.float32)
    targets = np.array([1, 2])
    nll, logp = score_logits(logits, targets)
    expected = [math.log(sum(math.exp(float(x) - float(row.max())) for x in row))
                + float(row.max()) - float(row[target]) for row, target in zip(logits, targets, strict=True)]
    np.testing.assert_allclose(nll, expected, rtol=1e-12)
    np.testing.assert_allclose(np.exp(logp).sum(axis=1), 1, rtol=1e-12)


def test_reference_matches_engine_binary_format(tmp_path: Path) -> None:
    ids = np.array([2, 1, 0], dtype='<i4'); lengths = np.array([3], dtype='<i4')
    mask = np.array([0, 1, 1], dtype=np.uint8)
    logits = np.array([[1.0001, 1, -2], [0, 5, 4]], dtype=np.float32)
    _, logp = score_logits(logits, ids[1:])
    directory = tmp_path / 'ref'
    with ReferenceWriter(directory, 3, ids, lengths, mask) as writer:
        writer.append(logp, logits.argmax(axis=1))
        assert not (directory / 'complete').exists()
        writer.finish()
    assert (directory / 'format.txt').read_bytes() == b'PPL_REF_V1\n3\n2\n'
    assert (directory / 'tokens.bin').read_bytes() == ids.tobytes()
    assert (directory / 'mask.bin').read_bytes() == mask.tobytes()
    assert (directory / 'lengths.bin').read_bytes() == lengths.tobytes()
    assert (directory / 'logp.fp16').read_bytes() == logp.astype('<f4').astype('<f2').tobytes()
    assert np.fromfile(directory / 'argmax.i32', dtype='<i4').tolist() == [0, 1]
    assert (directory / 'complete').read_bytes() == b'complete\n'


def test_incomplete_reference_never_marked_complete(tmp_path: Path) -> None:
    ids = np.array([2, 0, 1], dtype='<i4'); lengths = np.array([3], dtype='<i4')
    mask = np.array([0, 1, 1], dtype=np.uint8)
    with ReferenceWriter(tmp_path / 'ref', 2, ids, lengths, mask) as writer:
        writer.append(np.array([[-1., -1.]]), np.array([0]))
        with pytest.raises(RuntimeError, match='row accounting'):
            writer.finish()
    assert not (tmp_path / 'ref' / 'complete').exists()


def test_mask_rejects_bos_and_window_overflow() -> None:
    with pytest.raises(RuntimeError, match='BOS'):
        validate_dialogs(np.array([2, 3]), np.array([2]), np.array([1, 1]))
    with pytest.raises(RuntimeError, match='length'):
        validate_dialogs(np.array([2, 3]), np.array([1025]), np.array([0, 1]))


def test_reference_rejects_oversize_batch_without_completion(tmp_path: Path) -> None:
    ids = np.array([2] + [0] * 129, dtype='<i4')
    lengths = np.array([130], dtype='<i4'); mask = np.array([0] + [1] * 129, dtype='u1')
    with ReferenceWriter(tmp_path / 'ref', 2, ids, lengths, mask) as writer:
        with pytest.raises(RuntimeError, match='batch shape'):
            writer.append(np.full((129, 2), -math.log(2)), np.zeros(129, dtype=np.int32))
    assert not (tmp_path / 'ref' / 'complete').exists()
    assert (tmp_path / 'ref' / 'logp.fp16').stat().st_size == 0


def test_checkpoint_headers_account_bytes_without_loading_torch(tmp_path: Path) -> None:
    header = json.dumps({'layer.weight': {'dtype': 'BF16', 'shape': [2, 3], 'data_offsets': [0, 12]}}).encode()
    path = tmp_path / 'model.safetensors'
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 12)
    tensor = original_tensors(tmp_path)['layer.weight']
    assert tensor.shape == (2, 3) and tensor.dtype == 'BF16'
    assert tensor.offset == 8 + len(header) and tensor.end - tensor.offset == 12
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(RuntimeError, match='bounds invalid'):
        original_tensors(tmp_path)


def test_reference_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / 'ref'; destination.mkdir()
    marker = destination / 'preserve'; marker.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        ReferenceWriter(destination, 2, np.array([2, 0]), np.array([2]), np.array([0, 1]))
    assert marker.read_bytes() == b'original'


def test_dtype_memory_plan_preserves_fp32_and_limits_bf16() -> None:
    # Given the QAT checkpoint's full parameter count, including nontext tensors.
    from ppl_hf_qat import memory_budget
    elements = 11_959_730_224
    # When planning each explicitly supported execution dtype.
    bf16_source, bf16_budget = memory_budget(elements, 'bfloat16')
    fp32_source, fp32_budget = memory_budget(elements, 'float32')
    # Then BF16 has a conservative 40 GiB budget and FP32 retains 64 GiB.
    assert (bf16_source, bf16_budget) == (23_919_460_448, 40 * 1024**3)
    assert (fp32_source, fp32_budget) == (47_838_920_896, 64 * 1024**3)


def test_memory_plan_keeps_workspace_reserve_for_larger_checkpoints() -> None:
    from ppl_hf_qat import memory_budget
    assert memory_budget(40_000_000_000, 'bfloat16') == (80_000_000_000, 96_000_000_000)
