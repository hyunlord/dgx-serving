"""CPU checks for stored-distribution KL diagnostics."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))


def write_reference(path: Path, probabilities: np.ndarray, tops: list[int]) -> None:
    path.mkdir()
    rows, vocab = probabilities.shape
    (path / 'format.txt').write_text(f'PPL_REF_V1\n{vocab}\n{rows}\n')
    (path / 'complete').write_bytes(b'complete\n')
    np.array([2] + [0] * rows, dtype='<i4').tofile(path / 'tokens.bin')
    np.array([rows + 1], dtype='<i4').tofile(path / 'lengths.bin')
    np.array([0] + [1] * rows, dtype='u1').tofile(path / 'mask.bin')
    np.log(probabilities).astype('<f2').tofile(path / 'logp.fp16')
    np.array(tops, dtype='<i4').tofile(path / 'argmax.i32')


def test_streamed_kl_matches_independent_float64_and_original_top1(tmp_path: Path) -> None:
    # Given 129 rows to cross the 128-row streaming boundary.
    from compare_ppl_refs import compare_references
    p = np.tile([.6, .3, .1], (129, 1)); q = np.tile([.2, .4, .4], (129, 1))
    write_reference(tmp_path / 'ref', p, [0] * 129)
    write_reference(tmp_path / 'cur', q, [0] + [1] * 128)
    a = np.log(p[0]).astype('<f2').astype(np.float64); b = np.log(q[0]).astype('<f2').astype(np.float64)
    pa = np.exp(a); pa /= pa.sum(); pb = np.exp(b); pb /= pb.sum()
    expected = sum(float(x) * math.log(float(x / y)) for x, y in zip(pa, pb, strict=True))
    # When comparing stored probabilities in <=128-row chunks.
    result = compare_references(tmp_path / 'ref', tmp_path / 'cur')
    # Then independent double KL agrees and top1 comes from original logits' argmax file.
    assert result['mean_kl'] == pytest.approx(expected, rel=1e-6)
    assert result['kl_p99'] == pytest.approx(expected, rel=1e-6)
    assert result['top1_matches'] == 1 and result['top1_match_rate'] == 1 / 129
    assert result['ntok'] == 129 and result['max_batch_rows'] == 128


@pytest.mark.parametrize('mutation', ['incomplete', 'header', 'tokens', 'lengths', 'mask', 'logp_size', 'argmax_size', 'nan', 'argmax_range'])
def test_invalid_reference_is_rejected(tmp_path: Path, mutation: str) -> None:
    from compare_ppl_refs import compare_references, ReferenceError
    # Given a valid pair with one corrupt candidate component.
    for name in ['ref', 'cur']:
        write_reference(tmp_path / name, np.array([[.5, .3, .2], [.2, .7, .1]]), [0, 1])
    cur = tmp_path / 'cur'
    if mutation == 'incomplete': (cur / 'complete').unlink()
    elif mutation == 'header': (cur / 'format.txt').write_text('PPL_REF_V2\n3\n2\n')
    elif mutation == 'tokens': np.array([2, 1, 0], dtype='<i4').tofile(cur / 'tokens.bin')
    elif mutation == 'lengths': np.array([2, 1], dtype='<i4').tofile(cur / 'lengths.bin')
    elif mutation == 'mask': (cur / 'mask.bin').write_bytes(bytes([1, 0, 1]))
    elif mutation == 'logp_size': (cur / 'logp.fp16').write_bytes(b'')
    elif mutation == 'argmax_size': (cur / 'argmax.i32').write_bytes(b'')
    elif mutation == 'nan': np.array([[np.nan, -1, -2], [-1, -2, -3]], dtype='<f2').tofile(cur / 'logp.fp16')
    elif mutation == 'argmax_range': np.array([3, 0], dtype='<i4').tofile(cur / 'argmax.i32')
    # When validating and reading the pair, then it fails explicitly.
    with pytest.raises(ReferenceError): compare_references(tmp_path / 'ref', cur)


def test_cli_writes_diagnostic_json_and_refuses_overwrite(tmp_path: Path) -> None:
    # Given equal stored distributions and a fresh JSON destination.
    for name in ['ref', 'cur']:
        write_reference(tmp_path / name, np.array([[.5, .3, .2]]), [0])
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'tools/compare_ppl_refs.py'),
               '--reference', str(tmp_path / 'ref'), '--candidate', str(tmp_path / 'cur'), '--out', str(tmp_path / 'result.json')]
    # When the actual CLI runs.
    completed = subprocess.run(command, capture_output=True, text=True)
    # Then it records zero KL, exact stored top1 and refuses to replace the artifact.
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['mean_kl'] == 0 and result['top1_match_rate'] == 1
    assert result['diagnostic'] == 'renormalized_fp16_stored_distributions_not_raw_logits'
    assert subprocess.run(command, capture_output=True).returncode != 0
