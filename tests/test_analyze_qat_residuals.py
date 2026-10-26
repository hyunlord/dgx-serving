"""CPU checks for paired QAT residual analysis."""
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_qat_residuals import CANDIDATES, distribution, sample_indices


def test_known_grid_has_zero_residual() -> None:
    # Given signed int4 grid points with an exactly representable scale.
    values = torch.tensor([-8, -7, -3, -1, 0, 2, 6, 7] * 4).reshape(1, 32) * 0.125
    # When using the signed-maximum candidate.
    result = distribution(values, CANDIDATES[0])
    # Then every element lies within every strict near-integer threshold.
    assert result['lt_005'] == result['lt_01'] == result['lt_02'] == 32
    assert result['ge_04'] == 0


def test_sampling_is_repeatable_unique_and_name_specific() -> None:
    # Given a fixed tensor name, seed and population.
    args = (100_000, 'layers.0.self_attn.q_proj.weight', 20261027)
    # When requesting paired indices twice.
    a, ah = sample_indices(*args)
    b, bh = sample_indices(*args)
    # Then both sources receive the same unique bounded sample.
    assert a == b and ah == bh and len(a) == len(set(a)) == 20_000
    assert sample_indices(100_000, 'layers.1.self_attn.q_proj.weight', 20261027)[1] != ah


def test_zero_blocks_are_explicitly_excluded_from_undefined_residual() -> None:
    # Given an all-zero block whose scale is zero.
    values = torch.zeros(1, 32)
    # When calculating residuals.
    result = distribution(values, CANDIDATES[0])
    # Then no synthetic exact-grid votes enter the denominator.
    assert result['zero_blocks'] == 1
    assert result['elements'] == result['lt_01'] == 0


def test_real_checkpoint_cli_uses_paired_blocks(tmp_path: Path) -> None:
    # Given two identical checkpoints with different supported HF prefixes.
    qat = tmp_path / 'qat'; original = tmp_path / 'original'
    qat.mkdir(); original.mkdir()
    tensor = torch.tensor([-8, -7, -3, -1, 0, 2, 6, 7] * 16).bfloat16().reshape(1, 128)
    save_file({'model.language_model.layers.0.self_attn.q_proj.weight': tensor}, qat / 'model.safetensors')
    save_file({'model.layers.0.self_attn.q_proj.weight': tensor}, original / 'model.safetensors')
    output = tmp_path / 'result'
    # When the public CLI performs all candidates on shared indices.
    run = subprocess.run([sys.executable, 'tools/analyze_qat_residuals.py', '--qat', str(qat),
                          '--original', str(original), '--out', str(output)], capture_output=True, text=True, check=False)
    # Then evidence matches and ties select the first registered candidate.
    assert run.returncode == 0, run.stderr
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['winner_global'] == 'signed_absmax_div_minus8'
    assert summary['global']['qat'] == summary['global']['original']
    assert summary['samples'][0]['sample_blocks'] == 4
    assert len(summary['candidates']) == 42


def test_sharded_checkpoint_reads_same_indices(tmp_path: Path) -> None:
    # Given indexed shards and an unrelated safetensors file that must be ignored.
    qat = tmp_path / 'qat'; original = tmp_path / 'original'
    qat.mkdir(); original.mkdir()
    names = ['model.layers.0.self_attn.q_proj.weight', 'model.layers.1.mlp.down_proj.weight']
    values = torch.arange(128).bfloat16().reshape(1, 128)
    for folder in (qat, original):
        for index, name in enumerate(names):
            save_file({name: values}, folder / f'part-{index}.safetensors')
        save_file({'unrelated': values}, folder / 'ignore.safetensors')
        (folder / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {n: f'part-{i}.safetensors' for i, n in enumerate(names)}}))
    output = tmp_path / 'result'
    command = [sys.executable, 'tools/analyze_qat_residuals.py', '--qat', str(qat), '--original', str(original), '--out', str(output)]
    # When the CLI runs on the indexed checkpoints.
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    # Then only indexed text projections contribute and outputs cannot be overwritten.
    assert run.returncode == 0, run.stderr
    result = json.loads((output / 'summary.json').read_text())
    assert len(result['samples']) == 2
    assert result['global']['qat'] == result['global']['original']
    assert subprocess.run(command, capture_output=True, check=False).returncode != 0
