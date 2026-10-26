"""CPU checks for exact QAT grid evidence, never inference."""
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_qat_grid import analyze_blocks


def test_signed_h1_grid_preserves_bf16_bits() -> None:
    # Given a signed Q4_0 grid with representable bf16 scale and no zero ambiguity.
    q = torch.tensor([-8, -7, -3, -1, 1, 2, 6, 7] * 4)
    weight = (q * -0.125).to(torch.bfloat16).reshape(1, 32)
    # When every registered hypothesis is evaluated.
    rows = analyze_blocks(weight)
    # Then H1 recovers every bit and the H2 grid does not.
    assert rows['H1_bf16']['exact_blocks'] == 1
    assert rows['H2_bf16']['exact_blocks'] == 0


def test_off_grid_element_is_counted() -> None:
    # Given one block with one deliberately off-grid non-extremal weight.
    weight = torch.ones((1, 32), dtype=torch.bfloat16)
    weight[0, 0] = -8
    weight[0, 1] = 1.125
    # When analyzed.
    row = analyze_blocks(weight)['H1_bf16']
    # Then exactness fails without a relaxed tolerance.
    assert row['exact_blocks'] == 0
    assert row['mismatch_elements'] == 1


def test_cli_scans_real_safetensors_and_refuses_existing_output(tmp_path: Path) -> None:
    # Given a real safetensors checkpoint with a linear and a norm tensor.
    model = tmp_path / 'model'; model.mkdir()
    q = torch.tensor([-8, -7, -3, -1, 1, 2, 6, 7] * 16)
    save_file({'model.layers.0.self_attn.q_proj.weight': (q * 0.125).to(torch.bfloat16).reshape(1, 128),
               'model.norm.weight': torch.ones(8, dtype=torch.bfloat16)}, model / 'model.safetensors')
    output = tmp_path / 'result'
    command = [sys.executable, 'tools/analyze_qat_grid.py', '--model', str(model), '--out', str(output)]
    # When the public CPU CLI runs once.
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    # Then it records the exact inventory and passes the registered gate.
    assert run.returncode == 0, run.stderr
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['gate_pass'] is True
    assert summary['tensor_count'] == 2
    assert subprocess.run(command, capture_output=True, check=False).returncode != 0


def test_signed_zero_difference_is_reported_not_tolerated() -> None:
    # Given negative H1 scale producing negative zero from a positive-zero source.
    weight = torch.zeros((1, 32), dtype=torch.bfloat16)
    weight[0, 0] = 1
    # When literal-bit reconstruction is checked.
    result = analyze_blocks(weight)['H1_bf16']
    # Then signed-zero mismatches remain explicit gate failures.
    assert result['signed_zero_mismatch_elements'] == 31
    assert result['exact_blocks'] == 0


def test_zero_block_uses_positive_zero_scale() -> None:
    # Given an all-positive-zero block.
    weight = torch.zeros((1, 32), dtype=torch.bfloat16)
    # When every candidate is evaluated.
    results = analyze_blocks(weight)
    # Then all reconstruct exact bits with an explicit zero-block count.
    assert all(r['exact_blocks'] == 1 and r['zero_blocks'] == 1 for r in results.values())


def test_bf16_grid_does_not_imply_fp32_product_identity() -> None:
    # Given rounded bf16 grid points from a nontrivial bf16 scale.
    scale = torch.tensor(0.10009765625, dtype=torch.bfloat16)
    q = torch.tensor([-8, -7, -3, -1, 1, 2, 6, 7] * 4)
    weight = (q.float() * scale.float()).bfloat16().reshape(1, 32)
    # When bf16-grid and actual fp32-product identities are checked separately.
    row = analyze_blocks(weight)['H1_bf16']
    # Then the two counters expose precision loss without changing the B gate.
    assert row['exact_blocks'] == 1
    assert row['exact_fp32_product_blocks'] == 0


def test_positive_h3_extremum_fails_integer_range() -> None:
    # Given H3's positive maximum mapping to +8 outside [-8,7].
    weight = torch.ones((1, 32), dtype=torch.bfloat16)
    # When tested under H3.
    row = analyze_blocks(weight)['H3_bf16']
    # Then clamping cannot turn the invalid lattice into a passing block.
    assert row['out_of_range_blocks'] == 1
    assert row['exact_blocks'] == 0


def test_unsupported_required_tensor_blocks_gate(tmp_path: Path) -> None:
    # Given an otherwise exact grid and a present FP32 projection tensor.
    model = tmp_path / 'model'; model.mkdir()
    q = torch.tensor([-8, -7, -3, -1, 1, 2, 6, 7] * 16)
    save_file({'model.layers.0.self_attn.q_proj.weight': (q * 0.125).bfloat16().reshape(1, 128),
               'model.layers.0.self_attn.k_proj.weight': torch.ones(1, 128)}, model / 'model.safetensors')
    output = tmp_path / 'result'
    # When the public analyzer runs.
    run = subprocess.run([sys.executable, 'tools/analyze_qat_grid.py', '--model', str(model), '--out', str(output)],
                         capture_output=True, text=True, check=False)
    # Then the skipped required tensor is explicit and cannot produce a passing gate.
    assert run.returncode == 3
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['unsupported_required'] == ['model.layers.0.self_attn.k_proj.weight']
    assert summary['gate_pass'] is False
