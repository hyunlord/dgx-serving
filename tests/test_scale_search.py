"""R10 fp16 scale search: independent CPU oracle, no checkpoints or GPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import extract_weights as quant


def unpack(packed: np.ndarray) -> np.ndarray:
    q = np.empty(packed.size * 2, dtype=np.int8)
    q[::2] = (packed & 15).astype(np.int8) - 8
    q[1::2] = (packed >> 4).astype(np.int8) - 8
    return q.reshape(-1, 32)


def oracle(block: np.ndarray, scale: np.float16) -> tuple[np.float16, np.ndarray]:
    candidates = [scale]
    lo, hi = scale, scale
    for _ in range(4):
        with np.errstate(over="ignore", under="ignore"):
            lo = np.nextafter(lo, np.float16(-np.inf))
            hi = np.nextafter(hi, np.float16(np.inf))
        candidates.extend((lo, hi))
    choices = []
    for order, d in enumerate(candidates):
        if not np.isfinite(d) or d == 0 or np.signbit(d) != np.signbit(scale):
            continue
        q = np.clip(np.rint(block / np.float32(d)), -8, 7).astype(np.int8)
        error = sum((float(int(v)) * float(d) - float(w)) ** 2 for v, w in zip(q, block, strict=True))
        choices.append((error, order, d, q))
    _, _, selected, selected_q = min(choices, key=lambda row: (row[0], row[1]))
    return selected, selected_q


@pytest.mark.parametrize("rule,c", [("positive-max", 8.05), ("signed-max", None)])
def test_search_matches_nine_candidate_oracle_and_reassigns_q(rule: str, c: float | None) -> None:
    w = np.random.default_rng(703).normal(size=(100, 32)).astype(np.float32)
    w[0] = 0
    original_q, rounded = quant.quantize_variant(w, "qat-ident", rule, c, scale_dtype="fp16")
    packed, scales = quant.quantize_variant(w, "qat-ident", rule, c, scale_dtype="fp16", scale_search=4)
    actual_q = unpack(packed)
    for i, block in enumerate(w):
        d, q = oracle(block, rounded[i])
        assert scales[i].tobytes() == d.tobytes()
        np.testing.assert_array_equal(actual_q[i], q)
    assert np.count_nonzero(scales != rounded) > 0
    assert np.count_nonzero(actual_q != unpack(original_q)) > 0
    again = quant.quantize_variant(w, "qat-ident", rule, c, scale_dtype="fp16", scale_search=4)
    assert packed.tobytes() == again[0].tobytes()
    assert scales.tobytes() == again[1].tobytes()


@pytest.mark.parametrize("d", [np.float16(65504), np.float16(-65504),
                               np.nextafter(np.float16(0), np.float16(1)),
                               np.nextafter(np.float16(0), np.float16(-1))])
def test_candidate_boundaries_skip_zero_infinity_and_opposite_sign(d: np.float16) -> None:
    block = np.tile(np.arange(-8, 8, dtype=np.float32), 2)[None, :] * np.float32(d)
    scale, q = quant.search_fp16_scale(block, np.array([d], np.float32), 4)
    expected_d, expected_q = oracle(block[0], d)
    assert scale[0] == expected_d
    np.testing.assert_array_equal(q[0], expected_q)
    assert scale[0] == d


def test_zero_block_tie_keeps_existing_fallback() -> None:
    block = np.zeros((1, 32), np.float32)
    baseline = quant.quantize_variant(block, "qat-ident", "positive-max", 8.05, scale_dtype="fp16")
    searched = quant.quantize_variant(block, "qat-ident", "positive-max", 8.05, scale_dtype="fp16", scale_search=4)
    assert searched[0].tobytes() == baseline[0].tobytes()
    assert searched[1].tobytes() == baseline[1].tobytes()


@pytest.mark.parametrize("extra", [[], ["--quant", "qat-ident", "--quant-rule", "signed-max"],
                                    ["--quant", "qat-ident", "--quant-rule", "signed-max", "--scale-dtype", "fp16", "--scale-search", "-1"]])
def test_cli_rejects_invalid_search_combinations(extra: list[str]) -> None:
    parser = argparse.ArgumentParser()
    quant.add_quant_arguments(parser)
    args = parser.parse_args(["--scale-search", "4", *extra])
    with pytest.raises(SystemExit) as error:
        quant.validate_quant_arguments(args, parser)
    assert error.value.code == 2


def test_zero_search_preserves_legacy_paths() -> None:
    w = np.random.default_rng(11).normal(size=(4, 32)).astype(np.float32)
    for mode in quant.QUANT_MODES:
        for dtype in (("fp32", "fp16") if mode in ("rtn", "qat-ident") else ("fp32",)):
            options = {"scale_dtype": dtype}
            args = (w, mode, "positive-max", 8.05) if mode == "qat-ident" else (w, mode)
            a = quant.quantize_variant(*args, **options)
            b = quant.quantize_variant(*args, **options, scale_search=0)
            assert a[0].tobytes() == b[0].tobytes()
            assert a[1].tobytes() == b[1].tobytes()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_rejected(value: float) -> None:
    w = np.ones((1, 32), np.float32)
    w[0, 0] = value
    with pytest.raises(ValueError, match="nonfinite weight"):
        quant.quantize_variant(w, "qat-ident", "signed-max", scale_dtype="fp16", scale_search=4)


@pytest.mark.parametrize("script,basename", [("extract_weights.py", "L0_q"),
                                          ("extract_weights_qwen35.py", "L0_self_attn.q_proj")])
def test_cli_forwards_search_writes_manifest_and_preserves_lm_head(tmp_path: Path, script: str, basename: str) -> None:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(811)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                       num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                                       layer_types=["full_attention"]))
    matrix = model.model.layers[0].self_attn.q_proj.weight.detach().numpy()
    expected = quant.quantize_variant(matrix, "qat-ident", "positive-max", 8.05, scale_dtype="fp16", scale_search=4)
    original = quant.quantize_variant(matrix, "qat-ident", "positive-max", 8.05, scale_dtype="fp16")
    assert expected[1].tobytes() != original[1].tobytes()
    lm_expected = quant.quantize_variant(model.lm_head.weight.detach().numpy(), "rtn", scale_dtype="fp16")
    source, output = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    entry = Path(__file__).resolve().parents[1] / "tools" / script
    command = [sys.executable, str(entry), "--model", str(source), "--out", str(output),
               "--quant", "qat-ident", "--quant-rule", "positive-max", "--quant-c", "8.05",
               "--scale-dtype", "fp16", "--scale-search", "4"]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["quant_scale_search_ulp"] == 4
    assert manifest["quant_scale_search_objective"] == "block_sse_float64"
    assert (output / f"{basename}_q4.bin").read_bytes() == expected[0].tobytes()
    assert (output / f"{basename}_scale.bin").read_bytes() == expected[1].tobytes()
    assert (output / "w_lm_head_q4.bin").read_bytes() == lm_expected[0].tobytes()
    assert (output / "w_lm_head_scale.bin").read_bytes() == lm_expected[1].tobytes()
    rerun = subprocess.run(command, capture_output=True, text=True, check=False)
    assert rerun.returncode != 0 and "FileExistsError" in rerun.stderr
    assert (output / f"{basename}_scale.bin").read_bytes() == expected[1].tobytes()
