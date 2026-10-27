"""CPU checks for opt-in extraction policies; no model or GPU required."""
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import extract_weights as quant


def unpack(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.empty(packed.size * 2, dtype=np.float32)
    values[::2] = packed & 15
    values[1::2] = packed >> 4
    return (values.reshape(-1, 32) - 8) * scale[:, None]


def test_default_matches_frozen_rtn_formula() -> None:
    # Given fixed nonzero and zero blocks.
    w = np.random.default_rng(31).normal(size=(5, 64)).astype(np.float32)
    w[0, :32] = 0
    block = w.reshape(-1, 32)
    scale = np.max(np.abs(block), axis=1) / 7
    scale[scale == 0] = 1e-8
    q = np.clip(np.round(block / scale[:, None]) + 8, 0, 15).astype(np.uint8).ravel()
    expected = q[::2] | (q[1::2] << 4)
    # When the default policy runs.
    packed, actual_scale = quant.quantize_variant(w)
    # Then both engine files retain their bytes.
    assert packed.tobytes() == expected.tobytes()
    assert actual_scale.tobytes() == scale.tobytes()


def test_signed_scale_uses_minus_eight_and_preserves_nibble_order() -> None:
    # Given signed max +8 and values on the negative scale grid.
    w = np.tile(np.arange(-7, 9, dtype=np.float32), 2).reshape(1, 32)
    # When Q4_0 quantization runs.
    packed, scale = quant.quantize_variant(w, "qat-q40")
    # Then +8 is represented by q=-8, and low nibble is the even index.
    assert scale[0] == -1
    assert packed[7] >> 4 == 0
    np.testing.assert_array_equal(unpack(packed, scale), w)


def test_ident_requires_explicit_rule_and_positive_divisor() -> None:
    w = np.ones((1, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        quant.quantize_variant(w, "qat-ident")
    with pytest.raises(ValueError):
        quant.quantize_variant(w, "qat-ident", "positive-max", 0)


def test_ident_positive_rule_is_fp32_and_exact_on_grid() -> None:
    w = np.tile(np.arange(-8, 8, dtype=np.float32), 2).reshape(1, 32) * np.float32(.125)
    packed, scale = quant.quantize_variant(w, "qat-ident", "positive-max", 8)
    assert scale.dtype == np.float32
    np.testing.assert_array_equal(unpack(packed, scale), w)


def test_clip_is_deterministic_no_worse_than_full_range_and_ties_keep_first() -> None:
    w = np.random.default_rng(19).normal(size=(8, 64)).astype(np.float32)
    q0, s0 = quant.quantize_variant(w, "orig-rtnfull")
    q1, s1 = quant.quantize_variant(w, "orig-clip")
    q2, s2 = quant.quantize_variant(w, "orig-clip")
    assert q1.tobytes() == q2.tobytes() and s1.tobytes() == s2.tobytes()
    assert np.square(unpack(q1, s1).astype(np.float64) - w.reshape(-1, 32)).sum() <= np.square(unpack(q0, s0).astype(np.float64) - w.reshape(-1, 32)).sum()
    _, zero_scale = quant.quantize_variant(np.zeros((1, 32), np.float32), "orig-clip")
    assert zero_scale[0] == np.float32(1e-8)


def test_projection_selection_excludes_lm_head_and_nonstandard_projections() -> None:
    assert quant.variant_projection("L0_self_attn.q_proj")
    assert quant.variant_projection("L2_mlp.down_proj")
    assert not quant.variant_projection("w_lm_head")
    assert not quant.variant_projection("L0_linear_attn.in_proj_qkv")


def test_metrics_match_float64_reference() -> None:
    w = np.random.default_rng(7).normal(size=(2, 32)).astype(np.float32)
    packed, scale = quant.quantize_int4(w)
    metrics = quant.reconstruction_stats(w, packed, scale)
    difference = unpack(packed, scale).astype(np.float64) - w.reshape(-1, 32)
    assert metrics["relative_error"] == np.sqrt(np.square(difference).sum() / np.square(w.astype(np.float64)).sum())


@pytest.mark.parametrize("script", ["extract_weights.py", "extract_weights_qwen35.py"])
def test_cli_rejects_existing_destination_without_loading_model(tmp_path: Path, script: str) -> None:
    # Given a pre-existing output with a sentinel, and no accessible model.
    (tmp_path / "sentinel").write_text("original")
    entry = Path(__file__).resolve().parents[1] / "tools" / script
    # When an experimental extraction is requested.
    run = subprocess.run([sys.executable, str(entry), "--model", "not-a-model", "--out", str(tmp_path),
                          "--quant", "qat-q40"], capture_output=True, text=True, check=False)
    # Then it fails before any model load and preserves the destination.
    assert run.returncode != 0 and "FileExistsError" in run.stderr
    assert (tmp_path / "sentinel").read_text() == "original"


def test_cli_rejects_fp16_for_experimental_policy(tmp_path: Path) -> None:
    entry = Path(__file__).resolve().parents[1] / "tools/extract_weights_qwen35.py"
    run = subprocess.run([sys.executable, str(entry), "--model", "not-a-model", "--out", str(tmp_path / "new"),
                          "--quant", "qat-q40", "--scale-dtype", "fp16"], capture_output=True, text=True, check=False)
    assert run.returncode == 2 and "requires fp32 scales" in run.stderr
    assert not (tmp_path / "new").exists()


def test_clipping_search_matches_independent_per_block_enumeration() -> None:
    w = np.random.default_rng(73).normal(size=(3, 32)).astype(np.float32)
    packed, scale = quant.quantize_variant(w, "orig-clip")
    expected = []
    for block in w:
        d = block[np.argmax(np.abs(block))] / np.float32(-8)
        candidates = []
        for step in range(11):
            s = d * np.float32((100 - 3 * step) / 100)
            q = np.clip(np.rint(block / s), -8, 7)
            error = np.square((q * s).astype(np.float64) - block).sum()
            candidates.append((error, s, q * s))
        expected.append(min(enumerate(candidates), key=lambda pair: (pair[1][0], pair[0]))[1])
    np.testing.assert_array_equal(scale, [row[1] for row in expected])
    np.testing.assert_array_equal(unpack(packed, scale), [row[2] for row in expected])


def test_ident_fp16_rounds_scale_before_integer_assignment() -> None:
    # Given a half-step value whose q changes when scale is rounded first.
    w = np.zeros((1, 32), dtype=np.float32)
    w[0, 0] = np.float32(8.05 * 0.1)
    w[0, 1] = np.float32(0.14998)
    # When the selected grid uses fp16 scales.
    packed, scale = quant.quantize_variant(w, "qat-ident", "positive-max", 8.05, scale_dtype="fp16")
    # Then q=2, unlike post-casting the fp32 q=1, and storage is fp16.
    assert scale.dtype == np.float16
    assert packed[0] >> 4 == 10
    old_packed, _ = quant.quantize_variant(w, "qat-ident", "positive-max", 8.05)
    assert old_packed[0] >> 4 == 9


def test_ident_fp16_zero_blocks_stay_finite() -> None:
    # Given all-zero weights.
    w = np.zeros((1, 32), dtype=np.float32)
    # When a scale-first grid is selected.
    packed, scale = quant.quantize_variant(w, "qat-ident", "positive-max", 8.05, scale_dtype="fp16")
    # Then zero is represented exactly without zero-scale division.
    assert np.isfinite(scale).all() and (scale != 0).all()
    np.testing.assert_array_equal(unpack(packed, scale), w)


def test_reconstruction_metrics_promote_fp16_scale_to_engine_fp32_product() -> None:
    # Given a finite engine product that would overflow fp16 arithmetic.
    w = np.full((1, 32), 420000, dtype=np.float32)
    packed = np.full(16, 255, dtype=np.uint8)
    scales = np.array([60000], dtype=np.float16)
    # When reconstruction metrics combine fp32 q with serialized fp16 scales.
    metrics = quant.reconstruction_stats(w, packed, scales)
    # Then exact fp32 reconstruction has no spurious overflow error.
    assert metrics["relative_error"] == 0


@pytest.mark.parametrize("magnitude", [1e-10, 1e7])
def test_ident_fp16_rejects_unrepresentable_nonzero_scale(magnitude: float) -> None:
    # Given scales that underflow to zero or overflow to infinity.
    w = np.full((1, 32), magnitude, dtype=np.float32)
    # When scale-first conversion runs, then extraction stops instead of corrupting q.
    with pytest.raises(ValueError, match="fp16 scale"):
        quant.quantize_variant(w, "qat-ident", "positive-max", 8.05, scale_dtype="fp16")


@pytest.mark.parametrize("script", ["extract_weights.py", "extract_weights_qwen35.py"])
def test_ident_fp16_cli_preserves_existing_destination(tmp_path: Path, script: str) -> None:
    # Given an existing directory, the new option must pass parsing but never overwrite it.
    entry = Path(__file__).resolve().parents[1] / "tools" / script
    # When the scale-first policy is requested.
    run = subprocess.run([sys.executable, str(entry), "--model", "not-a-model", "--out", str(tmp_path),
                          "--quant", "qat-ident", "--quant-rule", "positive-max", "--quant-c", "8.05",
                          "--scale-dtype", "fp16"], capture_output=True, text=True, check=False)
    # Then destination protection precedes model loading.
    assert run.returncode != 0 and "FileExistsError" in run.stderr


@pytest.mark.parametrize("script,basename", [("extract_weights.py", "L0_q"),
                                          ("extract_weights_qwen35.py", "L0_self_attn.q_proj")])
def test_cli_serializes_fp16_scales_and_manifest_for_local_model(tmp_path: Path, script: str, basename: str) -> None:
    # Given a tiny locally constructed model (no network or GPU).
    import json
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                       num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                                       layer_types=["full_attention"]))
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.zero_()
        model.model.layers[0].self_attn.q_proj.weight[:, 0] = 8.05 * 0.1
        model.model.layers[0].self_attn.q_proj.weight[:, 1] = 0.14998
    source, output = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    entry = Path(__file__).resolve().parents[1] / "tools" / script
    # When the real CLI extracts a fresh output.
    run = subprocess.run([sys.executable, str(entry), "--model", str(source), "--out", str(output),
                          "--quant", "qat-ident", "--quant-rule", "positive-max", "--quant-c", "8.05",
                          "--scale-dtype", "fp16"], capture_output=True, text=True, check=False)
    # Then serialized bytes and manifest agree with the requested format.
    assert run.returncode == 0, run.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["scale_dtype"] == 1
    assert manifest["quant_scale_rounding"] == "before_integer_assignment"
    scales = np.fromfile(output / f"{basename}_scale.bin", dtype=np.float16)
    packed = np.fromfile(output / f"{basename}_q4.bin", dtype=np.uint8)
    assert scales.size == 32 and (output / f"{basename}_scale.bin").stat().st_size == 64
    assert np.all(packed[::16] >> 4 == 10)
