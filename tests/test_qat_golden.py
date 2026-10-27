"""Small CPU fixtures for exact engine restoration and golden isolation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from qat_golden_loader import dequant_chunks, engine_key
from make_reference_gemma4_qat import template_inputs


def test_low_nibble_first_and_negative_eight(tmp_path: Path) -> None:
    q = tmp_path / 'q.bin'; s = tmp_path / 's.bin'
    np.arange(16, dtype=np.uint8).tofile(q)
    np.array([0.125], dtype='<f4').tofile(s)
    blocks = list(dequant_chunks(q, s, 0))
    expected = np.array([v for low in range(16) for v in ((low-8)*0.125, -1)], dtype=np.float32)
    assert blocks[0][0] == 0
    np.testing.assert_array_equal(blocks[0][1].view(np.uint32), expected.view(np.uint32))


def test_fp16_scales_and_short_scale_rejected(tmp_path: Path) -> None:
    q = tmp_path / 'q.bin'; s = tmp_path / 's.bin'
    q.write_bytes(bytes([0xf0])*16)
    np.array([0.5], dtype='<f2').tofile(s)
    np.testing.assert_array_equal(list(dequant_chunks(q, s, 1))[0][1], [-4, 3.5]*16)
    with pytest.raises(RuntimeError, match='scale'):
        list(dequant_chunks(q, s, 0))


def test_mapping_does_not_quantize_embedding() -> None:
    assert engine_key('w_lm_head') == 'lm_head.weight'
    assert engine_key('L17_self_attn.o_proj') == 'model.language_model.layers.17.self_attn.o_proj.weight'
    with pytest.raises(RuntimeError, match='Unsupported'):
        engine_key('w_embed')


def test_template_reuses_ids_and_exact_fp32_depths(tmp_path: Path) -> None:
    ids = np.array([818,5279,529,7001,563], dtype='<i4')
    ids.tofile(tmp_path / 'token_ids.bin')
    for depth in range(9):
        (tmp_path / f'q4fp32_hs_{depth}.bin').write_bytes(b'')
    got, depths = template_inputs(tmp_path)
    assert got.tobytes() == ids.tobytes()
    assert depths == list(range(9))
    np.array([2,3], dtype='<i4').tofile(tmp_path / 'token_ids.bin')
    with pytest.raises(RuntimeError, match='five'):
        template_inputs(tmp_path)
