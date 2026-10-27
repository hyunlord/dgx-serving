"""CPU/GPU timestamp alignment regressions using real in-memory SQLite rows."""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import r10_nsys_decode as parser
import r10_trace_base as base


def fixture_rows(db: sqlite3.Connection) -> list[base.Launch]:
    """Model a kernel starting 1ns before prior sync but launched 5ns after it."""
    db.executescript("""
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        INSERT INTO StringIds VALUES(1,'cudaDeviceSynchronize');
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,
            globalTid INTEGER,nameId INTEGER);
    """)
    rows = []
    for token in range(34):
        t = 1000 + token * 100
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,1,1)", (t+16, t+80))
        for index, (name, a, z, api) in enumerate((
            ("embedding_kernel", -21, -10, -15),
            ("gemv_w4a8_kernel", 20, 60, 10),
            ("softcap_kernel", 55, 70, 13),
        )):
            rows.append(base.Launch(t+a, t+z, token*3+index, name, name,
                t+api, t+api+2, 1, "cudaLaunchKernel", (1,1,1), (32,1,1), 16, 0, 0, None, token))
    return rows


def test_reports_alignment_when_gpu_start_precedes_prior_sync() -> None:
    # Given: GPU timestamps invert CPU causal order by 6ns, with valid host ordering.
    with sqlite3.connect(":memory:") as db:
        rows = fixture_rows(db)
        with pytest.raises(base.TraceError, match="boundaries overlap"):
            base.decode(db, rows, True, 0)
        # When: retaining CPU boundaries and explicitly clipping GPU interval union.
        selected = parser.decode_aligned(db, rows, 0)
    # Then: raw duration survives, outside coverage is explicit, and no wall shift occurs.
    stats = selected.stats
    assert stats["wall_ns"] == 3200
    assert stats["kernel_sum_ns"] == 2112
    assert stats["raw_gpu_union_ns"] == 1952
    assert stats["gpu_union_ns"] == 1951
    assert stats["gpu_union_outside_wall_ns"] == 1
    assert stats["kernel_overlap_ns"] == 160
    assert stats["wall_minus_gpu_union_ns"] == 1249
    assert len(selected.anomalies) == 32
    assert selected.anomalies[0]["gpu_before_own_api_ns"] == 6


def test_keeps_original_metrics_when_kernels_fit_cpu_boundaries() -> None:
    # Given: a small positive gap between sync completion and first GPU kernel.
    with sqlite3.connect(":memory:") as db:
        rows = fixture_rows(db)
        rows = [replace(r, start=r.start+2) if r.short == "embedding_kernel" else r for r in rows]
        original = base.decode(db, rows, True, 0)
        # When: using the aligned implementation on an already valid trace.
        selected = parser.decode_aligned(db, rows, 0)
    # Then: original aggregate metrics remain unchanged, with no anomalies.
    for key, value in original.items():
        if key != "tokens":
            assert selected.stats[key] == value
    assert selected.anomalies == []


@pytest.mark.parametrize("defect", ["cpu_order", "wholly_outside", "terminal_after_completion"])
def test_rejects_corrupt_ordering_when_not_a_start_alignment_inversion(defect: str) -> None:
    # Given: an ordering violation distinct from the observed cross-domain inversion.
    with sqlite3.connect(":memory:") as db:
        rows = fixture_rows(db)
        index = 2 * 3
        if defect == "cpu_order":
            rows[index] = replace(rows[index], api_start=1179)
        elif defect == "wholly_outside":
            rows[index] = replace(rows[index], end=1179)
        else:
            rows[index+2] = replace(rows[index+2], end=1281)
        # When/Then: a boundary error remains a hard failure, without epsilon recovery.
        with pytest.raises(base.TraceError):
            parser.decode_aligned(db, rows, 0)
