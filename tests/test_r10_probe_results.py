"""Frozen probe format checks using synthetic line logs and KST clock CSVs."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import r10_probe_results as parser


def evidence(directory: Path, variant: str = "d1") -> tuple[Path, Path, Path, Path]:
    lines, guard, clock, output = (directory/name for name in ("lines.jsonl", "guard.json", "clock.csv", "out.json"))
    records = []
    def emit(line: str) -> None:
        records.append({"phase": "benchmark", "epoch_seconds": 1700000000, "line": line})
    emit("CONFIG shape=q_sliding in=3840 out=4096 dtype=0 QBLK=32 buffers=11 weight_bytes=9830400 rotating_weight_bytes=108134400 logical_bytes_per_launch=9851104 rows_per_block=1 warps_per_row=8 chunk=8")
    for name in ("baseline", variant):
        emit(f"DETERMINISTIC shape=q_sliding dtype=0 variant={name} exact=30/30")
    for name in ("old_gpu_vs_cpu", f"{variant}_vs_cpu", f"{variant}_vs_old_gpu"):
        emit(f"CORRECT shape=q_sliding dtype=0 comparison={name} denominator=CPUdouble_sum_abs_terms max=1e-7 median=1e-8 p99=5e-8 first_fail_index=-1 threshold=1e-5 status=PASS")
    for name, reps in (("baseline", 11000), (variant, 22000)):
        emit(f"CALIBRATE shape=q_sliding dtype=0 variant={name} reps=11 event_ms=1.000000")
        emit(f"CALIBRATE shape=q_sliding dtype=0 variant={name} reps={reps} event_ms=1200.000000")
    timestamp = 1700000000000000
    for pair in range(5):
        for order, name in enumerate(("baseline", variant)):
            warm_end = timestamp+10000000
            emit(f"WARM shape=q_sliding dtype=0 variant={name} reps=110000 event_ms=10000.000000 epoch_begin_us={timestamp} epoch_end_us={warm_end}")
            reps = 11000 if order == 0 else 22000
            end = warm_end+1200000
            emit(f"PAIR shape=q_sliding dtype=0 pair={pair} order={order} variant={name} reps={reps} event_ms=1200.000000 us_per_launch={1200000/reps:.9f} logical_bytes=9851104 epoch_begin_us={warm_end} epoch_end_us={end}")
            timestamp = end
    emit("PROBE_COMPLETE mode=bench (clock stability and performance acceptance require external analysis)")
    lines.write_text("".join(json.dumps(r)+"\n" for r in records))
    guard.write_text(json.dumps({"status": "clean", "command_exit": 0, "exit_code": 0,
        "monitors_failed": False, "unexpected_processes": [], "resident_memory_increases": [],
        "clock_gate_failed": False, "command_start_unix": 1699999999, "command_end_unix": 1700000120}))
    samples = ["timestamp, clocks.current.sm [MHz], clocks_event_reasons.active"]
    for i in range(1201):
        instant = datetime.fromtimestamp(1700000000+i/10, ZoneInfo("Asia/Seoul"))
        samples.append(f"{instant.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}, {2400+i%3} MHz, 0x0000000000000000")
    clock.write_text("\n".join(samples)+"\n")
    return lines, guard, clock, output


def test_exact_arm_clock_and_five_ratios_via_cli(tmp_path: Path) -> None:
    # Given: five valid baseline/candidate pairs and 100ms KST samples.
    paths = evidence(tmp_path)
    # When: invoking the standalone parser through its public CLI.
    subprocess.run([sys.executable, str(Path(parser.__file__)), *map(str, paths)], check=True)
    # Then: timestamps, unrounded source strings and logical bandwidth remain auditable.
    result = json.loads(paths[-1].read_text())["results"][0]
    assert len(result["arms"]) == 10
    assert result["paired_ratio_median"] == pytest.approx(2)
    assert result["arms"][0]["clock"]["samples"] == 13
    assert result["arms"][0]["clock"]["sm_median_mhz"] == 2401
    assert result["arms"][0]["clock"]["first_epoch_us"] == 1700000010000000
    assert result["arms"][0]["clock"]["max_sample_gap_us"] == 100000
    assert result["arms"][0]["event_ms"] == "1200.000000"
    assert result["arms"][0]["logical_GB_s"] == pytest.approx(9851104/(1200000/11000*1000))
    before = paths[-1].read_bytes()
    with pytest.raises(parser.EvidenceError, match="overwrite"):
        parser.run(*paths)
    assert paths[-1].read_bytes() == before


@pytest.mark.parametrize("before,after", [
    ("pair=4 order=1", "pair=4 order=0"),
    ("event_ms=10000.000000", "event_ms=9999.000000"),
    ("event_ms=1200.000000 us_per_launch", "event_ms=999.000000 us_per_launch"),
    ("max=1e-7", "max=1e-5"),
    ("exact=30/30", "exact=29/30"),
    ("PAIR shape=q_sliding dtype=0 pair=0 order=0 variant=baseline reps=11000", "PAIR shape=q_sliding dtype=0 pair=0 order=0 variant=baseline reps=11011"),
    ("rotating_weight_bytes=108134400", "rotating_weight_bytes=100000000"),
    ("PROBE_COMPLETE", "MISSING_COMPLETE"),
])
def test_rejects_changed_contract(tmp_path: Path, before: str, after: str) -> None:
    # Given: one frozen-contract field is invalid.
    paths = evidence(tmp_path)
    paths[0].write_text(paths[0].read_text().replace(before, after, 1))
    # When/Then: parsing rejects it and produces no valid summary.
    with pytest.raises(parser.EvidenceError):
        parser.run(*paths)
    assert not paths[-1].exists()


@pytest.mark.parametrize("defect", ["missing_arm", "guard", "missing_guard", "empty_clock", "missing_clock", "unbracketed_clock", "no_arm_samples"])
def test_rejects_missing_or_unguarded_evidence(tmp_path: Path, defect: str) -> None:
    paths = evidence(tmp_path)
    if defect == "missing_arm":
        records = paths[0].read_text().splitlines()
        paths[0].write_text("\n".join(r for r in records if "pair=4 order=1" not in r)+"\n")
    elif defect == "guard":
        paths[1].write_text(paths[1].read_text().replace('"clean"', '"failed"'))
    elif defect == "missing_guard":
        paths[1].unlink()
    elif defect == "empty_clock":
        paths[2].write_text(paths[2].read_text().splitlines()[0]+"\n")
    elif defect == "missing_clock":
        paths[2].unlink()
    elif defect == "no_arm_samples":
        rows = paths[2].read_text().splitlines()
        paths[2].write_text("\n".join(rows[:101]+rows[114:])+"\n")
    else:
        paths[2].write_text("\n".join(paths[2].read_text().splitlines()[:5])+"\n")
    with pytest.raises((parser.EvidenceError, FileNotFoundError)):
        parser.run(*paths)
    assert not paths[-1].exists()


def test_explicit_variant_parameter(tmp_path: Path) -> None:
    paths = evidence(tmp_path, "d3")
    parser.run(*paths, variant="d3")
    assert json.loads(paths[-1].read_text())["variant"] == "d3"
