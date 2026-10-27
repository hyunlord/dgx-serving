#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Validate guarded GEMV probe evidence without deciding performance acceptance.

How to run: python3 tools/r10_probe_results.py LINES.jsonl GUARD.json CLOCK.csv OUT.json [d1]
Clock timestamps are Asia/Seoul; arm intervals use the probe's exact epoch_us.
Output is exclusive. Invalid/incomplete evidence exits nonzero without a summary.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


class EvidenceError(RuntimeError):
    """Evidence cannot establish the frozen measurement contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    require(math.isfinite(value), f"Nonfinite {key}")
    return value


def integer(row: dict[str, str], key: str) -> int:
    require(re.fullmatch(r"-?[0-9]+", row[key]) is not None, f"Noninteger {key}")
    return int(row[key])


def clock_rows(path: Path) -> list[tuple[int, int, int]]:
    """Parse every row; never silently discard missing or malformed clock samples."""
    result = []
    with path.open(newline="") as stream:
        for source in csv.DictReader(stream):
            row = {k.strip(): v.strip() for k, v in source.items()}
            stamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=ZoneInfo("Asia/Seoul"))
            mhz = row["clocks.current.sm [MHz]"]
            require(re.fullmatch(r"[0-9]+(?: MHz)?", mhz) is not None, "Invalid SM clock")
            result.append((round(stamp.timestamp()*1_000_000), int(mhz.split()[0]),
                           int(row["clocks_event_reasons.active"], 16)))
    require(bool(result), "Clock CSV has no samples")
    require(all(a[0] < b[0] for a, b in zip(result, result[1:])), "Clock timestamps not strictly increasing")
    return result


def arm_clock(samples: list[tuple[int, int, int]], bounds: tuple[int, int]) -> dict:
    """Report exact-interval samples and cadence without a new stability threshold."""
    begin, end = bounds
    require(samples[0][0] <= begin < end <= samples[-1][0], "Clock samples do not bracket arm")
    inside = [(t, m, mask) for t, m, mask in samples if begin <= t <= end]
    require(bool(inside), "No clock sample inside arm")
    clocks = [m for _, m, _ in inside]
    adjacent = [t for t, _, _ in samples if begin <= t <= end]
    previous = max(t for t, _, _ in samples if t <= begin)
    following = min(t for t, _, _ in samples if t >= end)
    cadence = sorted(set([previous, *adjacent, following]))
    return {"samples": len(inside), "sm_median_mhz": statistics.median(clocks),
            "sm_min_mhz": min(clocks), "sm_max_mhz": max(clocks),
            "first_epoch_us": inside[0][0], "last_epoch_us": inside[-1][0],
            "max_sample_gap_us": max((b-a for a, b in zip(cadence, cadence[1:])), default=0),
            "nonzero_throttle_mask_samples": sum(mask != 0 for _, _, mask in inside)}


def read_groups(path: Path) -> list[list[tuple[str, dict[str, str]]]]:
    groups: list[list[tuple[str, dict[str, str]]]] = []
    complete = 0
    with path.open() as stream:
        for raw in stream:
            record = json.loads(raw)
            require(isinstance(record, dict) and isinstance(record.get("line"), str), "Malformed JSONL record")
            line = record["line"]
            require("PROBE_FAILED" not in line and "INVALID_SHORT_ARM" not in line, "Probe reports failure")
            if record.get("phase") != "benchmark":
                continue
            if line.startswith("PROBE_COMPLETE"):
                require("mode=bench " in line, "Only bench results supported")
                complete += 1
                continue
            kind = line.split(" ", 1)[0]
            if kind not in {"CONFIG", "DETERMINISTIC", "CORRECT", "CALIBRATE", "WARM", "PAIR"}:
                continue
            require(complete == 0, "Records follow PROBE_COMPLETE")
            fields = [part.split("=", 1) for part in line.split()[1:]]
            require(all(len(x) == 2 for x in fields), "Malformed probe field")
            values = dict(fields)
            require(len(values) == len(fields), "Duplicate probe field")
            if kind == "CONFIG":
                groups.append([])
            require(bool(groups), "Record precedes CONFIG")
            groups[-1].append((kind, values))
    require(complete == 1 and bool(groups), "Missing/duplicate completion or configuration")
    return groups


def summarize(group: list[tuple[str, dict[str, str]]], variant: str,
              clocks: list[tuple[int, int, int]]) -> dict:
    config = group[0][1]
    width, out, dtype = (integer(config, k) for k in ("in", "out", "dtype"))
    require(width > 0 and width % 32 == 0 and out > 0 and dtype in (0, 1), "Invalid shape/dtype")
    require(integer(config, "QBLK") == 32, "QBLK differs")
    buffers, weight, rotating, logical = (integer(config, k) for k in
        ("buffers", "weight_bytes", "rotating_weight_bytes", "logical_bytes_per_launch"))
    scratch = width + width//32*4 + out*4
    require(weight == width*out//2 + width//32*out*(2 if dtype else 4), "Weight byte accounting differs")
    require(buffers >= 2 and rotating == buffers*weight and rotating >= 96*1024**2,
            "Rotating weights fail 96MiB/two-buffer contract")
    require(rotating+scratch <= 2_000_000_000 and logical == weight+scratch, "Memory cap/logical byte mismatch")
    arms = ("baseline", variant)
    correct: dict[str, dict[str, str]] = {}
    deterministic: set[str] = set()
    calibration: dict[str, list[dict[str, str]]] = {a: [] for a in arms}
    pairs: list[dict] = []
    warm: dict[str, str] | None = None
    previous_end = 0
    comparisons = {"old_gpu_vs_cpu", f"{variant}_vs_cpu", f"{variant}_vs_old_gpu"}
    for kind, row in group[1:]:
        require(row.get("shape") == config["shape"] and row.get("dtype") == config["dtype"], "Mixed configuration")
        if kind == "CORRECT":
            name = row["comparison"]
            require(name in comparisons and name not in correct, "Duplicate/unknown correctness comparison")
            require(row["status"] == "PASS" and row["denominator"] == "CPUdouble_sum_abs_terms"
                and number(row, "threshold") == 1e-5 and integer(row, "first_fail_index") == -1
                and 0 <= number(row, "median") <= number(row, "p99") <= number(row, "max") < 1e-5,
                "Correctness gate failed or changed")
            correct[name] = row
        elif kind == "DETERMINISTIC":
            name = row["variant"]
            require(name in arms and name not in deterministic and row["exact"] == "30/30", "Determinism gate failed")
            deterministic.add(name)
        elif kind == "CALIBRATE":
            name = row["variant"]
            require(name in arms and not pairs and warm is None, "Late/unknown calibration")
            reps, ms = integer(row, "reps"), number(row, "event_ms")
            prior = calibration[name]
            require(0 < reps <= 10_000_000 and reps % buffers == 0 and ms > 0, "Invalid calibration")
            require(bool(prior) or reps == buffers, "Initial calibration must cover one rotation")
            require(not prior or (reps > integer(prior[-1], "reps") and number(prior[-1], "event_ms") < 1100), "Calibration restarts/duplicates")
            prior.append(row)
        elif kind == "WARM":
            require(warm is None and len(pairs) < 10, "Unpaired/extra warmup")
            require(number(row, "event_ms") >= 10000 and integer(row, "reps") > 0, "Warmup below 10 seconds")
            require(integer(row, "epoch_begin_us") >= previous_end
                and integer(row, "epoch_end_us")-integer(row, "epoch_begin_us") >= 10_000_000,
                "Warm interval overlaps or is below 10 seconds")
            warm = row
        elif kind == "PAIR":
            index = len(pairs)
            name = arms[index % 2]
            require(index < 10 and integer(row, "pair") == index//2 and integer(row, "order") == index%2
                and row["variant"] == name and warm is not None and warm["variant"] == name, "Missing/duplicate/wrong pair arm")
            require(set(correct) == comparisons and deterministic == set(arms), "Timing before all correctness gates")
            require(all(calibration[a] and number(calibration[a][-1], "event_ms") >= 1100 for a in arms), "Calibration incomplete")
            reps, ms, us = integer(row, "reps"), number(row, "event_ms"), number(row, "us_per_launch")
            require(reps == integer(calibration[name][-1], "reps") and ms >= 1000 and us > 0, "Short arm or repetitions changed")
            require(abs(us-ms*1000/reps) <= .0005/reps + .000000000501, "Printed us inconsistent with event_ms")
            require(integer(row, "logical_bytes") == logical, "Arm logical bytes differ")
            begin, end = integer(row, "epoch_begin_us"), integer(row, "epoch_end_us")
            require(warm is not None and begin >= integer(warm, "epoch_end_us") and end-begin >= 1_000_000,
                    "Arm/warm interval overlap or wall duration below one second")
            pairs.append({**row, "logical_GB_s": logical/(us*1000), "clock": arm_clock(clocks, (begin, end)), "warm": warm})
            previous_end, warm = end, None
        else:
            raise EvidenceError(f"Unexpected record {kind}")
    require(len(pairs) == 10 and warm is None, "Five complete pairs required")
    ratios = [float(pairs[i]["us_per_launch"])/float(pairs[i+1]["us_per_launch"]) for i in range(0, 10, 2)]
    return {"config": config, "correctness": correct, "deterministic": sorted(deterministic),
            "calibration": calibration, "arms": pairs, "baseline_over_variant_ratios": ratios,
            "paired_ratio_median": statistics.median(ratios)}


def run(lines: Path, guard_path: Path, clock: Path, output: Path, variant: str = "d1") -> None:
    require(not output.exists(), "Refusing to overwrite evidence")
    require(re.fullmatch(r"d[0-9]+", variant) is not None, "Invalid variant name")
    guard = json.loads(guard_path.read_text())
    require(guard["status"] == "clean" and guard["command_exit"] == 0 and guard["exit_code"] == 0
        and guard["monitors_failed"] is False and guard["unexpected_processes"] == []
        and guard["resident_memory_increases"] == [] and guard["clock_gate_failed"] is False,
        "Guard is not clean")
    groups = read_groups(lines)
    keys = [(g[0][1]["shape"], g[0][1]["dtype"]) for g in groups]
    require(len(set(keys)) == len(keys), "Duplicate shape/dtype runs")
    samples = clock_rows(clock)
    results = [summarize(g, variant, samples) for g in groups]
    for result in results:
        for arm in result["arms"]:
            require(guard["command_start_unix"]*1e6 <= int(arm["warm"]["epoch_begin_us"])
                and int(arm["epoch_end_us"]) <= guard["command_end_unix"]*1e6, "Arm outside guarded command")
    encoded = json.dumps({"status": "valid", "variant": variant, "results": results,
        "sources": {"lines": str(lines), "guard": str(guard_path), "clock": str(clock)},
        "note": "Validity only, no performance acceptance. GB/s is logical bytes, not DRAM. Clock timezone Asia/Seoul; inclusive exact arm epoch bounds. Cadence reported without new stability threshold."}, indent=2)+"\n"
    require(len(encoded.encode()) <= 1_000_000, "Summary exceeds 1MB")
    with output.open("x") as stream:
        stream.write(encoded)


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        raise SystemExit(__doc__)
    run(*(Path(arg) for arg in sys.argv[1:5]), variant=sys.argv[5] if len(sys.argv) == 6 else "d1")
