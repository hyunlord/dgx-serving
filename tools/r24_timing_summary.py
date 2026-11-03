#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: python3 tools/r24_timing_summary.py RUN_DIR OUTPUT_PREFIX [Asia/Seoul]
# Stdlib only: the task prohibits adding dependencies. Output files must be new.
"""Summarize R24 synthetic intervals, preserving missing measurements and clocks."""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Final
from zoneinfo import ZoneInfo

SHAPES: Final = (("gate", 15360, 3840), ("down", 3840, 15360), ("o", 3840, 4096))
FIELDS: Final = re.compile(r"(\w+)=([^\s]+)")


@dataclass(frozen=True, slots=True)
class ParseError(Exception):
    detail: str


@dataclass(frozen=True, slots=True)
class Clock:
    epoch: float
    mhz: int


@dataclass(frozen=True, slots=True)
class Interval:
    variant: str
    m: int
    n: int
    k: int
    repeat: int
    launches: int
    total_ms: float
    mean_ms: float
    start_epoch: float
    end_epoch: float
    clock_samples: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Stats:
    status: str
    count: int
    mean_ms: float | None
    median_ms: float | None
    min_ms: float | None
    max_ms: float | None
    sample_stdev_ms: float | None
    measured_clock_median_mhz: float | None
    measured_clock_samples: int
    intervals_without_clocks: int
    intervals: tuple[Interval, ...]


@dataclass(frozen=True, slots=True)
class Comparison:
    kind: str
    shape: str
    m: int
    n: int
    k: int
    main: Stats
    other: Stats
    other_over_main_mean_time: float | None
    paired_time_ratios: tuple[float, ...]
    paired_ratio_mean: float | None
    paired_ratio_min: float | None
    paired_ratio_max: float | None


def clocks(path: Path, timezone: ZoneInfo) -> tuple[Clock, ...]:
    """Parse the six-column guard clock monitor using an explicit device timezone."""
    if not path.exists():
        return ()
    result: list[Clock] = []
    for row in csv.reader(path.read_text().splitlines()):
        if not row or row[0].strip() == "timestamp":
            continue
        if len(row) != 6:
            raise ParseError(f"Invalid clock row in {path}: {row!r}")
        stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        mhz = int(row[1].strip().removesuffix(" MHz"))
        result.append(Clock(stamp.replace(tzinfo=timezone).timestamp(), mhz))
    return tuple(result)


def intervals(path: Path, samples: tuple[Clock, ...]) -> tuple[Interval, ...]:
    """Include only measured intervals; reject malformed, duplicate or short runs."""
    if not path.exists():
        return ()
    result: list[Interval] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for line in path.read_text().splitlines():
        if not line.startswith("INTERVAL "):
            continue
        f = dict(FIELDS.findall(line))
        if f.get("phase") != "measure":
            continue
        variant, m, n, k, repeat = f["variant"], int(f["M"]), int(f["N"]), int(f["K"]), int(f["pair"])
        if variant not in {"main", "swap", "llama"}:
            raise ParseError(f"Unknown variant {variant} in {path}")
        total, per, start, end = (float(f[key]) for key in ("total_ms", "mean_ms", "start_epoch", "end_epoch"))
        launches = int(f["launches"])
        key = (variant, m, n, k, repeat)
        if key in seen or repeat not in range(5) or launches <= 0:
            raise ParseError(f"Invalid or duplicate interval {key} in {path}")
        if not all(math.isfinite(v) for v in (total, per, start, end)) or total < 1000 or per <= 0 or end <= start:
            raise ParseError(f"Invalid or short interval {key} in {path}")
        if not math.isclose(total / launches, per, rel_tol=1e-6, abs_tol=1e-8):
            raise ParseError(f"Inconsistent mean time {key} in {path}")
        seen.add(key)
        selected = tuple(c.mhz for c in samples if start <= c.epoch <= end)
        result.append(Interval(variant, m, n, k, repeat, launches, total, per, start, end, selected))
    return tuple(result)


def stats(rows: tuple[Interval, ...]) -> Stats:
    """Aggregate run means equally; pool only clock samples inside measured intervals."""
    times = [r.mean_ms for r in rows]
    clock_values = [c for r in rows for c in r.clock_samples]
    status = "NOT_RUN" if not rows else "COMPLETE" if len(rows) == 5 else "PARTIAL"
    return Stats(status, len(rows), mean(times) if times else None, median(times) if times else None,
                 min(times) if times else None, max(times) if times else None,
                 stdev(times) if len(times) > 1 else None, median(clock_values) if clock_values else None,
                 len(clock_values), sum(not r.clock_samples for r in rows), rows)


def compare(rows: tuple[Interval, ...], shape: tuple[str, int, int, int], kind: str) -> Comparison:
    """Use paired ratios only for the alternating main/swap grid experiment."""
    name, m, n, k = shape
    other_name = "swap" if kind == "grid" else "llama"
    selected = tuple(r for r in rows if (r.m, r.n, r.k) == (m, n, k))
    main = stats(tuple(sorted((r for r in selected if r.variant == "main"), key=lambda r: r.repeat)))
    other = stats(tuple(sorted((r for r in selected if r.variant == other_name), key=lambda r: r.repeat)))
    base = {r.repeat: r.mean_ms for r in main.intervals}
    ratios = tuple(r.mean_ms / base[r.repeat] for r in other.intervals if r.repeat in base) if kind == "grid" else ()
    ratio = other.mean_ms / main.mean_ms if main.mean_ms is not None and other.mean_ms is not None else None
    return Comparison(kind, name, m, n, k, main, other, ratio, ratios, mean(ratios) if ratios else None,
                      min(ratios) if ratios else None, max(ratios) if ratios else None)


def render(rows: tuple[Comparison, ...]) -> str:
    """Render missing evidence explicitly, without treating synthetic data as engine timing."""
    lines = ["# R24 isolated timings", "", "Synthetic inputs; quantization and transfers excluded. Main and llama use different input distributions and quantization/layouts. Main/llama sweeps are sequential, not paired. Grid main/swap uses identical inputs and alternating pairs.", "", "M samples are all multiples of 128: these samples do not resolve tile-boundary steps. Clock medians pool samples only inside measured intervals. Guard cleanliness and byte-identity gates must be checked separately.", "", "| kind | shape | M | main count/status | other count/status | main ms | other ms | other/main time | main SM MHz | other SM MHz |", "|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
    def fmt(value: float | None) -> str:
        return "NOT_RUN" if value is None else f"{value:.6f}"
    for r in rows:
        lines.append(f"| {r.kind} | {r.shape} | {r.m} | {r.main.count}/{r.main.status} | {r.other.count}/{r.other.status} | {fmt(r.main.mean_ms)} | {fmt(r.other.mean_ms)} | {fmt(r.other_over_main_mean_time)} | {fmt(r.main.measured_clock_median_mhz)} | {fmt(r.other.measured_clock_median_mhz)} |")
    lines.extend(["", "JSON retains all five interval timings, min/max/median/sample standard deviation, clock sample counts and paired grid ratios. Time ratio below 1 means the other variant is faster; it is not a throughput ratio or adoption gate.", ""])
    return "\n".join(lines)


def main() -> None:
    """Read guarded command/clock files and create exclusive JSON and Markdown outputs."""
    if len(sys.argv) not in (3, 4):
        raise SystemExit("Usage: r24_timing_summary.py RUN_DIR OUTPUT_PREFIX [Asia/Seoul]")
    run, prefix = Path(sys.argv[1]), Path(sys.argv[2])
    timezone = ZoneInfo(sys.argv[3] if len(sys.argv) == 4 else "Asia/Seoul")
    logs: dict[str, tuple[Interval, ...]] = {}
    for name in ("grid_bench", "main_sweep", "llama_sweep"):
        logs[name] = intervals(run / f"command_{name}.log", clocks(run / f"clockmon_{name}.csv", timezone))
    comparisons = tuple(compare(logs["grid_bench"], (name, m, n, k), "grid")
                        for m in (1024, 2048) for name, n, k in SHAPES)
    comparisons += tuple(compare(logs["main_sweep"] + logs["llama_sweep"], (name, m, n, k), "sweep")
                         for m in (128, 256, 512, 1024, 2048) for name, n, k in SHAPES)
    outputs = (Path(f"{prefix}.json"), Path(f"{prefix}.md"))
    if any(p.exists() for p in outputs):
        raise ParseError("Refusing to overwrite output files")
    with outputs[0].open("x") as handle:
        json.dump({"run_directory": str(run), "clock_timezone": str(timezone),
                   "conditions": "synthetic; main/llama sequential and different data/quantization; guard status not audited",
                   "expected_grid_cases": 6, "expected_sweep_cases": 15,
                   "comparisons": [asdict(r) for r in comparisons]}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with outputs[1].open("x") as handle:
        handle.write(render(comparisons))


if __name__ == "__main__":
    main()
