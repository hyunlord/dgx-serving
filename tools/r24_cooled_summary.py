#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: python3 tools/r24_cooled_summary.py COOLED_RUN_DIR OUTPUT_PREFIX
# Stdlib only; imports sibling r24_timing_summary.py. Outputs must be new.
"""Report cooled R24 per-case timing, with measured-only thermal evidence."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

from r24_timing_summary import SHAPES, Comparison, Interval, ParseError, clocks, compare, intervals

TIMEZONE: Final = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True, slots=True)
class Coverage:
    repeat: int
    samples: int
    first_sample_offset_seconds: float | None
    last_sample_gap_seconds: float | None
    max_sample_gap_seconds: float | None


@dataclass(frozen=True, slots=True)
class EngineEvidence:
    name: str
    guard_status: str
    guard_exit_code: int | None
    status: str
    start_temperature_c: float | None
    measured_clock_masks: dict[str, int]
    measured_temperature_min_c: float | None
    measured_temperature_max_c: float | None
    clock_coverage: tuple[Coverage, ...]
    llama_wrapper_byte_identical: bool | None
    intervals: tuple[Interval, ...]


@dataclass(frozen=True, slots=True)
class Case:
    timing: Comparison
    main: EngineEvidence
    llama: EngineEvidence
    usable_cooled_comparison: bool


def evidence(run: Path, name: str) -> EngineEvidence:
    """Read completed guard evidence; retain incomplete runs without partial parsing."""
    result_file = run / f"result_{name}.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    status = str(result.get("status", "NOT_RUN"))
    code_value = result.get("exit_code")
    code = int(code_value) if code_value is not None else None
    command = run / f"command_{name}.log"
    clock_path = run / f"clockmon_{name}.csv"
    rows = intervals(command, clocks(clock_path, TIMEZONE)) if code is not None else ()
    raw: list[tuple[float, str, float]] = []
    if clock_path.exists() and code is not None:
        for row in csv.reader(clock_path.read_text().splitlines()):
            if not row or row[0].strip() == "timestamp":
                continue
            if len(row) != 6:
                raise ParseError(f"Malformed clock row in {clock_path}")
            stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=TIMEZONE).timestamp()
            raw.append((stamp, hex(int(row[5].strip(), 0)), float(row[4].strip())))
    masks: dict[str, int] = {}
    temperatures: list[float] = []
    coverage: list[Coverage] = []
    for interval in rows:
        selected = [(stamp, mask, temp) for stamp, mask, temp in raw if interval.start_epoch <= stamp <= interval.end_epoch]
        for _, mask, temp in selected:
            masks[mask] = masks.get(mask, 0) + 1
            temperatures.append(temp)
        stamps = [stamp for stamp, _, _ in selected]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        coverage.append(Coverage(interval.repeat, len(stamps), stamps[0] - interval.start_epoch if stamps else None,
                                 interval.end_epoch - stamps[-1] if stamps else None, max(gaps) if gaps else None))
    start_file = run / f"start_{name}.csv"
    start_temp: float | None = None
    if start_file.exists():
        start_rows = list(csv.reader(start_file.read_text().splitlines()))
        if len(start_rows) != 2 or len(start_rows[1]) != 4:
            raise ParseError(f"Unexpected start-temperature CSV: {start_file}")
        start_temp = float(start_rows[1][1].strip())
    wrapper: bool | None = None
    if name.startswith("llama_"):
        wrapper = any(line.startswith("CHECK variant=llama ") and "wrapper_byte_identical=1" in line
                      for line in command.read_text().splitlines()) if command.exists() else False
    good_clock = len(coverage) == 5 and all(c.samples > 0 for c in coverage)
    thermal_zero = bool(masks) and all(int(mask, 0) == 0 for mask in masks)
    state = "NOT_RUN"
    if code is not None:
        state = "PASS" if status == "clean" and code == 0 and len(rows) == 5 and good_clock and thermal_zero and wrapper is not False else "INCOMPLETE_OR_CONDITION_FAILED"
    elif status != "NOT_RUN":
        state = "RUNNING_OR_UNFINISHED"
    return EngineEvidence(name, status, code, state, start_temp, masks,
                          min(temperatures) if temperatures else None, max(temperatures) if temperatures else None,
                          tuple(coverage), wrapper, rows)


def markdown(cases: tuple[Case, ...]) -> str:
    """Keep thermal condition failures visible beside timing values."""
    lines = ["# R24 cooled synthetic sweep", "", "Main/llama use different synthetic inputs and quantization/layouts. Each engine is cooled separately. These are unpaired isolated timings, not engine-endpoint performance. All M are multiples of 128; linear versus tile-step scaling is not identifiable.", "", "PASS requires clean exit-0 guard, five >=1-second intervals, clock samples in each interval, zero measured throttle masks, and llama wrapper byte identity. Coverage offsets/gaps are preserved in JSON; start temperature is recorded before guard eviction. Existing main/swap correctness evidence remains separate.", "", "| shape | M | main status | llama status | main ms | llama ms | main/llama SM MHz | start C main/llama | measured masks main/llama |", "|---|---:|---|---|---:|---:|---|---|---|"]
    def fmt(value: float | None) -> str:
        return "NOT_RUN" if value is None else f"{value:.6f}"
    for case in cases:
        t = case.timing
        lines.append(f"| {t.shape} | {t.m} | {case.main.status} | {case.llama.status} | {fmt(t.main.mean_ms)} | {fmt(t.other.mean_ms)} | {fmt(t.main.measured_clock_median_mhz)}/{fmt(t.other.measured_clock_median_mhz)} | {fmt(case.main.start_temperature_c)}/{fmt(case.llama.start_temperature_c)} | {case.main.measured_clock_masks}/{case.llama.measured_clock_masks} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write new summary files, including all 30 expected engine cases."""
    if len(sys.argv) != 3:
        raise SystemExit("Usage: r24_cooled_summary.py COOLED_RUN_DIR OUTPUT_PREFIX")
    run, prefix = Path(sys.argv[1]), Path(sys.argv[2])
    cases: list[Case] = []
    for m in (128, 256, 512, 1024, 2048):
        for shape, n, k in SHAPES:
            ours = evidence(run, f"main_{shape}_M{m}")
            llama = evidence(run, f"llama_{shape}_M{m}")
            for expected_variant, item in (("main", ours), ("llama", llama)):
                if any((r.variant, r.m, r.n, r.k) != (expected_variant, m, n, k) for r in item.intervals):
                    raise ParseError(f"Unexpected variant or shape in {item.name}")
            timing = compare(ours.intervals + llama.intervals, (shape, m, n, k), "sweep")
            cases.append(Case(timing, ours, llama, ours.status == llama.status == "PASS"))
    paths = (Path(f"{prefix}.json"), Path(f"{prefix}.md"))
    if any(p.exists() for p in paths):
        raise ParseError("Refusing to overwrite evidence")
    with paths[0].open("x") as handle:
        json.dump({"run_directory": str(run), "clock_timezone": str(TIMEZONE), "expected_engine_cases": 30,
                   "completed_engine_cases": sum(e.guard_exit_code is not None for c in cases for e in (c.main, c.llama)),
                   "passing_engine_cases": sum(e.status == "PASS" for c in cases for e in (c.main, c.llama)),
                   "usable_comparisons": sum(c.usable_cooled_comparison for c in cases),
                   "conditions": "synthetic inputs/quantization differ; sequential unpaired; no tile-step inference",
                   "cases": [asdict(c) for c in cases]}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with paths[1].open("x") as handle:
        handle.write(markdown(tuple(cases)))


if __name__ == "__main__":
    main()
