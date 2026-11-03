#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: python3 tools/r24_ncu_summary.py raw.csv --source source.csv
"""Summarize one NCU launch without treating L2 or PC samples as DRAM/time."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

MISS: Final = "lts__t_sectors_op_read_lookup_miss.sum"
STALL: Final = "smsp__average_warps_issue_stalled_"
SAMPLE: Final = "smsp__pcsamp_warps_issue_stalled_"
MISSING: Final = {"", "n/a", "nan", "inf", "-inf", "not available"}


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    value: float | None
    unit: str
    raw: str


@dataclass(frozen=True, slots=True)
class Launch:
    kernel: str
    metrics: tuple[Metric, ...]

    def value(self, name: str) -> float | None:
        return next((m.value for m in self.metrics if m.name == name), None)


@dataclass(frozen=True, slots=True)
class SourceLine:
    address: str
    instruction: str
    samples: float | None
    shared_excessive: float | None
    category: str
    sample_reason_count: int
    available_sample_reason_count: int


@dataclass(frozen=True, slots=True)
class InputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def number(raw: str) -> float | None:
    """Parse unavailable counters as absent, never as zero."""
    if raw.strip().lower() in MISSING:
        return None
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def metric(name: str, raw: str, unit: str) -> Metric:
    value = number(raw)
    factors = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    if len(unit) > 1 and unit[0] in factors:
        if value is not None:
            value *= factors[unit[0]]
        unit = unit[1:]
    return Metric(name, value, unit, raw)


def parse_raw(text: str) -> Launch:
    """Accept NCU wide raw CSV and traditional long metric CSV; require one launch."""
    rows = list(csv.reader(io.StringIO(text)))
    start = next((i for i, r in enumerate(rows) if "Kernel Name" in r and "ID" in r), None)
    if start is None:
        raise InputError("Missing NCU raw CSV header")
    header = rows[start]
    body = [r for r in rows[start + 1:] if len(r) == len(header)]
    kernel_index = header.index("Kernel Name")
    if "Metric Name" in header:
        records = [dict(zip(header, r)) for r in body if r[header.index("ID")]]
        if len({r["ID"] for r in records}) != 1:
            raise InputError("Expected exactly one profiled launch")
        metrics = tuple(metric(r["Metric Name"], r["Metric Value"], r.get("Metric Unit", "")) for r in records)
        return Launch(records[0]["Kernel Name"], metrics)
    if len(body) != 2:
        raise InputError("Expected units row and exactly one wide CSV launch")
    units, values = body
    metrics = tuple(metric(n, v, u) for n, v, u in zip(header, values, units) if "__" in n)
    return Launch(values[kernel_index], metrics)


def category(instruction: str) -> str:
    """Assign a mnemonic category, leaving causal interpretation to SASS review."""
    tokens = instruction.split()
    opcode = next((t.split(".")[0] for t in tokens if not t.startswith("@")), "")
    groups = {"LDSM": "A_ldmatrix", "LDS": "shared_load", "STS": "shared_store", "LDGSTS": "async_copy", "IMMA": "integer_mma", "FMUL": "scale_accumulate", "FFMA": "scale_accumulate", "BAR": "barrier", "DEPBAR": "barrier", "LOP3": "integer_alu", "SHF": "integer_alu", "PRMT": "integer_alu"}
    return groups.get(opcode, opcode or "unknown")


def parse_source(text: str) -> tuple[SourceLine, ...]:
    rows = list(csv.reader(io.StringIO(text)))
    starts = [i for i, r in enumerate(rows) if "Address" in r and "Source" in r]
    if len(starts) != 1:
        raise InputError("Expected one SASS source CSV table")
    start = starts[0]
    aliases = {"Instructions Executed": "inst_executed", "L1 Wavefronts Shared": "memory_l1_wavefronts_shared", "L1 Wavefronts Shared Ideal": "memory_l1_wavefronts_shared_ideal"}
    reasons = {"barrier": "barrier", "branch_resolving": "branch_resolving", "dispatch": "dispatch_stall", "drain": "drain", "lg": "lg_throttle", "long_sb": "long_scoreboard", "math": "math_pipe_throttle", "membar": "membar", "mio": "mio_throttle", "misc": "misc", "no_inst": "no_instructions", "not_selected": "not_selected", "selected": "selected", "short_sb": "short_scoreboard", "sleep": "sleeping", "tex": "tex_throttle", "wait": "wait"}
    aliases.update({"stall_" + label: SAMPLE + reason for label, reason in reasons.items()})
    header = [aliases.get(name, name) for name in rows[start]]
    # Use all-sample stall reasons only: never add the duplicate not-issued family.
    sample_names = [n for n in header if n.startswith(SAMPLE) and "not_issued" not in n and n != SAMPLE + "selected"]
    result = []
    for row in rows[start + 1:]:
        if len(row) != len(header):
            continue
        record = dict(zip(header, row))
        if not record["Address"].startswith("0x"):
            continue
        samples = [number(record[n]) for n in sample_names]
        total = sum(v for v in samples if v is not None) if samples and all(v is not None for v in samples) else None
        actual = number(record.get("memory_l1_wavefronts_shared", ""))
        ideal = number(record.get("memory_l1_wavefronts_shared_ideal", ""))
        excess = actual - ideal if actual is not None and ideal is not None else number(record.get("derived__memory_l1_wavefronts_shared_excessive", ""))
        result.append(SourceLine(record["Address"], record["Source"], total, excess, category(record["Source"]), len(sample_names), sum(v is not None for v in samples)))
    return tuple(result)


def summarize(launch: Launch, source: tuple[SourceLine, ...], calibration_bytes: int | None) -> str:
    """Emit JSON with explicit unavailable values and sample/cycle denominators."""
    misses = launch.value(MISS)
    estimated = misses * 32 if misses is not None else None
    ratio = estimated / calibration_bytes if estimated is not None and calibration_bytes else None
    calibration = "NOT_REQUESTED" if calibration_bytes is None else "UNAVAILABLE"
    if ratio is not None:
        calibration = "PASS" if 0.9 <= ratio <= 1.1 else "FAIL_STOP_PROXY"
    stalls = []
    for item in launch.metrics:
        if item.name.startswith(STALL) and item.name.endswith("_per_issue_active.ratio"):
            reason = item.name[len(STALL):-len("_per_issue_active.ratio")]
            percent = launch.value(f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct")
            stalls.append({"reason": reason, "cycles_per_issue": item.value, "active_warp_cycle_pct": percent, "overlapping_subreason": reason.startswith(("long_scoreboard_pipe_", "mio_throttle_pipe_")), "is_stall": reason != "selected"})
    total_samples = sum(s.samples or 0 for s in source)
    top = sorted((s for s in source if s.samples is not None), key=lambda s: s.samples or 0, reverse=True)[:10]
    top_rows = [{**asdict(s), "all_exported_stall_samples_pct": 100 * (s.samples or 0) / total_samples if total_samples else None} for s in top]
    attribution: dict[str, float] = {}
    for line in source:
        if line.shared_excessive is not None:
            attribution[line.category] = attribution.get(line.category, 0) + line.shared_excessive
    report = {
        "kernel": launch.kernel, "metrics": [asdict(m) for m in launch.metrics], "stalls": stalls,
        "l2_read_miss_requested_bytes_proxy": estimated, "dram_bytes": None,
        "calibration": {"status": calibration, "known_read_bytes": calibration_bytes, "proxy_ratio": ratio, "tolerance": [0.9, 1.1]},
        "source": {"top10_pc": top_rows, "exported_stall_samples": total_samples if any(s.samples is not None for s in source) else None, "source_pc_count": len(source), "sample_reason_count": max((s.sample_reason_count for s in source), default=0), "minimum_available_sample_reason_count": min((s.available_sample_reason_count for s in source), default=0), "shared_excessive_wavefronts_by_instruction_category": attribution},
        "limitations": ["L2 proxy is not measured DRAM bytes", "Stall cycles per issue are not wall-clock time", "PC percentages cover exported stall columns only; export all reasons", "Sampled waiting PC can be the consumer of a load, not the load", "Shared load category does not alone distinguish B weights from scales"],
    }
    return json.dumps(report, indent=2, allow_nan=False)


def self_test() -> None:
    """Check missing counters, calibration boundaries and correlated source parsing."""
    raw = f'"ID","Kernel Name","{MISS}"\n"","","sector"\n"0","flat16","33,554,432"\n'
    launch = parse_raw(raw)
    assert launch.value(MISS) == 33554432
    assert '"status": "PASS"' in summarize(launch, (), 1073741824)
    assert '"status": "FAIL_STOP_PROXY"' in summarize(launch, (), 2000000000)
    assert '"status": "UNAVAILABLE"' in summarize(Launch("missing", ()), (), 1073741824)
    assert number("nan") is None
    assert metric("x", "2", "Msector").value == 2000000
    src = f'"Kernel Name","k"\n"Address","Source","{SAMPLE}barrier","memory_l1_wavefronts_shared","memory_l1_wavefronts_shared_ideal"\n"0x10","LDS R1, [R2]","4","12","4"\n'
    line = parse_source(src)[0]
    assert line.samples == 4 and line.shared_excessive == 8
    assert line.category == "shared_load"
    selected = src.replace(SAMPLE + "barrier", SAMPLE + "not_selected")
    assert parse_source(selected)[0].samples == 4
    display = src.replace(SAMPLE + "barrier", "stall_barrier").replace("memory_l1_wavefronts_shared_ideal", "L1 Wavefronts Shared Ideal").replace("memory_l1_wavefronts_shared", "L1 Wavefronts Shared")
    assert parse_source(display)[0].shared_excessive == 8
    assert parse_source(display)[0].samples == 4
    print("SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", nargs="?", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--calibration-bytes", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.raw is None or (args.calibration_bytes is not None and args.calibration_bytes <= 0):
        parser.error("raw CSV required; calibration bytes must be positive")
    try:
        launch = parse_raw(args.raw.read_text())
        source = parse_source(args.source.read_text()) if args.source else ()
        print(summarize(launch, source, args.calibration_bytes))
    except (InputError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
