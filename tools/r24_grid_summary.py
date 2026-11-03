#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: python3 tools/r24_grid_summary.py RUN_DIR OUTPUT_PREFIX
"""Summarize grid-axis NCU diagnostics; L2 proxy is not DRAM traffic."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from r24_ncu_summary import Launch, MISS, Metric, parse_raw

RESOURCE_NAMES: Final = (
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__waves_per_multiprocessor",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)


@dataclass(frozen=True, slots=True)
class GridReport:
    source_csv: str
    mode: str
    shape: str
    tokens: int
    kernel: str
    l2_read_miss_requested_bytes_proxy: float | None
    diagnostic_ncu_ns: float | None
    metrics: tuple[Metric, ...]


def duration_ns(launch: Launch) -> float | None:
    """Convert the diagnostic NCU duration without inferring missing units."""
    item = next((m for m in launch.metrics if m.name == "gpu__time_duration.sum"), None)
    if item is None or item.value is None:
        return None
    factors = {"ns": 1.0, "us": 1000.0, "ms": 1000000.0, "s": 1000000000.0}
    factor = factors.get(item.unit)
    return item.value * factor if factor is not None else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def fmt(value: float | None) -> str:
    return "UNAVAILABLE" if value is None else f"{value:,.9g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    reports = []
    for path in sorted(args.run_dir.glob("grid_*.csv")):
        match = re.fullmatch(r"grid_(main|swap)_(gate|down|o)_M(1024|2048)\.csv", path.name)
        if match is None:
            continue
        mode, shape, tokens = match.groups()
        launch = parse_raw(path.read_text())
        misses = launch.value(MISS)
        reports.append(GridReport(str(path), mode, shape, int(tokens), launch.kernel,
                                  misses * 32 if misses is not None else None,
                                  duration_ns(launch), launch.metrics))
    pairs = []
    lines = ["# R24 grid-axis NCU diagnostics", "",
             "All byte values are **근사(L2 miss 기반)**. NCU duration includes replay conditions and is diagnostic only; use the separate five-run continuous benchmark for absolute speed. Same resource limits do not imply identical achieved occupancy.", ""]
    for tokens in (1024, 2048):
        for shape in ("gate", "down", "o"):
            selected = {r.mode: r for r in reports if r.tokens == tokens and r.shape == shape}
            if not {"main", "swap"}.issubset(selected):
                continue
            main_report, swap_report = selected["main"], selected["swap"]
            resources = []
            for name in RESOURCE_NAMES:
                a = next((m.value for m in main_report.metrics if m.name == name), None)
                b = next((m.value for m in swap_report.metrics if m.name == name), None)
                resources.append(dict(metric=name, main=a, swap=b,
                                      equal=a == b if a is not None and b is not None else None))
            byte_ratio = ratio(swap_report.l2_read_miss_requested_bytes_proxy,
                               main_report.l2_read_miss_requested_bytes_proxy)
            time_ratio = ratio(swap_report.diagnostic_ncu_ns, main_report.diagnostic_ncu_ns)
            pairs.append(dict(shape=shape, tokens=tokens, proxy_swap_over_main=byte_ratio,
                              diagnostic_ncu_time_swap_over_main=time_ratio, resources=resources))
            lines += [f"## {shape}, M={tokens}", "", "| Metric | main | swapped |", "|---|---:|---:|"]
            for label, a, b in [("근사(L2 miss 기반) requested bytes", main_report.l2_read_miss_requested_bytes_proxy, swap_report.l2_read_miss_requested_bytes_proxy), ("NCU diagnostic duration ns", main_report.diagnostic_ncu_ns, swap_report.diagnostic_ncu_ns)]:
                lines.append(f"| {label} | {fmt(a)} | {fmt(b)} |")
            lines += ["", f"L2 proxy swap/main={fmt(byte_ratio)}; NCU diagnostic duration swap/main={fmt(time_ratio)}.", "", "| Resource metric | main | swapped | Equal |", "|---|---:|---:|---|"]
            for resource in resources:
                lines.append(f"| {resource['metric']} | {fmt(resource['main'])} | {fmt(resource['swap'])} | {resource['equal']} |")
            lines.append("")
    payload = dict(report_count=len(reports), complete_pair_count=len(pairs),
                   complete=len(reports) == 12 and len(pairs) == 6,
                   dram_bytes_status="UNSUPPORTED", proxy_label="근사(L2 miss 기반)",
                   reports=[asdict(r) for r in reports], pairs=pairs)
    json_path = Path(str(args.output_prefix) + ".json")
    md_path = Path(str(args.output_prefix) + ".md")
    if json_path.exists() or md_path.exists():
        parser.error("Output exists; choose a fresh prefix to preserve prior results")
    with json_path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"REPORTS={len(reports)} PAIRS={len(pairs)} JSON={json_path}")


if __name__ == "__main__":
    main()
