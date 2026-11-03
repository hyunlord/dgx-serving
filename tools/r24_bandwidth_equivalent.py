#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: python3 tools/r24_bandwidth_equivalent.py RUN_DIR OUTPUT_PREFIX
"""Compute L2-proxy memory-service equivalents, never additive attribution."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from r24_ncu_summary import MISS, parse_raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    base = args.run_dir
    guard = json.loads((base / "result_pure_read_bw.json").read_text())
    with (base / "clockmon_pure_read_bw.csv").open() as stream:
        clock_rows = list(csv.DictReader(stream, skipinitialspace=True))
    clock_samples = []
    for row in clock_rows:
        stamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
        epoch_ms = stamp.replace(tzinfo=ZoneInfo("Asia/Seoul")).timestamp() * 1000
        clock_samples.append((epoch_ms, int(row["clocks.current.sm [MHz]"].split()[0]),
                              int(row["temperature.gpu"]), row["clocks_event_reasons.active"]))
    variants = []
    pattern = re.compile(r"R20_BW variant=(\S+) rb=(\d+) GBps=([\d.]+) launches=(\d+) t0_ms=([\d.]+) t1_ms=([\d.]+)")
    for match in pattern.finditer((base / "command_pure_read_bw.log").read_text()):
        name, rb, bandwidth, launches, t0, t1 = match.groups()
        start, end = float(t0), float(t1)
        selected = [row for row in clock_samples if start <= row[0] <= end]
        mhz = [row[1] for row in selected]
        temperatures = [row[2] for row in selected]
        variants.append(dict(variant=name, row_bytes=int(rb), reported_gbps=float(bandwidth),
                             launches=int(launches), t0_ms=start, t1_ms=end,
                             measured_interval_ms=end-start, sample_count=len(selected),
                             sm_median_mhz=statistics.median(mhz) if mhz else None,
                             sm_min_mhz=min(mhz) if mhz else None,
                             sm_max_mhz=max(mhz) if mhz else None,
                             temperature_min_c=min(temperatures) if temperatures else None,
                             temperature_max_c=max(temperatures) if temperatures else None,
                             active_mask_counts=dict(collections.Counter(row[3] for row in selected))))
    registered = [row for row in variants if row["variant"] == "flat16_b384"]
    if len(registered) != 1:
        parser.error("Expected exactly one preregistered flat16_b384 result")
    chosen = registered[0]
    if not chosen["sample_count"]:
        parser.error("No measured-interval clock samples for selected bandwidth")
    bandwidth = chosen["reported_gbps"]
    references = [row for row in variants if row["variant"] in ("row16_w4", "row16_w8")]
    trace = json.loads((base / "trace_summary.json").read_text())
    results = []
    for tokens in (1024, 2048):
        for shape in ("gate", "down", "o"):
            observations = {}
            for engine, suffix in (("main", ""), ("llama", "_r2")):
                stem = f"ncu_{engine}_{shape}_M{tokens}{suffix}"
                launch = parse_raw((base / f"{stem}.csv").read_text())
                sectors = launch.value(MISS)
                if sectors is None:
                    parser.error(f"Missing L2 miss counter: {stem}")
                projections = [p for t in trace["traces"] if t["engine"] == engine and t["m"] == tokens
                               for p in t["projections"] if p["layer"] == 0 and p["role"] == shape]
                if len(projections) != 1:
                    parser.error(f"Expected one layer0 projection: {stem}")
                projection = projections[0]
                observations[engine] = dict(l2_proxy_bytes=sectors*32,
                                            nsys_mmq_ns=projection["mmq_ns"],
                                            nsys_fixup_ns=projection["fixup_ns"])
            ours, llama = observations["main"], observations["llama"]
            delta = ours["l2_proxy_bytes"] - llama["l2_proxy_bytes"]
            gap = (ours["nsys_mmq_ns"]-llama["nsys_mmq_ns"])/1e6
            with_fixup = gap + (ours["nsys_fixup_ns"]-llama["nsys_fixup_ns"])/1e6
            results.append(dict(shape=shape, tokens=tokens, observations=observations,
                                delta_l2_proxy_bytes=delta,
                                memory_service_equivalent_ms=delta/(bandwidth*1e6),
                                nsys_layer0_mmq_gap_ms=gap,
                                nsys_layer0_mmq_plus_fixup_gap_ms=with_fixup))
    payload = dict(selected_preregistered_variant=chosen, row16_reference_only=references,
                   clock_timestamp_timezone="Asia/Seoul", all_variant_intervals=variants,
                   bytes_label="근사(L2 miss 기반)", guard=guard, comparisons=results,
                   limitations=["Bytes are L2 read-miss proxy, not DRAM measurements",
                                "Bandwidth comes from a separate pure-read workload",
                                "Equivalent is not additive attribution or causal time savings",
                                "No percentage of gap or unexplained residual is inferred",
                                "Row16 reported GBps uses the original probe byte convention",
                                "Guard condition string is retained verbatim; actual baseline/final process lists are authoritative"])
    lines = ["# Pure-read bandwidth and memory-service equivalent", "",
             f"Preregistered flat16_b384: **{bandwidth:.2f} GB/s**, 1 GiB read once per launch, 16-byte loads, 384 CTAs × 256 threads. Ten-second warm-up precedes reported intervals.", "",
             f"Selected measured interval {chosen['measured_interval_ms']:.1f} ms, {chosen['sample_count']} clock samples, SM median {chosen['sm_median_mhz']} MHz, temperature {chosen['temperature_min_c']}–{chosen['temperature_max_c']} °C, active masks {chosen['active_mask_counts']}.", "",
             "L2 bytes below are **근사(L2 miss 기반)**. Δproxy bytes / selected measured bandwidth is a memory-service equivalent alongside the layer0 nsys gap. It is not additive attribution, a causal saving, or a subtraction used to manufacture an unexplained remainder.", "",
             "| M | Shape | Δproxy bytes (main−llama) | Memory-service equivalent ms | Layer0 nsys MMQ gap ms | Including fixup gap ms |", "|---:|---|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['tokens']} | {row['shape']} | {row['delta_l2_proxy_bytes']:.0f} | {row['memory_service_equivalent_ms']:.9f} | {row['nsys_layer0_mmq_gap_ms']:.9f} | {row['nsys_layer0_mmq_plus_fixup_gap_ms']:.9f} |")
    lines += ["", "## Row16 references (not used in the equivalent)", "", "| Variant | Row bytes | Reported GB/s | SM median MHz | Temperature °C | Masks |", "|---|---:|---:|---:|---|---|"]
    for row in references:
        lines.append(f"| {row['variant']} | {row['row_bytes']} | {row['reported_gbps']:.2f} | {row['sm_median_mhz']} | {row['temperature_min_c']}–{row['temperature_max_c']} | {row['active_mask_counts']} |")
    lines += ["", f"Guard status={guard['status']}, exit={guard['exit_code']}; unexpected processes={guard['unexpected_processes']}; monitors_failed={guard['monitors_failed']}. Actual baseline/final PID lists are preserved in JSON; the guard's condition string is historical metadata, not a replacement for those lists."]
    for suffix, text in ((".json", json.dumps(payload, indent=2, allow_nan=False)), (".md", "\n".join(lines))):
        path = Path(str(args.output_prefix)+suffix)
        with path.open("x") as stream:
            stream.write(text+"\n")
    print(f"BANDWIDTH_EQUIVALENT_COMPLETE pairs={len(results)} GBps={bandwidth}")


if __name__ == "__main__":
    main()
