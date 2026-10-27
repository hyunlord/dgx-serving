#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""R08 stable-last32 W4A8 decode accounting plus CUDA runtime API intervals.

How to run: python3 tools/r10_nsys_decode.py INPUT.sqlite OUTPUT.json
Keep r10_trace_base.py beside this file. It preserves the R08 parser from
commit 3c7de480963cf0af7b4eb5530c4e7efee11bc4b1:tools/r08_trace.py unchanged.
Requires the original B1 benchmark's eight event records and per-token syncs.
No profiler or GPU execution occurs. Input is opened read-only; output is exclusive.

API wall intervals can overlap GPU work and one another. Category sums and gap
intersections are descriptive coverage, not additive causes or removable latency.
GPU-empty means no selected kernel; it can include GPU memcpy or other activity.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import r10_trace_base as base


@dataclass(frozen=True, slots=True)
class ApiSpan:
    """A runtime call clipped to the selected completion interval."""

    start: int
    end: int
    name: str
    tid: int


@dataclass(frozen=True, slots=True)
class ApiMetric:
    """Call sum, unique coverage, and intersection with kernel-empty intervals."""

    calls: int
    clipped_duration_sum_ns: int
    interval_union_ns: int
    kernel_empty_intersection_ns: int


def category(name: str) -> str:
    """Keep all unmatched runtime APIs visible under other."""
    for prefix, label in (("cudaMalloc", "malloc"), ("cudaFree", "free"),
                          ("cudaMemcpy", "memcpy"), ("cudaMemset", "memset")):
        if name.startswith(prefix):
            return label
    if "Synchronize" in name:
        return "sync"
    if "GraphLaunch" in name:
        return "graph_launch"
    if "Launch" in name:
        return "kernel_launch"
    return "other"


def gaps(rows: list[base.Launch], bounds: tuple[int, int]) -> list[tuple[int, int]]:
    """Complement of selected kernel union inside the completion interval."""
    cursor, end = bounds
    result: list[tuple[int, int]] = []
    for row in sorted(rows, key=lambda r: r.start):
        if row.start < bounds[0] or row.end > end:
            raise base.TraceError("Kernel crosses selected completion interval")
        if row.start > cursor:
            result.append((cursor, row.start))
        cursor = max(cursor, row.end)
    if cursor < end:
        result.append((cursor, end))
    return result


def metric(spans: list[ApiSpan], empty: list[tuple[int, int]]) -> ApiMetric:
    """Union intersections remove repeated coverage from overlapping calls."""
    merged: list[tuple[int, int]] = []
    for span in sorted(spans, key=lambda s: s.start):
        if merged and span.start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span.end))
        else:
            merged.append((span.start, span.end))
    overlap = 0
    i = j = 0
    while i < len(merged) and j < len(empty):
        a, b = merged[i]
        c, d = empty[j]
        overlap += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return ApiMetric(len(spans), sum(s.end - s.start for s in spans),
                     sum(b-a for a, b in merged), overlap)


def phase(db: sqlite3.Connection, rows: list[base.Launch]) -> list[base.Launch]:
    """Select R08 event pair three, the timed W4A8 single-decode phase."""
    events = db.execute("""SELECT r.* FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE 'cudaEventRecord%'
        ORDER BY r.start""").fetchall()
    if len(events) != 8:
        raise base.TraceError("Expected exactly eight timing events (B1, one prefill)")
    a, b = events[4:6]
    if a["globalTid"] != b["globalTid"]:
        raise base.TraceError("Phase events belong to different host threads")
    return [r for r in rows if a["end"] < r.api_start < b["start"] and r.tid == a["globalTid"]]


def run(source: Path, output: Path) -> None:
    """Read pinned phase semantics, then attach non-causal API coverage."""
    if output.exists():
        raise base.TraceError("Refusing to overwrite existing evidence")
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = phase(db, base.read_launches(db))
        epoch = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        groups = base.terminal_groups(rows, "softcap_kernel")
        if len(groups) < 33:
            raise base.TraceError("Need at least 33 tokens including predecessor boundary")
        selected = [r for group in groups[-32:] for r in group]
        start = base.sync_end(db, groups[-33][-1], True)
        end = base.sync_end(db, groups[-1][-1], True)
        decoded = base.decode(db, rows, True, epoch)
        empty = gaps(selected, (start, end))
        thread = selected[0].tid
        runtime = db.execute("""SELECT r.start,r.end,r.globalTid,s.value
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
            WHERE r.start < ? AND r.end > ? ORDER BY r.start""", (end, start)).fetchall()
        spans = [ApiSpan(max(start, r["start"]), min(end, r["end"]), r["value"], r["globalTid"])
                 for r in runtime if r["globalTid"] == thread]
        by_category: dict[str, list[ApiSpan]] = defaultdict(list)
        by_name: dict[str, list[ApiSpan]] = defaultdict(list)
        for span in spans:
            by_category[category(span.name)].append(span)
            by_name[span.name].append(span)
        combined = metric(spans, empty)
        result = {
            "source": str(source.resolve()), "mode": "engine_w4a8_last32", "epoch_ns": epoch,
            "decode": decoded,
            "api_scope": "same host thread as W4A8 launch; intervals clipped to completion boundaries",
            "api_note": "Overlapping API categories are not additive causes. Synchronize can overlap GPU work. "
                        "Kernel-empty intervals can contain GPU copies; they are not proven device idle.",
            "excluded_other_thread_calls": sum(r["globalTid"] != thread for r in runtime),
            "runtime_api": asdict(combined),
            "runtime_api_by_category": {k: asdict(metric(v, empty)) for k, v in by_category.items()},
            "runtime_api_by_name": {k: asdict(metric(v, empty)) for k, v in by_name.items()},
            "kernel_empty_without_selected_thread_api_ns": sum(b-a for a, b in empty)
                - combined.kernel_empty_intersection_ns,
            "runtime_api_intervals": [asdict(s) for s in spans],
        }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "tokens": 32, "api_calls": len(spans)}))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    run(Path(sys.argv[1]), Path(sys.argv[2]))
