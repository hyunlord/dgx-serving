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


@dataclass(frozen=True, slots=True)
class DecodeSelection:
    """CPU-aligned bounds plus unchanged raw kernels and explicit anomalies."""

    stats: dict[str, base.Json]
    rows: list[base.Launch]
    bounds: tuple[int, int]
    anomalies: list[dict[str, int]]


def completion(db: sqlite3.Connection, last: base.Launch) -> int:
    """Use the first host-ordered sync; never skip one due to GPU timestamps."""
    row = db.execute("""SELECT r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE 'cudaDeviceSynchronize%'
        AND r.globalTid=? AND r.start>=? ORDER BY r.start LIMIT 1""",
        (last.tid, last.api_end)).fetchone()
    if row is None or row[0] < last.end:
        raise base.TraceError("Missing completion or terminal GPU ends after first host completion")
    return row[0]


def aligned_stats(rows: list[base.Launch], bounds: tuple[int, int], epoch: int) -> dict[str, base.Json]:
    """Keep duration/category sums raw and disclose union clipped at CPU bounds."""
    start, end = bounds
    groups = base.terminal_groups(rows, "softcap_kernel")
    result = base.aggregate(rows, end-start, {base.last_linear(g) for g in groups}, epoch, start, end)
    raw_union = base.union_ns([(r.start, r.end) for r in rows])
    clipped_union = base.union_ns([(max(start, r.start), min(end, r.end)) for r in rows])
    result.update(raw_gpu_union_ns=raw_union, gpu_union_ns=clipped_union,
                  gpu_union_outside_wall_ns=raw_union-clipped_union,
                  wall_minus_gpu_union_ns=end-start-clipped_union)
    return result


def decode_aligned(db: sqlite3.Connection, rows: list[base.Launch], epoch: int) -> DecodeSelection:
    """Accept only observed leading cross-domain inversions with valid host order."""
    groups = base.terminal_groups(rows, "softcap_kernel")
    if len(groups) < 33:
        raise base.TraceError("Need at least 33 tokens including predecessor boundary")
    previous_group = groups[-33]
    previous = completion(db, previous_group[-1])
    start = previous
    anomalies: list[dict[str, int]] = []
    tokens: list[base.Json] = []
    for index, group in enumerate(groups[-32:]):
        end = completion(db, group[-1])
        if end <= previous:
            raise base.TraceError("Non-increasing token completion boundaries")
        for row in group:
            if row.api_start < previous or row.api_end > end:
                raise base.TraceError("Token launch API crosses CPU completion boundary")
            if row.end <= previous or row.end > end:
                raise base.TraceError("Kernel wholly outside or ending after CPU completion interval")
            if row.start < previous:
                if row.start < previous_group[-1].end or row.start >= row.api_start:
                    raise base.TraceError("Leading kernel crosses previous GPU work or lacks API inversion")
                anomalies.append({"selected_index": index, "correlation": row.correlation,
                    "gpu_start_ns": row.start, "previous_sync_end_ns": previous,
                    "api_start_ns": row.api_start, "outside_token_start_ns": previous-row.start,
                    "gpu_before_own_api_ns": row.api_start-row.start})
        stat = aligned_stats(group, (previous, end), epoch)
        stat.pop("shapes")
        stat["selected_index"] = index
        tokens.append(stat)
        previous, previous_group = end, group
    selected = [r for group in groups[-32:] for r in group]
    result = aligned_stats(selected, (start, previous), epoch)
    result.update(status="ok", token_count=32, tokens_seen=len(groups), tokens=tokens,
        wall_scope="completion-sync to completion-sync; includes previous-token cleanup and current-token host/copies")
    return DecodeSelection(result, selected, (start, previous), anomalies)


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
        if row.end <= bounds[0] or row.end > end:
            raise base.TraceError("Kernel outside selected completion interval")
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
        selection = decode_aligned(db, rows, epoch)
        selected, decoded = selection.rows, selection.stats
        start, end = selection.bounds
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
            "token_start_alignment_anomalies": selection.anomalies,
            "gpu_before_correlated_api": [{"correlation": r.correlation, "gpu_start_ns": r.start,
                "api_start_ns": r.api_start, "inversion_ns": r.api_start-r.start}
                for r in selected if r.start < r.api_start],
            "alignment_note": "CPU wall and raw kernel sums are unchanged; GPU union is clipped to CPU wall. "
                "Kernel-empty time is a CPU-aligned estimate, not exact causality. Observed inversion sizes "
                "are not a calibrated uncertainty bound. wall = raw kernel sum - raw overlap "
                "- gpu_union_outside_wall_ns + kernel-empty time.",
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
