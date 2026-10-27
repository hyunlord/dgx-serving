#!/usr/bin/env python3
"""Read Nsight SQLite into R08 phase summaries and raw launch rows; never execute GPU work.

Usage: r08_trace.py INPUT.sqlite engine|llama_decode|llama_prefill OUTPUT.json
Engine requires B=1 and one timed prefill repetition (eight event records).
Llama requires one benchmark instance and -r 1, retaining its ordinary warmup.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypeAlias

Json: TypeAlias = None | bool | int | float | str | list["Json"] | dict[str, "Json"]
LINEAR: Final = ("gemv_", "gemm_mmq", "gemm_tiled", "mul_mat_vec_q", "mul_mat_q", "mul_mat_f")


class TraceError(RuntimeError):
    """The trace cannot establish the requested measurement boundaries."""


@dataclass(frozen=True, slots=True)
class Launch:
    start: int
    end: int
    correlation: int
    name: str
    short: str
    api_start: int
    api_end: int
    tid: int
    api: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    registers: int
    static_shared: int
    dynamic_shared: int
    graph_node: int | None
    ordinal: int


def category(name: str) -> str:
    """Conservative naming classification; ambiguous BLAS kernels remain other."""
    if name.startswith(LINEAR):
        return "linear"
    if "attn" in name or "soft_max" in name:
        return "attention"
    if "quantize" in name:
        return "activation_quantize"
    if any(x in name for x in ("norm", "rope", "add_", "scale_", "geglu", "softcap", "k_bin_", "silu", "gelu")):
        return "norm_elementwise_rope"
    return "other"


def union_ns(intervals: list[tuple[int, int]]) -> int:
    """Union avoids interpreting overlapping GPU intervals as wall time."""
    total = 0
    end = -1
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def read_launches(db: sqlite3.Connection) -> list[Launch]:
    rows = db.execute("""SELECT k.*,n.value AS full_name,s.value AS short_name,
        r.start AS api_start,r.end AS api_end,r.globalTid,a.value AS api
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds n ON n.id=k.demangledName JOIN StringIds s ON s.id=k.shortName
        LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
        LEFT JOIN StringIds a ON a.id=r.nameId ORDER BY k.start,k.end""").fetchall()
    counts: Counter[str] = Counter()
    result = []
    for r in rows:
        if r["api_start"] is None:
            raise TraceError("Kernel lacks runtime correlation; cannot establish launch boundaries")
        short = r["short_name"]
        result.append(Launch(r["start"], r["end"], r["correlationId"], r["full_name"], short,
            r["api_start"], r["api_end"], r["globalTid"], r["api"],
            (r["gridX"], r["gridY"], r["gridZ"]), (r["blockX"], r["blockY"], r["blockZ"]),
            r["registersPerThread"], r["staticSharedMemory"], r["dynamicSharedMemory"],
            r["graphNodeId"], counts[short]))
        counts[short] += 1
    if len(result) != db.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]:
        raise TraceError("Runtime join duplicated kernel rows (multiple processes/correlations)")
    return result


def sync_end(db: sqlite3.Connection, last: Launch, engine: bool) -> int:
    name = "cudaDeviceSynchronize%" if engine else "cudaStreamSynchronize%"
    r = db.execute("""SELECT r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE ? AND r.globalTid=?
        AND r.start>=? AND r.end>=? ORDER BY r.end LIMIT 1""",
        (name, last.tid, last.api_end, last.end)).fetchone()
    if r is None:
        raise TraceError("No terminal synchronization after softcap")
    return r[0]


def aggregate(rows: list[Launch], wall: int, lm_heads: set[int], epoch: int,
              start: int, end: int) -> dict[str, Json]:
    sums: dict[str, int] = defaultdict(int)
    groups: dict[str, list[Launch]] = defaultdict(list)
    for r in rows:
        cat = "lm_head" if r.start in lm_heads else category(r.short)
        sums[cat] += r.end - r.start
        groups[r.name + str(r.grid) + str(r.block)].append(r)
    kernel_sum = sum(sums.values())
    kernel_union = union_ns([(r.start, r.end) for r in rows])
    shapes: list[Json] = []
    for group in groups.values():
        r = group[0]
        shapes.append({"name": r.name, "short": r.short, "grid": list(r.grid), "block": list(r.block),
            "registers": r.registers, "static_shared": r.static_shared, "dynamic_shared": r.dynamic_shared,
            "calls": len(group), "sum_ns": sum(k.end-k.start for k in group),
            "first_function_ordinal": r.ordinal, "first_correlation": r.correlation})
    return {"kernel_count": len(rows), "kernel_sum_ns": kernel_sum,
        "gpu_union_ns": kernel_union, "wall_ns": wall,
        "kernel_overlap_ns": kernel_sum-kernel_union, "wall_minus_gpu_union_ns": wall-kernel_union,
        "non_kernel_remainder_ns": wall-kernel_sum, "categories_ns": dict(sums),
        "start_ns": start, "end_ns": end, "start_epoch": (epoch+start)/1e9,
        "end_epoch": (epoch+end)/1e9, "graph_launches": len({r.correlation for r in rows if "cudaGraphLaunch" in r.api}),
        "shapes": sorted(shapes, key=lambda x: x["sum_ns"], reverse=True)}


def terminal_groups(rows: list[Launch], terminal: str) -> list[list[Launch]]:
    groups = []
    current = []
    for r in rows:
        current.append(r)
        if r.short == terminal:
            groups.append(current)
            current = []
    if current:
        raise TraceError("Trailing kernels after final softcap; phase boundary ambiguous")
    return groups


def last_linear(rows: list[Launch]) -> int:
    candidates = [r for r in rows if category(r.short) == "linear"]
    if not candidates:
        raise TraceError("No identifiable lm_head linear kernel before terminal softcap")
    return candidates[-1].start


def decode(db: sqlite3.Connection, rows: list[Launch], engine: bool, epoch: int) -> dict[str, Json]:
    groups = terminal_groups(rows, "softcap_kernel" if engine else "softcap_f32")
    if len(groups) < 33:
        return {"status": "unavailable", "reason": "Need at least33 tokens including predecessor boundary", "tokens": len(groups)}
    chosen = groups[-32:]
    previous = sync_end(db, groups[-33][-1], engine)
    start = previous
    token_stats: list[Json] = []
    for index, group in enumerate(chosen):
        end = sync_end(db, group[-1], engine)
        if previous > group[0].start or end <= previous:
            raise TraceError("Token synchronization boundaries overlap next token")
        stat = aggregate(group, end-previous, {last_linear(group)}, epoch, previous, end)
        stat.pop("shapes")
        stat["selected_index"] = index
        token_stats.append(stat)
        previous = end
    flat = [r for group in chosen for r in group]
    result = aggregate(flat, previous-start, {last_linear(g) for g in chosen}, epoch, start, previous)
    result.update(status="ok", token_count=32, tokens_seen=len(groups), tokens=token_stats,
        wall_scope="completion-sync to completion-sync; includes previous-token cleanup and current-token host/copies")
    return result


def run(path: Path, mode: str, output: Path) -> None:
    if output.exists() or output.with_suffix(".kernels.csv").exists():
        raise TraceError("Refusing to overwrite existing evidence")
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = read_launches(db)
        epoch = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        result: dict[str, Json] = {"source": str(path), "mode": mode, "epoch_ns": epoch,
            "classification_note": "Unidentified BLAS/copy kernels remain other; inspect raw rows before attribution",
            "ordinal_note": "Function ordinals use GPU start order; NCU launch order may differ for graphs/concurrent streams. Verify profiled shape."}
        if mode == "engine":
            events = db.execute("""SELECT r.* FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
                WHERE s.value LIKE 'cudaEventRecord%' ORDER BY r.start""").fetchall()
            if len(events) != 8:
                raise TraceError("Engine profile must have B=1 and exactly one timed prefill repetition (8 events)")
            def phase(i: int) -> list[Launch]:
                a,b = events[2*i], events[2*i+1]
                if a["globalTid"] != b["globalTid"]:
                    raise TraceError("Timing events belong to different host threads")
                return [r for r in rows if a["end"] < r.api_start < b["start"] and r.tid==a["globalTid"]]
            result["decode"] = decode(db, phase(2), True, epoch)
            pre = phase(3)
            if len([r for r in pre if r.short=="softcap_kernel"]) != 1:
                raise TraceError("Expected one timed engine prefill")
            result["prefill"] = aggregate(pre, events[7]["start"]-events[6]["end"], {last_linear(pre)},
                epoch, events[6]["end"], events[7]["start"])
        elif mode == "llama_decode":
            result["decode"] = decode(db, rows, False, epoch)
        elif mode == "llama_prefill":
            groups = terminal_groups(rows, "softcap_f32")
            if len(groups) != 2:
                raise TraceError("Expected warmup plus one measured prefill, each with one terminal softcap")
            start = sync_end(db, groups[0][-1], False)
            end = sync_end(db, groups[1][-1], False)
            result["prefill"] = aggregate(groups[1], end-start, {last_linear(groups[1])}, epoch, start, end)
            result["wall_scope"] = "warmup-completion to measured-completion, includes memory-clear gap; all microbatches retained"
        else:
            raise TraceError("Mode must be engine, llama_decode or llama_prefill")
    encoded = json.dumps(result, indent=2)+"\n"
    if len(encoded.encode()) > 1_000_000:
        raise TraceError("Summary exceeds1MB; raw rows belong only in DGX artifacts")
    with output.open("x") as stream:
        stream.write(encoded)
    with output.with_suffix(".kernels.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    print(json.dumps({"summary": str(output), "kernel_rows": len(rows), "bytes": len(encoded.encode())}))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
