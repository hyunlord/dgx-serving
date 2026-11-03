#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ─── How to run ───
# python3 tools/r24_trace_summary.py ARTIFACT_DIR MANIFEST_JSON
# Stdlib only: no dependencies permitted without project approval.
"""R24 trace attribution, adapted from R22 trace/NVTX parsers (2026-11-02)."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

ROLES: Final = {"attn_q": "q", "attn_k": "k", "attn_v": "v", "attn_output": "o", "ffn_gate": "gate", "ffn_up": "up", "ffn_down": "down"}


@dataclass(frozen=True, slots=True)
class EvidenceError(Exception):
    detail: str


@dataclass(frozen=True, slots=True)
class Kernel:
    rid: int
    start: int
    end: int
    cid: int
    name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    registers: int
    shared: int


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    tid: int
    label: str


@dataclass(frozen=True, slots=True)
class Projection:
    layer: int
    role: str
    n: int
    k: int
    m: int
    mmq_ns: int
    fixup_ns: int
    kernel: Kernel


@dataclass(frozen=True, slots=True)
class Category:
    name: str
    count: int
    ms: float


@dataclass(frozen=True, slots=True)
class Trace:
    path: str
    sha256: str
    engine: str
    m: int
    categories: list[Category]
    kernel_count: int
    kernel_sum_ms: float
    projections: list[Projection]
    selection_skips: tuple[int, int, int] | None


def kernels(conn: sqlite3.Connection) -> list[Kernel]:
    """Read launches with exact grid/resources and correlation identity."""
    sql = """SELECT k.rowid,k.start,k.end,k.correlationId,s.value,k.gridX,k.gridY,k.gridZ,
    k.blockX,k.blockY,k.blockZ,k.registersPerThread,k.staticSharedMemory+k.dynamicSharedMemory
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"""
    return [Kernel(r[0], r[1], r[2], r[3], r[4], tuple(r[5:8]), tuple(r[8:11]), r[11], r[12]) for r in conn.execute(sql)]


def sequence(manifest: Path) -> list[tuple[int, str, int, int]]:
    """Reuse R22 source order, including shared global K/V and all 48 layers."""
    cfg = json.loads(manifest.read_text())
    result: list[tuple[int, str, int, int]] = []
    h, f = int(cfg["hidden_size"]), int(cfg["intermediate_size"])
    for layer, kind in enumerate(cfg["layer_types"]):
        full = kind == "full_attention"
        hd = int(cfg["global_head_dim"] if full else cfg["head_dim"])
        kv = int(cfg["num_global_kv_heads"] if full else cfg["num_kv_heads"])
        q = int(cfg["num_q_heads"]) * hd
        parts = [("q", q, h), ("k", kv * hd, h)]
        if not (full and cfg["attention_k_eq_v"]):
            parts.append(("v", kv * hd, h))
        parts.extend([("o", h, q), ("gate", f, h), ("up", f, h), ("down", h, f)])
        result.extend((layer, role, n, k) for role, n, k in parts)
    if len(result) != 328:
        raise EvidenceError("Manifest no longer has 328 projections")
    return result


def main_trace(rows: list[Kernel], expected: list[tuple[int, str, int, int]], m: int) -> tuple[list[Kernel], list[Projection], tuple[int, int, int]]:
    """Require two final complete prefill blocks and prove filtered skip indices."""
    starts = [i for i, row in enumerate(rows) if row.name.startswith("embedding_kernel(")]
    if len(starts) < 2:
        raise EvidenceError(f"Expected two final embedding boundaries, got {len(starts)}")
    if any("gemm_mmq_kernel" in row.name for row in rows[:starts[-2]]):
        raise EvidenceError("Unexpected MMQ before final warm prefill")
    warm, selected = rows[starts[-2]:starts[-1]], rows[starts[-1]:]
    for block in (warm, selected):
        if not block[-1].name.startswith("softcap_kernel("):
            raise EvidenceError("Incomplete prefill ending")
    mmq = [row for row in selected if "gemm_mmq_kernel" in row.name]
    warm_mmq = [row for row in warm if "gemm_mmq_kernel" in row.name]
    prefix = sum("gemm_mmq_kernel" in row.name for row in rows[:starts[-1]])
    if len(mmq) != 328 or len(warm_mmq) != 328 or prefix != 328:
        raise EvidenceError("MMQ count/prefix differs from 328 warm + 328 measured")
    detail: list[Projection] = []
    for (layer, role, n, k), launch, warm_launch in zip(expected, mmq, warm_mmq, strict=True):
        grid = ((n + 127)//128, (m + 127)//128, 1)
        if launch.grid != grid or warm_launch.grid != grid:
            raise EvidenceError(f"Grid mismatch layer={layer} role={role}")
        detail.append(Projection(layer, role, n, k, m, launch.end-launch.start, 0, launch))
    if [detail[i].role for i in (3, 4, 6)] != ["o", "gate", "down"]:
        raise EvidenceError("Target projection order changed")
    return selected, detail, (prefix+3, prefix+4, prefix+6)


def llama_trace(conn: sqlite3.Connection, rows: list[Kernel], m: int) -> tuple[list[Kernel], list[Projection]]:
    """Match GPU -> CPU launch -> unique MMQ NVTX -> measured prefill."""
    spans = [Span(*r) for r in conn.execute("SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text LIKE 'R22_%'")]
    outers = [s for s in spans if s.label.startswith("R22_PREFILL phase=measured ")]
    if len(outers) != 1:
        raise EvidenceError("Expected one measured llama NVTX range")
    outer = outers[0]
    markers = [s for s in spans if s.label.startswith("R22_MMQ ") and s.tid == outer.tid and outer.start <= s.start <= s.end <= outer.end]
    selected = [row for row in rows if outer.start <= row.start <= row.end <= outer.end]
    runtime: defaultdict[int, list[Span]] = defaultdict(list)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            sql = f"SELECT r.start,r.end,r.globalTid,s.value,r.correlationId FROM {table} r JOIN StringIds s ON s.id=r.nameId"
            for a, b, tid, name, cid in conn.execute(sql):
                if "LaunchKernel" in name or "LaunchCooperativeKernel" in name:
                    runtime[cid].append(Span(a, b, tid, name))
    correlated = [row for row in rows if any(s.tid == outer.tid and outer.start <= s.start <= s.end <= outer.end for s in runtime[row.cid])]
    if {r.rid for r in selected} != {r.rid for r in correlated}:
        raise EvidenceError("Temporal and correlated llama sets differ")
    mapped: defaultdict[Span, list[Kernel]] = defaultdict(list)
    for row in selected:
        launches = [s for s in runtime[row.cid] if s.tid == outer.tid and outer.start <= s.start <= s.end <= outer.end]
        if len(launches) != 1:
            raise EvidenceError("Nonunique launch")
        if "void mul_mat_q<" in row.name or "void mul_mat_q_stream_k_fixup<" in row.name:
            launch = launches[0]
            matches = [s for s in markers if s.start <= launch.start <= launch.end <= s.end]
            if len(matches) != 1:
                raise EvidenceError("Nonunique MMQ attribution")
            mapped[matches[0]].append(row)
    detail: list[Projection] = []
    for marker in markers:
        fields = dict(piece.split("=", 1) for piece in marker.label.split()[1:])
        w = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", fields["weight"])
        a, b, d = (tuple(map(int, fields[key].split(","))) for key in ("src0", "src1", "dstne"))
        if w is None or w[2] not in ROLES or a[0] != b[0] or a[1] != d[0] or b[1] != m or d[1] != m or a[2:] != (1, 1) or b[2:] != (1, 1):
            raise EvidenceError(f"Invalid dense shape marker {marker.label}")
        mmq = [r for r in mapped[marker] if "void mul_mat_q<" in r.name]
        if len(mmq) != 1:
            raise EvidenceError("Expected one MMQ per projection")
        fixup = sum(r.end-r.start for r in mapped[marker] if "stream_k_fixup" in r.name)
        detail.append(Projection(int(w[1]), ROLES[w[2]], a[1], a[0], m, mmq[0].end-mmq[0].start, fixup, mmq[0]))
    if len(detail) != 325:
        raise EvidenceError(f"Expected 325 llama projections, got {len(detail)}")
    return selected, detail


def summarize(path: Path, m: int, llama: bool, manifest: Path) -> Trace:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        rows = kernels(conn)
        skips = None
        if llama:
            selected, detail = llama_trace(conn, rows, m)
            filtered = [r for r in rows if "void mul_mat_q<" in r.name]
            indices = {r.rid: i for i, r in enumerate(filtered)}
            targets = {p.role: p for p in detail if p.layer == 0}
            skips = tuple(indices[targets[role].kernel.rid] for role in ("o", "gate", "down"))
            if len(filtered) != 650 or skips != (328, 329, 331):
                raise EvidenceError(f"Unexpected llama filtered launch sequence: {len(filtered)}, {skips}")
        else:
            selected, detail, skips = main_trace(rows, sequence(manifest), m)
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for row in selected:
        groups[row.name.split("(")[0]].append(row.end-row.start)
    categories = [Category(k, len(v), sum(v)/1e6) for k, v in sorted(groups.items(), key=lambda x: -sum(x[1]))]
    return Trace(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), "llama" if llama else "main", m, categories, len(selected), sum(r.end-r.start for r in selected)/1e6, detail, skips)


def run(root: Path, manifest: Path) -> None:
    """Write fresh evidence only after all four traces pass coverage checks."""
    traces: list[Trace] = []
    comparisons = []
    lines = ["# R24 same-session NVTX comparison", "", "Diagnostic condition: engine default graph setting; llama Graph OFF. MMQ includes llama fixup.", "", "|M|N|K|count|ours ms|llama ms|ratio|", "|---:|---:|---:|---:|---:|---:|---:|"]
    for m in (1024, 2048):
        pair: list[Trace] = []
        for llama in (False, True):
            name = f"{'llama' if llama else 'main'}_P{m}.sqlite"
            paths = list(root.rglob(name))
            if len(paths) != 1:
                raise EvidenceError(f"Expected one {name}, got {paths}")
            pair.append(summarize(paths[0], m, llama, manifest))
        ours, other = pair
        exact = {(p.layer, p.role, p.n, p.k, p.m): p for p in other.projections}
        matched = [(p, exact[(p.layer, p.role, p.n, p.k, p.m)]) for p in ours.projections if (p.layer, p.role, p.n, p.k, p.m) in exact]
        if len(matched) != 325 or len(exact) != 325:
            raise EvidenceError("Exact intersection differs from 325 unique projections")
        unmatched = {(p.layer, p.role) for p in ours.projections if (p.layer, p.role, p.n, p.k, p.m) not in exact}
        if unmatched != {(47, "gate"), (47, "up"), (47, "down")}:
            raise EvidenceError(f"Unexpected nonintersection: {unmatched}")
        groups: defaultdict[tuple[int, int], list[tuple[Projection, Projection]]] = defaultdict(list)
        for a, b in matched:
            groups[(a.n, a.k)].append((a, b))
        for (n, k), values in sorted(groups.items()):
            a = sum(x.mmq_ns for x, _ in values)/1e6
            b = sum(y.mmq_ns+y.fixup_ns for _, y in values)/1e6
            comparisons.append({"M": m, "N": n, "K": k, "count": len(values), "ours_ms": a, "llama_ms": b, "ratio": a/b})
            lines.append(f"|{m}|{n}|{k}|{len(values)}|{a:.6f}|{b:.6f}|{a/b:.6f}|")
        traces.extend(pair)
    (root/"trace_summary.json").write_text(json.dumps({"traces": [asdict(t) for t in traces], "intersection_shapes": comparisons}, indent=2)+"\n")
    (root/"trace_summary.md").write_text("\n".join(lines)+"\n")
    (root/"selection_verified.txt").write_text("PASS: both main traces contain 328 warm MMQ then 328 measured MMQ; o/gate/down filtered launch skips=331/332/334. Both llama traces map exactly325 projections through unique measured NVTX and CPU correlation. See trace_summary.json for hashes, resources and grids.\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
