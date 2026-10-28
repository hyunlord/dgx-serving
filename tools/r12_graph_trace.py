#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# ─── How to run ───
# python3 tools/r12_graph_trace.py INPUT.sqlite OUTPUT.json [CLOCK.csv]
# Alternatively: uv run tools/r12_graph_trace.py INPUT.sqlite OUTPUT.json [CLOCK.csv]
"""Strict R12 node-trace accounting; read-only SQLite, exclusive JSON output.

Adapted accounting semantics from R10 r10_nsys_decode.py/r10_trace_base.py:
completion-aligned last32, raw sums, clipped union, disclosed clock inversions.
Graph replay correlations replace individual launch correlations. No GPU work.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypeAlias
from zoneinfo import ZoneInfo

SqlValue: TypeAlias = None | int | float | str | bytes
SqlRow: TypeAlias = tuple[SqlValue, ...]


class SqlRows(Protocol):
    """SQLite scalar rows, checked by field parsers before domain construction."""

    def __iter__(self) -> Iterator[SqlRow]: ...
    def fetchone(self) -> SqlRow | None: ...


class SqlConnection(Protocol):
    """Narrow standard SQLite execute surface; no adapters or row factories."""

    def execute(self, sql: str, /) -> SqlRows: ...


def integer(value: SqlValue) -> int:
    """Reject unexpected SQLite storage classes instead of coercing them."""
    if not isinstance(value, int):
        raise TraceError(f'Expected SQLite INTEGER, got {type(value).__name__}')
    return value


def string(value: SqlValue) -> str:
    """Require text-valued names at the SQL boundary."""
    if not isinstance(value, str):
        raise TraceError(f'Expected SQLite TEXT, got {type(value).__name__}')
    return value


Json: TypeAlias = None | bool | int | float | str | list['Json'] | dict[str, 'Json']


class TraceError(RuntimeError):
    """Observed trace cannot prove the requested measurement boundaries."""


@dataclass(frozen=True, slots=True)
class Api:
    row: int
    start: int
    end: int
    tid: int
    correlation: int
    name: str


@dataclass(frozen=True, slots=True)
class Kernel:
    row: int
    start: int
    end: int
    name: str
    node: int
    context: tuple[int, int, int, int]
    api: Api


@dataclass(frozen=True, slots=True)
class Token:
    kernels: tuple[Kernel, ...]
    completion: int


def union_ns(intervals: list[tuple[int, int]]) -> int:
    """Return union duration, ignoring empty clipped intervals."""
    total, end = 0, -1
    for a, b in sorted(intervals):
        if b > a:
            total += max(0, b - max(a, end))
            end = max(end, b)
    return total


def category(name: str) -> str:
    """Keep R10 naming categories; lm_head is assigned by terminal order."""
    for prefixes, label in ((('gemv_', 'gemm_mmq', 'gemm_tiled'), 'linear'),
                            (('quantize',), 'activation_quantize')):
        if name.startswith(prefixes):
            return label
    if 'attn' in name:
        return 'attention'
    if any(s in name for s in ('norm', 'rope', 'add_', 'scale_', 'geglu', 'softcap', 'gelu')):
        return 'norm_elementwise_rope'
    return 'other'


def read_trace(db: SqlConnection) -> tuple[list[Api], list[Kernel]]:
    """Parse runtime records and reject missing/ambiguous GPU correlations."""
    apis = [Api(integer(r[0]), integer(r[1]), integer(r[2]), integer(r[3]), integer(r[4]), string(r[5]))
            for r in db.execute('''SELECT r.rowid,r.start,r.end,r.globalTid,r.correlationId,s.value
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId ORDER BY r.start''')]
    by_correlation: dict[int, list[Api]] = defaultdict(list)
    for api in apis:
        by_correlation[api.correlation].append(api)
    kernels: list[Kernel] = []
    for r in db.execute('''SELECT k.rowid,k.start,k.end,s.value,k.graphNodeId,
            k.globalPid,k.deviceId,k.contextId,k.streamId,k.correlationId
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName ORDER BY k.start,k.end'''):
        candidates = [a for a in by_correlation[integer(r[9])]
                      if (a.tid >> 24) == (integer(r[5]) >> 24)]
        if len(candidates) != 1:
            raise TraceError(f'Kernel {r[0]} has {len(candidates)} runtime correlations')
        kernels.append(Kernel(integer(r[0]), integer(r[1]), integer(r[2]), string(r[3]), integer(r[4] or 0),
                              (integer(r[5]), integer(r[6]), integer(r[7]), integer(r[8])), candidates[0]))
    return apis, kernels


def select_tokens(trace: tuple[list[Api], list[Kernel]]) -> list[Token]:
    """Require third of four benchmark timing phases and complete graph steps."""
    apis, kernels = trace
    events = [a for a in apis if a.name.startswith('cudaEventRecord')]
    if len(events) != 8 or events[4].tid != events[5].tid:
        raise TraceError('Expected eight timing events and same-thread third pair')
    a, b = events[4:6]
    selected = [k for k in kernels if a.end < k.api.start < b.start and k.api.tid == a.tid]
    if len({k.context for k in selected}) != 1:
        raise TraceError('W4A8 phase must use exactly one process/device/context/stream')
    tokens: list[Token] = []
    group: list[Kernel] = []
    for k in selected:
        group.append(k)
        if k.name != 'softcap_kernel':
            continue
        launches = {v.api.row for v in group}
        if len(group) != 1334 or len(launches) != 1:
            raise TraceError(f'Step has {len(group)} kernels and {len(launches)} replay launches')
        if any(not v.node or not v.api.name.startswith('cudaGraphLaunch') for v in group):
            raise TraceError('Missing graph node ID or graph-launch correlation')
        if not any(category(v.name) == 'linear' for v in group[:-1]):
            raise TraceError('Missing lm_head linear before softcap')
        syncs = [v for v in apis if v.tid == k.api.tid and v.start >= k.api.end
                 and v.name.startswith('cudaStreamSynchronize')]
        if not syncs or syncs[0].end < k.end:
            raise TraceError('Terminal kernel ends after first host-ordered stream completion')
        tokens.append(Token(tuple(group), syncs[0].end))
        group = []
    if group or len(tokens) < 33:
        raise TraceError('Need at least33 complete steps and no trailing kernels')
    return tokens


def summarize(tokens: list[Token], epoch: int) -> dict[str, Json]:
    """Preserve raw kernels and CPU-boundary clipping, never hide anomalies."""
    previous = tokens[-33]
    start = previous.completion
    stats: list[Json] = []
    anomalies: list[Json] = []
    totals: dict[str, int] = defaultdict(int)
    selected: list[Kernel] = []
    for index, token in enumerate(tokens[-32:]):
        end = token.completion
        if end <= previous.completion:
            raise TraceError('Non-increasing token completion')
        lm_head = [k.row for k in token.kernels if category(k.name) == 'linear'][-1]
        categories: dict[str, int] = {}
        for k in token.kernels:
            if k.api.start < previous.completion or k.api.end > end or k.end <= previous.completion or k.end > end:
                raise TraceError('Replay API or GPU kernel outside CPU completion boundaries')
            if k.start < previous.completion:
                if k.start < previous.kernels[-1].end or k.start >= k.api.start:
                    raise TraceError('Leading GPU kernel overlap lacks observed API inversion')
                anomalies.append({'index': index, 'kernel': k.row, 'gpu_start_ns': k.start,
                                  'previous_completion_ns': previous.completion, 'api_start_ns': k.api.start})
            label = 'lm_head' if k.row == lm_head else category(k.name)
            totals[label] += k.end-k.start
            categories[label] = categories.get(label, 0) + k.end-k.start
        raw = union_ns([(k.start, k.end) for k in token.kernels])
        clipped = union_ns([(max(previous.completion, k.start), min(end, k.end)) for k in token.kernels])
        stats.append({'index': index, 'wall_ns': end-previous.completion, 'kernel_count': len(token.kernels),
                      'kernel_sum_ns': sum(k.end-k.start for k in token.kernels), 'gpu_union_ns': clipped,
                      'raw_gpu_union_ns': raw, 'gpu_union_outside_wall_ns': raw-clipped,
                      'kernel_empty_ns': end-previous.completion-clipped, 'categories_ns': {k: v for k, v in categories.items()},
                      'graph_launches': 1, 'softcap_count': 1, 'lm_head_count': 1})
        selected.extend(token.kernels)
        previous = token
    end = previous.completion
    raw = union_ns([(k.start, k.end) for k in selected])
    clipped = union_ns([(max(start, k.start), min(end, k.end)) for k in selected])
    return {'status': 'ok', 'token_count': 32, 'tokens_seen': len(tokens), 'tokens': stats,
            'start_ns': start, 'end_ns': end, 'start_epoch': (epoch+start)/1e9, 'end_epoch': (epoch+end)/1e9,
            'kernel_count': len(selected), 'kernel_sum_ns': sum(totals.values()), 'categories_ns': dict(totals),
            'wall_ns': end-start, 'raw_gpu_union_ns': raw, 'gpu_union_ns': clipped,
            'kernel_overlap_ns': sum(totals.values())-raw, 'gpu_union_outside_wall_ns': raw-clipped,
            'kernel_empty_ns': end-start-clipped, 'graph_launches': 32, 'alignment_anomalies': anomalies,
            'gpu_before_correlated_api': [{'kernel': k.row, 'inversion_ns': k.api.start-k.start}
                                          for k in selected if k.start < k.api.start]}


def api_summary(apis: list[Api], bounds: tuple[int, int]) -> dict[str, Json]:
    """Capture spans include all host launch submission work, not just two APIs."""
    start, end = bounds
    inside = [a for a in apis if start <= a.start and a.end <= end]
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pending: Api | None = None
    for a in inside:
        spans[a.name].append((a.start, a.end))
        if a.name.startswith('cudaStreamBeginCapture'):
            if pending is not None:
                raise TraceError('Nested capture intervals')
            pending = a
        if a.name.startswith('cudaStreamEndCapture'):
            if pending is None:
                raise TraceError('Unpaired capture end')
            spans['capture_host_envelope'].append((pending.start, a.end))
            pending = None
    if pending is not None or len(spans['capture_host_envelope']) != 32:
        raise TraceError('Expected32 complete capture envelopes')
    return {name: {'calls': len(v), 'duration_sum_ns': sum(b-a for a, b in v), 'mean_ns': sum(b-a for a, b in v)/len(v), 'union_ns': union_ns(v)}
            for name, v in spans.items()}


def clock_summary(path: Path, bounds: tuple[float, float]) -> dict[str, Json]:
    """Parse nvidia-smi CSV local timestamps explicitly in Asia/Seoul."""
    samples: list[int] = []
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            row = {k.strip(): v.strip() for k, v in raw.items()}
            stamp = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
            if bounds[0] <= stamp <= bounds[1]:
                key = next(k for k in row if k.startswith('clocks') and '.sm' in k)
                samples.append(int(row[key].split()[0]))
    if not samples:
        raise TraceError('No clock samples inside exact decode interval')
    return {'count': len(samples), 'median_mhz': statistics.median(samples), 'min_mhz': min(samples), 'max_mhz': max(samples)}


def session_epoch(db: SqlConnection) -> int:
    """Read the single session epoch through the typed SQL boundary."""
    row = db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()
    if row is None or len(row) != 1:
        raise TraceError('Missing session epoch')
    return integer(row[0])


def main() -> None:
    """Analyze without modifying source evidence; refuse output replacement."""
    if len(sys.argv) not in (3, 4):
        raise SystemExit('Usage: r12_graph_trace.py INPUT.sqlite OUTPUT.json [CLOCK.csv]')
    source, output = map(Path, sys.argv[1:3])
    if output.exists():
        raise TraceError(f'Output exists: {output}')
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro', uri=True) as db:
        typed_db: SqlConnection = db
        trace = read_trace(typed_db)
        tokens = select_tokens(trace)
        epoch = session_epoch(typed_db)
        decoded = summarize(tokens, epoch)
    bounds = (tokens[-33].completion, tokens[-1].completion)
    apis = [a for a in trace[0] if a.tid == tokens[-1].kernels[-1].api.tid]
    result: dict[str, Json] = {'source': str(source.resolve()), 'decode': decoded,
        'runtime_api': api_summary(apis, bounds), 'epoch_ns': epoch,
        'note': 'Kernel-empty is wall minus clipped kernel union, not proven GPU idle. API/capture spans may overlap GPU work; do not add as causal components.'}
    if len(sys.argv) == 4:
        result['clock'] = clock_summary(Path(sys.argv[3]), ((epoch+bounds[0])/1e9, (epoch+bounds[1])/1e9))
    with output.open('x') as stream:
        _ = stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'tokens': 32}))


if __name__ == '__main__':
    main()
