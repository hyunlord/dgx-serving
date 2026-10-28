#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: python3 tools/r13_graph_trace.py TRACE.sqlite RESULT.json [CLOCK.csv]
"""R13 generation trace accounting with GPU-completion-aligned steady steps.

Use nsys --cuda-graph-trace=node. Unlike R12, host waits do not delimit GPU
steps: overlap mode has already submitted the next step before the wait.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import NotRequired, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.r12_graph_trace import (
    Api,
    Json,
    Kernel,
    SqlConnection,
    Token,
    TraceError,
    category,
    clock_summary,
    read_trace,
    session_epoch,
    union_ns,
)


class Distribution(TypedDict):
    count: int
    sum_ns: int
    median_ns: float
    p99_ns: int
    max_ns: int


class ApiCosts(TypedDict):
    calls: int
    full_duration_sum_ns: int
    clipped_union_ns: int


class Decode(TypedDict):
    token_count: int
    tokens_seen: int
    start_ns: int
    end_ns: int
    start_epoch: float
    end_epoch: float
    wall_ns: int
    kernel_count: int
    kernel_sum_ns: int
    gpu_union_ns: int
    kernel_empty_ns: int
    kernel_overlap_ns: int
    categories_ns: dict[str, int]
    gaps: Distribution
    inter_token_gaps: Distribution
    per_token: list[dict[str, int]]


class Report(TypedDict):
    source: str
    decode: Decode
    runtime_api: dict[str, ApiCosts]
    epoch_ns: int
    note: str
    clock: NotRequired[dict[str, Json]]


def select_tokens(kernels: list[Kernel]) -> list[Token]:
    """Require 128 complete graph launches, excluding the eager prefix."""
    groups: dict[int, list[Kernel]] = defaultdict(list)
    for kernel in kernels:
        if kernel.api.name.startswith('cudaGraphLaunch'):
            if not kernel.node:
                raise TraceError('Missing graph node ID; enable node tracing')
            groups[kernel.api.row].append(kernel)
    if len(groups) != 128:
        raise TraceError(f'Expected128 generation launches, observed{len(groups)}')
    ordered = sorted(groups.values(), key=lambda group: group[0].api.start)
    counts = {len(group) for group in ordered}
    if len(counts) != 1 or not counts.issubset({1334, 1336}):
        raise TraceError(f'Unexpected graph kernel counts: {counts}')
    contexts = {kernel.context for group in ordered for kernel in group}
    if len(contexts) != 1:
        raise TraceError('Generation must use one process/device/context/stream')
    tokens: list[Token] = []
    for group in ordered:
        group.sort(key=lambda kernel: (kernel.start, kernel.end))
        if sum(kernel.name == 'softcap_kernel' for kernel in group) != 1:
            raise TraceError('Graph must have exactly one final softcap')
        tail = [kernel.name for kernel in group[-3:]]
        expected_tail = (['softcap_kernel', 'g4g_snapshot', 'g4g_argmax']
                         if len(group) == 1336 else [group[-3].name, group[-2].name, 'softcap_kernel'])
        if tail != expected_tail:
            raise TraceError(f'Unexpected terminal kernels: {tail}')
        if tokens and group[0].start < tokens[-1].completion:
            raise TraceError('GPU graph steps overlap despite serialized stream')
        tokens.append(Token(tuple(group), max(kernel.end for kernel in group)))
    return tokens


def distribution(values: list[int]) -> Distribution:
    """Preserve integer-nanosecond samples; p99 uses the nearest-rank rule."""
    ordered = sorted(values)
    return {'count': len(values), 'sum_ns': sum(values),
            'median_ns': float(statistics.median(values)) if values else 0.0,
            'p99_ns': ordered[(99 * len(values) + 99) // 100 - 1] if values else 0,
            'max_ns': max(values, default=0)}


def gaps(intervals: list[tuple[int, int]], bounds: tuple[int, int]) -> list[int]:
    """Complement of GPU kernel union; transfers may occupy these gaps."""
    cursor, end = bounds
    result: list[int] = []
    for left, right in sorted(intervals):
        left, right = max(left, bounds[0]), min(right, end)
        if right <= left:
            continue
        if left > cursor:
            result.append(left - cursor)
        cursor = max(cursor, right)
    if cursor < end:
        result.append(end - cursor)
    return result


def summarize(tokens: list[Token], epoch: int) -> Decode:
    """Last32 steps begin at the preceding graph's GPU completion."""
    previous = tokens[-33]
    start = previous.completion
    categories: dict[str, int] = defaultdict(int)
    intervals: list[tuple[int, int]] = []
    rows: list[dict[str, int]] = []
    inter_token: list[int] = []
    for index, token in enumerate(tokens[-32:], start=96):
        end = token.completion
        lm_heads = [kernel for kernel in token.kernels if category(kernel.name) == 'linear']
        if not lm_heads:
            raise TraceError('Missing terminal linear/lm_head kernel')
        local = [(kernel.start, kernel.end) for kernel in token.kernels]
        intervals.extend(local)
        for kernel in token.kernels:
            label = 'lm_head' if kernel.row == lm_heads[-1].row else category(kernel.name)
            categories[label] += kernel.end - kernel.start
        inter_token.append(token.kernels[0].start - previous.completion)
        rows.append({'step': index, 'wall_ns': end - previous.completion,
                     'kernel_count': len(token.kernels),
                     'kernel_sum_ns': sum(right-left for left, right in local),
                     'kernel_empty_ns': end - previous.completion - union_ns(local)})
        previous = token
    end = previous.completion
    raw_sum = sum(categories.values())
    union = union_ns(intervals)
    return {'token_count': 32, 'tokens_seen': len(tokens), 'start_ns': start,
            'end_ns': end, 'start_epoch': (epoch+start)/1e9, 'end_epoch': (epoch+end)/1e9,
            'wall_ns': end-start, 'kernel_count': sum(row['kernel_count'] for row in rows),
            'kernel_sum_ns': raw_sum, 'gpu_union_ns': union,
            'kernel_empty_ns': end-start-union, 'kernel_overlap_ns': raw_sum-union,
            'categories_ns': dict(categories), 'gaps': distribution(gaps(intervals, (start, end))),
            'inter_token_gaps': distribution(inter_token), 'per_token': rows}


def api_costs(apis: list[Api], bounds: tuple[int, int]) -> dict[str, ApiCosts]:
    """CPU spans intersecting GPU bounds; these overlap execution, not overhead sums."""
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pending: dict[int, Api] = {}
    for api in apis:
        spans[api.name].append((api.start, api.end))
        if api.name.startswith('cudaStreamBeginCapture'):
            if api.tid in pending:
                raise TraceError('Nested capture envelope')
            pending[api.tid] = api
        if api.name.startswith('cudaStreamEndCapture'):
            begin = pending.pop(api.tid, None)
            if begin is None:
                raise TraceError('Unpaired capture end')
            spans['capture_host_envelope'].append((begin.start, api.end))
    if pending:
        raise TraceError('Unpaired capture begin')
    start, end = bounds
    result: dict[str, ApiCosts] = {}
    for name, values in spans.items():
        intersect = [(left, right) for left, right in values if left < end and right > start]
        if intersect:
            result[name] = {'calls': len(intersect),
                            'full_duration_sum_ns': sum(right-left for left, right in intersect),
                            'clipped_union_ns': union_ns([(max(start,left), min(end,right)) for left,right in intersect])}
    return result


def main() -> None:
    """Read evidence without replacement; reject unavailable trace semantics."""
    if len(sys.argv) not in (3, 4):
        raise SystemExit('Usage: r13_graph_trace.py TRACE.sqlite RESULT.json [CLOCK.csv]')
    source, output = map(Path, sys.argv[1:3])
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro', uri=True) as db:
        typed_db: SqlConnection = db
        apis, kernels = read_trace(typed_db)
        tokens = select_tokens(kernels)
        epoch = session_epoch(typed_db)
    decoded = summarize(tokens, epoch)
    bounds = (decoded['start_ns'], decoded['end_ns'])
    selected_tid = tokens[-1].kernels[0].api.tid
    costs = api_costs([api for api in apis if api.tid == selected_tid], bounds)
    result: Report = {'source': str(source.resolve()), 'decode': decoded, 'runtime_api': costs,
              'epoch_ns': epoch,
              'note': 'GPU-completion-aligned32 steps, not CPU request wall time. Kernel-empty includes transfer gaps; it is not proven GPU idle. CPU capture/update costs overlap GPU and must not be added causally.'}
    if len(sys.argv) == 4:
        result['clock'] = clock_summary(Path(sys.argv[3]), (decoded['start_epoch'], decoded['end_epoch']))
    with output.open('x') as stream:
        _ = stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'tokens': 32}))


if __name__ == '__main__':
    main()
