#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: python3 tools/r13_compare_graph.py --reference PREFIX --candidate PREFIX --mode direct
"""Fail closed on the first differing graph token/logit or invalid graph counters."""
from __future__ import annotations

import argparse
import array
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

Numbers: TypeAlias = dict[str, int | float]
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def decode_json(text: str, decoder: Callable[[str], JsonValue] = json.loads) -> JsonValue:
    """Bound the standard decoder result to the JSON value grammar."""
    return decoder(text)


class Arguments(argparse.Namespace):
    reference: Path = Path()
    candidate: Path = Path()
    mode: str = "off"
    request: int = 0
    reference_summary: Path | None = None
    candidate_summary: Path | None = None


class GateFailure(RuntimeError):
    """A diagnostic artifact violated the registered gate."""


@dataclass(frozen=True, slots=True)
class Run:
    prefix: Path
    summary: Path
    request: int

    def stats(self) -> Numbers:
        """Parse exactly one requested summary row and finite numeric fields."""
        rows: list[Numbers] = []
        for line in self.summary.read_text().splitlines():
            raw = decode_json(line)
            if not isinstance(raw, dict):
                raise GateFailure("summary row is not an object")
            parsed: Numbers = {}
            for key, value in raw.items():
                if isinstance(value, bool):
                    raise GateFailure("invalid summary field")
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise GateFailure(f"nonfinite/nonnumeric summary field {key}")
                parsed[key] = value
            if parsed.get("request") == self.request:
                rows.append(parsed)
        if len(rows) != 1:
            raise GateFailure("summary request missing or duplicated")
        return rows[0]

    def path(self, suffix: str) -> Path:
        return Path(str(self.prefix) + suffix)


def integer(stats: Numbers, key: str) -> int:
    value = stats[key]
    if value < 0 or int(value) != value:
        raise GateFailure(f"invalid integer counter {key}: {value}")
    return int(value)


def check_counters(stats: Numbers, mode: str) -> None:
    """Apply the registered capture counters, independently of timing."""
    for name in ("fallback_recaptures", "update_failures", "setparams_failures", "reinstantiations"):
        if integer(stats, name):
            raise GateFailure(f"{name} must be zero")
    executed = integer(stats, "executed_steps")
    accepted = integer(stats, "accepted_steps")
    if accepted != integer(stats, "accepted") or executed < accepted:
        raise GateFailure("executed/accepted counter mismatch")
    if integer(stats, "speculative_steps") != executed - accepted:
        raise GateFailure("speculative counter mismatch")
    match mode:
        case "overlap":
            if integer(stats, "planned_captures") != executed or integer(stats, "instantiations") > 2:
                raise GateFailure("overlap requires one planned capture per executed step and <=2 instances")
        case "direct" | "both":
            if integer(stats, "planned_captures") != 1 or integer(stats, "mapping_fallbacks"):
                raise GateFailure("direct/both requires one capture and zero mapping fallbacks")
        case "off" | "v1":
            pass
        case _:
            raise GateFailure(f"unknown mode {mode}")


def compare(reference: Run, candidate: Run, mode: str) -> Numbers:
    """Stream rows, report first differing float, and independently count ties."""
    ref, got = reference.stats(), candidate.stats()
    check_counters(got, mode)
    steps = integer(ref, "accepted")
    if steps <= 0 or steps != integer(got, "accepted"):
        raise GateFailure("accepted step counts differ or are empty")
    for key in ("start", "n_past"):
        if ref[key] != got[key]:
            raise GateFailure(f"{key} differs")
    for suffix in (".inputs.i32", ".argmax.i32"):
        a, b = reference.path(suffix).read_bytes(), candidate.path(suffix).read_bytes()
        if len(a) != steps * 4 or len(b) != steps * 4:
            raise GateFailure(f"invalid token file size {suffix}")
        if a != b:
            offset = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            raise GateFailure(f"first mismatch {suffix} step={offset // 4}")
    ids = array.array("i")
    ids.frombytes(reference.path(".argmax.i32").read_bytes())
    size = reference.path(".logits.f32").stat().st_size
    if size <= 0 or size % (steps * 4) or candidate.path(".logits.f32").stat().st_size != size:
        raise GateFailure("invalid or different logits size")
    vocab, ties = size // (steps * 4), 0
    with reference.path(".logits.f32").open("rb") as a, candidate.path(".logits.f32").open("rb") as b:
        for step in range(steps):
            left, right = a.read(vocab * 4), b.read(vocab * 4)
            if left != right:
                offset = next(i for i, (x, y) in enumerate(zip(left, right)) if x != y)
                raise GateFailure(f"first logits mismatch step={step} vocab={offset // 4}")
            values = array.array("f")
            values.frombytes(left)
            maximum, runner, argmax = -math.inf, -math.inf, -1
            for index, value in enumerate(values):
                if not math.isfinite(value):
                    raise GateFailure(f"nonfinite logit step={step} vocab={index}")
                if value > maximum:
                    runner, maximum, argmax = maximum, value, index
                elif value > runner:
                    runner = value
            if argmax != ids[step]:
                raise GateFailure(f"first-index CPU argmax mismatch step={step}")
            ties += int(runner == maximum)
    for stats in (ref, got):
        if integer(stats, "top2_checked_steps") != steps or integer(stats, "exact_top2_ties") != ties:
            raise GateFailure("recorded top-2 tie counters disagree with logits")
    return {"steps": steps, "vocab": vocab, "logits_bytes": size, "exact_top2_ties": ties,
            "executed_steps": integer(got, "executed_steps"), "accepted_steps": steps,
            "planned_captures": integer(got, "planned_captures")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--reference", type=Path, required=True)
    _ = parser.add_argument("--candidate", type=Path, required=True)
    _ = parser.add_argument("--mode", choices=("off", "v1", "overlap", "direct", "both"), required=True)
    _ = parser.add_argument("--request", type=int, default=0)
    _ = parser.add_argument("--reference-summary", type=Path)
    _ = parser.add_argument("--candidate-summary", type=Path)
    args = Arguments()
    _ = parser.parse_args(namespace=args)
    try:
        result = compare(Run(args.reference, args.reference_summary or Path(str(args.reference) + ".summary.jsonl"), args.request),
                         Run(args.candidate, args.candidate_summary or Path(str(args.candidate) + ".summary.jsonl"), args.request), args.mode)
    except (OSError, ValueError, KeyError, GateFailure) as error:
        print(json.dumps({"pass": False, "error": str(error)}, allow_nan=False))
        return 1
    print(json.dumps({"pass": True, "mode": args.mode, **result}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
