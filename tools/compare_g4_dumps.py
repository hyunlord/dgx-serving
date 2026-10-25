#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# How to run: python3 tools/compare_g4_dumps.py GOLDEN R1 R2 R3 R4 OUTPUT_PREFIX
"""CPU-only Request05 dump comparison; registered F decision uses full16 R2/R1 vs R3/R1."""
from __future__ import annotations

import csv
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

THRESHOLD: Final = 0.015
LIMIT: Final = 1_000_000
Array = NDArray[np.float64]
Scalar = str | int | float | bool
Verdict = Literal["investigate_prior", "acceptable", "borderline", "hold"]


class ComparisonError(ValueError):
    """Malformed, inconsistent, or incomplete input dumps."""


@dataclass(frozen=True, slots=True)
class Dump:
    directory: Path
    rows: int
    prefix: int
    hidden: int
    vocab: int
    layers: int
    path: str
    scale_dtype: int

    @classmethod
    def load(cls, directory: Path, expected_path: str, expected_dtype: int) -> Dump:
        raw = json.loads((directory / "metadata.json").read_text())
        if raw.get("complete") is not True:
            raise ComparisonError(f"Incomplete dump: {directory}")
        names = ("rows", "golden_prefix_rows", "hidden_width", "vocab_size", "layers", "scale_dtype")
        if any(type(raw.get(key)) is not int for key in names):
            raise ComparisonError(f"Metadata integer fields invalid: {directory}")
        result = cls(directory, *(raw[key] for key in names[:-1]), raw["path"], raw["scale_dtype"])
        if (result.rows, result.prefix) != (16, 5) or min(result.hidden, result.vocab, result.layers) < 1:
            raise ComparisonError(f"Invalid shape metadata: {directory}")
        if result.layers < 8 or result.vocab < 2:
            raise ComparisonError(f"Need depths0..8 and at least2 logits: {directory}")
        if result.path != expected_path or result.scale_dtype != expected_dtype:
            raise ComparisonError(f"Run role mismatch: {directory}")
        return result

    def hidden_at(self, depth: int) -> Array:
        return read_array(self.directory / f"hs_d{depth}.bin", self.rows, self.hidden)


class Measure(TypedDict):
    maxabs: float
    refmax: float
    m: float | str


def read_array(path: Path, rows: int, cols: int) -> Array:
    if path.stat().st_size != rows * cols * 4:
        raise ComparisonError(f"Wrong float32 byte count: {path}")
    values = np.fromfile(path, dtype="<f4").reshape(rows, cols).astype(np.float64)
    if not np.isfinite(values).all():
        raise ComparisonError(f"Nonfinite values: {path}")
    return values


def quotient(numerator: float, denominator: float) -> float:
    """Zero denominator: equal zero gives0; positive numerator gives infinity. No epsilon."""
    return numerator / denominator if denominator else (0.0 if numerator == 0 else math.inf)


def number(value: float) -> float | str:
    return value if math.isfinite(value) else ("Infinity" if value > 0 else "undefined")


def measure(candidate: Array, reference: Array) -> tuple[Measure, float]:
    if candidate.shape != reference.shape or candidate.size == 0:
        raise ComparisonError(f"Comparison shape mismatch: {candidate.shape}, {reference.shape}")
    delta = float(np.max(np.abs(candidate - reference)))
    refmax = float(np.max(np.abs(reference)))
    normalized = quotient(delta, refmax)
    return {"maxabs": delta, "refmax": refmax, "m": number(normalized)}, normalized


def decide(mmq: list[float], fp16: list[float], off_fail: bool, on_fail: bool) -> Verdict:
    """Literal F precedence; only full16 depth0..L R2/R1 and R3/R1 enter this rule."""
    if len(mmq) != len(fp16) or not mmq:
        raise ComparisonError("Decision arrays must have equal nonzero lengths")
    if any(a > 3 * b for a, b in zip(mmq, fp16)):
        return "investigate_prior"
    within_prior = all(a <= b for a, b in zip(mmq, fp16))
    if off_fail and within_prior:
        return "acceptable"
    if not off_fail and on_fail and within_prior:
        return "borderline"
    return "hold"


def top_two(values: Array) -> tuple[int, int, float]:
    """Argmax ties choose smallest token ID; second excludes the chosen top token."""
    first = int(np.argmax(values))
    rest = values.copy()
    rest[first] = -math.inf
    second = int(np.argmax(rest))
    return first, second, float(values[first] - values[second])


def csv_text(records: list[dict[str, Scalar]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def compare(golden: Path, directories: list[Path], output: Path) -> Verdict:
    dumps = [Dump.load(path, kind, dtype) for path, kind, dtype in zip(
        directories, ["SB", "MMQ", "SB", "MMQ"], [0, 0, 1, 1], strict=True)]
    first = dumps[0]
    signature = (first.rows, first.prefix, first.hidden, first.vocab, first.layers)
    for dump in dumps:
        if (dump.rows, dump.prefix, dump.hidden, dump.vocab, dump.layers) != signature:
            raise ComparisonError("R1..R4 dimensions differ")
    invariant_files = ("token_ids.bin", "rope_full_cos.bin", "rope_full_sin.bin",
                       "rope_sliding_cos.bin", "rope_sliding_sin.bin")
    for name in invariant_files:
        content = (first.directory / name).read_bytes()
        if not content or any((dump.directory / name).read_bytes() != content for dump in dumps[1:]):
            raise ComparisonError(f"R1..R4 input identity mismatch: {name}")
    table1: list[dict[str, Scalar]] = []
    table2: list[dict[str, Scalar]] = []
    g2: list[dict[str, Scalar]] = []
    golden_errors: list[list[float]] = [[], [], [], []]
    mmq: list[float] = []
    fp16: list[float] = []
    for depth in range(first.layers + 1):
        arrays = [dump.hidden_at(depth) for dump in dumps]
        for candidate, reference, label in [(1, 0, "R2_vs_R1"), (2, 0, "R3_vs_R1"), (3, 2, "R4_vs_R3")]:
            for rows, scope in [(16, "full16"), (5, "prefix5")]:
                result, value = measure(arrays[candidate][:rows], arrays[reference][:rows])
                table2.append({"comparison": label, "depth": depth, "scope": scope, **result})
                if rows == 16 and candidate == 1:
                    mmq.append(value)
                if rows == 16 and candidate == 2:
                    fp16.append(value)
        if depth <= 8:
            ref = read_array(golden / f"q4fp32_hs_{depth}.bin", 5, first.hidden)
            for run, array in enumerate(arrays):
                result, value = measure(array[:5], ref)
                golden_errors[run].append(value)
                table1.append({"run": f"R{run + 1}", "depth": depth, "scope": "prefix5",
                               **result, "threshold": THRESHOLD, "pass": value < THRESHOLD})
            prior, current = golden_errors[0][-1], golden_errors[2][-1]
            g2.append({"depth": depth, "R1_golden_m": number(prior), "R3_golden_m": number(current),
                       "difference_R3_minus_R1": number(current - prior),
                       "ratio_R3_over_R1": number(quotient(current, prior))})
    logits = [read_array(dump.directory / "logits.bin", 16, first.vocab) for dump in dumps]
    argmax_rows: list[dict[str, Scalar]] = []
    for run, array in enumerate(logits):
        for position in range(16):
            a, a2, agap = top_two(array[position])
            b, b2, bgap = top_two(logits[0][position])
            argmax_rows.append({"run": f"R{run + 1}", "position": position, "match_R1": a == b,
                                "actual_top1": a, "actual_top2": a2, "actual_gap": agap,
                                "R1_top1": b, "R1_top2": b2, "R1_gap": bgap})
    off_fail = any(value >= THRESHOLD for value in golden_errors[0])
    on_fail = any(value >= THRESHOLD for value in golden_errors[1])
    verdict = decide(mmq, fp16, off_fail, on_fail)
    report = {"metric": "max(abs(candidate-reference))/max(abs(reference)); no epsilon",
              "zero_denominator": "0/0=0; positive/0=Infinity; Infinity-Infinity=undefined",
              "argmax_ties": "smallest token ID; second excludes top1", "dimensions": signature,
              "run_directories": [str(path) for path in directories], "table1": table1, "table2": table2,
              "final_logits_argmax": argmax_rows, "G2": g2,
              "F": {"verdict": verdict, "OFF_R1_golden_fail": off_fail, "ON_R2_golden_fail": on_fail,
                    "greater_than_3x_depths": [d for d, (a, b) in enumerate(zip(mmq, fp16)) if a > 3*b],
                    "all_mmq_within_fp16": all(a <= b for a, b in zip(mmq, fp16)),
                    "scope": "full16 depth0..L; R2vsR1 MMQ, R3vsR1 fp16; R4/prefix diagnostic only"}}
    payloads = {".json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
    for label, records in [("table1", table1), ("table2", table2), ("argmax", argmax_rows), ("g2", g2)]:
        payloads[f"_{label}.csv"] = csv_text(records)
    for suffix, payload in payloads.items():
        path = Path(str(output) + suffix)
        if len(payload.encode()) > LIMIT or path.exists():
            raise ComparisonError(f"Output exceeds1MB or already exists: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, payload in payloads.items():
        with Path(str(output) + suffix).open("x") as handle:
            handle.write(payload)
    return verdict


if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise SystemExit("usage: compare_g4_dumps.py GOLDEN R1 R2 R3 R4 OUTPUT_PREFIX")
    print(compare(Path(sys.argv[1]), [Path(arg) for arg in sys.argv[2:6]], Path(sys.argv[6])))
