#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 tools/r11_timing_sources.py REPO_ROOT NEW_OUTPUT_DIR
# Self-test: python3 tools/r11_timing_sources.py --self-test REPO_ROOT
"""Generate pinned timing-only CUDA copies without changing production sources.

Compile copies with the original dependencies/flags plus -Itools -Itests -I.
Epoch boundaries enclose existing timers; prints occur after timers stop. Spec
warm and measured calls each emit markers: select the second call per wl/K.
No CUDA call, loop, original timer, input, reset, or test criterion is changed.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

TAG: Final = " // R11_TIMING\n"
HELPER: Final = (
    "static double r11_epoch(){ return std::chrono::duration<double>("
    "std::chrono::system_clock::now().time_since_epoch()).count(); }"
)


class TimingError(RuntimeError):
    """A source or destination violates the frozen generation contract."""


@dataclass(frozen=True, slots=True)
class Insertion:
    """Insert marked lines immediately before one exact source span."""

    anchor: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Source:
    """Pinned input and output names with ordered, unique insertions."""

    path: str
    sha256: str
    output: str
    insertions: tuple[Insertion, ...]
    markers: tuple[str, ...]


class FileRecord(TypedDict):
    source: str
    source_sha256: str
    generated: str
    generated_sha256: str
    injected_lines: int
    unified_patch: str


class Manifest(TypedDict):
    version: int
    scope: str
    compile_include_flags: list[str]
    files: list[FileRecord]


BENCH: Final = Source(
    "tools/bench_gemma4.cu",
    "9700b64c9192a9acb42f4bf8eddea4822c67c3e6bd5b38afb01702c5a87867b6",
    "bench_gemma4_timed.cu",
    (
        Insertion("int main(int argc,char**argv){\n", (HELPER,)),
        Insertion("        CK(cudaEventRecord(t0));\n        for(int s=0;s<ngen;s++){ if(is_moe) gemma4_moe_decode_batch",
                  ("        const double r11_begin=r11_epoch();",)),
        Insertion("        double ms=ev(t0,t1); double per_stream=ngen/(ms/1000.0);",
                  ("        const double r11_end=r11_epoch();",)),
        Insertion('        printf("  %-4d | %10.2f %10.2f | %9s | %.3f ms/%d스텝\\n",',
                  ('        printf("BATCH_INTERVAL start=%.9f end=%.9f B=%d ngen=%d elapsed_ms=%.9f\\n",r11_begin,r11_end,B,ngen,ms);',)),
        Insertion("        CK(cudaEventRecord(t0));\n        for(int s=0;s<ngen;s++){ if(is_moe) gemma4_moe_decode(c,w,st,d1,1,",
                  ("        const double r11_begin=r11_epoch();",)),
        Insertion("        double ms=ev(t0,t1); double ts=ngen/(ms/1000.0);",
                  ("        const double r11_end=r11_epoch();",)),
        Insertion('        printf("  %-6s single decode: %.2f t/s',
                  ('        printf("DECODE_INTERVAL start=%.9f end=%.9f uq=%d ngen=%d elapsed_ms=%.9f\\n",r11_begin,r11_end,sq,ngen,ms);',)),
    ),
    ("BATCH_INTERVAL", "DECODE_INTERVAL"),
)
SPEC: Final = Source(
    "tests/test_gemma4_spec_e2e.cu",
    "aadbf17f4a266c60eb75da5c568a390537181b75501c414985b8eaf0e9a4adb1",
    "spec_e2e_timed.cu",
    (
        Insertion("int main(int argc,char**argv){\n", (HELPER,)),
        Insertion("        auto t0=clk::now();\n        for(int t=0;t<N;t++){",
                  ("        const double r11_begin=r11_epoch();",)),
        Insertion("        free_g4_decode_state(st);\n    };\n\n    // spec 사이클:",
                  ("        const double r11_end=r11_epoch();",
                   '        printf("PLAIN_INTERVAL wl=%s start=%.9f end=%.9f N=%d elapsed_ms=%.9f\\n",cur_wl,r11_begin,r11_end,N,msec);')),
        Insertion("        auto t0=clk::now();\n        while((int)toks.size()<N+1",
                  ("        const double r11_begin=r11_epoch();",)),
        Insertion("        avg_k = cycles ? (double)ndraft/cycles : 0.0;",
                  ("        const double r11_end=r11_epoch();",
                   '        printf("SPEC_INTERVAL wl=%s K=%d start=%.9f end=%.9f N=%d elapsed_ms=%.9f\\n",cur_wl,K,r11_begin,r11_end,N,msec);')),
    ),
    ("PLAIN_INTERVAL", "SPEC_INTERVAL"),
)
SOURCES: Final = (BENCH, SPEC)


def digest(text: str) -> str:
    """Hash exact UTF-8 source bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instrument(original: str, spec: Source) -> str:
    """Validate the frozen source and preserve it as a literal subsequence."""
    if digest(original) != spec.sha256:
        raise TimingError(f"Source SHA mismatch: {spec.path}")
    result = original
    for insertion in spec.insertions:
        if result.count(insertion.anchor) != 1:
            raise TimingError(f"Expected one anchor in {spec.path}: {insertion.anchor!r}")
        addition = "".join(line + TAG for line in insertion.lines)
        result = result.replace(insertion.anchor, addition + insertion.anchor, 1)
    stripped = "".join(line for line in result.splitlines(keepends=True) if not line.endswith(TAG))
    if stripped != original:
        raise TimingError(f"Original lines changed: {spec.path}")
    if result.count(TAG) != sum(len(item.lines) for item in spec.insertions):
        raise TimingError(f"Wrong injection count: {spec.path}")
    for marker in spec.markers:
        if result.count(marker) != 1:
            raise TimingError(f"Wrong marker count: {marker}")
    return result


def generate(root: Path, output: Path) -> Manifest:
    """Validate all inputs and outputs before exclusively creating artifacts."""
    records: list[FileRecord] = []
    generated: list[tuple[Path, str]] = []
    for spec in SOURCES:
        original = (root / spec.path).read_text(encoding="utf-8")
        revised = instrument(original, spec)
        target = output / spec.output
        generated.append((target, revised))
        records.append({
            "source": str((root / spec.path).resolve()), "source_sha256": digest(original),
            "generated": str(target.resolve()), "generated_sha256": digest(revised),
            "injected_lines": revised.count(TAG),
            "unified_patch": "".join(difflib.unified_diff(
                original.splitlines(keepends=True), revised.splitlines(keepends=True),
                fromfile=spec.path, tofile=spec.output)),
        })
    manifest: Manifest = {
        "version": 1, "scope": "epoch/print-only; original CUDA calls, timers and criteria unchanged",
        "compile_include_flags": ["-Itools", "-Itests", "-I."], "files": records,
    }
    generated.append((output / "timing_sources.json", json.dumps(manifest, indent=2) + "\n"))
    for target, _ in generated:
        if target.exists() or target.is_symlink():
            raise TimingError(f"Refusing existing output: {target}")
    output.mkdir(parents=True, exist_ok=True)
    for target, text in generated:
        with target.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
    return manifest


def self_test(root: Path) -> None:
    """Exercise actual pinned sources, output refusal and lossless injections."""
    before = [(root / spec.path).read_bytes() for spec in SOURCES]
    with tempfile.TemporaryDirectory(prefix="r11-timing-") as temp:
        output = Path(temp) / "generated"
        manifest = generate(root, output)
        for spec, original, record in zip(SOURCES, before, manifest["files"], strict=True):
            result = (output / spec.output).read_text(encoding="utf-8")
            expected_lines = 7
            if record["injected_lines"] != expected_lines or digest(result) != record["generated_sha256"]:
                raise TimingError(f"Self-test manifest mismatch: {spec.path}")
            if (root / spec.path).read_bytes() != original:
                raise TimingError(f"Self-test source modified: {spec.path}")
        try:
            generate(root, output)
        except TimingError as error:
            if "Refusing existing output" not in str(error):
                raise
        else:
            raise TimingError("Self-test failed: existing files accepted")
        try:
            instrument(before[0].decode("utf-8") + "\n", BENCH)
        except TimingError as error:
            if "Source SHA mismatch" not in str(error):
                raise
        else:
            raise TimingError("Self-test failed: changed source accepted")
    print("R11_TIMING_SELF_TEST PASS sources=2 injected_lines=7,7 markers=4 originals_unchanged=1 exclusive_outputs=1")


def main(args: list[str]) -> None:
    """Use a dependency-free CLI because the measurement environment is frozen."""
    if len(args) != 2:
        raise SystemExit(__doc__ + "\nUsage: r11_timing_sources.py REPO OUTPUT | --self-test REPO")
    if args[0] == "--self-test":
        self_test(Path(args[1]).resolve())
    else:
        manifest = generate(Path(args[0]).resolve(), Path(args[1]).resolve())
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
