#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 tools/r11_llama_timing.py PINNED_LLAMA_CPP NEW_OUTPUT_DIR
# Self-test: python3 tools/r11_llama_timing.py --self-test
"""Generate a timing-only copy of df9beac llama-bench without editing upstream.

The existing high_resolution_clock timer is reused, never replaced. A compile-time
assertion requires it to alias system_clock so its epoch can select GPU samples.
The self-test uses a fixture; the pinned upstream input is checked on generation.
"""
from __future__ import annotations

import difflib
import json
import sys
import tempfile
from pathlib import Path
from typing import Final

from r11_timing_sources import (
    FileRecord,
    Insertion,
    Source,
    TimingError,
    digest,
    instrument,
)

REVISION: Final = "df9beac7153166550571c70cc667f270b745da49"
ANCHOR: Final = "            t.samples_ns.push_back(t_ns);"
SOURCE: Final = Source(
    "tools/llama-bench/llama-bench.cpp",
    "8eed21016aabf399c7223f1c8588840edc1d15398e87f2048056d381b8a182ec",
    "llama_bench_timed.cpp",
    (
        Insertion("#include <chrono>\n", ("#include <type_traits>",)),
        Insertion(ANCHOR, (
            '            static_assert(std::is_same<std::chrono::high_resolution_clock, std::chrono::system_clock>::value, "LLAMA_INTERVAL requires system-clock epoch");',
            '            fprintf(stderr,"LLAMA_INTERVAL start=%.9f end=%.9f rep=%d ngen=%d elapsed_ns=%llu\\n",t_start/1e9,(t_start+t_ns)/1e9,i+1,t.n_gen,(unsigned long long)t_ns);',
        )),
    ),
    ("LLAMA_INTERVAL start=",),
)


def generate(source: Path, output: Path) -> FileRecord:
    """Validate pinned source before exclusively writing the copy and manifest."""
    original = source.read_text(encoding="utf-8")
    generated = instrument(original, SOURCE)
    target = output / SOURCE.output
    manifest_path = output / "llama_timing_sources.json"
    for path in (target, manifest_path):
        if path.exists() or path.is_symlink():
            raise TimingError(f"Refusing existing output: {path}")
    record: FileRecord = {
        "source": str(source.resolve()), "source_sha256": digest(original),
        "generated": str(target.resolve()), "generated_sha256": digest(generated),
        "injected_lines": 3,
        "unified_patch": "".join(difflib.unified_diff(
            original.splitlines(keepends=True), generated.splitlines(keepends=True),
            fromfile=SOURCE.path, tofile=SOURCE.output)),
    }
    output.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as stream:
        stream.write(generated)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump({"upstream_revision": REVISION, "file": record,
                   "epoch_contract": "high_resolution_clock aliases system_clock; static_assert required"}, stream, indent=2)
        stream.write("\n")
    return record


def self_test() -> None:
    """Exercise insertion safety on a fixture, without claiming upstream testing."""
    fixture = '#include <chrono>\nvoid f(){\n' + ANCHOR + '\n}\n'
    spec = Source(SOURCE.path, digest(fixture), SOURCE.output, SOURCE.insertions, SOURCE.markers)
    generated = instrument(fixture, spec)
    if generated.count("static_assert(") != 1 or generated.count("fprintf(stderr,") != 1:
        raise TimingError("Fixture insertion counts wrong")
    if generated.index("fprintf(stderr,") > generated.index(ANCHOR):
        raise TimingError("Interval log must precede sample push")
    with tempfile.TemporaryDirectory(prefix="r11-llama-timing-") as temp:
        source = Path(temp) / "fixture.cpp"
        source.write_text(fixture, encoding="utf-8")
        output = Path(temp) / "output"
        try:
            generate(source, output)
        except TimingError as error:
            if "Source SHA mismatch" not in str(error):
                raise
        else:
            raise TimingError("Unpinned fixture accepted")
        if output.exists() or source.read_text(encoding="utf-8") != fixture:
            raise TimingError("Rejected source caused writes")
    print("R11_LLAMA_TIMING_SELF_TEST PASS fixture_only=1 pinned_upstream_generation=NOT_RUN injected_lines=3")


def main(args: list[str]) -> None:
    """Run dependency-free generation or the explicitly fixture-only self-test."""
    if args == ["--self-test"]:
        self_test()
    elif len(args) == 2:
        print(json.dumps(generate(Path(args[0]).resolve(), Path(args[1]).resolve()), indent=2))
    else:
        raise SystemExit("Usage: r11_llama_timing.py PINNED_CPP NEW_DIR | --self-test")


if __name__ == "__main__":
    main(sys.argv[1:])
