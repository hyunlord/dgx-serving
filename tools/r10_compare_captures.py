#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Compare every byte listed by two identical captures.csv manifests.

How to run: python3 tools/r10_compare_captures.py LEFT_DIR RIGHT_DIR OUTPUT.json
CSV columns must be exactly file,elements,itemsize, with identical ordered rows.
Only unique leaf filenames and regular, non-symlink files are accepted. Both
directories must contain exactly their listed payloads plus captures.csv.
Exit 0: identical; 1: content mismatch; 2: malformed input or output refusal.
This verifies all declared captures; it cannot prove an emitter omitted no tensor.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

CHUNK: Final = 1024 * 1024
MAX_SUMMARY: Final = 1_000_000
MANIFEST: Final = "captures.csv"


class CaptureError(RuntimeError):
    """Capture metadata or file structure cannot support a valid comparison."""


@dataclass(frozen=True, slots=True)
class Entry:
    """Canonical manifest row describing one dense capture payload."""

    file: str
    elements: int
    itemsize: int

    @property
    def size(self) -> int:
        return self.elements * self.itemsize


@dataclass(frozen=True, slots=True)
class Difference:
    """First difference and full-file hashes, after exact-size validation."""

    file: str
    bytes: int
    first_differing_byte: int
    left_sha256: str
    right_sha256: str


def positive(value: str) -> int:
    """Accept canonical positive decimal counts only."""
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise CaptureError(f"Invalid positive integer: {value!r}")
    return int(value)


def read_manifest(directory: Path) -> list[Entry]:
    """Validate names, metadata, exact file inventory, and payload lengths."""
    path = directory / MANIFEST
    if not directory.is_dir() or path.is_symlink() or not path.is_file():
        raise CaptureError(f"Missing regular manifest: {path}")
    entries: list[Entry] = []
    names: set[str] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != ["file", "elements", "itemsize"]:
            raise CaptureError(f"Invalid CSV header: {path}")
        for number, row in enumerate(reader, 2):
            if len(row) != 3:
                raise CaptureError(f"Expected three columns at {path}:{number}")
            name, elements, itemsize = row
            if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name)
                    or name == MANIFEST or name in names):
                raise CaptureError(f"Unsafe, reserved, or duplicate leaf at {path}:{number}: {name!r}")
            entry = Entry(name, positive(elements), positive(itemsize))
            payload = directory / name
            if payload.is_symlink() or not payload.is_file():
                raise CaptureError(f"Missing regular capture: {payload}")
            if payload.stat().st_size != entry.size:
                raise CaptureError(f"Wrong capture size: {payload}; expected {entry.size}")
            names.add(name)
            entries.append(entry)
    if not entries:
        raise CaptureError(f"Empty capture manifest: {path}")
    actual = {p.name for p in directory.iterdir()}
    if actual != names | {MANIFEST}:
        raise CaptureError(f"Unlisted entries in {directory}: {sorted(actual - names - {MANIFEST})}")
    return entries


def compare_file(paths: tuple[Path, Path], entry: Entry) -> Difference | None:
    """Hash full files and directly compare chunks, retaining the first byte offset."""
    left_hash, right_hash = hashlib.sha256(), hashlib.sha256()
    offset = 0
    first: int | None = None
    with (paths[0] / entry.file).open("rb") as left, (paths[1] / entry.file).open("rb") as right:
        while offset < entry.size:
            length = min(CHUNK, entry.size - offset)
            a, b = left.read(length), right.read(length)
            if len(a) != length or len(b) != length:
                raise CaptureError(f"Capture shortened during read: {entry.file}")
            left_hash.update(a)
            right_hash.update(b)
            if first is None and a != b:
                first = offset + next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            offset += length
        if left.read(1) or right.read(1):
            raise CaptureError(f"Capture grew during read: {entry.file}")
    if first is not None:
        return Difference(entry.file, entry.size, first, left_hash.hexdigest(), right_hash.hexdigest())
    return None


def run(paths: tuple[Path, Path], output: Path) -> int:
    """Write one bounded result without overwriting prior evidence."""
    if output.exists() or output.is_symlink():
        raise CaptureError(f"Refusing to overwrite output: {output}")
    try:
        left, right = (read_manifest(path) for path in paths)
        if left != right:
            raise CaptureError("Manifest rows differ (including order or numeric metadata)")
        differences = [d for entry in left if (d := compare_file(paths, entry)) is not None]
        status = 1 if differences else 0
        result = {
            "status": "mismatch" if differences else "identical",
            "left": str(paths[0].resolve()), "right": str(paths[1].resolve()),
            "files_compared": len(left), "bytes_compared_per_side": sum(e.size for e in left),
            "elements_compared": sum(e.elements for e in left),
            "mismatch_files": len(differences), "differences": [asdict(d) for d in differences],
            "scope": "Every declared payload byte; identical manifests do not prove emitter coverage",
        }
        encoded = json.dumps(result, indent=2) + "\n"
        if len(encoded.encode("utf-8")) > MAX_SUMMARY:
            raise CaptureError("Comparison summary exceeds 1 MB; refusing truncated evidence")
    except (CaptureError, OSError, csv.Error, UnicodeError, ValueError) as error:
        status = 2
        encoded = json.dumps({"status": "malformed", "error": str(error)[:4000]}, indent=2) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return status


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    try:
        code = run((Path(sys.argv[1]), Path(sys.argv[2])), Path(sys.argv[3]))
    except (CaptureError, OSError) as error:
        print(str(error), file=sys.stderr)
        code = 2
    raise SystemExit(code)
