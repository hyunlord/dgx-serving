#!/usr/bin/env python3
"""Convert fp32 quantization scales to fp16 while preserving source weights.

Only *_scale.bin files with a matching same-prefix *_q4.bin are converted.
Other files (including unmatched router coefficients) are symlinked unchanged;
the destination manifest is copied with scale_dtype=1. Destination must be new.

Usage: python tools/convert_scale_fp16.py [--dry-run] SRC DST
Dry-run inspects filenames, metadata and manifest only; it writes nothing.
"""
import getopt
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """Plan the conversion before creating a fresh destination."""
    try:
        options, args = getopt.gnu_getopt(sys.argv[1:], "", ["dry-run"])
    except getopt.GetoptError as error:
        raise SystemExit(str(error)) from error
    if len(args) != 2:
        raise SystemExit("usage: convert_scale_fp16.py [--dry-run] SRC DST")
    dry_run = bool(options)
    src, dst = (Path(arg).absolute() for arg in args)
    source_root, destination_root = src.resolve(), dst.resolve()
    if not src.is_dir():
        raise SystemExit(f"source directory missing: {src}")
    if (source_root == destination_root
            or source_root in destination_root.parents
            or destination_root in source_root.parents):
        raise SystemExit("source and destination must be separate, non-nested directories")
    if os.path.lexists(dst):
        raise SystemExit(f"destination already exists; refusing to overwrite: {dst}")
    with (src / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("scale_dtype", 0) != 0:
        raise SystemExit("source scale_dtype must be 0 (fp32)")
    entries = sorted(src.iterdir())
    converted: list[Path] = []
    linked: list[Path] = []
    unmatched: list[str] = []
    old_bytes = 0
    for entry in entries:
        if entry.name == "manifest.json":
            continue
        is_scale = entry.name.endswith("_scale.bin")
        paired = is_scale and (src / (entry.name[:-10] + "_q4.bin")).is_file()
        if paired:
            if not entry.is_file() or entry.stat().st_size % 4:
                raise SystemExit(f"invalid fp32 scale file: {entry}")
            converted.append(entry)
            old_bytes += entry.stat().st_size
            continue
        linked.append(entry)
        if is_scale:
            unmatched.append(entry.name)
    print(f"mode={'dry-run' if dry_run else 'convert'} converted={len(converted)} "
          f"symlink={len(linked)} unmatched={len(unmatched)} "
          f"source_scale_bytes={old_bytes} converted_scale_bytes={old_bytes // 2}")
    print("unmatched_scale_files=" + json.dumps(unmatched, ensure_ascii=False))
    print(f"dst={dst} manifest_scale_dtype=1")
    if dry_run:
        return
    dst.mkdir(parents=True, exist_ok=False)
    for entry in converted:
        values = np.fromfile(entry, dtype=np.float32)
        with (dst / entry.name).open("xb") as handle:
            values.astype(np.float16).tofile(handle)
    for entry in linked:
        (dst / entry.name).symlink_to(entry)
    manifest["scale_dtype"] = 1
    with (dst / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("conversion complete; source files preserved")


if __name__ == "__main__":
    main()
