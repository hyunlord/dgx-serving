#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# Run: python3 tools/evict_file_cache.py --cuda-meminfo PATH FILE_OR_DIR [...]
"""Sync and advise away only explicitly selected files, with memory evidence.

Run after all writers have finished. Files are opened read-only; fsync completes
pending writes before DONTNEED. Directory and file symlinks are followed with
inode deduplication. This neither removes files nor uses system-wide drop_caches.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


class CacheEvictionError(ValueError):
    """A selected cache target or memory snapshot cannot be safely processed."""


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    cached_bytes: int
    cuda_free_bytes: int
    cuda_total_bytes: int


@dataclass(frozen=True, slots=True)
class EvictedFiles:
    files: int
    bytes: int


@dataclass(frozen=True, slots=True)
class EvictionReport:
    targets: list[str]
    before: MemorySnapshot
    after: MemorySnapshot
    evicted: EvictedFiles


def memory_snapshot(cuda_meminfo: Path) -> MemorySnapshot:
    cached = next((int(line.split()[1]) * 1024
                   for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("Cached:")), None)
    if cached is None:
        raise CacheEvictionError("Cached missing from /proc/meminfo")
    payload = json.loads(subprocess.check_output([str(cuda_meminfo.resolve())], text=True, timeout=30))
    if not isinstance(payload, dict):
        raise CacheEvictionError("cuda_meminfo must return a JSON object")
    free, total = payload.get("free_bytes"), payload.get("total_bytes")
    if type(free) is not int or type(total) is not int or not 0 <= free <= total or total <= 0:
        raise CacheEvictionError("cuda_meminfo must report valid integer free_bytes/total_bytes")
    return MemorySnapshot(cached, free, total)


def selected_files(paths: list[Path]) -> list[Path]:
    """Resolve regular files without revisiting directory cycles or file aliases."""
    pending = list(paths)
    seen: set[tuple[int, int]] = set()
    files: list[Path] = []
    while pending:
        path = pending.pop().resolve(strict=True)
        info = path.stat()
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        if stat.S_ISDIR(info.st_mode):
            pending.extend(sorted(path.iterdir()))
        elif stat.S_ISREG(info.st_mode):
            files.append(path)
        else:
            raise CacheEvictionError(f"Not a regular file or directory: {path}")
    return sorted(files)


def evict_files(paths: list[Path]) -> EvictedFiles:
    """Sync completed writes then issue DONTNEED once per selected regular inode."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise CacheEvictionError("This platform does not provide posix_fadvise DONTNEED")
    files = selected_files(paths)
    size = 0
    for path in files:
        with path.open("rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise CacheEvictionError(f"Target changed to a non-regular file: {path}")
            os.fsync(source.fileno())
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            size += info.st_size
    return EvictedFiles(len(files), size)


def evict_with_evidence(paths: list[Path], cuda_meminfo: Path) -> EvictionReport:
    """Capture Cached/CUDA free on both sides of one scoped eviction pass."""
    before = memory_snapshot(cuda_meminfo)
    evicted = evict_files(paths)
    after = memory_snapshot(cuda_meminfo)
    return EvictionReport([str(path.resolve()) for path in paths], before, after, evicted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-meminfo", type=Path, default=Path(__file__).with_name("cuda_meminfo"))
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        report = evict_with_evidence(args.paths, args.cuda_meminfo)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"File cache eviction failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
