#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# Run: python3 tests/test_evict_file_cache.py
"""CPU-only file-cache and scoped measurement guard regression checks."""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


class FileCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((TOOLS / "evict_file_cache.py").is_file(), "cache eviction tool required")
        self.cache = importlib.import_module("evict_file_cache")
        self.guard = importlib.import_module("probe_run_guard")

    def test_recursive_symlink_and_hardlink_files_are_synced_once(self) -> None:
        # Given: a recursive directory, two aliases and a directory cycle.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            data = sub / "weights.bin"
            data.write_bytes(b"unchanged weights")
            (root / "alias.bin").symlink_to(data)
            os.link(data, root / "hardlink.bin")
            (sub / "cycle").symlink_to(root, target_is_directory=True)
            events: list[str] = []
            before = data.stat().st_mtime_ns
            # When: completed writes are synced before per-file advice.
            with patch.object(os, "fsync", side_effect=lambda fd: events.append("sync")), \
                 patch.object(os, "posix_fadvise", create=True, side_effect=lambda *args: events.append("advice")), \
                 patch.object(os, "POSIX_FADV_DONTNEED", 4, create=True):
                result = self.cache.evict_files([root, data])
            # Then: aliases do not multiply advice and contents/mtime remain intact.
            self.assertEqual(events, ["sync", "advice"])
            self.assertEqual(result.files, 1)
            self.assertEqual(result.bytes, len(b"unchanged weights"))
            self.assertEqual(data.read_bytes(), b"unchanged weights")
            self.assertEqual(data.stat().st_mtime_ns, before)

    def test_missing_path_and_special_file_fail_without_advice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fifo = root / "pipe"
            os.mkfifo(fifo)
            with patch.object(os, "posix_fadvise", create=True) as advice, \
                 patch.object(os, "POSIX_FADV_DONTNEED", 4, create=True):
                with self.assertRaises(OSError):
                    self.cache.evict_files([root / "missing"])
                with self.assertRaises(self.cache.CacheEvictionError):
                    self.cache.evict_files([fifo])
                advice.assert_not_called()

    def test_sync_failure_does_not_discard_dirty_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "written.bin"
            data.write_bytes(b"keep")
            with patch.object(os, "fsync", side_effect=OSError("sync failed")), \
                 patch.object(os, "posix_fadvise", create=True) as advice, \
                 patch.object(os, "POSIX_FADV_DONTNEED", 4, create=True):
                with self.assertRaises(OSError):
                    self.cache.evict_files([data])
                advice.assert_not_called()

    def test_guard_infers_inputs_and_rejects_reference_outside_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            weights, reference, hf = artifacts / "weights", artifacts / "ref", root / "hf"
            for path in (weights, reference, hf):
                path.mkdir(parents=True)
            command = ["env", "X=1", "ppl", "--weights", str(weights), "--kld-ref", str(reference)]
            paths = self.guard.eviction_targets(command, [], [], [hf], artifacts)
            self.assertEqual(set(paths), {weights.resolve(), reference.resolve(), hf.resolve()})
            with self.assertRaises(ValueError):
                self.guard.eviction_targets(["ppl", "--kld-ref", str(hf)], [], [], [], artifacts)


    def test_guard_uses_post_eviction_cuda_free_and_keeps_eighty_percent_limit(self) -> None:
        # Given: eviction reports recovery, but the requested budget still exceeds 80%.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights"
            weights.mkdir()
            log_dir = root / "logs"
            report = {"before": {"cuda_free_bytes": 50}, "after": {"cuda_free_bytes": 100}}
            events: list[str] = []
            def eviction_output(*args, **kwargs):
                events.append("evict")
                return json.dumps(report)
            def fresh_cuda(path):
                events.append("cuda")
                return self.guard.CUDAMemory(100, 200)
            arguments = ["guard", "--name", "budget", "--log-dir", str(log_dir),
                         "--expected-bytes", "81", "--evict-weights", str(weights), "--", "true"]
            # When: the guard runs preflight without launching the workload.
            with patch.object(sys, "argv", arguments), \
                 patch.object(self.guard, "capture", return_value="free output"), \
                 patch.object(self.guard, "available_bytes", return_value=1000), \
                 patch.object(self.guard, "gpu_processes", return_value=[]), \
                 patch.object(self.guard.subprocess, "check_output", side_effect=eviction_output), \
                 patch.object(self.guard, "cuda_memory", side_effect=fresh_cuda), \
                 patch.object(self.guard.subprocess, "Popen") as launch:
                status = self.guard.main()
                launch.assert_not_called()
            # Then: evidence records the new count, and 81 bytes is rejected against 80.
            result = json.loads((log_dir / "result_budget.json").read_text())
            self.assertEqual(status, 88)
            self.assertEqual(events, ["evict", "cuda"])
            self.assertEqual(result["limit_bytes"], 80)
            self.assertEqual(result["cache_eviction"], report)


if __name__ == "__main__":
    unittest.main()
