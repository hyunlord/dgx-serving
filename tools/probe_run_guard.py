#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# Run: python3 tools/probe_run_guard.py --name NAME --log-dir DIR --expected-bytes N -- COMMAND ...
"""Serialize one caller-owned GPU measurement with memory and resident-process evidence.

Exit 86: resident memory increased; 87: unexpected GPU process; 88: memory
budget rejected; 89: monitor/evidence failure; 90: clock gate rejected.
The caller owns retry policy and must not run two instances concurrently.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Final, TextIO

GPU_QUERY: Final = "--query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm"
APP_QUERY: Final = "--query-compute-apps=pid,process_name,used_memory"
CLOCK_QUERY: Final = ("--query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,"
                      "temperature.gpu,clocks_throttle_reasons.active")
# NVIDIA nvml.h: https://github.com/NVIDIA/go-nvml/blob/main/pkg/nvml/nvml.h
HW_CLOCK_MASK: Final = 0x8 | 0x40 | 0x80  # HW slowdown, thermal slowdown, power brake.
CONDITION: Final = "ComfyUI 상주(유휴, 3.2GB)"


@dataclass(frozen=True, slots=True)
class GPUProcess:
    pid: int
    name: str
    memory_mib: int


def capture(*command: str) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)


def memory_number(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*(?:MiB)?\s*", value)
    if match is None:
        raise ValueError(f"Per-process GPU memory unavailable: {value!r}")
    return int(match[1])


def gpu_processes() -> list[GPUProcess]:
    root = ET.fromstring(capture("nvidia-smi", "-q", "-x"))
    return [GPUProcess(int(p.findtext("pid", "")), p.findtext("process_name", ""),
                       memory_number(p.findtext("used_memory", "")))
            for p in root.findall(".//process_info")]


def approved(process: GPUProcess) -> bool:
    if process.pid == 5292:
        return process.name.endswith("/Xorg")
    if process.pid == 5349:
        return process.name.endswith("/gnome-shell")
    if process.pid == 2278032:
        return "/ComfyUI/" in process.name and Path(process.name).name.startswith("python")
    if process.pid == 3557110:
        return "/research/make-meme-b/" in process.name and Path(process.name).name.startswith("python")
    return False


def available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable missing from /proc/meminfo")


@dataclass(frozen=True, slots=True)
class CUDAMemory:
    free_bytes: int
    total_bytes: int


def cuda_memory(executable: Path) -> CUDAMemory:
    """Read the helper's strict byte counts; malformed evidence fails closed."""
    payload = json.loads(capture(str(executable.resolve())))
    message = "CUDA memory helper must return integer free_bytes/total_bytes with 0 <= free <= total"
    if not isinstance(payload, dict):
        raise ValueError(message)
    free, total = payload.get("free_bytes"), payload.get("total_bytes")
    if type(free) is not int or type(total) is not int:
        raise ValueError(message)
    if not 0 <= free <= total or total <= 0:
        raise ValueError(message)
    return CUDAMemory(free, total)


def eviction_targets(command: list[str], weights: list[Path], references: list[Path],
                     checkpoints: list[Path], artifacts: Path) -> list[Path]:
    """Collect only run-specific inputs; PPL references must stay under artifacts."""
    selected_weights, selected_refs = list(weights), list(references)
    for index, argument in enumerate(command):
        if argument in ("--weights", "--kld-ref"):
            if index + 1 >= len(command):
                raise ValueError(f"Missing path after {argument}")
            target = Path(command[index + 1])
            (selected_weights if argument == "--weights" else selected_refs).append(target)
        elif argument.startswith("--weights="):
            selected_weights.append(Path(argument.split("=", 1)[1]))
        elif argument.startswith("--kld-ref="):
            selected_refs.append(Path(argument.split("=", 1)[1]))
    root = artifacts.resolve()
    for target in selected_refs:
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"PPL reference cache target is outside artifacts: {target}")
    return list(dict.fromkeys(path.resolve(strict=True)
                             for path in selected_weights + selected_refs + checkpoints))


def descendants(owned: set[int]) -> None:
    pairs = [tuple(map(int, line.split())) for line in capture("ps", "-eo", "pid=,ppid=").splitlines()]
    while True:
        additions = {pid for pid, parent in pairs if parent in owned} - owned
        if not additions:
            return
        owned.update(additions)


def app_samples(path: Path) -> list[GPUProcess]:
    samples: list[GPUProcess] = []
    for row in csv.reader(path.read_text().splitlines()):
        if not row or row[0].strip() == "pid":
            continue
        if len(row) != 3 or not row[0].strip().isdigit():
            raise ValueError(f"Invalid application monitor row: {row!r}")
        samples.append(GPUProcess(int(row[0]), row[1].strip(), memory_number(row[2])))
    return samples


def validate_gpu_log(path: Path) -> None:
    rows = list(csv.reader(path.read_text().splitlines()))
    data = [r for r in rows if r and r[0].strip() != "timestamp"]
    if not data:
        raise ValueError("GPU monitor contains no samples")
    for row in data:
        if len(row) != 4 or not row[1].strip().endswith("%"):
            raise ValueError(f"Invalid GPU monitor row: {row!r}")
        float(row[1].strip().removesuffix("%"))
    # GB10 reports memory.used as [N/A]; retain it verbatim, never invent a value.


def stop_monitor(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def clock_samples(path: Path) -> list[tuple[float, int, int, int]]:
    samples: list[tuple[float, int, int, int]] = []
    for row in csv.reader(path.read_text().splitlines()):
        if not row or row[0].strip() == "timestamp":
            continue
        if len(row) != 6:
            raise ValueError(f"Invalid clock monitor row: {row!r}")
        stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
        clock = int(row[1].strip().removesuffix(" MHz"))
        max_clock = int(row[2].strip().removesuffix(" MHz"))
        mask = int(row[5].strip(), 0)
        samples.append((stamp, clock, mask, max_clock))
    if not samples:
        raise ValueError("Clock monitor contains no valid samples")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--cuda-meminfo", type=Path,
                        default=Path(__file__).resolve().with_name("cuda_meminfo"))
    parser.add_argument("--evict-weights", action="append", type=Path, default=[],
                        help="Run-specific weights to evict once before the memory gate (repeatable)")
    parser.add_argument("--evict-ppl-ref", action="append", type=Path, default=[],
                        help="Existing PPL reference under repository artifacts (repeatable)")
    parser.add_argument("--evict-hf-checkpoint", action="append", type=Path, default=[],
                        help="Exact last-used HF checkpoint, never the entire HF cache (repeatable)")
    parser.add_argument("--clock-gate", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command: list[str] = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.expected_bytes <= 0 or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error("Require command, positive expected bytes and a safe measurement name")
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.log_dir / f"result_{args.name}.json"
    # Refuse reusing an evidence name; retries must get a new name.
    with result_path.open("x") as initial:
        json.dump({"status": "starting", "command": command}, initial)
    gpu_path = args.log_dir / f"gpumon_{args.name}.csv"
    apps_path = args.log_dir / f"apps_{args.name}.csv"
    clock_path = args.log_dir / f"clockmon_{args.name}.csv"
    monitors: list[subprocess.Popen[str]] = []
    handles: list[TextIO] = []
    child: subprocess.Popen[str] | None = None
    status = 89
    result = {"condition": CONDITION, "command": command, "expected_bytes": args.expected_bytes,
              "started_unix": time.time(), "status": "evidence_failure"}
    try:
        free = capture("free", "-h")
        (args.log_dir / f"memory_{args.name}.txt").write_text(free)
        available = available_bytes()
        baseline = gpu_processes()
        result.update(available_bytes=available,
                      baseline=[asdict(p) for p in baseline])
        (args.log_dir / f"baseline_{args.name}.json").write_text(json.dumps(result, indent=2))
        if any(not approved(p) for p in baseline):
            result["status"] = "unexpected_baseline_process"
            status = 87
            return status
        # Check the allowlist before the helper creates its transient CUDA context.
        targets = eviction_targets(command, args.evict_weights, args.evict_ppl_ref,
                                   args.evict_hf_checkpoint, Path(__file__).resolve().parents[1] / "artifacts")
        if targets:
            eviction = json.loads(subprocess.check_output(
                [sys.executable, str(Path(__file__).with_name("evict_file_cache.py")),
                 "--cuda-meminfo", str(args.cuda_meminfo), *map(str, targets)],
                text=True, timeout=600))
            result["cache_eviction"] = eviction
            (args.log_dir / f"evict_{args.name}.json").write_text(json.dumps(eviction, indent=2) + "\n")
        else:
            result["cache_eviction"] = {"targets": [], "status": "no_run_specific_targets"}
        memory = cuda_memory(args.cuda_meminfo)
        limit = memory.free_bytes * 8 // 10
        result.update(cuda_free_bytes=memory.free_bytes, cuda_total_bytes=memory.total_bytes,
                      limit_bytes=limit, memory_budget_basis="cuda_free_bytes",
                      cuda_meminfo=str(args.cuda_meminfo))
        (args.log_dir / f"baseline_{args.name}.json").write_text(json.dumps(result, indent=2))
        if args.expected_bytes > limit:
            result["status"] = "memory_budget_rejected"
            status = 88
            return status
        for path, query, flag, interval in ((gpu_path, GPU_QUERY, "-l", "1"),
                (apps_path, APP_QUERY, "-l", "5"), (clock_path, CLOCK_QUERY, "-lms", "100")):
            handle = path.open("x")
            handles.append(handle)
            monitors.append(subprocess.Popen(["nvidia-smi", query, "--format=csv", flag, interval],
                                             stdout=handle, stderr=subprocess.STDOUT, text=True))
        deadline = time.monotonic() + 10
        while not all(p.stat().st_size for p in (gpu_path, apps_path, clock_path)):
            if any(p.poll() is not None for p in monitors) or time.monotonic() >= deadline:
                raise ValueError("Monitor startup failed or produced no data within 10 seconds")
            time.sleep(0.05)
        validate_gpu_log(gpu_path)
        clock_samples(clock_path)
        if not app_samples(apps_path):
            header = "pid, process_name, used_gpu_memory [MiB]"
            rows = [row.strip() for row in apps_path.read_text().splitlines() if row.strip()]
            current = gpu_processes()
            graphics_only = all(p.name in ("/usr/lib/xorg/Xorg", "/usr/bin/gnome-shell") and approved(p) for p in current)
            if rows != [header] or not graphics_only or current != baseline:
                raise ValueError("Application monitor did not capture resident baseline")
            result["empty_compute_baseline_verified"] = True
        if args.clock_gate:
            (args.log_dir / f"clock_before_{args.name}.txt").write_text(capture("nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE"))
        during = False
        with (args.log_dir / f"command_{args.name}.log").open("x") as output:
            command_start = time.time()
            monotonic_start = time.monotonic()
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True)
            owned = {child.pid}
            while True:
                descendants(owned)
                if child.poll() is not None:
                    break
                if args.clock_gate and not during and time.monotonic() - monotonic_start >= 10:
                    (args.log_dir / f"clock_during_{args.name}.txt").write_text(capture("nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE"))
                    during = True
                time.sleep(0.2)
            command_status = child.wait()
            command_end = time.time()
        result.update(command_start_unix=command_start, command_end_unix=command_end)
        if args.clock_gate:
            (args.log_dir / f"clock_after_{args.name}.txt").write_text(capture("nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE"))
        monitor_failed = any(p.poll() is not None for p in monitors)
        for monitor in monitors:
            stop_monitor(monitor)
        for handle in handles:
            handle.close()
        with apps_path.open("a") as output:
            output.write(capture("nvidia-smi", APP_QUERY, "--format=csv"))
        final = gpu_processes()
        validate_gpu_log(gpu_path)
        interval_start, interval_end = command_start, command_end
        if args.clock_gate:
            log = (args.log_dir / f"command_{args.name}.log").read_text()
            markers = dict(re.findall(r"interval=load edge=(start|end)[^\n]*epoch_seconds=([0-9.eE+-]+)", log))
            if set(markers) != {"start", "end"}:
                raise ValueError("Clock gate requires load start/end timestamp markers")
            interval_start, interval_end = float(markers["start"]), float(markers["end"])
            if not command_start <= interval_start < interval_end <= command_end or interval_end - interval_start < 30:
                raise ValueError("Clock gate requires an actual load interval of at least 30 seconds")
        result.update(clock_interval_start_unix=interval_start, clock_interval_end_unix=interval_end)
        clock_rows = [r for r in clock_samples(clock_path) if interval_start <= r[0] <= interval_end]
        if not clock_rows or (args.clock_gate and not during):
            raise ValueError("Missing in-command clock samples or during-load CLOCK,PERFORMANCE snapshot")
        clocks = [r[1] for r in clock_rows]
        blocked_clock = median(clocks) < 1800 or any(r[2] & HW_CLOCK_MASK for r in clock_rows)
        result.update(clock_samples=len(clocks), clock_min_mhz=min(clocks), clock_max_mhz=max(clocks),
                      clock_median_mhz=median(clocks), clock_distribution=dict(Counter(clocks)),
                      clock_mask_distribution=dict(Counter(hex(r[2]) for r in clock_rows)),
                      clock_reported_max_mhz=max(r[3] for r in clock_rows),
                      clock_gate_enabled=args.clock_gate, clock_gate_failed=args.clock_gate and blocked_clock)
        samples = app_samples(apps_path) + final
        baseline_by_pid = {p.pid: p for p in baseline}
        unexpected = [asdict(p) for p in samples if p.pid not in owned and
                      (p.pid not in baseline_by_pid or p.name != baseline_by_pid[p.pid].name)]
        increased = [asdict(p) for p in samples if p.pid in baseline_by_pid
                     and p.memory_mib > baseline_by_pid[p.pid].memory_mib]
        result.update(command_exit=command_status, owned_pids=sorted(owned), final=[asdict(p) for p in final],
                      resident_memory_increases=increased, unexpected_processes=unexpected,
                      monitors_failed=monitor_failed, sample_count=len(samples) - len(final))
        if any(p.pid in owned for p in final):
            raise ValueError("Measurement descendants still have GPU allocations after command exit")
        status = (89 if monitor_failed else 87 if unexpected else 86 if increased
                  else 90 if args.clock_gate and blocked_clock else command_status)
        result["status"] = ("monitor_failed" if monitor_failed else "unexpected_process" if unexpected
                            else "contaminated" if increased else "clock_gate_failed" if status == 90
                            else "clean" if status == 0 else "command_failed")
        return status
    except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        status = 89
        return status
    finally:
        if child is not None and child.poll() is None:
            # Never strand a GPU workload if evidence collection itself fails.
            result["command_exit"] = child.wait()
        for monitor in monitors:
            stop_monitor(monitor)
        for handle in handles:
            handle.close()
        result["ended_unix"] = time.time()
        result["exit_code"] = status
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
