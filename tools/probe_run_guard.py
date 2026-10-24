#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# Run: python3 tools/probe_run_guard.py --name NAME --log-dir DIR --expected-bytes N -- COMMAND ...
"""Serialize one caller-owned GPU measurement with memory and resident-process evidence.

Exit 86: resident memory increased; 87: unexpected GPU process; 88: memory
budget rejected; 89: monitor/evidence failure. Otherwise preserve command status.
The caller owns retry policy and must not run two instances concurrently.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TextIO

GPU_QUERY: Final = "--query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm"
APP_QUERY: Final = "--query-compute-apps=pid,process_name,used_memory"
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
    if process.pid == 4363:
        return process.name.endswith("/Xorg")
    if process.pid == 4681:
        return process.name.endswith("/gnome-shell")
    if process.pid == 2278032:
        return "/ComfyUI/" in process.name and Path(process.name).name.startswith("python")
    return False


def available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable missing from /proc/meminfo")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--expected-bytes", required=True, type=int)
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
        result.update(available_bytes=available, limit_bytes=available * 7 // 10,
                      baseline=[asdict(p) for p in baseline])
        (args.log_dir / f"baseline_{args.name}.json").write_text(json.dumps(result, indent=2))
        if any(not approved(p) for p in baseline):
            result["status"] = "unexpected_baseline_process"
            status = 87
            return status
        if args.expected_bytes > available * 7 // 10:
            result["status"] = "memory_budget_rejected"
            status = 88
            return status
        for path, query, interval in ((gpu_path, GPU_QUERY, "1"), (apps_path, APP_QUERY, "5")):
            handle = path.open("x")
            handles.append(handle)
            monitors.append(subprocess.Popen(["nvidia-smi", query, "--format=csv", "-l", interval],
                                             stdout=handle, stderr=subprocess.STDOUT, text=True))
        deadline = time.monotonic() + 10
        while not (gpu_path.stat().st_size and apps_path.stat().st_size):
            if any(p.poll() is not None for p in monitors) or time.monotonic() >= deadline:
                raise ValueError("Monitor startup failed or produced no data within 10 seconds")
            time.sleep(0.05)
        validate_gpu_log(gpu_path)
        if not app_samples(apps_path):
            raise ValueError("Application monitor did not capture resident baseline")
        with (args.log_dir / f"command_{args.name}.log").open("x") as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True)
            owned = {child.pid}
            while True:
                descendants(owned)
                if child.poll() is not None:
                    break
                time.sleep(0.2)
            command_status = child.wait()
        monitor_failed = any(p.poll() is not None for p in monitors)
        for monitor in monitors:
            stop_monitor(monitor)
        for handle in handles:
            handle.close()
        with apps_path.open("a") as output:
            output.write(capture("nvidia-smi", APP_QUERY, "--format=csv"))
        final = gpu_processes()
        validate_gpu_log(gpu_path)
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
        status = 89 if monitor_failed else 87 if unexpected else 86 if increased else command_status
        result["status"] = ("monitor_failed" if monitor_failed else "unexpected_process" if unexpected
                            else "contaminated" if increased else "clean" if status == 0 else "command_failed")
        return status
    except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        return 89
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
