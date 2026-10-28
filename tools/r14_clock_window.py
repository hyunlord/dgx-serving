#!/usr/bin/env python3
"""Median SM clock of the 100 ms nvidia-smi samples inside [start, end] epochs.

Usage: python tools/r14_clock_window.py CLOCKMON.csv START_EPOCH END_EPOCH
Prints JSON {samples, median_mhz, min_mhz, max_mhz, nonzero_reason_samples}. Timestamps in
the CSV are local time of the measuring host (this script must run on that host).
"""
import csv, json, statistics, sys
from datetime import datetime


def window(path, start, end):
    vals, reasons = [], 0
    with open(path) as f:
        rows = csv.reader(f)
        next(rows)
        for r in rows:
            if len(r) < 6:
                continue
            try:
                t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                mhz = int(r[1].split()[0])
            except ValueError:
                continue
            if start <= t <= end:
                vals.append(mhz)
                reasons += int(r[5].strip(), 16) != 0
    if not vals:
        return {"samples": 0, "median_mhz": None}
    return {"samples": len(vals), "median_mhz": statistics.median(vals), "min_mhz": min(vals),
            "max_mhz": max(vals), "nonzero_reason_samples": reasons}


if __name__ == "__main__":
    print(json.dumps(window(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))))
