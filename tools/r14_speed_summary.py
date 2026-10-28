#!/usr/bin/env python3
"""Summarize guarded R14 speed runs: t/s and SM clock median inside each measured interval.

Usage: python tools/r14_speed_summary.py DIR NAME [NAME ...] > summary.json
For each NAME (probe_run_guard --name) in DIR the measured interval is taken from:
  bench   : the last DECODE_INTERVAL line in command_NAME.log (start/end epochs, token_per_second)
  llama   : lines_NAME.jsonl — the llama-bench jsonl result line; interval = [t_line - n_gen/avg_ts, t_line]
            (n_gen and avg_ts from that line; this assumes the generation run ends just before printing)
  daemon  : out_NAME.json from r14_daemon_client (first..last token arrival, t/s after the first token)
Clock = median of clockmon_NAME.csv samples inside the interval (tools/r14_clock_window.py).
Guard status, resident processes and unexpected processes come from result_NAME.json.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r14_clock_window import window


def one(d, name):
    res = json.load(open(os.path.join(d, f"result_{name}.json")))
    row = {"name": name, "guard_status": res.get("status"), "exit": res.get("exit_code"),
           "unexpected_processes": res.get("unexpected_processes"),
           "baseline_processes": [p["name"] for p in res.get("baseline", [])]}
    log = os.path.join(d, f"command_{name}.log")
    text = open(log).read() if os.path.exists(log) else ""
    m = re.findall(r"DECODE_INTERVAL .*?start=(\S+) end=(\S+) steps=(\d+) \S+ token_per_second=(\S+)(.*)", text)
    out = os.path.join(d, f"out_{name}.json")
    if m:
        s, e, n, ts, rest = m[-1]
        row.update(kind="bench", start=float(s), end=float(e), steps=int(n), tok_s=float(ts), detail=rest.strip())
    elif os.path.exists(out):
        r = json.load(open(out))["requests"]["1"]
        row.update(kind="daemon", start=r["decode_start_epoch"], end=r["decode_end_epoch"],
                   steps=len(r["tokens"]) - 1, tok_s=r["decode_tok_s_after_first"], prompt_tokens=r["prompt_tokens"])
    else:
        for ln in open(os.path.join(d, f"lines_{name}.jsonl")):
            j = json.loads(ln)
            if j.get("phase") == "benchmark" and j["line"].startswith("{") and "avg_ts" in j["line"]:
                lb = json.loads(j["line"])
                dur = lb["n_gen"] / lb["avg_ts"]
                row.update(kind="llama", start=j["epoch_seconds"] - dur, end=j["epoch_seconds"], steps=lb["n_gen"],
                           tok_s=lb["avg_ts"], n_depth=lb.get("n_depth"), stddev_ts=lb.get("stddev_ts"),
                           samples_ts=lb.get("samples_ts"))
    if "start" in row:
        row["interval_s"] = row["end"] - row["start"]
        row["clock"] = window(os.path.join(d, f"clockmon_{name}.csv"), row["start"], row["end"])
    return row


if __name__ == "__main__":
    d = sys.argv[1]
    print(json.dumps([one(d, n) for n in sys.argv[2:]], indent=1))
