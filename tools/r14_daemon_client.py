#!/usr/bin/env python3
"""Drive serve_daemon_gemma4_online over its tagged stdin/stdout protocol (R14 serving gates).

Usage:
  python tools/r14_daemon_client.py --daemon BIN --weights DIR [--maxlen 2048 --bmax 8 --uq 2]
         --plan PLAN.json --out RESULT.json [--stop-ids 1]

PLAN.json is a list of actions executed in order of their trigger:
  {"op": "send",   "rid": 1, "prompt": "short.bin", "max_tokens": 128,
   "smpl": [temp, top_p, rep_penalty, seed] (optional), "when": TRIGGER}
  {"op": "cancel", "rid": 1, "when": TRIGGER}
TRIGGER: "start" | {"tokens": [rid, n]} (after rid's n-th token arrives) | {"end": rid}
Prompt paths are int32 token files, relative to the plan file. The result records,
per request, the token list, END reason, client arrival epochs of every token and
the daemon stderr lines (PATH/FASTPATH counters). The daemon environment is inherited.
"""
import argparse, json, os, struct, subprocess, sys, threading, time


def load_ids(path):
    data = open(path, "rb").read()
    return list(struct.unpack(f"<{len(data)//4}i", data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--uq", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--bmax", type=int, default=8)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-ids", default="1")  # serve_api default tokenizer generation_config eos
    ap.add_argument("--assistant", default="")  # R17: drafter weights dir (5th daemon argument)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    base = os.path.dirname(os.path.abspath(a.plan))
    env = {**os.environ, "STOP_IDS": a.stop_ids}
    cmd = [a.daemon, a.weights, str(a.uq), str(a.maxlen), str(a.bmax)] + ([a.assistant] if a.assistant else [])
    p = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    err_lines, ready = [], threading.Event()

    def pump_err():
        for line in p.stderr:
            err_lines.append([time.time(), line.rstrip("\n")])
            if "[online] ready" in line:
                ready.set()
        ready.set()
    threading.Thread(target=pump_err, daemon=True).start()
    if not ready.wait(1800) or p.poll() is not None:
        sys.exit("daemon failed to become ready")

    reqs, pending = {}, list(plan)

    def fire(trigger_ok):
        nonlocal pending
        keep = []
        for act in pending:
            if trigger_ok(act["when"]):
                if act["op"] == "send":
                    ids = load_ids(os.path.join(base, act["prompt"]))
                    smpl = act.get("smpl")
                    head = f'{act["rid"]} {act["max_tokens"]} '
                    if smpl:
                        head += "SMPL {:g} {:g} {:g} {} ".format(*smpl)
                    reqs[act["rid"]] = {"prompt": act["prompt"], "prompt_tokens": len(ids),
                                        "max_tokens": act["max_tokens"], "smpl": smpl,
                                        "sent": time.time(), "tokens": [], "times": [], "end": None}
                    p.stdin.write(head + " ".join(map(str, ids)) + "\n")
                elif act["op"] == "cancel":
                    reqs.setdefault(act["rid"], {})["cancel_sent"] = time.time()
                    p.stdin.write(f'CANCEL {act["rid"]}\n')
                p.stdin.flush()
            else:
                keep.append(act)
        changed = len(keep) != len(pending)
        pending = keep
        return changed

    def met(when, ev):
        if when == "start":
            return True
        if "tokens" in when:
            rid, n = when["tokens"]
            return rid in reqs and len(reqs[rid]["tokens"]) >= n
        if "end" in when:
            return when["end"] in reqs and reqs[when["end"]]["end"] is not None
        return False

    while fire(lambda w: met(w, None)):
        pass
    while any(r.get("end") is None for r in reqs.values() if "tokens" in r) or pending:
        line = p.stdout.readline()
        if not line:
            break
        now = time.time()
        parts = line.split()
        rid = int(parts[0])
        r = reqs[rid]
        if parts[1] == "END":
            r["end"] = parts[2] if len(parts) > 2 else ""
            r["end_time"] = now
        else:
            r["tokens"].append(int(parts[1]))
            r["times"].append(now)
        while fire(lambda w: met(w, None)):
            pass
    p.stdin.write("SHUTDOWN\n"); p.stdin.flush()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.terminate()
    for r in reqs.values():
        t = r.get("times") or []
        if len(t) >= 2:
            r["decode_tok_s_after_first"] = (len(t) - 1) / (t[-1] - t[0])
            r["decode_start_epoch"], r["decode_end_epoch"] = t[0], t[-1]
    with open(a.out, "x") as f:
        json.dump({"command": cmd, "env": {k: os.environ[k] for k in sorted(os.environ)
                                           if k.startswith(("G4_", "GEMV_", "STOP_IDS"))},
                   "requests": {str(k): v for k, v in reqs.items()}, "daemon_stderr": err_lines,
                   "unfired": pending}, f)
    print(json.dumps({str(k): {"n": len(v.get("tokens", [])), "end": v.get("end"),
                               "tok_s": v.get("decode_tok_s_after_first")} for k, v in reqs.items()}))


if __name__ == "__main__":
    main()
