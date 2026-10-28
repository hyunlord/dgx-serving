#!/usr/bin/env python3
"""Compare r14_daemon_client outputs request-by-request (tokens and END reason).

Usage:
  python tools/r14_compare_gates.py same A.json B.json [C.json ...]
      every request id must have identical tokens and END reason in all files
  python tools/r14_compare_gates.py solo MIXED.json RID:SOLO.json:SOLO_RID [...]
      request RID of MIXED must equal request SOLO_RID of SOLO (tokens and END reason);
      append ':prefix' to require only that MIXED's tokens are a prefix of SOLO's (cancelled requests)
Prints one line per comparison and a final PASS/FAIL; exit status 1 on any mismatch.
"""
import json, sys


def load(path):
    return json.load(open(path))["requests"]


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    mode, args, ok = sys.argv[1], sys.argv[2:], True
    if mode == "same":
        ref = load(args[0])
        for other in args[1:]:
            cur = load(other)
            for rid in sorted(set(ref) | set(cur), key=int):
                a, b = ref.get(rid, {}), cur.get(rid, {})
                d = first_diff(a.get("tokens", []), b.get("tokens", []))
                same = d is None and a.get("end") == b.get("end")
                ok &= same
                print(f"{'PASS' if same else 'FAIL'} req={rid} {args[0]} vs {other} n={len(a.get('tokens', []))}/{len(b.get('tokens', []))} "
                      f"end={a.get('end')}/{b.get('end')} first_divergence={d}")
    elif mode == "solo":
        mixed = load(args[0])
        for spec in args[1:]:
            parts = spec.split(":")
            rid, solo_path, solo_rid = parts[0], parts[1], parts[2]
            prefix = len(parts) > 3 and parts[3] == "prefix"
            a, b = mixed[rid], load(solo_path)[solo_rid]
            ta, tb = a.get("tokens", []), b.get("tokens", [])
            if prefix:
                same = tb[:len(ta)] == ta
                d = first_diff(ta, tb[:len(ta)])
            else:
                d = first_diff(ta, tb)
                same = d is None and a.get("end") == b.get("end")
            ok &= same
            print(f"{'PASS' if same else 'FAIL'} mixed req={rid} vs {solo_path} req={solo_rid} {'prefix ' if prefix else ''}"
                  f"n={len(ta)}/{len(tb)} end={a.get('end')}/{b.get('end')} first_divergence={d}")
    else:
        sys.exit("mode must be same or solo")
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
