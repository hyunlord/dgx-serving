#!/usr/bin/env python3
"""spec daemon 다중 슬롯 throughput 벤치(daemon-직결·태그드 프로토콜·client-observed).
verify concat 분할상환 A/B용: 같은 워크로드 N개를 동시 주입 → aggregate tok/s(전 요청 토큰합/벽시계)와
요청별 t/s·TTFT. daemon 1회 로드로 wl×N 격자 순회(각 조합 rounds회 — 마지막 round가 warm 측정). 사용:
  python3 tools/bench_spec_multi.py <daemon_bin> [Ns=1,2,4] [wls=code,chat,prose] [max_tokens=96] [rounds=2]
프롬프트=3관문 wl_*_ids.bin(실 채팅 템플릿). STOP_IDS 기본 serve_api와 동일.
"""
import os, subprocess, sys, threading, time, collections, queue, struct

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TW   = os.path.join(REPO, "artifacts/gemma4-12b/weights")
AW   = os.path.join(REPO, "artifacts/gemma4-12b-assistant/weights")
WLD  = os.path.join(REPO, "artifacts/gemma4-12b-assistant/golden")

BIN  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "tools/serve_daemon_gemma4_spec_online")
NS   = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4").split(",")]
WLS  = (sys.argv[3] if len(sys.argv) > 3 else "code,chat,prose").split(",")
MT   = int(sys.argv[4]) if len(sys.argv) > 4 else 96
RND  = int(sys.argv[5]) if len(sys.argv) > 5 else 2
UQ, MAXLEN, BMAX, K = "2", "384", "4", "4"

def wl_ids(name):
    b = open(os.path.join(WLD, f"wl_{name}_ids.bin"), "rb").read()
    return list(struct.unpack(f"{len(b)//4}i", b))

env = dict(os.environ); env.setdefault("STOP_IDS", "1,106,50")
proc = subprocess.Popen([BIN, TW, AW, UQ, MAXLEN, BMAX, K],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, bufsize=1, env=env)
for line in proc.stderr:
    if "ready" in line: break

out_q = collections.defaultdict(queue.Queue)
def reader():                                     # ★도착 시각을 리더에서 기록(수집 순서 아티팩트 방지)
    for line in proc.stdout:
        t = time.monotonic()
        parts = line.split()
        if len(parts) >= 2: out_q[int(parts[0])].put((t, parts[1:]))
threading.Thread(target=reader, daemon=True).start()

print(f"[bench_spec_multi] bin={os.path.basename(BIN)} Ns={NS} wls={WLS} max_tokens={MT} Bmax={BMAX} K={K}", flush=True)
rid = 100
for wl in WLS:
    ids = wl_ids(wl)
    for N in NS:
        last = None
        for r in range(RND):
            reqs = []
            t0 = time.monotonic()
            for i in range(N):
                rid += 1
                proc.stdin.write(f"{rid} {MT} " + " ".join(map(str, ids)) + "\n")
                reqs.append(rid)
            proc.stdin.flush()
            stats = {}
            for q_rid in reqs:
                ntok = 0; t_first = None
                while True:
                    t_arr, parts = out_q[q_rid].get(timeout=600)
                    if parts[0] == "END":
                        stats[q_rid] = (ntok, t_first, t_arr); break
                    if t_first is None: t_first = t_arr
                    ntok += 1
            t_end = max(v[2] for v in stats.values())
            tot = sum(v[0] for v in stats.values())
            last = (tot / (t_end - t0),
                    [v[0] / (v[2] - v[1]) if (v[1] and v[2] > v[1]) else float("nan") for v in stats.values()],
                    [(v[1] - t0) if v[1] else float("nan") for v in stats.values()], tot, t_end - t0)
        agg, per, ttft, tot, dt = last
        print(f"  wl={wl:<5} N={N}: aggregate {agg:6.1f} tok/s ({tot}tok/{dt:.2f}s) · "
              f"요청별 {' '.join(f'{p:.1f}' for p in per)} · TTFT {' '.join(f'{t:.2f}' for t in ttft)}s", flush=True)

proc.stdin.write("SHUTDOWN\n"); proc.stdin.flush()
proc.wait(timeout=30)
