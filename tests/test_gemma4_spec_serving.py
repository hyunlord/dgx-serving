#!/usr/bin/env python3
"""draft-spec 서빙 게이트(4관문) — serve_daemon_gemma4_spec_online.
[1] ★서빙 spec==greedy: spec daemon 출력 == plain daemon(serve_daemon_gemma4_online) 출력 토큰-동일
    (같은 프롬프트·같은 max_tokens·같은 온디바이스 rope — 사이클 e2e bit-id가 서빙 경로에서도 유지).
[2] 동시 격리: 동시 N요청 각각 == 자기 단독(solo) 출력 동일(슬롯별 순차 사이클 = 격리 구성적 — e2e 검증).
[3] stop/length 종결 정상 + 생존(크래시 없음).
front-style 연속 reader(test_spec_online 패턴). raw ID(토크나이저 불요) — 프롬프트는 3관문 wl_*_ids.bin(실 채팅 템플릿).
사용법: python3 tests/test_gemma4_spec_serving.py
"""
import collections, os, queue, struct, subprocess, sys, threading

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TW   = os.path.join(REPO, "artifacts/gemma4-12b/weights")
AW   = os.path.join(REPO, "artifacts/gemma4-12b-assistant/weights")
WLD  = os.path.join(REPO, "artifacts/gemma4-12b-assistant/golden")
PLAIN= os.path.join(REPO, "tools", "serve_daemon_gemma4_online")
SPEC = os.path.join(REPO, "tools", "serve_daemon_gemma4_spec_online")
UQ, MAXLEN, BMAX, K, MAXTOK = "2", "256", "3", "4", 48
STOP_IDS = os.environ.get("STOP_IDS", "1,106,50")     # gemma generation_config(serve_api와 동일 주입)

def wl_ids(name):
    with open(os.path.join(WLD, f"wl_{name}_ids.bin"), "rb") as f:
        b = f.read()
    return list(struct.unpack(f"{len(b)//4}i", b))

class Daemon:
    def __init__(self, argv, tag):
        env = {**os.environ, "STOP_IDS": STOP_IDS}
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        for line in self.p.stderr:
            print(f"  [{tag}] {line.rstrip()}")
            if "ready" in line: break
            if self.p.poll() is not None: raise RuntimeError(f"{tag} 起動 실패")
        self.q = collections.defaultdict(queue.Queue)
        self.reason = {}
        threading.Thread(target=self._rd, daemon=True).start()
    def _rd(self):
        for line in self.p.stdout:
            parts = line.split()
            if len(parts) < 2: continue
            rid = int(parts[0])
            if parts[1] == "END":
                self.reason[rid] = parts[2] if len(parts) > 2 else "?"
                self.q[rid].put(None)
            else:
                self.q[rid].put(int(parts[1]))
        for qq in self.q.values(): qq.put(None)
    def send(self, rid, ids, maxtok=MAXTOK):
        self.p.stdin.write(f"{rid} {maxtok} " + " ".join(map(str, ids)) + "\n"); self.p.stdin.flush()
    def collect(self, rid):
        out = []
        while True:
            t = self.q[rid].get(timeout=600)
            if t is None: break
            out.append(t)
        return out
    def close(self):
        try:
            self.p.stdin.write("SHUTDOWN\n"); self.p.stdin.flush(); self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.terminate()

def main():
    wls = ["chat", "code", "prose"]
    prompts = {w: wl_ids(w) for w in wls}
    fails = 0

    # ---- plain daemon 기준(솔로 순차) ----
    print("[1/3] plain daemon 기준 수집...")
    plain = Daemon([PLAIN, TW, UQ, MAXLEN, BMAX], "plain")
    ref = {}
    for i, w in enumerate(wls):
        plain.send(100+i, prompts[w]); ref[w] = plain.collect(100+i)
        print(f"  plain {w}: {len(ref[w])}tok END={plain.reason.get(100+i)} head={ref[w][:6]}")
    plain.close()

    # ---- spec daemon: 솔로 == plain (★서빙 spec==greedy) ----
    print("[2/3] spec daemon 솔로 — spec==greedy 게이트...")
    spec = Daemon([SPEC, TW, AW, UQ, MAXLEN, BMAX, K], "spec")
    solo = {}
    for i, w in enumerate(wls):
        spec.send(200+i, prompts[w]); solo[w] = spec.collect(200+i)
        same = solo[w] == ref[w]
        print(f"  spec  {w}: {len(solo[w])}tok END={spec.reason.get(200+i)} == plain: {'YES' if same else '★NO'}")
        if not same:
            fails += 1
            n = min(len(solo[w]), len(ref[w]))
            div = next((j for j in range(n) if solo[w][j] != ref[w][j]), n)
            print(f"    첫 분기 idx={div} plain={ref[w][div:div+3] if div<len(ref[w]) else '-'} spec={solo[w][div:div+3] if div<len(solo[w]) else '-'}")

    # ---- 동시 격리: 동시 3요청 각각 == 자기 솔로 ----
    print("[3/3] spec daemon 동시 3요청 — 슬롯 격리...")
    for i, w in enumerate(wls): spec.send(300+i, prompts[w])
    conc = {w: spec.collect(300+i) for i, w in enumerate(wls)}
    for w in wls:
        same = conc[w] == solo[w]
        print(f"  동시 {w}: == 솔로: {'YES' if same else '★NO'}")
        if not same: fails += 1
    alive = spec.p.poll() is None
    print(f"  생존: daemon {'alive' if alive else '★DEAD'}")
    if not alive: fails += 1
    spec.close()

    print(f"\n[test_gemma4_spec_serving] {'FAIL' if fails else 'PASS (서빙 spec==greedy·동시 격리·생존)'}")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
