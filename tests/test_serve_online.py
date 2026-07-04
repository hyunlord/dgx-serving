#!/usr/bin/env python3
"""M5 온라인 STEP C 정확성 게이트(daemon 직접) — ★온라인 연속 배칭 각 요청 == 단독 실행 bit-identical.
검증하는 불변식: 온라인 배칭/스케줄이 각 요청의 계산을 안 바꾼다 = 슬롯 격리(한 요청이 다른 요청에 안 샘).
  - SOLO: 각 요청을 혼자(활성 슬롯 1개) 실행 → 기준 출력.
  - CONCURRENT: 같은 요청들을 동시 주입(활성 슬롯 N개, ragged 배치) → 슬롯 인터리브 출력을 req_id로 demux.
  - 게이트: 각 req_id에 대해 concurrent[req] == solo[req](bit-id). + 결정성(동시 2회 동일) + 도착순서 무관.
★ragged 격리 시험: 프롬프트 길이를 일부러 다르게(슬롯별 다른 P). test_gemma4_ragged가 ragged==single 증명 —
  이 테스트는 그 위에 실시간 비동기 주입 경로까지 슬롯 격리가 유지되는지 확인.
Phase: gemma4 plain·greedy·dummy rope(품질 아닌 격리/bit-id 검증). 사용법: <venv>/python tests/test_serve_online.py
"""
import os, subprocess, sys, threading, collections

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS = os.path.join(REPO, "artifacts/gemma4-31b/weights"); TOKID = "google/gemma-4-E4B"
DAEMON = os.path.join(REPO, "tools", "serve_daemon_gemma4_online")
MAXTOK = 10; BMAX = 8
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKID)

# 서로 다른 프롬프트 + 서로 다른 길이(ragged 슬롯 격리 시험). req_id: 1..N.
PROMPTS = [
    "The capital of France is",              # 짧음
    "Water is composed of hydrogen and oxygen atoms, and its chemical formula is",  # 김
    "Two plus two equals",                   # 짧음
    "The largest planet in our solar system is called",  # 중간
]
REQS = []  # (req_id, ids)
for i, p in enumerate(PROMPTS, start=1):
    REQS.append((i, tok(p).input_ids))
for rid, ids in REQS:
    print(f"[cfg] req {rid}: {len(ids)} ids  prompt={PROMPTS[rid-1]!r}")

def spawn():
    p = subprocess.Popen([DAEMON, WEIGHTS, "2", "384", str(BMAX)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stderr:            # "[online] ready" 대기
        print("  "+line.rstrip())
        if "ready" in line: break
        if p.poll() is not None: raise RuntimeError("daemon 起動 실패")
    return p

def send(p, rid, ids):
    p.stdin.write(f"{rid} {MAXTOK} " + " ".join(map(str, ids)) + "\n"); p.stdin.flush()

def read_until_done(p, expected_ids):
    """stdout 라인 파싱 → req_id별 토큰 누적. expected_ids 전부 END 받을 때까지."""
    out = collections.defaultdict(list); done = set()
    while done != set(expected_ids):
        l = p.stdout.readline()
        if not l:
            raise RuntimeError(f"daemon EOF(미완: {set(expected_ids)-done})")
        parts = l.split()
        if len(parts) != 2: continue
        rid = int(parts[0])
        if parts[1] == "END": done.add(rid)
        else: out[rid].append(int(parts[1]))
    return out

# ── SOLO: 각 요청 혼자(활성 슬롯 1개) ──
solo = {}
p = spawn()
for rid, ids in REQS:
    send(p, rid, ids)
    r = read_until_done(p, [rid])
    solo[rid] = r[rid]
    print(f"[solo] req {rid}: {solo[rid]}  → {tok.decode(solo[rid], skip_special_tokens=True)!r}")
p.stdin.write("SHUTDOWN\n"); p.stdin.flush(); p.stdin.close(); p.terminate(); p.wait()

# ── CONCURRENT: 동시 주입(활성 슬롯 N개, ragged 배치) ──
def run_concurrent(order):
    p = spawn()
    for rid in order:
        ids = dict(REQS)[rid]; send(p, rid, ids)
    conc = read_until_done(p, [rid for rid,_ in REQS])
    p.stdin.write("SHUTDOWN\n"); p.stdin.flush(); p.stdin.close(); p.terminate(); p.wait()
    return conc

conc1 = run_concurrent([rid for rid,_ in REQS])          # 도착순서 1: 정순
conc2 = run_concurrent([rid for rid,_ in REQS])          # 결정성(동일 순서 2회)
conc3 = run_concurrent([rid for rid,_ in reversed(REQS)])  # 도착순서 2: 역순(도착 타이밍 무관 시험)

# ── 게이트 ──
print()
bitid = all(conc1[rid] == solo[rid] for rid,_ in REQS)
det   = all(conc2[rid] == conc1[rid] for rid,_ in REQS)
order = all(conc3[rid] == solo[rid] for rid,_ in REQS)
for rid,_ in REQS:
    ok = conc1[rid]==solo[rid]
    print(f"[req {rid}] solo=={('CONCURRENT' if ok else 'DIFFER!')}  conc={conc1[rid]}")
print(f"\n[게이트] 동시==단독 bit-id={bitid} | 결정성={det} | 도착순서 무관(역순==단독)={order}")
ok = bitid and det and order
print(f"[test_serve_online] {'PASS(온라인 연속 배칭 각 요청==단독 bit-id, 슬롯 격리 유지)' if ok else 'FAIL(슬롯 격리 위반 — 계층 격리 조사 필요)'}")
sys.exit(0 if ok else 1)
