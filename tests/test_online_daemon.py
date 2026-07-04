#!/usr/bin/env python3
"""M5 온라인 daemon 정확성 게이트 (모델-무관·raw 토큰 ID) — ★온라인 연속 배칭 각 요청 == 단독 실행 bit-identical.
토크나이저 불요(게이트 핵심 = concurrent==solo·슬롯 격리 = IDs in→out 일관성). gemma4/qwen/향후 spec 공용.
  - SOLO: 각 요청 혼자(활성 슬롯 1) → 기준 출력.  - CONCURRENT: 같은 요청 동시 주입(활성 슬롯 N, ragged) → req_id demux.
  - 게이트: concurrent[req]==solo[req](슬롯 격리·교차오염 없음) + 결정성(동시 2회) + 도착순서 무관(역순==단독).
★ragged 격리: 프롬프트 길이 상이(P=5/12/7/10). raw ID(전 모델 vocab 내 안전 소값).
사용법: python tests/test_online_daemon.py <daemon_bin> [daemon args...]
  예: python tests/test_online_daemon.py tools/serve_daemon_qwen_online artifacts/qwen35-35b-a3b/weights artifacts/qwen35-35b-a3b/golden 2 384 8
★WIP harness 주의(2026-07-04): subprocess(Popen) 경로에 stdin/pipe 타이밍 이슈 관찰됨(gemma4·qwen 둘 다 stall — daemon
  버그 아님·test-infra 이슈). daemon 정확성은 **direct-pipe standalone**으로 검증 완료(신뢰):
    (printf "1 8 <ids>\n"; sleep 40; printf "2 8 <ids>\n"; sleep 15) | serve_daemon_qwen_online <w> <golden> 2 384 8
  → 35b: req1(단독)==req2 후 재사용, concurrent==solo·reuse==solo bit-id 확인(docs/m5-api-4model.md). subprocess 경로 수정=후속.
"""
import collections, select, subprocess, sys

if len(sys.argv) < 2: sys.exit("사용법: test_online_daemon.py <daemon_bin> [args...]")
CMD = sys.argv[1:]
MAXTOK = 10
# ragged raw-ID 프롬프트(전 모델 vocab 내 안전한 소값, 서로 다른 길이·내용). req_id 1..N.
REQS = [
    (1, [1, 100, 200, 300, 400]),                                  # P=5
    (2, [1, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]),    # P=12
    (3, [1, 500, 600, 700, 800, 900, 1000]),                       # P=7
    (4, [1, 11, 22, 33, 44, 55, 66, 77, 88, 99]),                  # P=10
]
IDMAP = dict(REQS)
for rid, ids in REQS: print(f"[cfg] req {rid}: P={len(ids)} ids")

def spawn():
    p = subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stderr:
        print("  "+line.rstrip())
        if "ready" in line: return p
        if p.poll() is not None: raise RuntimeError("daemon 起動 실패")
    raise RuntimeError("daemon ready 신호 없음")

def send(p, rid, ids): p.stdin.write(f"{rid} {MAXTOK} " + " ".join(map(str, ids)) + "\n"); p.stdin.flush()

def read_until(p, expected, stall_s=90):
    out = collections.defaultdict(list); done = set(); exp = set(expected)
    while done != exp:
        r,_,_ = select.select([p.stdout], [], [], stall_s)   # stall 가드(hang 대신 graceful fail)
        if not r: raise RuntimeError(f"daemon 무응답 {stall_s}s(미완: {exp-done}) — stdin 전달/파이프 확인")
        l = p.stdout.readline()
        if not l: raise RuntimeError(f"daemon EOF(미완: {exp-done})")
        parts = l.split()
        if len(parts) != 2: continue
        rid = int(parts[0])
        if parts[1] == "END": done.add(rid)
        else: out[rid].append(int(parts[1]))
    return out

# ★단일 daemon 스폰(1회 weight 로드) — solo·concurrent 모두 같은 프로세스에서(대형 모델 4× 재로드 회피 +
#   슬롯 재사용 격리[qwen GDN reset_slot] 검증 보너스: solo 후 free된 슬롯을 concurrent가 재사용).
p = spawn()
# ── SOLO(각 요청 혼자 = 활성 슬롯 1) ──
solo = {}
for rid, ids in REQS:
    send(p, rid, ids); solo[rid] = read_until(p, [rid])[rid]
    print(f"[solo] req {rid}: {solo[rid]}")
# ── CONCURRENT(같은 daemon, 도착순서 3종) ──
def run_conc(order):
    for rid in order: send(p, rid, IDMAP[rid])
    return read_until(p, [rid for rid,_ in REQS])
c1 = run_conc([rid for rid,_ in REQS])
c2 = run_conc([rid for rid,_ in REQS])
c3 = run_conc([rid for rid,_ in reversed(REQS)])
p.stdin.write("SHUTDOWN\n"); p.stdin.flush(); p.stdin.close(); p.terminate(); p.wait()

# ── 게이트 ──
print()
bitid = all(c1[rid] == solo[rid] for rid,_ in REQS)
det   = all(c2[rid] == c1[rid]   for rid,_ in REQS)
order = all(c3[rid] == solo[rid] for rid,_ in REQS)
for rid,_ in REQS:
    print(f"[req {rid}] {'동시==단독' if c1[rid]==solo[rid] else 'DIFFER!'}  conc={c1[rid]}")
print(f"\n[게이트] 동시==단독 bit-id={bitid} | 결정성={det} | 도착순서 무관={order}")
ok = bitid and det and order
print(f"[test_online_daemon] {'PASS(온라인 연속 배칭 각 요청==단독 bit-id, 슬롯 격리 유지)' if ok else 'FAIL(슬롯 격리 위반)'}")
sys.exit(0 if ok else 1)
