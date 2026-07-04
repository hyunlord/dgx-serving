#!/usr/bin/env python3
"""M5 STEP C 게이트 — 27b spec 온라인 daemon == canonical 단독 spec bit-identical(+ 곱 accept 보존·슬롯 격리).
★spec 사이클(draft→verify→accept/promote)이 API/온라인 배칭 경로서도 각 요청 계산 안 바꿈:
  - SOLO 기준: daemon SOLO 모드(canonical solo_spec — bench_serve_spec ISO 기준과 동일 함수) → 요청별 토큰.
  - ONLINE: daemon 온라인 모드(태그드 연속 배칭) → concurrent(2슬롯 격리) + sequential(slot 재사용·MtpKV clear).
  - 게이트: online[req] == solo_ref[req] bit-identical(스킴 내 W4A8+spec) = accept 곱 보존(같은 토큰=같은 accept 패턴).
★robust: front-style 연속 reader 스레드로 stdout demux(35b subprocess quirk 회피 — 검증된 패턴). raw ID(토크나이저 불요).
사용법: <venv>/python tests/test_spec_online.py   (또는 python3 — HF 불필요)
"""
import collections, os, queue, subprocess, sys, threading, time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WD = os.path.join(REPO,"artifacts/qwen35-27b/weights"); GD = os.path.join(REPO,"artifacts/qwen35-27b/mtp_golden")
DAEMON = os.path.join(REPO,"tools","serve_daemon_qwen_spec_online")
UQ="2"; BMAX="8"; MAXTOK=16
# ragged raw-ID 프롬프트(27b vocab 내 안전 소값, 서로 다른 길이·내용)
REQS = [ (1,[1,100,200,300,400]), (2,[1,50,60,70,80,90,110]), (3,[1,500,600,700]) ]
IDMAP = dict(REQS)
for rid,ids in REQS: print(f"[cfg] req {rid}: P={len(ids)}")

def wait_ready(p):
    for line in p.stderr:
        print("  "+line.rstrip())
        if "ready" in line: return
        if p.poll() is not None: raise RuntimeError("daemon 起動 실패")
    raise RuntimeError("ready 신호 없음")

# ── SOLO 기준(canonical 단독 spec) ── daemon SOLO 모드: "<maxtok> <ids>" → "<tok>"*N + "###END"
def solo_refs():
    env=dict(os.environ); env["SOLO"]="1"
    p=subprocess.Popen([DAEMON,WD,GD,UQ,BMAX],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    wait_ready(p)
    q=queue.Queue()
    def rd():
        for line in p.stdout: q.put(line.strip())
        q.put(None)
    threading.Thread(target=rd,daemon=True).start()
    ref={}
    for rid,ids in REQS:
        p.stdin.write(f"{MAXTOK} "+" ".join(map(str,ids))+"\n"); p.stdin.flush()
        toks=[]
        while True:
            l=q.get(timeout=180)
            if l is None: raise RuntimeError("SOLO daemon EOF")
            if l=="###END": break
            if l: toks.append(int(l))
        ref[rid]=toks; print(f"[solo] req {rid}: {toks}")
    p.stdin.close(); p.terminate(); p.wait()
    return ref

# ── ONLINE(태그드 연속 배칭) ── front-style 연속 reader 스레드 demux
class Online:
    def __init__(self):
        self.p=subprocess.Popen([DAEMON,WD,GD,UQ,BMAX],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        wait_ready(self.p)
        self.q=collections.defaultdict(queue.Queue); self.wlock=threading.Lock()
        threading.Thread(target=self._rd,daemon=True).start()
    def _rd(self):
        for line in self.p.stdout:
            parts=line.split()
            if len(parts)!=2: continue
            rid=int(parts[0])
            self.q[rid].put(None if parts[1]=="END" else int(parts[1]))
        # EOF
        for qq in self.q.values(): qq.put(None)
    def send(self,rid,ids):
        with self.wlock: self.p.stdin.write(f"{rid} {MAXTOK} "+" ".join(map(str,ids))+"\n"); self.p.stdin.flush()
    def collect(self,rid):
        toks=[]
        while True:
            t=self.q[rid].get(timeout=180)
            if t is None: break
            toks.append(t)
        return toks
    def close(self):
        try: self.p.stdin.write("SHUTDOWN\n"); self.p.stdin.flush(); self.p.stdin.close(); self.p.terminate()
        except Exception: pass

ref = solo_refs()

on = Online()
# concurrent: 3요청 동시 주입 → 스레드로 각 collect(격리)
conc={}; errs={}
def worker(rid):
    try: conc[rid]=on.collect(rid)
    except Exception as e: errs[rid]=repr(e)
ths=[threading.Thread(target=worker,args=(rid,)) for rid,_ in REQS]
for t in ths: t.start()
for rid,ids in REQS: on.send(rid,ids)      # 동시 주입(로드 완료 후 배치)
for t in ths: t.join(timeout=240)
# sequential(slot 재사용): 같은 요청 재주입(다른 req_id)
seq={}
for rid,ids in REQS:
    on.send(1000+rid,ids); seq[rid]=on.collect(1000+rid)
on.close()
if errs: raise RuntimeError(f"online 오류: {errs}")

# ── 게이트 ──
print()
bitid_c = all(conc[rid]==ref[rid] for rid,_ in REQS)   # concurrent == canonical 단독 spec
bitid_s = all(seq[rid]==ref[rid] for rid,_ in REQS)    # sequential(재사용) == canonical
for rid,_ in REQS:
    okc = conc[rid]==ref[rid]; oks = seq[rid]==ref[rid]
    print(f"[req {rid}] concurrent=={'단독' if okc else 'DIFFER!'} · reuse=={'단독' if oks else 'DIFFER!'} · gen={len(ref[rid])} conc={conc[rid]}")
print(f"\n[게이트] concurrent==단독 spec bit-id={bitid_c} | reuse==단독 bit-id={bitid_s} (스킴: W4A8+spec, 같은 토큰=accept 곱 보존)")
ok = bitid_c and bitid_s
print(f"[test_spec_online] {'PASS(27b spec 온라인 각 요청==canonical 단독 spec bit-id, accept 곱 보존·슬롯 격리)' if ok else 'FAIL(spec 사이클/격리 조사)'}")
sys.exit(0 if ok else 1)
