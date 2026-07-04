#!/usr/bin/env python3
"""26B-A4B MoE 온라인 서빙 게이트 — serve_daemon_gemma4_online(MoE 분기) 각 요청 == 단독 bit-identical.
★배치 decode 정확성은 Gate C(test_gemma4_moe_decode batch==solo abs 0.0)서 이미 증명 — 이 게이트는 온라인 e2e:
  프리필(gemma4_moe_prefill_batch_slot)+decode(gemma4_moe_decode_batch)+태그드 스트림 배선이 각 요청 계산 안 바꿈.
front-style 연속 reader(test_spec_online.py 검증 패턴 — subprocess quirk 회피). raw ID(토크나이저 불요). 스킴 내 q4.
사용법: python3 tests/test_moe26b_online.py
"""
import collections, os, queue, subprocess, sys, threading

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WD = os.path.join(REPO,"artifacts/gemma4-26b-a4b/weights")
DAEMON = os.path.join(REPO,"tools","serve_daemon_gemma4_online")
UQ="1"; MAXLEN="64"; BMAX="4"; MAXTOK=10
REQS = [ (1,[2,651,1596,573]), (2,[2,235292,611,573,1001]), (3,[2,1596,573,2]) ]   # ragged raw-ID
IDMAP=dict(REQS)
for rid,ids in REQS: print(f"[cfg] req {rid}: P={len(ids)}")

class Online:
    def __init__(self):
        self.p=subprocess.Popen([DAEMON,WD,UQ,MAXLEN,BMAX],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        for line in self.p.stderr:
            print("  "+line.rstrip())
            if "ready" in line: break
            if self.p.poll() is not None: raise RuntimeError("daemon 起動 실패")
        self.q=collections.defaultdict(queue.Queue)
        threading.Thread(target=self._rd,daemon=True).start()
    def _rd(self):
        for line in self.p.stdout:
            parts=line.split()
            if len(parts)!=2: continue
            rid=int(parts[0]); self.q[rid].put(None if parts[1]=="END" else int(parts[1]))
        for qq in self.q.values(): qq.put(None)
    def send(self,rid,ids): self.p.stdin.write(f"{rid} {MAXTOK} "+" ".join(map(str,ids))+"\n"); self.p.stdin.flush()
    def collect(self,rid):
        out=[]
        while True:
            t=self.q[rid].get(timeout=300)
            if t is None: break
            out.append(t)
        return out
    def close(self):
        try: self.p.stdin.write("SHUTDOWN\n"); self.p.stdin.flush(); self.p.stdin.close(); self.p.terminate()
        except Exception: pass

on = Online()
# solo: 각 요청 하나씩(활성 슬롯 1)
solo={}
for rid,ids in REQS:
    on.send(rid,ids); solo[rid]=on.collect(rid); print(f"[solo] req {rid}: {solo[rid]}")
# concurrent: 동시 주입(스레드로 각 collect) + reuse(slot 재사용, 다른 req_id)
conc={}; errs={}
def wk(rid):
    try: conc[rid]=on.collect(rid)
    except Exception as e: errs[rid]=repr(e)
ths=[threading.Thread(target=wk,args=(rid,)) for rid,_ in REQS]
for t in ths: t.start()
for rid,ids in REQS: on.send(rid,ids)
for t in ths: t.join(timeout=320)
seq={}
for rid,ids in REQS: on.send(1000+rid,ids); seq[rid]=on.collect(1000+rid)   # reuse
on.close()
if errs: raise RuntimeError(f"online 오류: {errs}")

print()
bc = all(conc[rid]==solo[rid] for rid,_ in REQS)
bs = all(seq[rid]==solo[rid] for rid,_ in REQS)
for rid,_ in REQS:
    print(f"[req {rid}] concurrent=={'단독' if conc[rid]==solo[rid] else 'DIFFER!'} · reuse=={'단독' if seq[rid]==solo[rid] else 'DIFFER!'}")
print(f"\n[게이트] concurrent==단독={bc} | reuse==단독={bs} (26B MoE 온라인 e2e·스킴 내 q4·slot 격리)")
ok = bc and bs
print(f"[test_moe26b_online] {'PASS(26B-A4B MoE 온라인 서빙 각 요청==단독 bit-id — 4모델 완성)' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
