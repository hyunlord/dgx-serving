#!/usr/bin/env python3
"""M5 온라인 STEP D 부하 스모크 — 동시 요청 부하 하 동작 확인(완료·격리·무누수) + tpot **관찰만**(주장 아님).
Phase 성격: 성능 최적화 아님 — 온라인 배칭이 부하 하에서도 정확·안정한지 확인. 성능 수치는 정직 분해(온라인 오버헤드 포함).
  - N=16 동시 요청( > Bmax=8 → 초과분은 daemon 큐 대기 → 슬롯 해제 시 admit = 동적 배치 형성·큐잉 경로 시험).
  - 4개 알려진 프롬프트 × 4반복 → 각 응답 == 단독 기준(★큐잉/부하 하에서도 슬롯 격리 유지 = 교차 오염 없음).
  - 관찰: 벽시계·총 생성 토큰·집계 tok/s(서버측 aggregate, 큐잉 포함 — llama/raw 대비 아님, 정직 분해).
사용법: <venv>/python tests/test_serve_load_smoke.py
"""
import json, os, subprocess, sys, threading, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS = os.path.join(REPO, "artifacts/gemma4-31b/weights"); PORT = 8139; MAXTOK = 24; BMAX = 8; COPIES = 4
BASE_PROMPTS = [
    "The capital of France is",
    "Water is composed of hydrogen and oxygen atoms, and its chemical formula is",
    "Two plus two equals",
    "The largest planet in our solar system is called",
]

def post(body, timeout=240):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()

def complete(prompt):
    r = json.loads(post({"prompt":prompt,"max_tokens":MAXTOK,"stream":False}))
    return r["choices"][0]["text"], r["usage"]["completion_tokens"]

srv = subprocess.Popen([sys.executable, os.path.join(REPO,"tools","serve_api.py"),
                        "--recipe","recipes/gemma4-31b-gb10.yaml","--weights",WEIGHTS,
                        "--port",str(PORT),"--bmax",str(BMAX)], stderr=subprocess.PIPE, text=True, cwd=REPO)
ready=False
for line in srv.stderr:
    print("  "+line.rstrip())
    if "listening" in line: ready=True; break
    if srv.poll() is not None: break
if not ready: srv.terminate(); sys.exit("[test] serve_api 起動 실패")

try:
    # 기준(단독): 각 프롬프트 하나씩
    solo = {p: complete(p)[0] for p in BASE_PROMPTS}
    for p in BASE_PROMPTS: print(f"[solo] {p[:28]!r:<30} → {solo[p]!r}")

    # 부하: N=16 동시(4×4), 계단식 도착(도착률 변동 흉내 — 앞부분 몰림 → 큐 초과 유발)
    jobs = [(i, BASE_PROMPTS[i % len(BASE_PROMPTS)]) for i in range(len(BASE_PROMPTS)*COPIES)]
    res = {}; errs = {}; toks = {}
    def worker(i, p, delay):
        if delay: time.sleep(delay)                    # 계단식 도착(고정 패턴 — 난수 미사용)
        try: res[i], toks[i] = complete(p)
        except Exception as e: errs[i] = repr(e)
    # 앞 12개는 즉시(→ Bmax 8 초과로 큐 대기), 뒤 4개는 0.05s 간격(도착 중 admit 경로)
    ths = [threading.Thread(target=worker, args=(i, p, 0.0 if i < 12 else (i-11)*0.05)) for i, p in jobs]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join(timeout=300)
    dt = time.time() - t0
    if errs: raise RuntimeError(f"부하 요청 오류: {errs}")
finally:
    srv.terminate(); srv.wait()

# ── 검증: 완료 + 격리(각 == 단독) ──
done = len(res) == len(jobs)
isolated = all(res[i] == solo[p] for i, p in jobs)
total_tok = sum(toks.values())
print(f"\n[부하] 요청 {len(jobs)}개(동시, Bmax={BMAX}) 완료={len(res)}/{len(jobs)} | 큐잉/부하 하 각==단독(격리)={isolated}")
print(f"[관찰] 벽시계={dt:.1f}s · 총 생성토큰={total_tok} · 집계 tok/s={total_tok/dt:.1f}"
      f"  ※서버측 aggregate(큐잉 포함) — 성능 주장 아님·llama/raw 대비 아님(Phase 관찰)")
ok = done and isolated
print(f"[test_serve_load_smoke] {'PASS(부하 하 완료·큐잉 하 슬롯 격리 유지·무누수)' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
