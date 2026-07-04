#!/usr/bin/env python3
"""M5 온라인 STEP C 정확성 게이트(API 레벨/전체 스택) — ★동시 HTTP 요청 각각 == 단독 요청 bit-identical.
daemon 레벨 게이트(test_serve_online.py)가 슬롯 격리를 이미 증명 — 이 테스트는 그 위에 **front demux**까지:
  동시 HTTP 연결이 각자 자기 토큰만 받는가(교차 오염 없음)를 전체 스택(HTTP→토큰화→태그드 주입→demux→디코드)에서 확인.
  - SOLO: 각 프롬프트를 API로 하나씩(순차) → 기준 텍스트.
  - CONCURRENT: 같은 프롬프트들을 스레드로 동시 POST(barrier로 동시 발사) → 온라인 배칭 → 텍스트.
  - 게이트: 각 프롬프트 concurrent_text == solo_text(demux 정확 + 슬롯 격리 유지) + 스트리밍 delta 합 == 논스트림.
Phase: gemma4 plain·greedy. 성능 아닌 정확성/격리 검증. 사용법: <venv>/python tests/test_serve_api_online.py
"""
import json, os, subprocess, sys, threading, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS = os.path.join(REPO, "artifacts/gemma4-31b/weights"); PORT = 8138; MAXTOK = 16
PROMPTS = [
    "The capital of France is",
    "Water is composed of hydrogen and oxygen atoms, and its chemical formula is",
    "Two plus two equals",
    "The largest planet in our solar system is called",
]

def post(body, timeout=180):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()

def complete(prompt, stream=False):
    raw = post({"prompt":prompt,"max_tokens":MAXTOK,"stream":stream})
    if not stream: return json.loads(raw)["choices"][0]["text"]
    deltas = [json.loads(l[6:])["choices"][0]["text"] for l in raw.splitlines()
              if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    return "".join(deltas)

# ── serve_api.py(온라인 daemon) 起動 ──
srv = subprocess.Popen([sys.executable, os.path.join(REPO,"tools","serve_api.py"),
                        "--recipe","recipes/gemma4-31b-gb10.yaml","--weights",WEIGHTS,
                        "--port",str(PORT),"--bmax","8"], stderr=subprocess.PIPE, text=True, cwd=REPO)
ready=False
for line in srv.stderr:
    print("  "+line.rstrip())
    if "listening" in line: ready=True; break
    if srv.poll() is not None: break
if not ready: srv.terminate(); sys.exit("[test] serve_api 起動 실패")

try:
    # ── SOLO(순차) ──
    solo = {p: complete(p) for p in PROMPTS}
    for p in PROMPTS: print(f"[solo] {p[:32]!r:<34} → {solo[p]!r}")

    # ── CONCURRENT(동시 발사 — barrier) ──
    conc = {}; errs = {}
    bar = threading.Barrier(len(PROMPTS))
    def worker(p):
        try: bar.wait(); conc[p] = complete(p)      # 모든 스레드가 동시에 POST 발사(온라인 배칭 유도)
        except Exception as e: errs[p] = repr(e)
    ths = [threading.Thread(target=worker, args=(p,)) for p in PROMPTS]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join(timeout=200)
    dt = time.time() - t0
    if errs: raise RuntimeError(f"동시 요청 오류: {errs}")

    # ── 스트리밍 일치(한 프롬프트) ──
    strm = complete(PROMPTS[0], stream=True)
finally:
    srv.terminate(); srv.wait()

# ── 게이트 ──
print()
demux_ok = True
for p in PROMPTS:
    ok = conc.get(p) == solo[p]; demux_ok &= ok
    print(f"[동시=={('단독' if ok else 'DIFFER!')}] {p[:32]!r:<34} conc={conc.get(p)!r}")
strm_ok = (strm == solo[PROMPTS[0]])
print(f"\n[게이트] 동시 {len(PROMPTS)}요청 각==단독(demux/슬롯격리)={demux_ok} | 스트림==논스트림={strm_ok} | 동시벽시계={dt:.1f}s")
ok = demux_ok and strm_ok
print(f"[test_serve_api_online] {'PASS(온라인 배칭 전체 스택 각 요청==단독 bit-id, front demux 정확·슬롯 격리)' if ok else 'FAIL(교차 오염 — front demux/daemon 격리 조사)'}")
sys.exit(0 if ok else 1)
