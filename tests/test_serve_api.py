#!/usr/bin/env python3
"""M5 STEP C-3 통합 게이트 — API 경유 == 직접 daemon bit-id + 결정성 + 스트림 일치 + /v1/models.
★정확성 불변 검증: front(serve_api.py)은 토큰화 왕복만·계산 안 함 → API 경유 생성 == 직접 daemon 생성 bit-identical.
  Part A(직접): HF 토큰화→daemon 직접 호출→gen ids→decode. Part B(API): serve_api.py 起動→/v1/completions→text.
  게이트: api_text == direct_text(bit-id) · 2회 동일(결정성) · SSE delta 합 == 논스트림(스트림 일치) · /v1/models.
Phase 0: gemma4 plain·greedy·dummy rope(품질 아닌 plumbing/bit-id 검증). 사용법: <venv>/python tests/test_serve_api.py
"""
import json, os, subprocess, sys, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS = os.path.join(REPO, "artifacts/gemma4-31b/weights"); TOKID = "google/gemma-4-E4B"
PROMPT = "The capital of France is"; MAXTOK = 6; PORT = 8137
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKID)
ids = tok(PROMPT).input_ids
print(f"[cfg] prompt={PROMPT!r} → {len(ids)} ids, max_tokens={MAXTOK}")

def wait_ready(proc, key):
    for line in proc.stderr:
        print("  "+line.rstrip())
        if key in line: return True
        if proc.poll() is not None: return False
    return False

# ── Part A: 직접 daemon(참조) ──
d = subprocess.Popen([os.path.join(REPO,"tools","serve_daemon_gemma4"), WEIGHTS, "2", "384"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
assert wait_ready(d, "ready"), "직접 daemon 起動 실패"
d.stdin.write(f"{MAXTOK} "+" ".join(map(str,ids))+"\n"); d.stdin.flush()
direct_ids = []
while True:
    l = d.stdout.readline().strip()
    if l == "###END": break
    if l: direct_ids.append(int(l))
d.stdin.close(); d.terminate(); d.wait()
direct_text = tok.decode(direct_ids, skip_special_tokens=True)
print(f"[direct] gen_ids={direct_ids} text={direct_text!r}")

# ── Part B: API 경유(serve_api.py) ──
srv = subprocess.Popen([sys.executable, os.path.join(REPO,"tools","serve_api.py"),
                        "--recipe","recipes/gemma4-31b-gb10.yaml","--weights",WEIGHTS,"--port",str(PORT)],
                       stderr=subprocess.PIPE, text=True, cwd=REPO)
assert wait_ready(srv, "listening"), "serve_api 起動 실패"

def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req, timeout=120).read().decode()

try:
    r1 = json.loads(post({"prompt":PROMPT,"max_tokens":MAXTOK,"stream":False}))
    api_text = r1["choices"][0]["text"]
    r2 = json.loads(post({"prompt":PROMPT,"max_tokens":MAXTOK,"stream":False}))
    det = (r2["choices"][0]["text"] == api_text)
    raw = post({"prompt":PROMPT,"max_tokens":MAXTOK,"stream":True})
    deltas = [json.loads(l[6:])["choices"][0]["text"] for l in raw.splitlines()
              if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    stream_text = "".join(deltas)
    models = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10).read())
    mid = models["data"][0]["id"]
finally:
    srv.terminate(); srv.wait()

print(f"[api] text={api_text!r} usage={r1['usage']} | stream_text={stream_text!r} | model={mid}")
bitid = (api_text == direct_text); strm = (stream_text == api_text)
print(f"\n[게이트] API==직접 bit-id={bitid} | 결정성={det} | 스트림==논스트림={strm} | /v1/models={mid}")
ok = bitid and det and strm
print(f"[test_serve_api] {'PASS(API 경유==직접 서빙 bit-id, 계층이 계산 불변)' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
