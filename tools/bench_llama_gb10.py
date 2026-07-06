#!/usr/bin/env python3
"""llama-on-GB10 공정 벤치(재현 도구): MTP fork llama-server를 우리 serve_api A/B와 동일 조건으로 측정.
공정 조건: 같은 12B(Q4_K_M vs 우리 w4a8 — 동급 4bit·동일 스킴 아님 명시)·같은 공식 0.4B assistant
(f16 GGUF vs 우리 W4A8 — llama에 고정밀 유리 제공)·같은 프롬프트(chat/code/prose = probe_assistant_spec
WORKLOADS)·같은 회계(client-observed submit→last token, 이 스크립트의 시계 — 우리 A/B와 동일 코드 패턴)·
같은 K(--spec-draft-n-max 4)·warm 1회·single-stream(--parallel 1)·전층 GPU(-ngl/-ngld 99)·greedy(temp 0).
환경변수: LLAMA_BIN(MTP fork llama-server)·GGUF_12B·GGUF_AST·MODES(plain,spec)·N_NEW.
assistant GGUF 생성: MTP fork의 convert_hf_to_gguf.py <assistant snapshot> --outtype f16.
측정 결과(GB10·2026-07): plain 25.6 / spec chat 56.2·code 69.6·prose 45.6 t/s → docs/performance.md §3b.
"""
import json, os, re, subprocess, sys, time, http.client

BIN   = os.environ.get("LLAMA_BIN", os.path.expanduser("~/repos/llama.cpp-mtp/build/bin/llama-server"))
M12B  = os.environ.get("GGUF_12B",  os.path.expanduser("~/models/poc/gemma-4-12B-it-Q4_K_M.gguf"))
MAST  = os.environ.get("GGUF_AST",  os.path.expanduser("~/models/poc/gemma-4-12B-it-assistant-f16.gguf"))
PORT  = 8124
N_NEW = int(os.environ.get("N_NEW", "128"))
PROMPTS = {
    "chat":  "Hi! How are you today? Tell me a little about yourself.",
    "code":  "Write a Python function that returns the nth Fibonacci number.",
    "prose": "Write a short descriptive paragraph about a rainy autumn morning.",
}

def start(spec):
    args = [BIN, "-m", M12B, "-ngl", "99", "--port", str(PORT), "--parallel", "1", "-c", "4096", "--no-webui"]
    if spec:
        args += ["-md", MAST, "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "-ngld", "99"]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    t0 = time.time(); log = []
    import threading, collections
    lines = collections.deque(maxlen=4000)
    def rd(stream):
        for line in stream: lines.append(line.rstrip())
    threading.Thread(target=rd, args=(p.stderr,), daemon=True).start()
    threading.Thread(target=rd, args=(p.stdout,), daemon=True).start()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            print("\n".join(list(lines)[-20:])); raise RuntimeError("llama-server 종료(起動 실패)")
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
            c.request("GET", "/health"); r = c.getresponse()
            if r.status == 200: c.close(); break
            c.close()
        except Exception: time.sleep(1.5)
    else:
        raise RuntimeError("起動 타임아웃")
    return p, lines

def ask(prompt):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": N_NEW, "temperature": 0, "stream": False})
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.loads(c.getresponse().read()); c.close()
    dt = time.time() - t0
    n = r.get("usage", {}).get("completion_tokens", 0)
    return n, dt, (n/dt if dt > 0 else 0), r["choices"][0]["message"]["content"][:60]

def drain_accept(lines):
    """직전 요청 구간의 draft 통계(마지막 항목) 파싱 — 포맷 가변 대비 폭넓게."""
    txt = "\n".join(lines)
    pats = [r"draft acceptance rate[^\d]*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated",
            r"n_draft\s*=\s*(\d+).{0,40}n_accept(?:ed)?\s*=\s*(\d+)"]
    for pat in pats:
        m = list(re.finditer(pat, txt))
        if m: return m[-1].group(0)
    return None

results = {}
for mode in (["plain", "spec"] if os.environ.get("MODES") is None else os.environ["MODES"].split(",")):
    print(f"=== llama-GB10 {mode} 起動(웨이트 로드)... ===", flush=True)
    p, lines = start(spec=(mode == "spec"))
    try:
        for name, pr in PROMPTS.items():
            ask(pr)                                        # warm-up
            lines.clear()
            n, dt, tps, txt = ask(pr)
            acc = drain_accept(lines) if mode == "spec" else None
            results[(mode, name)] = (n, dt, tps)
            print(f"[llama-{mode}] {name}: {tps:.2f} t/s (n={n} dt={dt:.2f}s){' | '+acc if acc else ''} | '{txt}'", flush=True)
    finally:
        p.terminate(); p.wait(timeout=20)
    time.sleep(2)

print("\n=== llama-on-GB10 요약 (client-observed·128tok·warm) ===")
ours = {"chat": (22.0, 50.5), "code": (21.9, 46.7), "prose": (None, None)}   # 우리 A/B(동일 회계) 참조
for name in PROMPTS:
    lp = results.get(("plain", name)); ls = results.get(("spec", name))
    op, osp = ours[name]
    print(f"{name}: llama plain {(lp[2] if lp else 0):.2f} · llama spec {(ls[2] if ls else 0):.2f} t/s"
          f"  || 우리 plain {op or '—'} · 우리 spec {osp or '—'}")
