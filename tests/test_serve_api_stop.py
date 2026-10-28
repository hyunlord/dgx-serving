#!/usr/bin/env python3
"""serve_api stop 문자열·취소 게이트 — 스크립트 daemon 프로세스(실 태그드 프로토콜, GPU 불요).

FakeDaemonProc는 "<rid> <max_tokens> <ids...>"를 받으면 정해진 토큰 열을 천천히 내보내고,
"CANCEL <rid>"를 받으면 그 요청을 멈추고 "<rid> END cancel"을 쓴다(실 run_online_daemon과 같은 계약).
받은 CANCEL은 로그 파일에 남겨 front가 실제로 보냈는지 검사한다. 검증:
  (1) 논스트리밍: stop 문자열 앞까지만 출력, finish_reason=stop, daemon이 CANCEL을 받음
  (2) 스트리밍: 청크 합 == 논스트리밍 텍스트, stop 문자열(및 그 일부)이 청크에 나오지 않음
  (3) 일치하지 않는 stop의 접두사 꼬리: 끝에서 보류분이 방출되고 finish_reason=length, CANCEL 없음
  (4) stop 없음: 기존 출력과 동일
  (5) 스트리밍 중 클라이언트 끊김 → CANCEL 전달
Usage: python tests/test_serve_api_stop.py [--tokenizer google/gemma-4-E4B]
"""
import os, sys, json, socket, tempfile, textwrap, threading, time, urllib.request
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import serve_api  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

FAKE = textwrap.dedent('''
    import os, sys, threading, time
    toks = [int(x) for x in os.environ["FAKE_TOKENS"].split(",")]
    log = open(os.environ["FAKE_LOG"], "a", buffering=1)
    cancelled, lock = set(), threading.Lock()
    print("[online] ready: fake", file=sys.stderr, flush=True)
    def run(rid, n):
        for i in range(min(n, len(toks))):
            time.sleep(0.03)
            with lock:
                if rid in cancelled:
                    print(f"{rid} END cancel", flush=True); return
                print(f"{rid} {toks[i]}", flush=True)
        print(f"{rid} END length", flush=True)
    for line in sys.stdin:
        p = line.split()
        if not p: continue
        if p[0] == "SHUTDOWN": break
        if p[0] == "CANCEL":
            log.write(f"CANCEL {p[1]}\\n")
            with lock: cancelled.add(int(p[1]))
            continue
        threading.Thread(target=run, args=(int(p[0]), int(p[1])), daemon=True).start()
''')


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode())


def post_stream(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps({**body, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    events, buf = [], ""
    with urllib.request.urlopen(req) as r:
        while True:
            chunk = r.read(1)
            if not chunk: break
            buf += chunk.decode()
            while "\n\n" in buf:
                evt, buf = buf.split("\n\n", 1)
                line = evt.replace("data: ", "").strip()
                if line and line != "[DONE]": events.append(json.loads(line))
    return events


def main():
    tokenizer = sys.argv[sys.argv.index("--tokenizer") + 1] if "--tokenizer" in sys.argv else "google/gemma-4-E4B"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    text = "alpha beta gamma delta epsilon zeta eta theta"
    toks = tok(text, add_special_tokens=False).input_ids
    full = tok.decode(toks, skip_special_tokens=True)
    tmp = tempfile.mkdtemp(prefix="r14stop")
    script, log = os.path.join(tmp, "fake.py"), os.path.join(tmp, "cancel.log")
    open(script, "w").write(FAKE); open(log, "w").close()
    env = {**os.environ, "FAKE_TOKENS": ",".join(map(str, toks)), "FAKE_LOG": log}
    serve_api.Handler.daemon = serve_api.OnlineDaemon([sys.executable, script], env=env, cancel=True)
    serve_api.Handler.tok = tok; serve_api.Handler.sampling = True
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_api.Handler); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cancels = lambda: open(log).read().split()[1::2]
    fails = []
    def check(ok, msg):
        print(("PASS " if ok else "FAIL ") + msg); fails.extend([] if ok else [msg])

    base = post(port, {"prompt": "x", "max_tokens": 64})
    check(base["choices"][0]["text"] == full and base["choices"][0]["finish_reason"] == "length", "(4) no stop == full text")

    n0 = len(cancels())
    r = post(port, {"prompt": "x", "max_tokens": 64, "stop": ["gamma"]})
    want = full[:full.find("gamma")]
    check(r["choices"][0]["text"] == want and r["choices"][0]["finish_reason"] == "stop", f"(1) non-stream stop text={r['choices'][0]['text']!r}")
    time.sleep(0.3); check(len(cancels()) == n0 + 1, "(1) daemon received CANCEL")

    ev = post_stream(port, {"prompt": "x", "max_tokens": 64, "stop": [" delta ep"]})
    parts = [e["choices"][0].get("text", "") for e in ev]
    joined = "".join(parts)
    check(joined == full[:full.find(" delta ep")] and ev[-1]["choices"][0]["finish_reason"] == "stop",
          f"(2) stream stop joined={joined!r}")
    check(all("delta" not in p for p in parts), "(2) no stop fragment emitted")
    nonstream = post(port, {"prompt": "x", "max_tokens": 64, "stop": [" delta ep"]})["choices"][0]["text"]
    check(joined == nonstream, "(2) stream == non-stream")

    time.sleep(0.3); n1 = len(cancels())
    ev = post_stream(port, {"prompt": "x", "max_tokens": 64, "stop": ["theta!"]})
    joined = "".join(e["choices"][0].get("text", "") for e in ev)
    check(joined == full and ev[-1]["choices"][0]["finish_reason"] == "length", f"(3) held-back tail released joined={joined!r}")
    time.sleep(0.3); check(len(cancels()) == n1, "(3) no CANCEL without match")

    s = socket.create_connection(("127.0.0.1", port))
    body = json.dumps({"prompt": "x", "max_tokens": 64, "stream": True}).encode()
    s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
              + str(len(body)).encode() + b"\r\n\r\n" + body)
    s.recv(256); s.close()
    deadline = time.time() + 5
    while time.time() < deadline and len(cancels()) == n1: time.sleep(0.05)
    check(len(cancels()) == n1 + 1, "(5) client disconnect -> CANCEL")

    serve_api.Handler.daemon.close(); srv.shutdown()
    print("[test_serve_api_stop]", "FAIL" if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
