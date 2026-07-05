#!/usr/bin/env python3
"""Serve API front 통합 게이트 — 가짜 daemon으로 HTTP 계층 실구동(가중치·GPU 불요).

daemon(계산)은 불변이므로, front 변경(chat completions·메트릭·웹UI)의 정확성은 **가짜 daemon**으로
검증한다. FakeDaemon은 실 daemon의 stream(ids,max_tokens) 계약만 흉내(결정적 토큰 반환) — 이걸로:
  (1) /v1/completions 기존 경로 무회귀(text 형식·usage),
  (2) /v1/chat/completions chat 규격(message/delta) + 템플릿 적용(chat_ids),
  (3) ★스트리밍 청크합 == 논스트리밍(기존 게이트 방식),
  (4) 메트릭(usage+timings) 존재,
  (5) 동시 요청 slot 격리(각자 자기 출력),
  (6) GET / 웹UI HTML·/v1/models 배지,
를 검증. FakeDaemon은 입력 ids를 그대로 echo(결정적)해 plumbing/격리를 드러낸다.
"""
import os, sys, json, threading, urllib.request, urllib.error
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import serve_api  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

CANDIDATES = ["Qwen/Qwen3-4B-Instruct-2507","Qwen/Qwen2.5-3B-Instruct","Qwen/Qwen2.5-0.5B-Instruct"]

class FakeDaemon:
    """실 OnlineDaemon.stream 계약만 흉내: 입력 ids를 max_tokens 만큼 결정적 echo(계산 대체)."""
    def stream(self, ids, max_tokens):
        for i in range(min(max_tokens, max(1, len(ids)))):
            yield ids[i % len(ids)]
    def close(self): pass

def post(port, path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req) as r: return r.status, r.read().decode()

def post_stream(port, path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type":"application/json"})
    events = []
    with urllib.request.urlopen(req) as r:
        buf = ""
        while True:
            chunk = r.read(1)
            if not chunk: break
            buf += chunk.decode()
            while "\n\n" in buf:
                evt, buf = buf.split("\n\n", 1)
                line = evt.replace("data: ", "").strip()
                if line and line != "[DONE]": events.append(json.loads(line))
    return events

def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
        return r.status, r.getheader("Content-Type"), r.read().decode()

def main():
    from transformers import AutoTokenizer
    tok = None
    for n in CANDIDATES:
        try:
            t = AutoTokenizer.from_pretrained(n)
            if getattr(t,"chat_template",None): tok = t; break
        except Exception: continue
    if tok is None:
        print("[test_serve_api_front] SKIP — chat 토크나이저 캐시 없음(오프라인)."); return 0

    serve_api.Handler.tok = tok
    serve_api.Handler.daemon = FakeDaemon()
    serve_api.Handler.model_id = "test-model"; serve_api.Handler.quant = "w4a8"; serve_api.Handler.spec = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve_api.Handler)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True); th.start()
    fails = []
    try:
        # (6) 웹UI + /v1/models
        st, ct, html = get(port, "/")
        if not (st==200 and "text/html" in ct and "<html" in html.lower()): fails.append("GET / 웹UI 실패")
        st, ct, models = get(port, "/v1/models"); md = json.loads(models)
        if not (md["data"][0]["id"]=="test-model" and md["data"][0]["quant"]=="w4a8"): fails.append("/v1/models 배지 실패")
        print(f"  GET / (웹UI {len(html)}B) · /v1/models 배지: {'PASS' if not fails else 'FAIL'}")

        # (1) /v1/completions 무회귀: text 형식 + usage
        st, body = post(port, "/v1/completions", {"prompt":"Hello world","max_tokens":8})
        c = json.loads(body)
        ok1 = (c["object"]=="text_completion" and "text" in c["choices"][0] and "usage" in c)
        print(f"  /v1/completions text 형식+usage: {'PASS' if ok1 else 'FAIL'}")
        if not ok1: fails.append("completions 형식")

        # (2) /v1/chat/completions chat 규격 + 템플릿 적용(chat_ids와 동일 ids echo되므로 decode 일관)
        msgs = [{"role":"user","content":"The capital of France is"}]
        st, body = post(port, "/v1/chat/completions", {"messages":msgs,"max_tokens":8})
        cc = json.loads(body)
        ok2 = (cc["object"]=="chat.completion" and cc["choices"][0]["message"]["role"]=="assistant"
               and "content" in cc["choices"][0]["message"])
        print(f"  /v1/chat/completions message 규격: {'PASS' if ok2 else 'FAIL'}")
        if not ok2: fails.append("chat 형식")

        # (4) 메트릭 usage+timings
        okm = ("timings" in cc and "tokens_per_s" in cc["timings"] and "usage" in cc)
        print(f"  메트릭 usage+timings: {'PASS' if okm else 'FAIL'} ({cc.get('timings',{})})")
        if not okm: fails.append("메트릭 누락")

        # (3) ★청크합 == 논스트림(같은 messages) — chat
        ns = json.loads(post(port, "/v1/chat/completions", {"messages":msgs,"max_tokens":8})[1])["choices"][0]["message"]["content"]
        evs = post_stream(port, "/v1/chat/completions", {"messages":msgs,"max_tokens":8,"stream":True})
        streamed = "".join(e["choices"][0].get("delta",{}).get("content","") for e in evs)
        ok3 = (streamed == ns)
        print(f"  청크합==논스트림(chat): {'PASS' if ok3 else 'FAIL'} (stream {len(streamed)}c == nonstream {len(ns)}c)")
        if not ok3: fails.append(f"청크합!=논스트림: {streamed!r} != {ns!r}")
        # 스트림 종료 이벤트에 timings 포함?
        okst = any("timings" in e for e in evs)
        print(f"  스트림 종료 메트릭: {'PASS' if okst else 'FAIL'}")
        if not okst: fails.append("스트림 timings 누락")

        # (5) 동시 요청 slot 격리: 서로 다른 프롬프트 → 각자 자기 출력(echo)
        results = {}
        def worker(key, prompt):
            results[key] = json.loads(post(port, "/v1/completions", {"prompt":prompt,"max_tokens":6})[1])["choices"][0]["text"]
        ta = threading.Thread(target=worker, args=("a","aaaa bbbb")); tb = threading.Thread(target=worker, args=("b","zzzz yyyy"))
        ta.start(); tb.start(); ta.join(); tb.join()
        # 각 응답이 자기 프롬프트 토큰 echo(격리) — 서로 안 섞임
        iso = (results["a"] != results["b"]) and all(results.values())
        print(f"  동시 slot 격리: {'PASS' if iso else 'FAIL'} (a={results['a']!r} b={results['b']!r})")
        if not iso: fails.append("slot 격리")
    finally:
        httpd.shutdown()

    if fails:
        print("[test_serve_api_front] FAIL:"); [print("   -",f) for f in fails]; return 1
    print("[test_serve_api_front] PASS — front 무회귀·chat 규격·청크합 일치·메트릭·격리·웹UI.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
