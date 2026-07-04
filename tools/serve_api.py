#!/usr/bin/env python3
"""M5 Phase 0 — Serve API front (얇은 계층: OpenAI 호환 HTTP + HF 토큰화 + SSE).
2계층 아키텍처(docs/m5-api-design.md): 이 front은 계산 안 함 — HTTP 표면·text↔ids·SSE만. 계산은 C++ daemon.
  起動: recipe.yaml 로드 → verify(지문·로컬 메모리 가드 — recipe.py 재사용) → daemon(serve_daemon_gemma4) spawn(영속 파이프).
  런타임: /v1/completions(greedy) → HF 토큰화 → daemon에 ID 전송 → 생성 ID 수신 → HF 디코드 → JSON/SSE.
★정확성 불변: daemon이 gemma4_decode를 통과만 → API 경유 출력 == 직접 서빙 bit-identical(front은 토큰화 왕복만).
Phase 0 스코프: gemma4 plain·greedy만(temperature/logprobs/n>1 미지원 — 명시 거부·무시). 요청 직렬화(단일 daemon; 온라인 배칭=다음).
사용법: python tools/serve_api.py --recipe recipes/gemma4-31b-gb10.yaml --weights artifacts/gemma4-31b/weights --port 8000
"""
import argparse, json, os, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "autotune"))

# ---- recipe verify 재사용(recipe.py — 지문·메모리 가드; 미니벤치는 생략=verify --skip-bench 등가) ----
def load_and_verify_recipe(recipe_path, device_json, model_json):
    from recipe import load_recipe, fingerprint_check, local_memory_check  # noqa: E402
    recipe = load_recipe(recipe_path)                      # 스키마 v0 + scheme 존재 검사(없으면 거부)
    dp = json.load(open(device_json)); mp = json.load(open(model_json))
    warns, fails = fingerprint_check(recipe, dp)
    for w in warns: print(f"  [지문 경고] {w}", file=sys.stderr)
    if fails:
        for f in fails: print(f"  [지문 거부] {f}", file=sys.stderr)
        sys.exit("[serve_api] REJECT — toolkit 필수 caps 미충족(이 장비 실행 불가).")
    ok_cfgs, rejected, avail = local_memory_check(recipe, mp, dp)
    print(f"  [메모리] 로컬 가용 {avail}GB — {len(ok_cfgs)} config OK, {len(rejected)} 거부", file=sys.stderr)
    if rejected and not ok_cfgs:
        sys.exit("[serve_api] REJECT — 모든 regime config가 로컬 메모리 예산 초과.")
    if warns:
        print("  [안내] 지문 오차 — claims 재현 미보장. 레시피를 prior로 로컬 재튜닝 권장(warm-start).", file=sys.stderr)
    return recipe

# ---- C++ daemon 래퍼(영속 프로세스, 요청 직렬화) ----
class Daemon:
    def __init__(self, weights_dir, uq, maxlen):
        bin_ = os.path.join(REPO, "tools", "serve_daemon_gemma4")
        if not os.path.exists(bin_): sys.exit(f"[serve_api] daemon 바이너리 없음: {bin_} (make tools/serve_daemon_gemma4)")
        self.p = subprocess.Popen([bin_, weights_dir, str(uq), str(maxlen)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()
        # readiness: daemon이 stderr에 "[daemon] ready" 찍을 때까지 대기(weight 로드 완료 신호)
        for line in self.p.stderr:
            print("  [daemon] "+line.rstrip(), file=sys.stderr)
            if "ready" in line: break
        if self.p.poll() is not None: sys.exit("[serve_api] daemon 起動 실패(로드 중 종료).")

    def generate(self, ids, max_tokens):
        """token-ID 요청 → 생성 token-ID 리스트(제너레이터: 토큰마다 yield = 스트리밍)."""
        with self.lock:                                    # 단일 daemon → 요청 직렬화(Phase 0)
            self.p.stdin.write(f"{max_tokens} " + " ".join(map(str, ids)) + "\n"); self.p.stdin.flush()
            while True:
                line = self.p.stdout.readline()
                if not line: raise RuntimeError("daemon 파이프 종료(EOF)")
                line = line.strip()
                if line == "###END": break
                if line: yield int(line)

    def close(self):
        try: self.p.stdin.close(); self.p.terminate()
        except Exception: pass

# ---- OpenAI 호환 HTTP 핸들러 ----
class Handler(BaseHTTPRequestHandler):
    daemon = None; tok = None; model_id = "gemma4"       # 클래스 변수(서버 起動 시 주입)
    def log_message(self, *a): pass                       # 접근 로그 억제

    def _json(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object":"list","data":[{"id":self.model_id,"object":"model","owned_by":"dgx-serving"}]})
        else: self._json(404, {"error":{"message":"not found","type":"invalid_request_error"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/completions":
            return self._json(404, {"error":{"message":"Phase 0: /v1/completions만 지원","type":"invalid_request_error"}})
        try:
            n = int(self.headers.get("Content-Length",0)); req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error":{"message":f"bad json: {e}","type":"invalid_request_error"}})
        # Phase 0 미지원 명시 거부(greedy 전제 깨는 것)
        for k in ("logprobs","n","best_of","suffix"):
            if req.get(k) not in (None,1): return self._json(400, {"error":{"message":f"Phase 0 미지원: {k}(greedy만)","type":"invalid_request_error"}})
        if req.get("temperature",0) not in (0,None,0.0):
            print(f"  [경고] temperature={req.get('temperature')} 무시(Phase 0 greedy만)", file=sys.stderr)
        prompt = req.get("prompt","");  max_tokens = int(req.get("max_tokens",16)); stream = bool(req.get("stream",False))
        if isinstance(prompt, list): prompt = prompt[0] if prompt else ""
        ids = self.tok(prompt).input_ids                  # HF 토큰화(BOS 포함 — Gemma 관례)
        cid = "cmpl-"+uuid.uuid4().hex[:24]; created = int(time.time())

        if stream:
            self.send_response(200); self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-cache"); self.end_headers()
            acc = []; prev_text = ""                       # incremental detok: 누적 디코드 후 신규 suffix가 delta
            for tid in self.daemon.generate(ids, max_tokens):
                acc.append(tid)
                cur = self.tok.decode(acc, skip_special_tokens=True)
                delta = cur[len(prev_text):]; prev_text = cur
                chunk = {"id":cid,"object":"text_completion","created":created,"model":self.model_id,
                         "choices":[{"text":delta,"index":0,"finish_reason":None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode()); self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        else:
            gen_ids = list(self.daemon.generate(ids, max_tokens))
            text = self.tok.decode(gen_ids, skip_special_tokens=True)
            self._json(200, {"id":cid,"object":"text_completion","created":created,"model":self.model_id,
                             "choices":[{"text":text,"index":0,"finish_reason":"length"}],
                             "usage":{"prompt_tokens":len(ids),"completion_tokens":len(gen_ids),"total_tokens":len(ids)+len(gen_ids)}})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B")   # vocab 262144 == 엔진 V(de-risked)
    ap.add_argument("--device", default=os.path.join(HERE,"autotune","profiles","gb10.json"))
    ap.add_argument("--model", default=os.path.join(HERE,"autotune","profiles","gemma4-31b.json"))
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    recipe = load_and_verify_recipe(args.recipe, args.device, args.model)
    quant = recipe["target"]["model"].get("quant","w4a8")
    uq = 2 if quant == "w4a8" else (1 if quant == "q4" else 0)
    model_id = recipe["target"]["model"].get("id","gemma4")
    print(f"[serve_api] recipe PASS(verify) → model={model_id} uq={uq}. daemon 起動 중(weight 로드)...", file=sys.stderr)

    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    from transformers import AutoTokenizer
    Handler.tok = AutoTokenizer.from_pretrained(args.tokenizer)
    Handler.daemon = Daemon(args.weights, uq, args.maxlen)
    Handler.model_id = model_id

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve_api] listening :{args.port} (/v1/completions greedy, /v1/models). recipe={os.path.basename(args.recipe)}", file=sys.stderr)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: Handler.daemon.close()

if __name__ == "__main__":
    main()
