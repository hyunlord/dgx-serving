#!/usr/bin/env python3
"""M5 온라인 — Serve API front (얇은 계층: OpenAI 호환 HTTP + HF 토큰화 + SSE + 온라인 연속 배칭).
2계층 아키텍처(docs/m5-api-design.md): 이 front은 계산 안 함 — HTTP 표면·text↔ids·SSE·태그드 demux만. 계산은 C++ daemon.
  起動: recipe.yaml 로드 → verify(지문·로컬 메모리 가드 — recipe.py 재사용) → daemon(serve_daemon_gemma4_online) spawn(영속 파이프).
  런타임: 동시 HTTP 연결(ThreadingHTTPServer) 각각 /v1/completions(greedy) → HF 토큰화 → 태그드 요청(req_id)을 daemon에
    비동기 주입 → daemon 배치 루프가 스텝 경계에서 admit → 연속 배칭. 단일 reader 스레드가 인터리브 출력을 req_id로 demux →
    연결별 큐 → HF 디코드 → JSON/SSE. ★front은 계산 안 함 = 온라인 배칭이 각 요청 계산 안 바꿈(daemon 게이트 증명).
★정확성 불변: daemon이 gemma4_decode를 통과만·슬롯 격리 → 동시 요청 각 출력 == 단독 서빙 bit-identical(test_serve_online).
Phase 스코프: gemma4 plain·greedy만(temperature/logprobs/n>1 미지원 — 명시 거부·무시). 동시 요청=온라인 배칭(직렬화 아님).
사용법: python tools/serve_api.py --recipe recipes/gemma4-31b-gb10.yaml --weights artifacts/gemma4-31b/weights --port 8000
"""
import argparse, json, os, queue, subprocess, sys, threading, time, uuid
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

# ---- C++ 온라인 배칭 daemon 래퍼(영속 프로세스, 태그드 비동기 주입 + 단일 reader 스레드 demux) ----
# ★직렬화 안 함: 동시 HTTP 스레드가 각자 태그드 요청을 daemon에 주입 → daemon 배치 루프가 연속 배칭.
#   stdin 쓰기만 락(라인 원자성). stdout은 단일 reader 스레드가 읽어 req_id로 연결별 큐에 분배(경합 방지).
class OnlineDaemon:
    def __init__(self, cmd):                               # cmd = [daemon_bin, args...] (모델별 레이아웃은 main이 구성)
        if not os.path.exists(cmd[0]): sys.exit(f"[serve_api] daemon 바이너리 없음: {cmd[0]} (make {os.path.relpath(cmd[0],REPO)})")
        self.p = subprocess.Popen(cmd,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.wlock = threading.Lock()                      # stdin 쓰기 직렬화(요청 라인 원자성)
        self.idlock = threading.Lock(); self._next = 0     # req_id 할당 카운터
        self.qlock = threading.Lock(); self.queues = {}    # req_id -> queue.Queue(연결별 토큰 큐)
        # readiness: daemon이 stderr에 "[online] ready" 찍을 때까지 대기(weight 로드 완료 신호)
        for line in self.p.stderr:
            print("  [daemon] "+line.rstrip(), file=sys.stderr)
            if "ready" in line: break
        if self.p.poll() is not None: sys.exit("[serve_api] daemon 起動 실패(로드 중 종료).")
        self.reader = threading.Thread(target=self._reader, daemon=True); self.reader.start()

    def _reader(self):
        """단일 스레드: daemon stdout 인터리브 출력 → req_id로 연결별 큐 분배. 'END'=완료 센티넬(None)."""
        for line in self.p.stdout:
            parts = line.split()
            if len(parts) != 2: continue
            rid = int(parts[0])
            with self.qlock: q = self.queues.get(rid)
            if q is None: continue                         # 이미 정리된 요청(방어)
            q.put(None if parts[1] == "END" else int(parts[1]))
        # daemon 종료(EOF): 대기 중 연결에 센티넬 방출(무한 대기 방지)
        with self.qlock:
            for q in self.queues.values(): q.put(None)

    def stream(self, ids, max_tokens):
        """token-ID 요청 → 태그드 비동기 주입 → 생성 token-ID를 큐에서 순서대로 yield(스트리밍)."""
        with self.idlock: rid = self._next; self._next += 1
        q = queue.Queue()
        with self.qlock: self.queues[rid] = q
        with self.wlock:                                   # 라인 원자성만(짧은 임계구역 — 배칭 안 막음)
            self.p.stdin.write(f"{rid} {max_tokens} " + " ".join(map(str, ids)) + "\n"); self.p.stdin.flush()
        try:
            while True:
                tok = q.get()                              # reader 스레드가 넣을 때까지 블록(다른 요청은 병행)
                if tok is None: break                      # END 센티넬
                yield tok
        finally:
            with self.qlock: self.queues.pop(rid, None)

    def close(self):
        try: self.p.stdin.write("SHUTDOWN\n"); self.p.stdin.flush(); self.p.stdin.close(); self.p.terminate()
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
            for tid in self.daemon.stream(ids, max_tokens):
                acc.append(tid)
                cur = self.tok.decode(acc, skip_special_tokens=True)
                delta = cur[len(prev_text):]; prev_text = cur
                chunk = {"id":cid,"object":"text_completion","created":created,"model":self.model_id,
                         "choices":[{"text":delta,"index":0,"finish_reason":None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode()); self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        else:
            gen_ids = list(self.daemon.stream(ids, max_tokens))
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
    ap.add_argument("--bmax", type=int, default=8)                 # 동시 활성 슬롯 상한(온라인 배칭 폭)
    ap.add_argument("--model", default="gemma4", choices=["gemma4","qwen"])  # daemon 종류(arg 레이아웃 상이)
    ap.add_argument("--golden", default="")                        # qwen rope golden dir(qwen 필수 — gemma4는 dummy rope)
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
    if args.model == "gemma4":
        cmd = [os.path.join(REPO,"tools","serve_daemon_gemma4_online"), args.weights, str(uq), str(args.maxlen), str(args.bmax)]
    else:                                                          # qwen(27b·35b) — golden(rope) 필수
        if not args.golden: sys.exit("[serve_api] --model qwen 은 --golden(rope dir) 필수")
        cmd = [os.path.join(REPO,"tools","serve_daemon_qwen_online"), args.weights, args.golden, str(uq), str(args.maxlen), str(args.bmax)]
    Handler.daemon = OnlineDaemon(cmd)
    Handler.model_id = model_id

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve_api] listening :{args.port} (/v1/completions greedy 온라인 배칭 Bmax={args.bmax}, /v1/models). recipe={os.path.basename(args.recipe)}", file=sys.stderr)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: Handler.daemon.close()

if __name__ == "__main__":
    main()
