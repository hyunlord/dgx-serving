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

# 웹 채팅 UI(정적 HTML 1장 — 외부 의존 0). tools/chat.html에서 로드(리포 자립); 없으면 최소 폴백.
try:
    with open(os.path.join(HERE, "chat.html"), encoding="utf-8") as _f: WEBUI_HTML = _f.read()
except OSError:
    WEBUI_HTML = "<!doctype html><meta charset=utf-8><title>dgx-serving</title><p>chat.html not found</p>"

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
    def __init__(self, cmd, env=None):                     # cmd = [daemon_bin, args...] · env = STOP_IDS 등 주입(EOS 종료)
        if not os.path.exists(cmd[0]): sys.exit(f"[serve_api] daemon 바이너리 없음: {cmd[0]} (make {os.path.relpath(cmd[0],REPO)})")
        self.p = subprocess.Popen(cmd, env=env,
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
            if len(parts) < 2: continue
            rid = int(parts[0])
            with self.qlock: q = self.queues.get(rid)
            if q is None: continue                         # 이미 정리된 요청(방어)
            if parts[1] == "END":                          # "rid END <reason>"(stop/length) — daemon 권위 종료 사유
                q.put(("__END__", parts[2] if len(parts) > 2 else "stop"))
            elif len(parts) == 2:                          # "rid <tok>"
                q.put(int(parts[1]))
        # daemon 종료(EOF): 대기 중 연결에 센티넬 방출(무한 대기 방지)
        with self.qlock:
            for q in self.queues.values(): q.put(("__END__", "length"))

    def stream(self, ids, max_tokens, out=None):
        """token-ID 요청 → 태그드 비동기 주입 → 생성 token-ID를 큐에서 순서대로 yield(스트리밍).
        out(dict, 선택): 종료 시 out['finish_reason']에 daemon 권위 사유(stop=EOS / length=max_tokens/maxlen) 기록."""
        with self.idlock: rid = self._next; self._next += 1
        q = queue.Queue()
        with self.qlock: self.queues[rid] = q
        with self.wlock:                                   # 라인 원자성만(짧은 임계구역 — 배칭 안 막음)
            self.p.stdin.write(f"{rid} {max_tokens} " + " ".join(map(str, ids)) + "\n"); self.p.stdin.flush()
        try:
            while True:
                tok = q.get()                              # reader 스레드가 넣을 때까지 블록(다른 요청은 병행)
                if isinstance(tok, tuple) and tok[0] == "__END__":  # 종료 센티넬(사유 포함)
                    if out is not None: out["finish_reason"] = tok[1]
                    break
                yield tok
        finally:
            with self.qlock: self.queues.pop(rid, None)

    def close(self):
        try: self.p.stdin.write("SHUTDOWN\n"); self.p.stdin.flush(); self.p.stdin.close(); self.p.terminate()
        except Exception: pass

# ---- 채팅 템플릿(정확성 지점) — messages를 모델별 chat template로 감싸 토큰화 ----
#   ★단일 소스: /v1/chat/completions과 template 게이트(test_chat_template.py)가 이 함수를 공유 →
#   손수 concat 같은 오적용 차단. HF apply_chat_template를 add_generation_prompt=True로 호출(모델별 자동).
def chat_ids(tok, messages):
    if getattr(tok, "chat_template", None) is None:
        raise ValueError(f"tokenizer({tok.name_or_path})에 chat_template 없음 — /v1/chat/completions 불가(--tokenizer로 chat 모델 지정)")
    # ★transformers 버전에 따라 tokenize=True가 list[int] 또는 BatchEncoding(dict) 반환 → 둘 다 안전하게 input_ids 추출.
    #   (dict를 그냥 list()하면 키 ['input_ids','attention_mask']를 토큰으로 오인 = 조용한 붕괴 — 방지.)
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(out, "input_ids"): out = out["input_ids"]   # BatchEncoding
    elif isinstance(out, dict):   out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)): out = out[0]   # 배치 중첩 해제
    return [int(x) for x in out]

# ---- OpenAI 호환 HTTP 핸들러 ----
class Handler(BaseHTTPRequestHandler):
    daemon = None; tok = None; model_id = "gemma4"; quant = "w4a8"; spec = False  # 클래스 변수(서버 起動 시 주입)
    def log_message(self, *a): pass                       # 접근 로그 억제

    def _json(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p == "/v1/models":
            self._json(200, {"object":"list","data":[{"id":self.model_id,"object":"model","owned_by":"dgx-serving",
                                                       "quant":self.quant,"spec":self.spec}]})
        elif p in ("", "/index.html"):                    # 웹 채팅 UI(정적 HTML 1장 — 의존성 0)
            b = WEBUI_HTML.encode()
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
        else: self._json(404, {"error":{"message":"not found","type":"invalid_request_error"}})

    # 미지원 파라미터(greedy·bit-id 전제 깨는 것) 명시 거부 — 공통.
    def _reject_unsupported(self, req):
        for k in ("logprobs","n","best_of","suffix"):
            if req.get(k) not in (None,1):
                self._json(400, {"error":{"message":f"unsupported: {k} (greedy only)","type":"invalid_request_error"}}); return True
        if req.get("temperature",0) not in (0,None,0.0):
            print(f"  [경고] temperature={req.get('temperature')} 무시(greedy만)", file=sys.stderr)
        return False

    def do_POST(self):
        p = self.path.rstrip("/")
        if p not in ("/v1/completions", "/v1/chat/completions"):
            return self._json(404, {"error":{"message":"지원: /v1/completions, /v1/chat/completions","type":"invalid_request_error"}})
        try:
            n = int(self.headers.get("Content-Length",0)); req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error":{"message":f"bad json: {e}","type":"invalid_request_error"}})
        if self._reject_unsupported(req): return
        chat = (p == "/v1/chat/completions")
        max_tokens = int(req.get("max_completion_tokens", req.get("max_tokens",256))); stream = bool(req.get("stream",False))
        # 입력 → token ids: chat=템플릿 적용(정확성 본체) / completions=원시 프롬프트 토큰화(기존 경로 불변)
        try:
            if chat:
                msgs = req.get("messages",[])
                if not isinstance(msgs, list) or not msgs:
                    return self._json(400, {"error":{"message":"messages 필요","type":"invalid_request_error"}})
                ids = chat_ids(self.tok, msgs)
            else:
                prompt = req.get("prompt","")
                if isinstance(prompt, list): prompt = prompt[0] if prompt else ""
                ids = self.tok(prompt).input_ids           # HF 토큰화(BOS 포함 — Gemma 관례)
        except Exception as e:
            return self._json(400, {"error":{"message":str(e),"type":"invalid_request_error"}})
        self._generate(ids, max_tokens, stream, chat)

    # 생성 실행(스트림/논스트림 공통) — daemon.stream(불변) + incremental detok + 메트릭(front 계산·daemon 불변).
    def _generate(self, ids, max_tokens, stream, chat):
        cid = ("chatcmpl-" if chat else "cmpl-")+uuid.uuid4().hex[:24]; created = int(time.time())
        obj = "chat.completion.chunk" if (chat and stream) else ("chat.completion" if chat else "text_completion")
        t0 = time.time(); t_first = None; acc = []; prev_text = ""
        rinfo = {"finish_reason": "length"}                # daemon이 END 사유(stop=EOS/length) 기록 → 권위 finish_reason
        if stream:
            self.send_response(200); self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-cache"); self.end_headers()
            if chat:                                       # chat 관례: 첫 청크에 role
                self._sse({"id":cid,"object":obj,"created":created,"model":self.model_id,
                           "choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":None}]})
            for tid in self.daemon.stream(ids, max_tokens, rinfo):
                if t_first is None: t_first = time.time()
                acc.append(tid)
                cur = self.tok.decode(acc, skip_special_tokens=True)
                delta = cur[len(prev_text):]; prev_text = cur
                ch = ({"delta":{"content":delta}} if chat else {"text":delta})
                self._sse({"id":cid,"object":obj,"created":created,"model":self.model_id,
                           "choices":[{**ch,"index":0,"finish_reason":None}]})
            # 종료 청크: finish_reason + usage + timings(메트릭 — 마지막 이벤트에 포함)
            tm = self._timings(len(ids), acc, t0, t_first)
            fr = rinfo["finish_reason"]                    # daemon 권위(stop=EOS 만남 / length=max_tokens·maxlen)
            fin = ({"delta":{}} if chat else {"text":""})
            self._sse({"id":cid,"object":obj,"created":created,"model":self.model_id,
                       "choices":[{**fin,"index":0,"finish_reason":fr}], "usage":tm["usage"], "timings":tm["timings"]})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        else:
            for tid in self.daemon.stream(ids, max_tokens, rinfo):
                if t_first is None: t_first = time.time()
                acc.append(tid)
            text = self.tok.decode(acc, skip_special_tokens=True)
            tm = self._timings(len(ids), acc, t0, t_first)
            fr = rinfo["finish_reason"]                    # daemon 권위(stop=EOS / length=max_tokens·maxlen)
            choice = ({"message":{"role":"assistant","content":text}} if chat else {"text":text})
            self._json(200, {"id":cid,"object":obj,"created":created,"model":self.model_id,
                             "choices":[{**choice,"index":0,"finish_reason":fr}], "usage":tm["usage"], "timings":tm["timings"]})

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode()); self.wfile.flush()

    # 메트릭(캡처의 "N tokens · S s · X t/s") — front 관측(daemon 불변). 측정 지점: 요청 주입~마지막 토큰.
    def _timings(self, n_prompt, gen_ids, t0, t_first):
        n = len(gen_ids); t_end = time.time()
        elapsed = t_end - t0; ttft = (t_first - t0) if t_first else 0.0
        tps = (n / elapsed) if elapsed > 0 else 0.0        # client-observed(주입~완료 기준) — 정직 명시
        return {"usage":{"prompt_tokens":n_prompt,"completion_tokens":n,"total_tokens":n_prompt+n},
                "timings":{"completion_tokens":n,"elapsed_s":round(elapsed,3),"ttft_s":round(ttft,3),
                           "tokens_per_s":round(tps,2),"measured":"client-observed (submit→last token)"}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="")                        # 있으면 verify(지문·메모리); 없으면 직접 서빙/벤치(--uq)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B")   # vocab 262144 == 엔진 V(de-risked)
    ap.add_argument("--device", default=os.path.join(HERE,"autotune","profiles","gb10.json"))
    ap.add_argument("--model-profile", default=os.path.join(HERE,"autotune","profiles","gemma4-31b.json"))  # recipe verify용 프로파일 json(--model과 별개)
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--bmax", type=int, default=8)                 # 동시 활성 슬롯 상한(온라인 배칭 폭)
    ap.add_argument("--model", default="gemma4", choices=["gemma4","qwen","spec"])  # daemon 종류(arg 레이아웃 상이; spec=27b MTP spec)
    ap.add_argument("--golden", default="")                        # qwen/spec rope+MTP golden dir(필수 — gemma4는 dummy rope)
    ap.add_argument("--uq", type=int, default=2)                   # recipe 없을 때 양자화(2=W4A8·1=q4·0=fp32)
    ap.add_argument("--model-id", default="")                      # recipe 없을 때 모델 표시명
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="바인드 주소(기본 127.0.0.1=로컬만). 0.0.0.0=내부망 전체 노출 — ★인증 없는 서버라 내부망 전용·공유기 포트포워딩으로 외부 노출 금지")
    args = ap.parse_args()

    if args.recipe:
        recipe = load_and_verify_recipe(args.recipe, args.device, args.model_profile)
        quant = recipe["target"]["model"].get("quant","w4a8")
        uq = 2 if quant == "w4a8" else (1 if quant == "q4" else 0)
        model_id = recipe["target"]["model"].get("id","gemma4")
        print(f"[serve_api] recipe PASS(verify) → model={model_id} uq={uq}. daemon 起動 중(weight 로드)...", file=sys.stderr)
    else:
        uq = args.uq; model_id = args.model_id or f"{args.model}-direct"
        print(f"[serve_api] recipe 없음(직접 서빙/벤치) → uq={uq} model={model_id}. daemon 起動 중(weight 로드)...", file=sys.stderr)

    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    from transformers import AutoTokenizer
    Handler.tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # ★EOS/stop 토큰 = generation_config.eos_token_id 권위(다중 가능 — gemma [1,106,50]·qwen [248046,248044]).
    #   tokenizer.eos_token_id는 단일이라 chat 종료 토큰(<end_of_turn>/<|im_end|>)을 놓칠 수 있어 gen_config 우선.
    stop_ids = []
    try:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained(args.tokenizer)
        e = gc.eos_token_id
        stop_ids = list(e) if isinstance(e,(list,tuple)) else ([e] if e is not None else [])
    except Exception as ex:
        print(f"  [경고] generation_config 로드 실패({ex}) — tokenizer.eos_token_id로 폴백", file=sys.stderr)
    if Handler.tok.eos_token_id is not None and Handler.tok.eos_token_id not in stop_ids:
        stop_ids.append(Handler.tok.eos_token_id)          # 폴백/보강(중복 제거)
    daemon_env = {**os.environ, "STOP_IDS": ",".join(str(x) for x in stop_ids)}
    print(f"  [stop] EOS 종료 토큰 IDs = {stop_ids} (generation_config 권위)", file=sys.stderr)
    if args.model == "gemma4":
        cmd = [os.path.join(REPO,"tools","serve_daemon_gemma4_online"), args.weights, str(uq), str(args.maxlen), str(args.bmax)]
    elif args.model == "spec":                                     # 27b MTP spec(draft→verify→accept) — golden(rope+MTP) 필수
        if not args.golden: sys.exit("[serve_api] --model spec 은 --golden(rope+MTP dir) 필수")
        cmd = [os.path.join(REPO,"tools","serve_daemon_qwen_spec_online"), args.weights, args.golden, str(uq), str(args.bmax)]  # spec: <w> <golden> <uq> <Bmax>(maxlen=PMAX·K/W 기본)
    else:                                                          # qwen(27b·35b plain) — golden(rope) 필수
        if not args.golden: sys.exit("[serve_api] --model qwen 은 --golden(rope dir) 필수")
        cmd = [os.path.join(REPO,"tools","serve_daemon_qwen_online"), args.weights, args.golden, str(uq), str(args.maxlen), str(args.bmax)]
    Handler.daemon = OnlineDaemon(cmd, env=daemon_env)
    Handler.model_id = model_id
    Handler.quant = {2:"w4a8", 1:"q4", 0:"fp32"}.get(uq, "w4a8")
    Handler.spec = (args.model == "spec")                  # 모델 배지(/v1/models·웹UI)용

    class _Server(ThreadingHTTPServer):
        request_queue_size = 128       # SYN backlog 상향(기본 5) — 동시 연결 폭주(초과구독)서 커널 리셋 방지
        daemon_threads = True          # 종료 시 핸들러 스레드 대기 안 함(빠른 셧다운)
        allow_reuse_address = True     # 재起動 시 TIME_WAIT 포트 재사용
    httpd = _Server((args.host, args.port), Handler)
    print(f"[serve_api] listening {args.host}:{args.port} — 웹 채팅 UI http://localhost:{args.port}/ · "
          f"/v1/completions · /v1/chat/completions · /v1/models (greedy 온라인 배칭 Bmax={args.bmax}, model={model_id})", file=sys.stderr)
    if args.host == "0.0.0.0":
        print("  [보안] 0.0.0.0 = 내부망 전체 노출·인증 없음 — 신뢰된 내부망에서만 사용, 외부 포트포워딩 금지", file=sys.stderr)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: Handler.daemon.close()

if __name__ == "__main__":
    main()
