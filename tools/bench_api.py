#!/usr/bin/env python3
"""M5 API 성능 측정 하네스 — API 계층 오버헤드 정량(3-way 선결·honest ⑦ 오귀속 방지).
★프레이밍: "dgx-serving vs vLLM"이 엔진 비교가 되게 = 우리 API 계층(HTTP/토큰화/IPC/스레딩) 오버헤드를 measured로 확인·제거.
동일 워크로드(같은 pre-tokenized 프롬프트·gen 길이·동시성)를 두 경로로 warm·back-to-back 측정:
  - L2 daemon-direct: online daemon에 pre-tokenized 태그드 요청을 파이프로 직접 주입(HTTP/HF토큰화 없음) = 엔진+IPC.
  - L3 full API: serve_api.py에 동시 HTTP 스트리밍 요청(HF토큰화 왕복·스레딩 포함) = 전체 API 스택.
갭(L2→L3) = API 계층 오버헤드. + 토크나이저 micro-bench(tokenize/detok 절대 비용, streaming detok O(G²) 확인).
지표: throughput(집계 tok/s)·TTFT(첫 토큰 지연)·tpot(토큰간 간격 — 배치 스텝시간). 측정 먼저·controlled·overclaim 금지.
사용법: <venv>/python tools/bench_api.py [N=8] [G=64]
"""
import collections, json, os, statistics, subprocess, sys, threading, time, urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS = os.path.join(REPO, "artifacts/gemma4-31b/weights"); TOKID = "google/gemma-4-E4B"
DAEMON = os.path.join(REPO, "tools", "serve_daemon_gemma4_online")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8       # 동시 요청 수(=Bmax → 한 배치로 admit)
G = int(sys.argv[2]) if len(sys.argv) > 2 else 64      # 요청당 생성 토큰 수
PORT = 8140
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKID)

BASE = ["The capital of France is", "Water is composed of hydrogen and oxygen atoms, so its formula is",
        "Two plus two equals", "The largest planet in our solar system is called",
        "The speed of light in vacuum is approximately", "Once upon a time in a distant land",
        "The mitochondria is the powerhouse of the", "In computer science, a binary tree is"]
PROMPTS = [BASE[i % len(BASE)] for i in range(N)]
PROMPT_IDS = [tok(p).input_ids for p in PROMPTS]

def sm_clock():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=clocks.sm","--format=csv,noheader,nounits"], text=True)
        return int(out.strip().split("\n")[0])
    except Exception: return -1

def stats(ttft, tpot, total_tok, wall):
    return {"throughput": total_tok/wall, "wall": wall, "total_tok": total_tok,
            "ttft_mean": statistics.mean(ttft), "ttft_p95": sorted(ttft)[int(len(ttft)*0.95)-1] if len(ttft)>1 else ttft[0],
            "tpot_mean": statistics.mean(tpot), "tpot_p95": sorted(tpot)[int(len(tpot)*0.95)-1] if len(tpot)>1 else tpot[0]}

# ─────────────────────── L2: daemon-direct(파이프, pre-tokenized) ───────────────────────
def measure_daemon_direct(warm):
    p = subprocess.Popen([DAEMON, WEIGHTS, "2", "384", str(N)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stderr:
        if "ready" in line: break
        if p.poll() is not None: raise RuntimeError("daemon 起動 실패")

    def run_batch(maxtok, base_rid):
        submit_t = {}; first_t = {}; last_t = {}; ntok = collections.Counter(); inter = collections.defaultdict(list)
        prev = {}
        t0 = time.perf_counter()
        for i in range(N):
            rid = base_rid + i
            p.stdin.write(f"{rid} {maxtok} " + " ".join(map(str, PROMPT_IDS[i])) + "\n")
            submit_t[rid] = time.perf_counter()
        p.stdin.flush()
        done = set(); expected = set(range(base_rid, base_rid+N)); gen = collections.defaultdict(list)
        while done != expected:
            l = p.stdout.readline()
            if not l: raise RuntimeError("daemon EOF")
            parts = l.split()
            if len(parts) != 2: continue
            rid = int(parts[0]); now = time.perf_counter()
            if parts[1] == "END": done.add(rid); last_t[rid] = prev.get(rid, now); continue
            gen[rid].append(int(parts[1]))
            if rid not in first_t: first_t[rid] = now
            else: inter[rid].append(now - prev[rid])
            prev[rid] = now; ntok[rid]+=1
        wall = time.perf_counter() - t0
        ttft = [first_t[r]-submit_t[r] for r in expected]
        tpot = [statistics.mean(inter[r]) for r in expected if inter[r]]
        return stats(ttft, tpot, sum(ntok.values()), wall), gen
    if warm: run_batch(8, 100000)                      # throwaway 워밍
    m, gen = run_batch(G, 0)
    p.stdin.write("SHUTDOWN\n"); p.stdin.flush(); p.stdin.close(); p.terminate(); p.wait()
    return m, gen

# ─────────────────────── L3: full API(HTTP 스트리밍) ───────────────────────
def measure_api(warm):
    srv = subprocess.Popen([sys.executable, os.path.join(REPO,"tools","serve_api.py"),
                            "--recipe","recipes/gemma4-31b-gb10.yaml","--weights",WEIGHTS,
                            "--port",str(PORT),"--bmax",str(N)], stderr=subprocess.PIPE, text=True, cwd=REPO)
    for line in srv.stderr:
        if "listening" in line: break
        if srv.poll() is not None: raise RuntimeError("serve_api 起動 실패")

    def one(prompt, maxtok, out, idx):
        body = json.dumps({"prompt":prompt,"max_tokens":maxtok,"stream":True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                     headers={"Content-Type":"application/json"})
        t_sub = time.perf_counter(); first=None; times=[]; text=""
        resp = urllib.request.urlopen(req, timeout=300)
        for raw in resp:                                # SSE 라인 스트림(이벤트=토큰 1개)
            s = raw.decode().strip()
            if not s.startswith("data: "): continue
            payload = s[6:]
            if payload == "[DONE]": break
            now = time.perf_counter()
            if first is None: first = now
            times.append(now)
            text += json.loads(payload)["choices"][0]["text"]
        out[idx] = (t_sub, first, times, text)

    def run_batch(maxtok):
        out = {}; bar = threading.Barrier(N)
        def worker(i):
            bar.wait(); one(PROMPTS[i], maxtok, out, i)
        ths=[threading.Thread(target=worker,args=(i,)) for i in range(N)]
        t0=time.perf_counter()
        for t in ths: t.start()
        for t in ths: t.join(timeout=300)
        wall=time.perf_counter()-t0
        ttft=[out[i][1]-out[i][0] for i in range(N)]
        tpot=[statistics.mean([out[i][2][k]-out[i][2][k-1] for k in range(1,len(out[i][2]))]) for i in range(N) if len(out[i][2])>1]
        total=sum(len(out[i][2]) for i in range(N))
        return stats(ttft,tpot,total,wall), {i:out[i][3] for i in range(N)}
    if warm: run_batch(8)                              # throwaway 워밍
    m, texts = run_batch(G)
    srv.terminate(); srv.wait()
    return m, texts

# ─────────────────────── 토크나이저 micro-bench(STEP A) ───────────────────────
def measure_tokenizer(gen_ids):
    K=200
    t=time.perf_counter()
    for _ in range(K): tok(PROMPTS[0])
    t_tok=(time.perf_counter()-t)/K*1000
    # 현 streaming detok(O(G²)): 매 토큰 누적 시퀀스 전체 재디코드
    t=time.perf_counter()
    for _ in range(20):
        acc=[]; prev=""
        for tid in gen_ids:
            acc.append(tid); cur=tok.decode(acc, skip_special_tokens=True); prev=cur
    t_o2=(time.perf_counter()-t)/20*1000
    # 단일 detok(전체 1회)
    t=time.perf_counter()
    for _ in range(200): tok.decode(gen_ids, skip_special_tokens=True)
    t_single=(time.perf_counter()-t)/200*1000
    return t_tok, t_o2, t_single

def show(name, m):
    print(f"  [{name}] throughput={m['throughput']:.1f} tok/s | TTFT mean={m['ttft_mean']*1000:.1f}ms p95={m['ttft_p95']*1000:.1f}ms"
          f" | tpot mean={m['tpot_mean']*1000:.1f}ms p95={m['tpot_p95']*1000:.1f}ms | wall={m['wall']:.2f}s tok={m['total_tok']}")

if __name__ == "__main__":
    print(f"[cfg] N={N} 동시 · G={G} gen토큰 · gemma4 plain greedy · SM clock={sm_clock()}MHz(측정 전)")
    print("[L2] daemon-direct(파이프·pre-tokenized, HTTP/토큰화 없음) 측정 중...")
    l2, gen = measure_daemon_direct(warm=True)
    show("L2 daemon-direct", l2)
    print(f"[warm 후 SM clock={sm_clock()}MHz]")
    print("[L3] full API(HTTP 스트리밍·HF토큰화) 측정 중...")
    l3, texts = measure_api(warm=True)
    show("L3 full API   ", l3)
    # 워크로드 동일성 교차확인: daemon-direct gen == API gen text(bit-id 유지)
    api0 = texts[0]; dir0 = tok.decode(gen[0], skip_special_tokens=True)
    print(f"[교차확인] req0 daemon-direct=={api0==dir0!s} API text (동일 워크로드·greedy)")
    print("[STEP A] 토크나이저 micro-bench...")
    t_tok, t_o2, t_single = measure_tokenizer(gen[0])
    print(f"  tokenize(prompt)={t_tok:.3f}ms/call | streaming detok O(G²) 누적(G={G})={t_o2:.2f}ms/req | 단일 detok(1회)={t_single:.3f}ms")

    # ── 갭 좌표(정직) ──
    g_tp = (1 - l3['throughput']/l2['throughput'])*100
    print(f"\n[갭] throughput L2 {l2['throughput']:.1f} → L3 {l3['throughput']:.1f} tok/s = API 계층 오버헤드 {g_tp:+.1f}%")
    print(f"[갭] TTFT L2 {l2['ttft_mean']*1000:.1f} → L3 {l3['ttft_mean']*1000:.1f}ms | tpot L2 {l2['tpot_mean']*1000:.1f} → L3 {l3['tpot_mean']*1000:.1f}ms")
    print(f"[참조] 엔진 기준선 bench_serve_gemma4(기록): ~41 tok/s(λ16 B16, nreq48) — L2가 근접하면 daemon IPC도 작음")
    print(f"[해석] tpot 스텝시간({l2['tpot_mean']*1000:.0f}ms) 대비 detok/tokenize({t_o2:.1f}/{t_tok:.2f}ms) 비중으로 지배 용의자 판별(measured)")
