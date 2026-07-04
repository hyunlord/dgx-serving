#!/usr/bin/env python3
"""3-way(2-way) HTTP 서빙 공정 비교 클라이언트 — dgx-serving vs llama.cpp(vs vLLM 되면).
methodology(docs/m5-3way-methodology.md §2-3): 단일 부하 생성기 → 여러 /v1/completions 엔드포인트를 같은 trace로.
  공정 조건: 같은 프롬프트·같은 max_tokens·같은 동시성·warm(cold-클럭 방지)·HTTP 측정 지점(TTFT/tpot/throughput).
  ★엔진별 양자화/토크나이저 다름 → 속도만 대조(품질 별도). tok/s는 각 엔진 완성 토큰 기준(max_tokens 고정=공정).
사용법: python3 tools/bench_3way.py "dgx=http://127.0.0.1:8000" "llama=http://127.0.0.1:8080" [N=8] [G=64]
"""
import json, statistics, sys, threading, time, urllib.request

eps = []; nums=[]
for a in sys.argv[1:]:
    if "=" in a and a.split("=")[1].startswith("http"): name,url=a.split("=",1); eps.append((name,url.rstrip("/")))
    elif a.isdigit(): nums.append(int(a))            # 위치 인자(순서대로 N, G) — 센티넬 충돌 버그 방지(첫 값이 8이어도 정확)
N = nums[0] if len(nums)>=1 else 8                    # 동시 요청 수
G = nums[1] if len(nums)>=2 else 64                   # 요청당 생성 토큰 수
if not eps: sys.exit("사용법: bench_3way.py name=url [name=url ...] [N] [G]")
PROMPTS=["The capital of France is","Water is composed of","Two plus two equals","The largest planet is",
         "In machine learning, a neural network","The speed of light is approximately","Once upon a time","The mitochondria is"]
PROMPTS=[PROMPTS[i%len(PROMPTS)] for i in range(N)]

def stream_one(url, prompt, maxtok, out, idx):
    body=json.dumps({"prompt":prompt,"max_tokens":maxtok,"stream":True,"temperature":0}).encode()
    req=urllib.request.Request(url+"/v1/completions",data=body,headers={"Content-Type":"application/json"})
    t_sub=time.perf_counter(); first=None; times=[]; n=0
    try:
        resp=urllib.request.urlopen(req,timeout=300)
        for raw in resp:
            s=raw.decode("utf-8","ignore").strip()
            if not s.startswith("data: "): continue
            p=s[6:]
            if p=="[DONE]": break
            try: j=json.loads(p)
            except Exception: continue
            now=time.perf_counter()
            if first is None: first=now
            times.append(now); n+=1
        out[idx]=(t_sub,first,times,n)
    except Exception as e: out[idx]=("ERR",repr(e)[:120],[],0)

def measure(name,url,maxtok):
    out={}; bar=threading.Barrier(N)
    def wk(i):
        try: bar.wait()
        except Exception: pass
        stream_one(url,PROMPTS[i],maxtok,out,i)
    ths=[threading.Thread(target=wk,args=(i,)) for i in range(N)]
    t0=time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join(timeout=300)
    wall=time.perf_counter()-t0
    errs=[out[i][1] for i in range(N) if i in out and out[i][0]=="ERR"]
    if errs: return {"err":errs[0]}
    ttft=[out[i][1]-out[i][0] for i in range(N) if out[i][1]]
    tpot=[statistics.mean([out[i][2][k]-out[i][2][k-1] for k in range(1,len(out[i][2]))]) for i in range(N) if len(out[i][2])>1]
    tot=sum(out[i][3] for i in range(N))
    return {"throughput":tot/wall if wall>0 else 0,"ttft_mean":statistics.mean(ttft) if ttft else 0,
            "ttft_p95":sorted(ttft)[int(len(ttft)*0.95)-1] if len(ttft)>1 else (ttft[0] if ttft else 0),
            "tpot_mean":statistics.mean(tpot) if tpot else 0,"wall":wall,"tok":tot}

def sm_clock():
    try: import subprocess; return int(subprocess.check_output(["nvidia-smi","--query-gpu=clocks.sm","--format=csv,noheader,nounits"],text=True).strip().split("\n")[0])
    except Exception: return -1

print(f"[cfg] N={N} 동시 · G={G} gen · 엔드포인트: {[n for n,_ in eps]} · SM clock={sm_clock()}MHz")
res={}
for name,url in eps:
    print(f"[warm] {name} ({url}) ...")
    try: measure(name,url,8)   # throwaway warm
    except Exception as e: print(f"  warm 실패: {e}")
    m=measure(name,url,G); res[name]=m
    if "err" in m: print(f"  [{name}] 측정 실패: {m['err']}")
    else: print(f"  [{name}] throughput={m['throughput']:.1f} tok/s | TTFT={m['ttft_mean']*1000:.0f}ms(p95 {m['ttft_p95']*1000:.0f}) | tpot={m['tpot_mean']*1000:.1f}ms | wall={m['wall']:.2f}s tok={m['tok']}")
print(f"[warm 후 SM clock={sm_clock()}MHz]")
print("\n[3-way 좌표] (같은 프롬프트·max_tokens·동시성·warm·HTTP 측정 — 속도만, 품질/양자화 별도)")
ok=[n for n in res if "err" not in res[n]]
for n in ok:
    m=res[n]; print(f"  {n:8s}: throughput {m['throughput']:6.1f} tok/s · TTFT {m['ttft_mean']*1000:5.0f}ms · tpot {m['tpot_mean']*1000:5.1f}ms")
if len(ok)>=2:
    base=ok[0]
    for n in ok[1:]:
        r=res[base]['throughput']/res[n]['throughput'] if res[n]['throughput'] else 0
        print(f"  → {base} vs {n} throughput 비율 {r:.2f}× ({'우위' if r>1 else '열위'})")
