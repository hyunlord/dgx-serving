#!/usr/bin/env python3
"""공식 Gemma4 assistant(google/gemma-4-12B-it-assistant, 0.4B MTP drafter)로 12B draft-spec 순 곱 measured.
구현 0(HF 공식 경로만) — E2B 판정(docs/gemma-draft-spec-verdict.md 순 곱 ~1.0×)의 재판정 도구.
판정: docs/gemma-assistant-spec-measured.md. 사용: [N_NEW=128 REPS=2 SMOKE=0] python tools/probe_assistant_spec.py
- plain greedy vs assisted greedy: 같은 프롬프트·같은 max_new_tokens, 벽시계 t/s 비율 = 순 곱.
- spec 불변식: assisted 출력 == plain greedy 출력(토큰 동일) 확인 — README "exact same quality" 검증.
- accept 계측: target/assistant forward 횟수 훅 카운트 → tokens/cycle = 1+평균 accept 길이.
- 워크로드 3종(채팅/코드/산문·E2B verdict와 같은 축) × K 스윕(constant 2/4/8 + 기본 heuristic).
GB10 warm 프로토콜: 측정 전 워밍업 1회·torch.cuda.synchronize·2회 반복 보고.
"""
import os, sys, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TARGET = "google/gemma-4-12B-it"
ASSIST = "google/gemma-4-12B-it-assistant"
N_NEW  = int(os.environ.get("N_NEW", "128"))
REPS   = int(os.environ.get("REPS", "2"))
SMOKE  = os.environ.get("SMOKE", "0") == "1"

def mem_avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) / 1e6
    return -1

print(f"[mem guard] MemAvailable={mem_avail_gb():.1f} GB (12B bf16 ~24GB 필요)", flush=True)
assert mem_avail_gb() > 40, "메모리 여유 부족 — 중단"

tok = AutoTokenizer.from_pretrained(TARGET)
print("[load] target 12B bf16 → cuda ...", flush=True)
t0 = time.time()
target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16).to("cuda")  # accelerate 없음 → CPU 로드 후 이동(unified라 등가)
target.eval()
print(f"[load] target {time.time()-t0:.0f}s | class={target.__class__.__name__}", flush=True)
t0 = time.time()
assistant = AutoModelForCausalLM.from_pretrained(ASSIST, dtype=torch.bfloat16).to("cuda")
assistant.eval()
n_params = sum(p.numel() for p in assistant.parameters())
print(f"[load] assistant {time.time()-t0:.0f}s | class={assistant.__class__.__name__} | params={n_params/1e9:.3f}B", flush=True)
print(f"[gen cfg] assistant num_assistant_tokens={assistant.generation_config.num_assistant_tokens} "
      f"schedule={assistant.generation_config.num_assistant_tokens_schedule}", flush=True)

# ★E2B verdict 세션(accept_measure.py)과 동일 프롬프트·동일 템플릿 호출 — accept 대조 공정성
WORKLOADS = {
    "chat":  [{"role": "user", "content": "Hi! How are you today? Tell me a little about yourself."}],
    "code":  [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number."}],
    "prose": [{"role": "user", "content": "Write a short descriptive paragraph about a rainy autumn morning."}],
}

def ids_for(msgs):
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(out, "input_ids"): out = out["input_ids"]
    elif isinstance(out, dict):   out = out["input_ids"]
    return out.to("cuda")

class Counter:
    """model.forward 호출 횟수(pre-hook) — prefill 포함. cycles = calls-1(첫 호출=prefill)."""
    def __init__(self, model): self.n = 0; self.h = model.register_forward_pre_hook(lambda m, a: self.inc())
    def inc(self): self.n += 1
    def close(self): self.h.remove()

@torch.inference_mode()
def run(ids, assisted, k=None, schedule=None):
    kw = dict(max_new_tokens=N_NEW, do_sample=False, use_cache=True,
              pad_token_id=tok.pad_token_id or 0)
    if assisted:
        kw["assistant_model"] = assistant
        if k is not None:
            assistant.generation_config.num_assistant_tokens = k
            assistant.generation_config.num_assistant_tokens_schedule = schedule or "constant"
    ct, ca = Counter(target), Counter(assistant)
    torch.cuda.synchronize(); t0 = time.time()
    out = target.generate(ids, **kw)
    torch.cuda.synchronize(); dt = time.time() - t0
    ct.close(); ca.close()
    new = out[0, ids.shape[1]:]
    return {"toks": new.tolist(), "n": len(new), "dt": dt, "tps": len(new)/dt,
            "target_calls": ct.n, "assist_calls": ca.n}

def fmt(r): return f"{r['tps']:6.2f} t/s (n={r['n']} dt={r['dt']:.2f}s tgt_fwd={r['target_calls']} ast_fwd={r['assist_calls']})"

results = {}
sweep = [(None, None)] if SMOKE else [(None, None), (2, "constant"), (4, "constant"), (8, "constant")]
for name, msgs in (list(WORKLOADS.items())[:1] if SMOKE else WORKLOADS.items()):
    ids = ids_for(msgs)
    print(f"\n=== workload {name} (prompt {ids.shape[1]} tok) ===", flush=True)
    _ = run(ids, assisted=False)                       # warm-up (클럭·캐시)
    plains = [run(ids, assisted=False) for _ in range(REPS)]
    plain = max(plains, key=lambda r: r["tps"])        # warm best(보수적: 최고 plain 대비)
    for r in plains: print(f"  plain    {fmt(r)}", flush=True)
    results[name] = {"plain": plain, "assisted": {}}
    for k, sch in sweep:
        tag = f"K={k or 'cfg-default'}/{sch or 'default'}"
        _ = run(ids, assisted=True, k=k, schedule=sch) # warm-up
        runs = [run(ids, assisted=True, k=k, schedule=sch) for _ in range(REPS)]
        best = max(runs, key=lambda r: r["tps"])
        same = (best["toks"] == plain["toks"])
        # 첫 target 호출=prefill → verify 사이클 = target_calls-1, tokens/cycle = n/cycles
        cyc = max(best["target_calls"] - 1, 1); tpc = best["n"] / cyc
        mult = best["tps"] / plain["tps"]
        for r in runs: print(f"  assisted[{tag}] {fmt(r)}", flush=True)
        print(f"  → {tag}: 순 곱 {mult:.2f}× | tokens/cycle {tpc:.2f} (accept≈{tpc-1:.2f}) | "
              f"출력==plain {'YES' if same else 'NO(불일치)'}", flush=True)
        results[name]["assisted"][tag] = {**best, "mult": mult, "tokens_per_cycle": tpc, "match_plain": same}

print("\n=== 요약 (순 곱 = assisted t/s ÷ plain t/s, warm best) ===")
for name, r in results.items():
    p = r["plain"]["tps"]
    line = " | ".join(f"{tag} {v['mult']:.2f}×(acc {v['tokens_per_cycle']-1:.2f},{'==' if v['match_plain'] else '≠'})"
                      for tag, v in r["assisted"].items())
    print(f"{name}: plain {p:.2f} t/s → {line}")
out = os.environ.get("OUT_JSON", "assistant_spec_results.json")   # CWD 기준(리포 오염 방지)
json.dump(results, open(out, "w"), indent=1)
print(f"[saved] {out}")
