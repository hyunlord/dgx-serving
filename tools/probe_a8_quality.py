#!/usr/bin/env python3
# 챕터2b STEP0 품질 사전측정: activation을 per-token int8 양자화했을 때 출력 토큰열이
# fp32 activation 대비 얼마나 바뀌나. 모든 nn.Linear 입력을 int8(symmetric, per-token=per-row)로
# round-trip(quantize→dequantize)하여 greedy 생성. W4A8의 activation 품질 영향 상한(worst-case:
# 작은 Linear까지 전부 양자화)을 본다. fp32 대비 토큰 첫 divergence + 변경 개수 보고.
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
NGEN  = int(sys.argv[2]) if len(sys.argv) > 2 else 64
BLK   = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # 0=per-token(whole row), 32=per-block-32 (llama.cpp q8_1식)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, trust_remote_code=True)
model.eval()

def quant_int8_per_row(x):
    # x: [..., IN]. symmetric int8. BLK=0 → per-token(whole row); BLK>0 → per-block(llama.cpp q8_1식).
    if BLK and x.shape[-1] % BLK == 0:
        sh = x.shape
        xb = x.reshape(*sh[:-1], sh[-1] // BLK, BLK)
        s = xb.abs().amax(dim=-1, keepdim=True) / 127.0
        s = torch.where(s == 0, torch.ones_like(s), s)
        xq = torch.clamp(torch.round(xb / s), -127, 127)
        return (xq * s).reshape(sh)
    s = x.abs().amax(dim=-1, keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    xq = torch.clamp(torch.round(x / s), -127, 127)
    return xq * s

_orig = torch.nn.functional.linear
SIM = {"on": False, "n": 0}
def patched_linear(input, weight, bias=None):
    if SIM["on"]:
        input = quant_int8_per_row(input)
        SIM["n"] += 1
    return _orig(input, weight, bias)

@torch.no_grad()
def greedy(prompt, sim):
    SIM["on"] = sim
    ids = tok(prompt, return_tensors="pt").input_ids
    out = []
    past = None
    cur = ids
    for _ in range(NGEN):
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        out.append(nxt)
        cur = torch.tensor([[nxt]])
    SIM["on"] = False
    return out

torch.nn.functional.linear = patched_linear

PROMPTS = [
    "The history of artificial intelligence began",
    "In a distant galaxy, a small spaceship",
    "def fibonacci(n):\n    if n <= 1:",
    "The capital of France is",
]
print(f"model={MODEL} ngen={NGEN}  (per-token int8 activation, all nn.Linear)")
total_div = 0; total = 0
for p in PROMPTS:
    a = greedy(p, False)   # fp32 X
    b = greedy(p, True)    # int8 X
    div = next((i for i in range(len(a)) if a[i] != b[i]), len(a))
    nchg = sum(1 for i in range(len(a)) if a[i] != b[i])
    total_div += nchg; total += len(a)
    print(f"  [{p[:30]!r:32}] first_div={div}/{NGEN}  changed={nchg}/{NGEN}")
print(f"linears quantized/gen-step ~ {SIM['n']// (NGEN*len(PROMPTS)) if SIM['n'] else 0}")
print(f"TOTAL token change: {total_div}/{total} = {100*total_div/total:.1f}%")
