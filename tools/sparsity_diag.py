#!/usr/bin/env python3
"""(C) sparsity 진단: FFN 활성 sparsity 실재 + 예측가능성 (measured 진단·구현 아님).
   SwiGLU: h = silu(gate(x)) * up(x) (중간활성, size=intermediate). h[j]≈0이면 down col j 안 읽어도 됨(skippable).
   측정: (1)|h| near-zero %(threshold별)·(2)zeroing 출력오차(정확성 영향)·(3)static(같은 뉴런?) vs dynamic(예측가능성)."""
import os, re, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",dtype=torch.float32).eval()
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
TEXT=("The mitochondrion is a double-membrane organelle in eukaryotic cells generating ATP. "
      "Machine learning models learn patterns from data. The capital of France is Paris. "
      "Quantum computing uses superposition and entanglement for parallel computation. "
      "Photosynthesis converts light energy into chemical energy in plants and algae.")
ids=tok(TEXT,return_tensors="pt").input_ids  # (1,T) — T 토큰 = decode 시점들 프록시
T=ids.shape[1]; print(f"토큰 {T}개 (decode 시점 프록시)")

# down_proj 입력 = h (SwiGLU 중간활성). hook으로 층별 h 수집.
H={}
def mkh(n):
    def hook(mod,inp,out): H[n]=inp[0].detach()[0]  # (T, intermediate)
    return hook
downs={n:mod for n,mod in m.named_modules() if re.search(r"layers\.\d+\.mlp\.down_proj$",n)}
hs=[mod.register_forward_hook(mkh(n)) for n,mod in downs.items()]
with torch.no_grad(): m(ids)
for h in hs: h.remove()
print(f"FFN down_proj {len(downs)}개 (intermediate={list(H.values())[0].shape[1]})")

# (1) near-zero % (|h| < thr × per-token max) + (2) 그 기여 비율(에너지)
import numpy as np
print("\n=== FFN 활성 sparsity (전 층·토큰 평균) ===")
print(f"{'thr(|h|<thr×max)':22s} {'skippable 뉴런%':>16s} {'잃는 에너지(L2)%':>18s}")
allh=torch.cat([H[n] for n in downs],0)  # (L*T, inter)
absh=allh.abs(); mx=absh.amax(1,keepdim=True)+1e-9
for thr in (0.001,0.01,0.05,0.1):
    mask=absh < thr*mx                      # skippable(작은 활성)
    skip_pct=float(mask.float().mean())*100
    lost=float((allh*mask).pow(2).sum()/allh.pow(2).sum())*100  # 잃는 L2 에너지
    print(f"  {thr:<20.3f} {skip_pct:15.1f}% {lost:17.3f}%")

# (3) static vs dynamic: 토큰마다 skippable 뉴런이 같은가(=예측 쉬움) 다른가(=predictor 필요)
print("\n=== static vs dynamic (예측가능성, thr=0.01) ===")
n0=list(downs)[len(downs)//2]  # 중간층
h=H[n0]; mask=(h.abs()<0.01*(h.abs().amax(1,keepdim=True)+1e-9))  # (T,inter)
per_tok_skip=mask.float().mean(1)*100
# 모든 토큰서 skip되는 뉴런(static) vs 토큰별로만(dynamic)
always_skip=float((mask.all(0)).float().mean())*100
ever_skip=float((mask.any(0)).float().mean())*100
print(f"  중간층: 토큰당 skippable {per_tok_skip.mean():.1f}%(±{per_tok_skip.std():.1f})")
print(f"  항상-skip(static·예측 쉬움) {always_skip:.1f}% · 가끔-skip(dynamic·predictor 필요) {ever_skip-always_skip:.1f}%")
print(f"\n[해석] skippable 높고 잃는 에너지 낮고 static 크면=sparsity lever 실재. SwiGLU면 soft(exact-zero 적음) 예상.")
