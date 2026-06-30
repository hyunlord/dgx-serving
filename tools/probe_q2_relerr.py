#!/usr/bin/env python3
"""챕터9 2bit 품질 de-risk (1단계): reconstruction relerr — 순진 2bit 붕괴도 + scheme/block 회복.
   forward 불요(GDN fla 크래시·108GB fp32 회피) — 실제 27B weight를 양자화→역양자화 relerr 측정.
   relerr = ||W - dequant(quant(W))|| / ||W||. q4(우리 엔진 천장) 대비 q2 변종이 얼마나 무너지나(붕괴도) +
   더 나은 scheme(minmax)·finer block이 회복하나. 유효 bit/weight(fp16 scale 오버헤드 포함)도 — 품질/바이트 트레이드.
   사용법: python tools/probe_q2_relerr.py [snapshot_dir]
"""
import sys, glob, json, os, numpy as np
try: import torch
except Exception as e: print("torch 필요:", e); sys.exit(1)
from safetensors import safe_open

SNAP = sys.argv[1] if len(sys.argv)>1 else glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/*/"))[0]
idx = json.load(open(glob.glob(SNAP+"*index.json")[0]))["weight_map"]

def get(key):
    shard = idx[key];
    with safe_open(os.path.join(SNAP, shard), framework="pt") as f:
        return f.get_tensor(key).float()

def relerr(W, Wdq): return (torch.norm(W-Wdq)/torch.norm(W)).item()

def q_sym(W, bits, blk):  # 우리 엔진 스킴: scale=maxabs/(2^(b-1)-1), offset 2^(b-1)
    OUT,IN=W.shape;
    if IN%blk: return W
    wb=W.reshape(OUT,IN//blk,blk); off=2**(bits-1); qmax=2**bits-1
    s=wb.abs().amax(-1,keepdim=True)/(off-1 if off>1 else 1); s=torch.where(s==0,torch.ones_like(s),s)
    q=torch.clamp(torch.round(wb/s)+off,0,qmax)
    return ((q-off)*s).reshape(OUT,IN)

def q_minmax(W, bits, blk):  # min-max 비대칭(저비트 최적): 4레벨 전부 활용
    OUT,IN=W.shape
    if IN%blk: return W
    wb=W.reshape(OUT,IN//blk,blk); qmax=2**bits-1
    lo=wb.amin(-1,keepdim=True); hi=wb.amax(-1,keepdim=True)
    s=(hi-lo)/qmax; s=torch.where(s==0,torch.ones_like(s),s)
    q=torch.clamp(torch.round((wb-lo)/s),0,qmax)
    return (q*s+lo).reshape(OUT,IN)

KEYS={
 "L0 gdn in_proj_qkv":"model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
 "L0 mlp gate":"model.language_model.layers.0.mlp.gate_proj.weight",
 "L0 mlp down":"model.language_model.layers.0.mlp.down_proj.weight",
 "L3 attn q_proj":"model.language_model.layers.3.self_attn.q_proj.weight",
 "L3 attn o_proj":"model.language_model.layers.3.self_attn.o_proj.weight",
 "lm_head":"lm_head.weight",
}
# scheme: (name, fn, bits, blk, eff_bits/weight[fp16 scale])  — minmax는 scale+lo 2개/블록
SCH=[("q4-sym b32(엔진)", lambda W:q_sym(W,4,32), 4+16/32),
     ("q2-sym b32(naive)", lambda W:q_sym(W,2,32), 2+16/32),
     ("q2-minmax b32",     lambda W:q_minmax(W,2,32), 2+2*16/32),
     ("q2-minmax b16",     lambda W:q_minmax(W,2,16), 2+2*16/16)]

print(f"snapshot={SNAP.split('/')[-2][:16]}")
print(f"{'matrix':22} | "+" | ".join(f"{n.split('(')[0]:14}" for n,_,_ in SCH))
print(f"{'eff bit/w':22} | "+" | ".join(f"{b:14.2f}" for _,_,b in SCH))
print("-"*100)
agg={n:[] for n,_,_ in SCH}
for label,key in KEYS.items():
    if key not in idx: print(f"{label:22} | (키 없음)"); continue
    W=get(key); row=[]
    for n,fn,_ in SCH:
        e=relerr(W,fn(W)); agg[n].append(e); row.append(f"{e:.4f}")
    print(f"{label:22} | "+" | ".join(f"{r:14}" for r in row)+f"   {tuple(W.shape)}")
print("-"*100)
print(f"{'MEAN relerr':22} | "+" | ".join(f"{np.mean(agg[n]):14.4f}" for n,_,_ in SCH))
q4m=np.mean(agg["q4-sym b32(엔진)"])
print(f"\n해석: q2/q4 relerr 배수 = 붕괴도. q2-sym(naive)이 q4 대비 ×N. minmax·finer가 q4쪽으로 회복하나.")
for n,_,b in SCH:
    if n.startswith("q2"): print(f"  {n:20} relerr {np.mean(agg[n]):.4f} = q4의 {np.mean(agg[n])/q4m:.1f}× (유효 {b:.2f}bit/w, q4 4.5 대비 {4.5/b:.2f}× 바이트↓)")
