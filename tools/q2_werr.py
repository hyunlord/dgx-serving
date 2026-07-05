#!/usr/bin/env python3
"""q2 극한양자화 STEP 0/A: weight 재구성 오차 study (품질 저하 근원 measured).
   대칭 b-bit 그룹 양자화: scale=absmax/(half-0.5), q=clip(round(w/scale+half-0.5),0,2^b-1), deq=(q-(half-0.5))*scale.
   측정: rel L2 오차(||W-deq||/||W||) per (bits, group) + bytes/weight(= bits + 32/group scale 오버헤드).
   ★품질 프록시: weight 오차↑ = 출력 오차↑ = 품질↓. full argmax/perplexity는 q2 커널 후속."""
import os, numpy as np
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from transformers import AutoModelForCausalLM

def quant_err(W, bits, G):
    # W: (out,in) fp32. 그룹=in축 G개씩. scale=absmax/(half-0.5).
    out,inn = W.shape
    if inn % G: G = [g for g in (32,16,8,inn) if inn%g==0][0]
    Wg = W.reshape(out, inn//G, G)
    absmax = np.abs(Wg).max(axis=2, keepdims=True)  # (out, inn/G, 1)
    levels = 1<<bits; half = levels//2
    scale = absmax/(half-0.5); scale[scale==0]=1
    q = np.clip(np.round(Wg/scale + (half-0.5)), 0, levels-1)
    deq = (q-(half-0.5))*scale
    err = np.linalg.norm((Wg-deq).ravel()) / (np.linalg.norm(Wg.ravel())+1e-12)
    bytes_per_w = bits + 32.0/G   # 데이터 비트 + fp32 scale/G
    return err, bytes_per_w

print("Qwen3.5-0.8B 로드(fp32 weight 오차 study)...")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32)
sd = m.state_dict()
# 대표 텐서: 중간층 gate/up/down/q/o
import re
L = 6  # 중간층
cands = [k for k in sd if re.search(rf"layers\.{L}\.(mlp|self_attn)\.(gate|up|down|q|k|v|o)_proj\.weight", k)]
print(f"대표 텐서 {len(cands)}개 (layer {L})")
schemes = [("q4-g32",4,32),("q2-g32",2,32),("q2-g16",2,16),("q2-g8",2,8),("q3-g32",3,32)]
print(f"\n{'텐서':32s} " + " ".join(f"{n:>10s}" for n,_,_ in schemes))
aggr = {n:[] for n,_,_ in schemes}
for k in cands:
    W = sd[k].numpy().astype(np.float32)
    row = f"{k.split('.',2)[-1][:32]:32s} "
    for n,b,g in schemes:
        e,bpw = quant_err(W,b,g); aggr[n].append(e)
        row += f"{e*100:9.2f}% "
    print(row)
print(f"\n{'평균 rel L2 오차':32s} " + " ".join(f"{np.mean(aggr[n])*100:9.2f}% " for n,_,_ in schemes))
print(f"{'bytes/weight':32s} " + " ".join(f"{(b+32.0/g):9.2f}  " for _,b,g in schemes))
print("\n[해석] q4-g32=5bit 기준. q2-g32=3bit(0.6×=목표). 오차 배수 = 품질 저하 프록시.")
