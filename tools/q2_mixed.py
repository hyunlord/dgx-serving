#!/usr/bin/env python3
"""정교 q2 STEP 0확인+A: AWQ alpha 확인 + 혼합 정밀도 스윕(민감 층 q4·품질↔bytes Pareto). 설계 0·최소 스크립트.
   혼합: 민감도(importance×q2오차) 상위 K% linear는 q4 유지·나머지 q2-AWQ. bytes=가중평균. 품질↔bytes 곡선."""
import os, re, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
MODEL="Qwen/Qwen3.5-0.8B"; G=32
tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32).eval()
def qd(W,bits,G):
    if bits>=16: return W
    out,inn=W.shape; g=G
    while inn%g: g//=2
    Wg=W.reshape(out,inn//g,g).float(); am=Wg.abs().amax(2,keepdim=True); lv=1<<bits; h=lv//2
    s=am/(h-0.5); s=torch.where(s==0,torch.ones_like(s),s)
    return ((torch.clamp(torch.round(Wg/s+(h-0.5)),0,lv-1)-(h-0.5))*s).reshape(out,inn).to(W.dtype)
def qd_awq(W,act,bits,G,alpha):
    s=(act.float()+1e-8)**alpha; s=s/s.mean(); s=s.clamp(1e-2,1e2); return qd(W*s[None,:],bits,G)/s[None,:]
targets={n:mod for n,mod in m.named_modules() if re.search(r"layers\.\d+\.(self_attn|mlp)\.\w+_proj$",n) and hasattr(mod,"weight")}
orig={n:mod.weight.data.clone() for n,mod in targets.items()}
TEXT=("The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. It generates most of "
      "the cell's supply of adenosine triphosphate. Machine learning models are trained on large datasets. "
      "The capital of France is Paris, and the capital of Japan is Tokyo.")
ids=tok(TEXT,return_tensors="pt").input_ids; pids=tok("The capital of France is",return_tensors="pt").input_ids
act={}
def mkh(n):
    def h(mod,inp,out):
        x=inp[0].detach().abs().reshape(-1,inp[0].shape[-1]).mean(0); act[n]=x if n not in act else act[n]+x
    return h
hs=[mod.register_forward_hook(mkh(n)) for n,mod in targets.items()]
with torch.no_grad(): m(ids)
for h in hs: h.remove()
# 민감도 = importance(act 합) × q2 재구성오차 × 파라미터수 (q2로 잃는 영향 프록시)
sens={}
for n in targets:
    W=orig[n]; werr=float((W-qd(W,2,G)).norm()/(W.norm()+1e-9)); sens[n]=werr*float(act[n].sum())*W.numel()
ranked=sorted(targets,key=lambda n:-sens[n])  # 민감 큰 순
def run(fn):
    for n,mod in targets.items(): mod.weight.data=fn(n,orig[n])
    with torch.no_grad():
        lg=m(ids).logits[0]; ppl=float(torch.exp(torch.nn.functional.cross_entropy(lg[:-1],ids[0,1:]))); am=lg.argmax(-1)
        gt=tok.decode(m.generate(pids,max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id)[0,pids.shape[1]:],skip_special_tokens=True)
    return ppl,am,gt
ppl_fp,am_fp,_=run(lambda n,W:W); print(f"fp32 ppl {ppl_fp:.3f}")
def bytesw(kq4):  # 상위 kq4 linear q4(5bit)·나머지 q2(3bit) 가중평균 bits/w
    tot=sum(orig[n].numel() for n in targets); q4p=sum(orig[n].numel() for n in ranked[:kq4]); return (q4p*5+(tot-q4p)*3)/tot
print("\n=== AWQ alpha 확인(오튜닝 아닌지) ===")
for a in (0.5,1.0):
    ppl,am,_=run(lambda n,W:qd_awq(W,act[n],2,G,a)); print(f"  q2-AWQ a={a}: ppl {ppl:.1f}({ppl/ppl_fp:.0f}×)·argmax {float((am==am_fp).float().mean())*100:.1f}%")
print("\n=== 혼합 정밀도(민감 상위 K% q4·나머지 q2-AWQ, 품질↔bytes) ===")
N=len(ranked)
for frac in (0.0,0.25,0.5,0.75,1.0):
    kq4=int(N*frac); q4set=set(ranked[:kq4])
    ppl,am,gt=run(lambda n,W:(qd(W,4,G) if n in q4set else qd_awq(W,act[n],2,G,0.5)))
    print(f"  q4={frac*100:3.0f}%({kq4}/{N}) bits/w={bytesw(kq4):.2f}: ppl {ppl:.2f}({ppl/ppl_fp:.1f}×)·argmax {float((am==am_fp).float().mean())*100:.1f}%·gen {gt[:40]!r}")
for n,mod in targets.items(): mod.weight.data=orig[n]
print("\n[해석] q4=5bit·q2=3bit. 품질 q4근접(1.1×) 되는 최소 bits/w = sub-q4 실현점. 안 되면 q2 한계.")
