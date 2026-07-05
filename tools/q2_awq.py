#!/usr/bin/env python3
"""정교 q2 STEP 0: 중요도 가중(AWQ 개념) 회복 숫자. 설계 0·한 기법·최소 스크립트.
   AWQ: calibration으로 채널별 활성 크기 s 측정 → salient 채널 weight 스케일업(W·s) 후 양자화 → dequant/s.
   → salient(활성 큰) 채널이 정밀 양자화(상대오차↓), 나머지에 오차 몰림. 시뮬 forward로 품질(perplexity/argmax)."""
import os, re, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
MODEL="Qwen/Qwen3.5-0.8B"; G=32; ALPHA=0.5
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

def qd(W,bits,G):  # 대칭 그룹 양자화→dequant
    if bits>=16: return W
    out,inn=W.shape; g=G
    while inn%g: g//=2
    Wg=W.reshape(out,inn//g,g).float(); am=Wg.abs().amax(2,keepdim=True); lv=1<<bits; h=lv//2
    s=am/(h-0.5); s=torch.where(s==0,torch.ones_like(s),s)
    q=torch.clamp(torch.round(Wg/s+(h-0.5)),0,lv-1)
    return ((q-(h-0.5))*s).reshape(out,inn).to(W.dtype)

def qd_awq(W,act,bits,G,alpha):  # 중요도 가중: s=act^alpha(정규화), W·s 양자화 후 /s
    s=(act.float()+1e-8)**alpha; s=s/s.mean(); s=s.clamp(1e-2,1e2)
    return qd(W*s[None,:],bits,G)/s[None,:]

targets={n:mod for n,mod in m.named_modules() if re.search(r"layers\.\d+\.(self_attn|mlp)\.\w+_proj$",n) and hasattr(mod,"weight")}
orig={n:mod.weight.data.clone() for n,mod in targets.items()}
print(f"대상 Linear {len(targets)}개")

TEXT=("The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. "
      "It generates most of the cell's supply of adenosine triphosphate, used as a source of chemical energy. "
      "Machine learning models are trained on large datasets to recognize patterns and make predictions. "
      "The capital of France is Paris, and the capital of Japan is Tokyo.")
ids=tok(TEXT,return_tensors="pt").input_ids
pids=tok("The capital of France is",return_tensors="pt").input_ids

# calibration: 채널별 활성 |X| 평균(입력 채널) — hook
act={};
def mkhook(n):
    def h(mod,inp,out):
        x=inp[0].detach().abs().reshape(-1,inp[0].shape[-1]).mean(0)
        act[n]=x if n not in act else act[n]+x
    return h
hs=[mod.register_forward_hook(mkhook(n)) for n,mod in targets.items()]
with torch.no_grad(): m(ids)
for h in hs: h.remove()

def run(quant_fn):
    for n,mod in targets.items(): mod.weight.data=quant_fn(n,orig[n])
    with torch.no_grad():
        lg=m(ids).logits[0]; loss=torch.nn.functional.cross_entropy(lg[:-1],ids[0,1:]); ppl=float(torch.exp(loss))
        am=lg.argmax(-1); gen=m.generate(pids,max_new_tokens=20,do_sample=False,pad_token_id=tok.eos_token_id)
        gt=tok.decode(gen[0,pids.shape[1]:],skip_special_tokens=True)
    return ppl,am,gt

ppl_fp,am_fp,_=run(lambda n,W: W)  # fp baseline
print(f"\nfp32: ppl {ppl_fp:.3f}")
for label,fn in [("q4-naive",lambda n,W:qd(W,4,G)),
                 ("q2-naive",lambda n,W:qd(W,2,G)),
                 ("q2-AWQ",  lambda n,W:qd_awq(W,act[n],2,G,ALPHA))]:
    ppl,am,gt=run(fn); agree=float((am==am_fp).float().mean())*100
    print(f"{label:9s}: ppl {ppl:.3f}({ppl/ppl_fp:.1f}×)·argmax {agree:.1f}%·gen {gt[:50]!r}")
for n,mod in targets.items(): mod.weight.data=orig[n]
print(f"\n[STEP0] AWQ가 naive q2(464×)를 얼마나 회복? (alpha={ALPHA})")
