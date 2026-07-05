#!/usr/bin/env python3
"""q2 measured-first: naive q2 forward 품질 숫자 (설계 0·최소 스크립트).
   PyTorch 시뮬레이션: weight를 fp/q4/naive-q2로 양자화(dequant to fp)→forward→perplexity + argmax 일치율 + 생성 스팟.
   ★실 출력 품질(weight 오차 프록시 아님). 0.8b(안전·fits·naive-q2 붕괴는 모델-일반). group=32(엔진 QBLK)."""
import os, re, numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
MODEL="Qwen/Qwen3.5-0.8B"; G=32
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

def quant_dequant(W, bits, G):
    if bits>=16: return W
    out,inn=W.shape; g=G
    while inn%g: g//=2
    Wg=W.reshape(out,inn//g,g).float()
    absmax=Wg.abs().amax(dim=2,keepdim=True); levels=1<<bits; half=levels//2
    scale=absmax/(half-0.5); scale=torch.where(scale==0,torch.ones_like(scale),scale)
    q=torch.clamp(torch.round(Wg/scale+(half-0.5)),0,levels-1)
    return ((q-(half-0.5))*scale).reshape(out,inn).to(W.dtype)

# 대상 Linear(transformer block matmul) — embed/norm/lm_head fp(엔진도 특수취급)
targets={n:mod for n,mod in m.named_modules() if re.search(r"layers\.\d+\.(self_attn|mlp)\.\w+_proj$",n) and hasattr(mod,"weight")}
orig={n:mod.weight.data.clone() for n,mod in targets.items()}
print(f"양자화 대상 Linear {len(targets)}개 (embed/norm/lm_head 제외)")

TEXT=("The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. "
      "It generates most of the cell's supply of adenosine triphosphate, used as a source of chemical energy. "
      "Machine learning models are trained on large datasets to recognize patterns and make predictions. "
      "The capital of France is Paris, and the capital of Japan is Tokyo.")
ids=tok(TEXT,return_tensors="pt").input_ids
PROMPT="The capital of France is"; pids=tok(PROMPT,return_tensors="pt").input_ids

def run(bits):
    for n,mod in targets.items(): mod.weight.data=quant_dequant(orig[n],bits,G)
    with torch.no_grad():
        out=m(ids); logits=out.logits[0]           # (T,V)
        loss=torch.nn.functional.cross_entropy(logits[:-1], ids[0,1:]); ppl=float(torch.exp(loss))
        am=logits.argmax(-1)                        # (T,)
        gen=m.generate(pids,max_new_tokens=20,do_sample=False,pad_token_id=tok.eos_token_id)
        gtext=tok.decode(gen[0,pids.shape[1]:],skip_special_tokens=True)
    return ppl, am, gtext

print("\n=== fp32 baseline ==="); ppl_fp,am_fp,gen_fp=run(16); print(f"  perplexity={ppl_fp:.3f} | gen: {gen_fp!r}")
res={}
for bits,name in [(4,"q4"),(2,"naive-q2"),(3,"q3")]:
    ppl,am,gt=run(bits); agree=float((am==am_fp).float().mean())*100
    res[name]=(ppl,agree,gt); print(f"\n=== {name} (bits={bits}, g={G}) ===\n  perplexity={ppl:.3f}(fp {ppl_fp:.3f})·argmax일치(vs fp)={agree:.1f}%\n  gen: {gt!r}")
for n,mod in targets.items(): mod.weight.data=orig[n]  # 복원
print("\n=== 요약 (품질 숫자) ===")
print(f"  fp32     : ppl {ppl_fp:.2f}")
for name in ("q4","q3","naive-q2"):
    ppl,agree,_=res[name]; print(f"  {name:8s} : ppl {ppl:.2f}({ppl/ppl_fp:.2f}×)·argmax {agree:.1f}%")
