#!/usr/bin/env python3
"""궤적-난이도 분리 프로브(accept 갭 진단 자산): 같은 HF bf16 스택(assistant+target)이
  (a) 자기 bf16-greedy 텍스트 vs (b) 엔진 궤적(테스트 DUMP_PLAIN 덤프) 위에서
위치별 1스텝 draft 일치율 a1 = P(assistant 1st draft == target greedy)이 얼마나 다른가.
★한 번의 full forward로 전 위치 hidden(pre-norm)·shared KV·target greedy를 얻음(위치 t 입력 = KV[:t]·hidden[t-1]).
a1(b) ≪ a1(a)면 → 궤적 차이(엔진 텍스트가 본질적으로 draft 어려움). a1(b) ≈ a1(a)면 → 텍스트 아님(엔진 축 잔존).
accept(K)↔a1 정합: E[accept]=p(1-p^K)/(1-p). ★검증: a1(HF,HF텍스트)=0.844 → accept4 환산 2.66 = HF 실측 2.66 정확 재현.
측정 결과(진단 완결): q4 궤적 a1 0.885(오히려 쉬움)·fp32 궤적 chat 0.844(정상)·prose는 HF 텍스트조차 0.604(본질 난이도)
→ 궤적 가설 기각. 상세 판정: docs/gemma-assistant-spec-measured.md "accept 갭 규명".
사용: python tools/probe_traj_difficulty.py [traj_dir] [label]
  traj_dir: DUMP_PLAIN=<dir> ./tests/test_gemma4_spec_e2e ... 로 덤프한 traj_{chat,code,prose}.bin 디렉토리.
"""
import os, struct, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WLD  = f"{REPO}/artifacts/gemma4-12b-assistant/golden"
TRAJ = sys.argv[1] if len(sys.argv)>1 else f"{REPO}/artifacts/gemma4-12b-assistant/traj"
LABEL = sys.argv[2] if len(sys.argv)>2 else "q4"
N = 96

def ri(p):
    with open(p, "rb") as f: b = f.read()
    return list(struct.unpack(f"{len(b)//4}i", b))

tok = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
print("[traj-diff] loading target+assistant (bf16 — HF probe와 동일 구성)...", flush=True)
tgt = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B-it", dtype=torch.bfloat16).to("cuda").eval()
ast = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B-it-assistant", dtype=torch.bfloat16).to("cuda").eval()
emb_mod = tgt.get_input_embeddings()

@torch.inference_mode()
def a1_over_text(full_ids, P):
    """full_ids = 프롬프트(P) + 생성 텍스트. 위치 t(P-1..len-2)에서:
    입력 h=hidden[-1][t]·KV[:t+1]·tok=full_ids[t+1]?? — 사이클 의미로: anchor 토큰 = full_ids[t](마지막 확정),
    assistant는 (embed(full_ids[t]), h[t-? ]).. 정합: 생성 위치 t의 draft = anchor full_ids[t] · h = hidden[t-1]?
    사이클 정의로 재정리: anchor a_t = full_ids[t](위치 t·KV에 미기재), KV len = t, h = hidden[-1][:, t-1](마지막 KV 위치).
    target greedy = argmax(logits[t]) (full forward의 위치 t 로짓 = P(next|..full_ids[t])).
    assistant 예측 = ast(embed(a_t)·h, KV[:t], pos=t) → argmax. 일치율 = P(ast == target greedy)."""
    ids = torch.tensor([full_ids], device="cuda")
    out = tgt(ids, output_hidden_states=True, return_shared_kv_states=True, use_cache=False)
    hs = out.hidden_states[-1].float()                     # [1, L, 3840] pre-norm(실측 확정)
    skv = out.shared_kv_states
    greedy = out.logits[0].float().argmax(-1)              # [L] — 위치 t: P(next|..t)
    L = len(full_ids)
    match = tot = 0
    for t in range(P, L - 1):                              # 생성 구간(anchor가 생성 토큰인 위치만 — 사이클과 동일 분포)
        kv = {k: (v[0][:, :, :t, :].float(), v[1][:, :, :t, :].float()) for k, v in skv.items()}
        h = hs[:, t - 1 : t]                               # 마지막 KV 위치의 hidden
        e = emb_mod(ids[:, t : t + 1]).float()
        pid = torch.tensor([[t]], device="cuda")
        o = ast(inputs_embeds=torch.cat([e, h], -1).to(torch.bfloat16),
                position_ids=pid, shared_kv_states={k: (a.to(torch.bfloat16), b.to(torch.bfloat16)) for k, (a, b) in kv.items()},
                attention_mask=None)
        d1 = int(o.logits[0, -1].argmax().item())
        match += int(d1 == int(greedy[t].item())); tot += 1
    return match / max(tot, 1), tot

@torch.inference_mode()
def hf_greedy(prompt_ids, n):
    ids = torch.tensor([prompt_ids], device="cuda")
    out = tgt.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=0)
    return out[0].tolist()

for wl in ["chat", "code", "prose"]:
    pids = ri(f"{WLD}/wl_{wl}_ids.bin"); P = len(pids)
    ours = ri(f"{TRAJ}/traj_{wl}.bin")                     # e2e plain 궤적(anchor+N)
    full_ours = pids + ours
    full_hf = hf_greedy(pids, N + 1)
    a_ours, n1 = a1_over_text(full_ours, P)
    a_hf, n2 = a1_over_text(full_hf, P)
    def acc4(p): return p * (1 - p**4) / (1 - p) if p < 1 else 4.0
    print(f"[{wl}] a1(HF스택, HF텍스트)={a_hf:.3f}(→accept4≈{acc4(a_hf):.2f}, n={n2}) | "
          f"a1(HF스택, 우리{LABEL}텍스트)={a_ours:.3f}(→accept4≈{acc4(a_ours):.2f}, n={n1}) | Δ={a_hf-a_ours:+.3f}", flush=True)
print("판정: Δ 크면=궤적(가)·Δ 작으면=텍스트 아님(나 잔존). 엔진 p≈0.55·HF p≈0.78 참조.")
