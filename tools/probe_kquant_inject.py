#!/usr/bin/env python3
"""K-quant 품질 주입 프로브(구현 0 진단 자산): llama Q4_K_M GGUF를 fp32로 dequant해 우리 weights
레이아웃의 *_f32.bin만 교체 → 엔진 uq=0로 "K-quant 품질 target"의 accept를 커널 구현 없이 측정.
norm/scalar/manifest는 우리 것 유지(llama의 Gemma norm +1 관례 등 함정 회피 — 격리 변수=큰 matmul 값만).
★정합 골든: 각 텐서 dequant vs 우리 bf16값 f32 — cosine<0.98(퍼뮤트/매핑) 또는 norm-ratio 이탈(구운 스케일)
시 중단(조용한 오답 금지). 측정 결과(GB10·2026-07): 가중치 rel-err 0.0695(blueprint 0.1016 대비 32% 정확)
이나 accept 회복은 워크로드 선택적 — code K4 1.74→1.91(소폭·bf16값 2.56 미달=code 벽은 4bit 자체)·
prose 0.71→1.13(HF 동률 완전 회복) → K-quant 커널 구현 보류 판정(q2 패턴). 상세: gemma-assistant-spec-measured.md.
사용: [SRC=... DST=... GGUF=... GGUF_PY=...] python tools/probe_kquant_inject.py → 이후 test_gemma4_spec_e2e uq=0.
"""
import os, sys, shutil
import numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/repos/llama.cpp-mtp/gguf-py")))
from gguf import GGUFReader
from gguf.quants import dequantize

SRC  = os.environ.get("SRC",  "artifacts/gemma4-12b/weights-fp32")
DST  = os.environ.get("DST",  "artifacts/gemma4-12b/weights-kquant")
GGUF = os.environ.get("GGUF", os.path.expanduser("~/models/poc/gemma-4-12B-it-Q4_K_M.gguf"))

r = GGUFReader(GGUF)
tmap = {t.name: t for t in r.tensors}

def gguf_name(our):   # 우리 f32 파일명 → GGUF 텐서명
    if our == "w_lm_head_f32.bin" or our == "w_embed.bin":
        return "token_embd.weight"          # tied
    # LN_self_attn.q_proj_f32.bin / LN_mlp.gate_proj_f32.bin
    base = our[:-len("_f32.bin")] if our.endswith("_f32.bin") else our[:-4]
    l, rest = base.split("_", 1)
    n = l[1:]
    m = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
         "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
    return f"blk.{n}.{m[rest]}.weight" if rest in m else None

# 1) 하드링크 복제(즉시) 후 f32만 제거 → 신규 작성(원본 inode 무손상)
if not os.path.isdir(DST):
    subprocess = __import__("subprocess")
    subprocess.run(["cp", "-al", SRC, DST], check=True)
    for f in os.listdir(DST):
        if f.endswith("_f32.bin") or f == "w_embed.bin":
            os.unlink(os.path.join(DST, f))
    print("[kquant] 하드링크 복제 + f32/embed 제거 완료")

targets = sorted([f for f in os.listdir(SRC) if f.endswith("_f32.bin")] + ["w_embed.bin"])
n_rep = n_fb = 0; worst = (1.0, "")
rels = []
for f in targets:
    dstp = os.path.join(DST, f)
    if os.path.exists(dstp): continue                      # 재개 가능
    g = gguf_name(f)
    ref = np.fromfile(os.path.join(SRC, f), dtype=np.float32)
    if g is None or g not in tmap:
        shutil.copyfile(os.path.join(SRC, f), dstp); n_fb += 1
        print(f"  [fallback-copy] {f} (GGUF 대응 없음: {g})")
        continue
    t = tmap[g]
    w = dequantize(t.data, t.tensor_type).astype(np.float32)
    shape = tuple(int(x) for x in reversed(t.shape))       # ne 역순 = [out, in]
    w = w.reshape(shape)
    if w.size != ref.size:
        print(f"  ★크기 불일치 {f}: {w.size} vs {ref.size} — 중단"); sys.exit(1)
    a = w.reshape(-1); b = ref
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    ratio = float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-12))     # ★cosine은 스케일-불변 — 구운 스케일(√H 등) 검출용
    rel = float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-12))
    rels.append(rel)
    if cos < worst[0]: worst = (cos, f)
    if cos < 0.98 or not (0.9 < ratio < 1.1):
        print(f"  ★정합 골든 FAIL {f}: cosine={cos:.4f} ratio={ratio:.3f} (퍼뮤트/매핑/스케일 의심) — 중단"); sys.exit(1)
    a.tofile(dstp); n_rep += 1
    if n_rep % 50 == 0: print(f"  ... {n_rep}/{len(targets)}", flush=True)

print(f"[kquant] 교체 {n_rep} · fallback {n_fb} · worst cosine {worst[0]:.4f} ({worst[1]}) · "
      f"mean rel-err {np.mean(rels):.4f} (참고: blueprint q4 rel 0.1016)")
print(f"[kquant] done → {DST}")
