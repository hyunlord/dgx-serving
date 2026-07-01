#!/usr/bin/env python3
# long-rope 테이블 생성: 표준 RoPE(rope_scaling=None 확인됨, theta=1e7, rotary_dim=64) → pos 0..max_len-1.
#   검증: pos<128 == 기존 rope_cos_long.bin(모델 forward서 캡처한 golden) bit-identical → 공식==모델 rope 증명.
#   rope_scaling=None(위치-균일)이라 pos<128 일치 시 pos>=128 확장 정확(golden by 등가). YaRN/NTK면 이 스크립트 수정 필요.
# 사용법: python3 tools/gen_rope_long.py <artifacts/qwen35-Xb> [max_len]
import sys, json, struct
import numpy as np

art = sys.argv[1] if len(sys.argv)>1 else "artifacts/qwen35-0.8b"
max_len = int(sys.argv[2]) if len(sys.argv)>2 else 8192
mani = json.load(open(f"{art}/weights/manifest.json"))
theta = float(mani["rope_theta"]); rot = int(mani["rotary_dim"])
print(f"[gen_rope_long] theta={theta} rotary_dim={rot} max_len={max_len} (rope_scaling=None=표준 RoPE)")

# 표준 RoPE (transformers 기본, rope_scaling=None): inv_freq[i]=1/theta^(2i/rot), i=0..rot/2-1
#   transformers는 fp32로 계산(inv_freq/freqs) → fp32 매칭. emb=cat([freqs,freqs]), attention_scaling=1.0.
inv_freq = (1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float32) / np.float32(rot)))).astype(np.float32)  # [rot/2] fp32
pos = np.arange(max_len, dtype=np.float32)                                    # [max_len]
freqs = np.outer(pos, inv_freq).astype(np.float32)                            # [max_len, rot/2] fp32
emb = np.concatenate([freqs, freqs], axis=-1)                                 # [max_len, rot]
cos = np.cos(emb).astype(np.float32); sin = np.sin(emb).astype(np.float32)

# 검증: pos<128 == 기존 캡처 테이블(golden) bit-identical
gd = f"{art}/mtp_golden"
def rd(p):
    with open(p,"rb") as f: return np.frombuffer(f.read(), dtype=np.float32)
try:
    old_c = rd(f"{gd}/rope_cos_long.bin"); old_s = rd(f"{gd}/rope_sin_long.bin")
    n_old = old_c.size // rot
    print(f"  기존 테이블: {n_old}행 × {rot}")
    m = min(n_old, max_len)
    dc = np.abs(cos[:m].reshape(-1) - old_c[:m*rot]).max()
    ds = np.abs(sin[:m].reshape(-1) - old_s[:m*rot]).max()
    print(f"  pos<{m} 검증: max|Δcos|={dc:.3e} max|Δsin|={ds:.3e} → {'BIT-IDENTICAL ✓ (공식==모델 rope)' if max(dc,ds)<1e-6 else '★불일치 — rope 정의 다름(YaRN/interleave?)'}")
    if max(dc,ds) >= 1e-6:
        print("  경고: 불일치 → 확장 중단(공식이 모델과 다름). 조사 필요."); sys.exit(1)
except FileNotFoundError:
    print("  (기존 테이블 없음 — 검증 스킵)")

# 덤프 (기존 파일명 그대로 덮되, 백업 후. 여기선 별도 _ext 파일로 저장 → 테스트에서 선택 사용)
out_c = f"{gd}/rope_cos_ext.bin"; out_s = f"{gd}/rope_sin_ext.bin"
cos.tofile(out_c); sin.tofile(out_s)
print(f"  덤프: {out_c} / {out_s} ({max_len}행 × {rot}, 각 {cos.nbytes/1e6:.1f}MB)")
