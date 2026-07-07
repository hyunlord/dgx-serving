#!/usr/bin/env python3
"""gemma4 daemon rope 테이블 절대 골든 게이트 — runtime/rope_gemma4.h 수식의 python 미러 vs HF 캡처 골든.

★존재 이유(버그 전례 2건): daemon의 자체 생성 rope가 ①full theta 1e4 하드코딩 ②partial_rotary_factor
0.25 누락(전체 회전)으로 두 번 틀렸고, 둘 다 **spec==plain 상대 게이트가 못 잡았음**(양쪽이 같은 틀린
테이블 공유 — 긴 생성(수백 tok+) 문자-붕괴로만 발현). 이 게이트는 HF에서 캡처한 절대 기준(gen_rope_*,
bf16 정밀)과 직접 대조한다. 문턱 4e-3 = bf16 캡처 1ulp(~2e-3) 여유 2×.

수식(확정 — HF config rope_parameters + 골든 역산 64쌍 일치, 12B/31B/26B 공통):
  full   : proportional — theta=1e6, 회전쌍 = partial(0.25)×hd/2 = 64만 회전(inv_i=theta^(-2i/hd)),
           나머지 쌍은 항등(cos=1·sin=0 — 기존 rope 커널에서 회전 항등과 등가).
  sliding: 표준 — theta=1e4, 전체 hd/2쌍 회전.
사용: python3 tests/test_gemma4_rope_table.py  (12B·31B golden 필요 — 없으면 해당 모델 skip)
"""
import os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THRESH = 4e-3

def gen(T, hd, theta, partial):
    half = hd // 2
    rot = int(partial * hd / 2)
    cos = np.ones((T, hd), dtype=np.float64); sin = np.zeros((T, hd), dtype=np.float64)
    i = np.arange(rot)
    ang = np.arange(T)[:, None] * (theta ** (-2.0 * i / hd))[None, :]
    cos[:, :rot] = np.cos(ang); sin[:, :rot] = np.sin(ang)
    cos[:, half:half+rot] = cos[:, :rot]; sin[:, half:half+rot] = sin[:, :rot]   # cat(freqs,freqs)
    return cos.astype(np.float32), sin.astype(np.float32)

def main():
    fails = 0; ran = 0
    for model, hd_f, hd_s in [("gemma4-12b", 512, 256), ("gemma4-31b", 512, 256)]:
        gd = os.path.join(REPO, "artifacts", model, "golden")
        p = os.path.join(gd, "gen_rope_full_cos.bin")
        if not os.path.exists(p):
            print(f"  [{model}] golden 없음 — skip"); continue
        ran += 1
        for name, hd, theta, partial in [("full", hd_f, 1e6, 0.25), ("sliding", hd_s, 1e4, 1.0)]:
            gc = np.fromfile(os.path.join(gd, f"gen_rope_{name}_cos.bin"), dtype=np.float32).reshape(-1, hd)
            gs = np.fromfile(os.path.join(gd, f"gen_rope_{name}_sin.bin"), dtype=np.float32).reshape(-1, hd)
            c, s = gen(gc.shape[0], hd, theta, partial)
            d = max(np.abs(c - gc).max(), np.abs(s - gs).max())
            ok = d < THRESH
            print(f"  [{model} {name}] max|Δ|={d:.3e} (<{THRESH:g}) {'PASS' if ok else '★FAIL'}")
            if not ok: fails += 1
        # 반증 대조(게이트 감도 자가 증명): partial 누락(전체 회전) 수식은 반드시 크게 어긋나야 함
        c, s = gen(gc.shape[0], hd_f, 1e6, 1.0)
        gcf = np.fromfile(os.path.join(gd, "gen_rope_full_cos.bin"), dtype=np.float32).reshape(-1, hd_f)
        d_bad = np.abs(c - gcf).max()
        # 구별 가능성 문턱 = 10×THRESH(테이블 위치 수에 따라 절대값은 달라짐 — 31B 골든 96행은 0.125 수준)
        print(f"  [{model} 감도] partial 누락 수식 max|Δ|={d_bad:.3f} (>{10*THRESH:g}면 게이트가 그 버그를 구별: "
              f"{'PASS' if d_bad > 10*THRESH else '★FAIL(게이트 무감)'})")
        if d_bad <= 10*THRESH: fails += 1
    if ran == 0:
        print("[test_gemma4_rope_table] SKIP (golden 자산 없음)"); return 0
    print(f"[test_gemma4_rope_table] {'PASS (daemon rope 수식 == HF 절대 골든)' if fails==0 else 'FAIL'}")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
