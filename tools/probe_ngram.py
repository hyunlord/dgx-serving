#!/usr/bin/env python3
"""챕터9 accept lever de-risk(STEP A): n-gram drafting accept 통계 (통합 전).
   입력 = 우리 27B greedy 시퀀스(gen_dump 출력) + 프롬프트 길이 P. spec cycle 시뮬:
     confirmed t(cur=seq[t]) → n-gram draft(seq[0..t]의 suffix 최장매칭 → 후속 K) → 실제 seq[t+1..]와 consecutive accept a.
     advance t += 1+a, avg accept = Σ(1+a)/cycles. (verify가 거르므로 mism=0 불변 — 여기선 greedy가 ground truth.)
   측정: n-gram 단독 accept(MTP 2.25/code 3.0 대비), n범위·K·scope(prompt-only vs gen포함) 영향.
   사용법: python tools/probe_ngram.py <seq.bin> <P> [label]
"""
import sys, numpy as np
seq = np.fromfile(sys.argv[1], dtype=np.int32)
P = int(sys.argv[2]); label = sys.argv[3] if len(sys.argv)>3 else sys.argv[1].split('/')[-1]
L = len(seq)

def sim(nmin, nmax, K, scope):
    t = P; cyc = 0; tot = 0; hit = 0
    while t < L-1:
        we = (t) if scope=='gen' else P          # 검색 윈도: seq[0:we] (cur 이전)
        a = 0
        for n in range(nmax, nmin-1, -1):         # 최장 suffix부터
            if t-n+1 < 0: continue
            suf = seq[t-n+1:t+1]
            found = -1
            for j in range(min(we, t)-1, n-2, -1):   # 가장 최근 earlier 발생
                if np.array_equal(seq[j-n+1:j+1], suf): found = j; break
            if found >= 0:
                cand = seq[found+1:found+1+K]; act = seq[t+1:t+1+K]
                aa = 0
                for i in range(min(len(cand),len(act))):
                    if cand[i]==act[i]: aa += 1
                    else: break
                a = aa
                if aa>0: hit += 1
                break
        tot += 1 + a; cyc += 1; t += 1 + a
    return tot/cyc, cyc, hit

print(f"=== n-gram accept de-risk: {label} (L={L}, P={P}, gen={L-P}) ===")
print(f"  {'config':34} accept  cycles  hit율")
for scope in ['gen','prompt']:
    for (nmin,nmax) in [(1,1),(2,2),(2,3),(2,4),(1,4)]:
        for K in [7]:
            acc,cyc,hit = sim(nmin,nmax,K,scope)
            print(f"  n[{nmin}-{nmax}] K={K} scope={scope:6} | {acc:5.2f}   {cyc:4d}    {100*hit/cyc:3.0f}%")
print(f"  (현 MTP 단일 accept: 2.25 평균 / code~3.0 / 산문~1.8. n-gram이 이보다↑면 lever.)")
