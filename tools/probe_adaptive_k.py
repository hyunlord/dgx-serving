#!/usr/bin/env python3
"""적응 draft 길이 STEP 0 분석(구현 0 진단 자산): test_gemma4_spec_e2e를 CONF_LOG=csv로 돌려 얻은
draft별 (margin=top1-top2 로짓, prob=softmax top-prob, m=그 사이클 accept 수)에서
① margin/prob이 accept를 예측하는가(AUC·조건부 accept율), ② 고정 K의 낭비율(뽑고 버린 draft —
그림자 draft=첫 reject 이후 뽑힌 것 포함), ③ 문턱 τ 후보표(P(accept|margin≥τ) vs 이하 비율)를 산출.
행 의미: accepted = j<=m · first-reject = j==m+1 · shadow = j>m+1(reject된 접두사 위에서 뽑힘 —
궤적-이탈이라 상관 분석에서 제외, 낭비 계산에는 포함). 사용: python tools/probe_adaptive_k.py <csv>
"""
import sys, csv
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "scratch/conf_k.csv"
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append((r["wl"], int(r["K"]), int(r["cycle"]), int(r["j"]),
                     float(r["margin"]), float(r["prob"]), int(r["m"])))
print(f"[adaptive_k] {path}: {len(rows)} draft 행")

def auc(pos, neg):                     # Mann-Whitney U 기반 AUC(신호가 accept를 얼마나 분리하나)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); rk = allv.argsort().argsort().astype(np.float64) + 1
    return (rk[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

wls = sorted(set(r[0] for r in rows), key=lambda w: ["chat","code","prose"].index(w))
Ks  = sorted(set(r[1] for r in rows))
for wl in wls:
    for K in Ks:
        sub = [r for r in rows if r[0]==wl and r[1]==K]
        if not sub: continue
        ncyc = len(set(r[2] for r in sub))
        m_by_cyc = {r[2]: r[6] for r in sub}
        tot_m = sum(m_by_cyc.values())
        n_draft = len(sub)
        waste = 1.0 - tot_m/n_draft                          # 뽑고 버린 비율 = (ΣK-Σm)/ΣK
        shadow = sum(1 for r in sub if r[3] > r[6]+1)        # 첫 reject 이후 뽑힌 draft(순수 낭비)
        m0 = sum(1 for m in m_by_cyc.values() if m == 0)     # 완전 낭비 사이클
        # 상관: on-trajectory 행만(j<=m+1)
        ont = [r for r in sub if r[3] <= r[6]+1]
        acc = np.array([1 if r[3]<=r[6] else 0 for r in ont])
        mg  = np.array([r[4] for r in ont]); pb = np.array([r[5] for r in ont])
        a_mg = auc(mg[acc==1], mg[acc==0]); a_pb = auc(pb[acc==1], pb[acc==0])
        print(f"\n== {wl} K={K}: cyc {ncyc} · draft {n_draft} · accept {tot_m} · "
              f"낭비 {waste*100:.0f}% (shadow {shadow/n_draft*100:.0f}%p · m=0 사이클 {m0/ncyc*100:.0f}%)")
        print(f"   on-traj {len(ont)}행 (accept율 {acc.mean()*100:.0f}%) · AUC margin {a_mg:.3f} · prob {a_pb:.3f}")
        print(f"   margin 중앙값: accept {np.median(mg[acc==1]) if acc.sum() else float('nan'):.2f} vs "
              f"first-reject {np.median(mg[acc==0]) if (acc==0).sum() else float('nan'):.2f}")
        # 문턱표: P(accept | margin>=τ) / P(accept | margin<τ) / margin<τ 비율(조기중단이 걸릴 빈도)
        print(f"   {'τ':>6} {'P(acc|≥τ)':>10} {'P(acc|<τ)':>10} {'<τ 비율':>8}")
        for tau in [0.5, 1, 2, 3, 4, 6, 8, 12]:
            hi = acc[mg >= tau]; lo = acc[mg < tau]
            print(f"   {tau:>6} {hi.mean()*100 if len(hi) else float('nan'):>9.0f}% "
                  f"{lo.mean()*100 if len(lo) else float('nan'):>9.0f}% {len(lo)/len(acc)*100 if len(acc) else 0:>7.0f}%")
        # 위치별 accept율(깊이 상방 판단: K8에서 j 5~8이 살아남나)
        pj = []
        for j in range(1, K+1):
            js = [r for r in sub if r[3]==j and r[3] <= r[6]+1]  # on-traj만
            if js: pj.append(f"j{j} {sum(1 for r in js if r[3]<=r[6])/len(js)*100:.0f}%({len(js)})")
        print(f"   위치별 accept(on-traj): {' '.join(pj)}")
