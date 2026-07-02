#!/usr/bin/env python3
"""dgx-serving auto-tune 하네스 Phase 1 — config(Bmax/K/W) 자동 최적화.

원칙(챕터10 헌법):
- config-주도: 커널 불변, bench_serve_spec의 argv/env만 조작.
- 메모리 사전 필터: trial 실행 전 admission 예산 계산(bench와 같은 공식) — 초과 config는
  실행 없이 prune(GB10 unified oversubscribe = 시스템 전체 다운 전례).
- 결정적 목적함수: 같은 seed/trace(controlled), λ4 단일점, tok/s 최대화 + tpot95 soft 제약.
- ISO 2단: trial은 성능만, 최종 best만 full ISO(12/12) — 실패 시 무효(차점 승격은 수동 판단).
- BO 코어 재발명 금지: Optuna TPE + sqlite(중단·재개). 우리 몫 = 래퍼/필터/게이트/아티팩트.

사용:
  python tools/autotune/autotune.py --model profiles/qwen35-27b.json --device profiles/gb10.json \
      --trials 30 --seed 42 --lam 4 --nreq 40 [--study NAME] [--verify-only]
"""
import argparse, json, os, re, subprocess, sys, time, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BENCH = os.path.join(REPO, "tools", "bench_serve_spec")


# ---------- 메모리 사전 필터 (bench_serve_spec.cu admission guard 공식 재현) ----------
def per_slot_bytes(mp: dict, K: int, W: int, pmax: int = None) -> int:
    """bench_serve_spec.cu:596-602 공식. Python 재현값이 bench 로그와 일치 검증됨
    (27b K6W3: KV 175.8MB + MTP 2.4MB + VA 1376.6MB)."""
    Nn = K + W
    pmax = pmax if pmax is not None else mp["pmax_default"]
    maxlen = max(288, pmax + 192 + K + Nn + 16)
    L = mp["num_layers"]
    nfull = L // mp["full_attention_interval"]
    ngdn = L - nfull
    NKV, HD = mp["num_kv_heads"], mp["head_dim"]
    nh, D = mp["linear_num_value_heads"], mp["linear_value_head_dim"]
    Kk = mp["linear_conv_kernel"]
    convd = mp["linear_num_key_heads"] * mp["linear_key_head_dim"] * 2 + nh * D
    per_kv = nfull * maxlen * NKV * HD * 2 * 2 + ngdn * (nh * D * D + (Kk - 1) * convd) * 4
    per_mtp = 2 * maxlen * NKV * HD * 4
    per_va = Nn * ngdn * (nh * D * D + convd) * 4
    return per_kv + per_mtp + per_va


def mem_available_bytes() -> int:
    """가드 정확화(Phase 7): MemAvailable(페이지캐시 등 회수 가능 포함)이 unified 실제 가용."""
    with open("/proc/meminfo") as f:
        for line in f:
            m = re.match(r"MemAvailable:\s+(\d+)\s+kB", line)
            if m:
                return int(m.group(1)) * 1024
    raise RuntimeError("MemAvailable not found")


def mem_filter(mp: dict, dp: dict, bmax: int, K: int, W: int):
    """(ok, budget_gb, avail_gb) — 실행 전 판정. bench 내부 가드와 같은 margin."""
    need = bmax * per_slot_bytes(mp, K, W)
    avail = dp["mem_guard_margin"] * mem_available_bytes()
    return need <= avail, need / 1e9, avail / 1e9


# ---------- 목적함수: bench_serve_spec 실행 + 파싱 ----------
CSPEC_RE = re.compile(r"c-spec\s+(\S+)\s+\|\s+(\S+)\s+(\S+)\s+\|\s+(\S+)\s+(\S+)\s+(\S+)\s+\|\s+(\S+)\s+(\S+)\s+\|\s+(\S+)\s+(\S+)\s+\|\s+(\S+)")


def run_bench(mp: dict, bmax: int, K: int, W: int, lam: float, nreq: int, seed: int,
              iso: bool = False, cyc: bool = False, timeout_s: int = 900):
    env = dict(os.environ, MTP_Q4="1")
    if iso:
        env["ISO"] = "1"
    if cyc:
        env["CYC"] = "1"
    cmd = [BENCH, mp["weights_dir"], mp["golden_dir"], str(mp["uq"]),
           str(bmax), str(nreq), str(seed), str(K), str(W), str(lam)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout_s)
    out = r.stdout + r.stderr
    res = {"cmd": " ".join(cmd), "returncode": r.returncode, "wall_s": round(time.time() - t0, 1)}
    if iso:
        m = re.search(r"\[ISO\] (\d+)/(\d+) .*(PASS|FAIL)", out)
        res["iso"] = m.group(0) if m else "NO_ISO_LINE"
        res["iso_pass"] = bool(m and m.group(3) == "PASS")
        return res, out
    m = CSPEC_RE.search(out)
    if not m:
        res["error"] = out[-2000:]
        return res, out
    res.update(lam=float(m.group(1)), thru_req=float(m.group(2)), tok_s=float(m.group(3)),
               e2e50=float(m.group(4)), e2e95=float(m.group(5)), e2e99=float(m.group(6)),
               ttft95=float(m.group(7)), tpot95=float(m.group(8)), avgB=float(m.group(9)),
               accept=float(m.group(11)))
    if cyc:
        res["cyc_lines"] = re.findall(r"\[CYC\].*", out)
    return res, out


def score(res: dict, tpot_cap: float) -> float:
    """tok/s 최대화 + tpot95 soft 페널티(초과 1ms당 -0.1 tok/s 스케일 → 초과 0.1s = -10)."""
    return res["tok_s"] - max(0.0, res["tpot95"] - tpot_cap) * 100.0


# ---------- 프로파일 → 탐색 공간/제약 (STEP B: 하드코딩 제거 — 프로파일만 교체하면 재사용) ----------
def search_space_from_profiles(mp: dict, dp: dict) -> dict:
    """K/W 범위는 트리 형상 일반 범위, Bmax 상한은 메모리에서 유도:
    최악 형상(K·W 최대)의 per-slot로 margin 내 들어가는 최대 짝수 B."""
    k_range, w_range = (3, 8), (1, 5)
    worst = per_slot_bytes(mp, k_range[1], w_range[1])
    avail = dp["mem_guard_margin"] * mem_available_bytes()
    bmax_hw = int(avail // worst)
    bmax_hi = min(16, bmax_hw - bmax_hw % 2)  # 16 = gemm_batch 템플릿 자연 상한
    return {"bmax": (4, bmax_hi), "K": k_range, "W": w_range}


# ---------- Optuna study ----------
def make_objective(mp, dp, args, cache):
    import optuna
    space = search_space_from_profiles(mp, dp)
    print(f"[autotune] 탐색 공간(프로파일 유도): {space}", flush=True)

    def objective(trial: "optuna.Trial"):
        bmax = trial.suggest_int("bmax", *space["bmax"], step=2)
        K = trial.suggest_int("K", *space["K"])
        W = trial.suggest_int("W", *space["W"])
        key = (bmax, K, W)
        # canonical 제약: Nn=K+W>8 (concat-mmq 곱 전 구간 + 단독 spec 자동 일치 = mism 원천 차단)
        if K + W <= 8:
            trial.set_user_attr("pruned", "canonical Nn<=8 (per-slot 폴백 — 곱 손실 영역)")
            raise optuna.TrialPruned()
        ok, need_gb, avail_gb = mem_filter(mp, dp, bmax, K, W)
        trial.set_user_attr("mem_need_gb", round(need_gb, 2))
        if not ok:
            trial.set_user_attr("pruned", f"memory {need_gb:.1f}GB > {avail_gb:.1f}GB")
            raise optuna.TrialPruned()
        if key in cache:  # TPE 중복 제안 → 재실행 없이 재사용(결정적 목적이라 유효)
            res = cache[key]
            trial.set_user_attr("cached", True)
        else:
            res, _ = run_bench(mp, bmax, K, W, args.lam, args.nreq, args.seed)
            if "tok_s" not in res:
                trial.set_user_attr("error", res.get("error", "?")[:500])
                raise optuna.TrialPruned()
            cache[key] = res
        for k in ("tok_s", "tpot95", "accept", "avgB", "wall_s"):
            trial.set_user_attr(k, res[k])
        s = score(res, args.tpot_cap)
        print(f"  trial#{trial.number}: B{bmax} K{K} W{W} -> {res['tok_s']:.1f} tok/s "
              f"tpot {res['tpot95']:.3f} accept {res['accept']:.2f} score {s:.1f}", flush=True)
        return s

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "profiles", "qwen35-27b.json"))
    ap.add_argument("--device", default=os.path.join(HERE, "profiles", "gb10.json"))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42, help="trace seed(목적함수 고정 — controlled)")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--nreq", type=int, default=40)
    ap.add_argument("--tpot-cap", type=float, default=0.35)
    ap.add_argument("--study", default=None)
    ap.add_argument("--sampler-seed", type=int, default=7)
    args = ap.parse_args()

    mp = json.load(open(args.model))
    dp = json.load(open(args.device))
    if not os.path.exists(BENCH):
        sys.exit(f"bench 바이너리 없음: {BENCH} (make 먼저)")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = args.study or f"autotune-{mp['name'].replace('/', '_').replace(' ', '')}"
    storage = f"sqlite:///{os.path.join(HERE, 'autotune.db')}"
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.sampler_seed))
    cache = {}
    # 재개 시 기존 완료 trial을 캐시에 복원(같은 config 재실행 방지)
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and "tok_s" in t.user_attrs:
            cache[(t.params["bmax"], t.params["K"], t.params["W"])] = dict(t.user_attrs)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[autotune] {study_name}: 기존 완료 {done}, 목표 {args.trials} trials "
          f"(seed={args.seed} λ={args.lam} nreq={args.nreq} tpot_cap={args.tpot_cap})", flush=True)
    study.optimize(make_objective(mp, dp, args, cache), n_trials=args.trials)

    best = study.best_trial
    print(f"\n[autotune] BEST trial#{best.number}: {best.params} score {best.value:.1f} "
          f"tok/s {best.user_attrs.get('tok_s')} tpot {best.user_attrs.get('tpot95')}", flush=True)

    # ---- 검증 프로토콜: 다른 seed 재측정(과적합 가드) + full ISO + CYC ----
    p = best.params
    alt_seed = args.seed + 735
    print(f"[verify] 다른 seed({alt_seed}) 재측정...", flush=True)
    alt, _ = run_bench(mp, p["bmax"], p["K"], p["W"], args.lam, args.nreq, alt_seed)
    print(f"[verify] alt-seed: {alt.get('tok_s')} tok/s tpot {alt.get('tpot95')}", flush=True)
    print("[verify] full ISO(12/12)...", flush=True)
    iso, _ = run_bench(mp, p["bmax"], p["K"], p["W"], args.lam, 12, args.seed, iso=True)
    print(f"[verify] {iso.get('iso')}", flush=True)
    print("[verify] CYC 사이클 리포트...", flush=True)
    cyc, _ = run_bench(mp, p["bmax"], p["K"], p["W"], args.lam, 30, args.seed, cyc=True)

    art = {
        "schema": "dgx-serving/autotune-config@1",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "device_profile": dp, "model_profile": mp,
        "objective": {"lam": args.lam, "nreq": args.nreq, "seed": args.seed,
                      "tpot_cap": args.tpot_cap, "metric": "tok_s - 100*max(0,tpot95-cap)"},
        "best_config": {"bmax": p["bmax"], "K": p["K"], "W": p["W"], "Nn": p["K"] + p["W"],
                        "env": {"MTP_Q4": "1"}},
        "best_measured": {k: best.user_attrs.get(k) for k in ("tok_s", "tpot95", "accept", "avgB")},
        "verify": {"alt_seed": alt_seed,
                   "alt_measured": {k: alt.get(k) for k in ("tok_s", "tpot95", "accept")},
                   "iso": iso.get("iso"), "iso_pass": iso.get("iso_pass"),
                   "cyc": cyc.get("cyc_lines", [])},
        "trials": [{"n": t.number, "params": t.params, "state": str(t.state),
                    "value": t.value, "attrs": t.user_attrs} for t in study.trials],
    }
    out = os.path.join(HERE, "artifacts",
                       f"{study_name}-{datetime.date.today()}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(art, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[autotune] 아티팩트: {out}  (ISO {'PASS' if iso.get('iso_pass') else 'FAIL — 아티팩트 무효, 차점 검토'})",
          flush=True)


if __name__ == "__main__":
    main()
