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


def per_slot_bytes_plain(mp: dict, maxlen: int = 384) -> int:
    """plain(GDN·MTP 無, Gemma4류) per-slot KV bytes = bench_serve_gemma4/decode_state_gemma4 per_seq_bytes 재현.
    full 층(hd_f·nkv_f) + sliding 층(hd_s·nkv_s), 선형 캡(max_len), ×2(K,V)×4(fp32)."""
    L = mp["num_layers"]
    ap = mp.get("attn_pattern", {})
    nfull = ap.get("full_layers", L)
    nsliding = ap.get("sliding_layers", L - nfull)
    hd_s, nkv_s = mp["head_dim"], mp["num_kv_heads"]
    hd_f = mp.get("global_head_dim", hd_s)
    nkv_f = mp.get("num_global_key_value_heads", nkv_s)
    return nfull * maxlen * nkv_f * hd_f * 2 * 4 + nsliding * maxlen * nkv_s * hd_s * 2 * 4


def per_slot_bytes_auto(mp: dict, K: int = 0, W: int = 0) -> int:
    """모델 유형 자동 분기: spec(mtp) → 기존 per_slot_bytes(K/W/MTP), plain → per_slot_bytes_plain(KV only)."""
    return per_slot_bytes(mp, K, W) if mp.get("mtp") else per_slot_bytes_plain(mp)


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
              iso: bool = False, cyc: bool = False, timeout_s: int = 900, extra_env: dict = None):
    env = dict(os.environ, MTP_Q4="1")
    if extra_env:
        env.update(extra_env)
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


# ---------- warm 서버 (Phase 2): 모델 1회 로드, stdin trial 루프 ----------
# env 축(MTP_Q4/DRAFT_GEMV/DRAFT_SEQ)은 bench 쪽 static 캐시라 프로세스 재시작 그룹 —
# env key가 바뀔 때만 재시작(1개 상주, 27b weights 중복 방지). 핫 축(B/K/W/λ/seed/nreq)은 stdin.
class WarmServer:
    def __init__(self, mp: dict):
        self.mp, self.proc, self.env_key = mp, None, None

    def _start(self, env_key: tuple):
        self.stop()
        env = dict(os.environ, TRIALS="1")
        env.update(dict(env_key))
        cmd = [BENCH, self.mp["weights_dir"], self.mp["golden_dir"], str(self.mp["uq"]),
               "8", "12", "42", "6", "3", "4"]
        self.proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        t0 = time.time()
        for line in self.proc.stdout:
            if "TRIALS_READY" in line:
                break
            if time.time() - t0 > 300:
                raise RuntimeError("warm server: no TRIALS_READY in 300s")
        self.env_key = env_key
        print(f"  [warm] server up env={dict(env_key)} ({time.time()-t0:.0f}s load)", flush=True)

    def run(self, env_key: tuple, bmax: int, K: int, W: int, lam: float, nreq: int, seed: int,
            timeout_s: int = 900) -> dict:
        if self.proc is None or self.proc.poll() is not None or env_key != self.env_key:
            self._start(env_key)
        t0 = time.time()
        self.proc.stdin.write(f"{bmax} {K} {W} {lam} {nreq} {seed}\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("warm server died mid-trial")
            if line.startswith("TRIAL_ERR"):
                raise RuntimeError(line.strip())
            if line.startswith("TRIAL "):
                f = line.split()
                return {"tok_s": float(f[1]), "tpot95": float(f[2]), "accept": float(f[3]),
                        "avgB": float(f[4]), "wall_s": round(time.time() - t0, 1)}
            if time.time() - t0 > timeout_s:
                self.stop()
                raise RuntimeError("warm server trial timeout")

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
            self.proc = None


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


# Phase 2 env 축: 실사 결과 spec 경로에 실효인 런타임 손잡이만(전부 bench static 캐시 → 재시작 그룹).
# 제외(정직 기록): SMAX·GDN state 타입=컴파일 타임(Phase 2b), chunk 크기·canonical 경계 M·split-K 게이트=env 부재,
#                 BM16/SPLITK=plain batch-decode 전용(spec 경로 미실행 — 무영향), DQ4=0은 27b accept 붕괴 기지.
P2_ENV_AXES = {"MTP_Q4": ["1", "0"], "DRAFT_GEMV": ["0", "1"], "DRAFT_SEQ": ["0", "1"]}

# Phase 2b: draft matmul 커널 variant(mw 타일 — 컴파일 축의 dispatch화, fat binary에 전 variant 탑재).
# P2에서 measured 확정된 축(MTP_Q4=1, DRAFT_GEMV=0, DRAFT_SEQ=0)은 고정하고 variant+형상만 탐색.
P2B_ENV_AXES = {"DRAFT_MW": ["0", "1", "2"]}
# regime 격자: λ{1,4,8} × mix{short-heavy 0.9, long-heavy 0.3} = 6 regime.
P2B_REGIMES = [{"lam": l, "shortp": s} for l in (1.0, 4.0, 8.0) for s in (0.9, 0.3)]


def params_env_key(params: dict) -> tuple:
    return tuple(sorted((k, params[k]) for k in P2_ENV_AXES if k in params))


# ---------- Optuna study ----------
def make_objective(mp, dp, args, cache, server=None):
    import optuna
    space = search_space_from_profiles(mp, dp)
    print(f"[autotune] 탐색 공간(프로파일 유도): {space}"
          + (f" + env축 {list(P2_ENV_AXES)}" if args.phase2 else ""), flush=True)

    def objective(trial: "optuna.Trial"):
        bmax = trial.suggest_int("bmax", *space["bmax"], step=2)
        K = trial.suggest_int("K", *space["K"])
        W = trial.suggest_int("W", *space["W"])
        params = {"bmax": bmax, "K": K, "W": W}
        if args.phase2:
            for name, choices in P2_ENV_AXES.items():
                params[name] = trial.suggest_categorical(name, choices)
        env_key = params_env_key(params)
        key = (bmax, K, W, env_key)
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
        elif server is not None:
            res = server.run(env_key, bmax, K, W, args.lam, args.nreq, args.seed)
            cache[key] = res
        else:
            res, _ = run_bench(mp, bmax, K, W, args.lam, args.nreq, args.seed,
                               extra_env=dict(env_key))
            if "tok_s" not in res:
                trial.set_user_attr("error", res.get("error", "?")[:500])
                raise optuna.TrialPruned()
            cache[key] = res
        for k in ("tok_s", "tpot95", "accept", "avgB", "wall_s"):
            trial.set_user_attr(k, res[k])
        s = score(res, args.tpot_cap)
        envtag = " ".join(f"{k}={v}" for k, v in env_key) if env_key else ""
        print(f"  trial#{trial.number}: B{bmax} K{K} W{W} {envtag} -> {res['tok_s']:.1f} tok/s "
              f"tpot {res['tpot95']:.3f} accept {res['accept']:.2f} score {s:.1f}", flush=True)
        return s

    return objective


# ---------- Phase 2b: regime별 튜닝(multi-fidelity) → regime→config 테이블 ----------
def run_phase2b(mp, dp, args):
    """regime(λ×mix)마다 소형 TPE study(핫 축 bmax/K/W + variant DRAFT_MW, 스크리닝 nreq)를 돌리고
    상위 후보만 고충실도(nreq=60, 콜드 실행)로 확정 — regime→config 테이블 산출.
    저충실도도 in-engine(같은 bench, 짧은 nreq) — 격리 마이크로벤치 금지(오도 전례)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{os.path.join(HERE, 'autotune.db')}"
    space = search_space_from_profiles(mp, dp)
    fixed_env = {"MTP_Q4": "1", "DRAFT_GEMV": "0", "DRAFT_SEQ": "0"}  # P2 measured 확정치 고정
    server = WarmServer(mp)
    table = []
    try:
        for rg in P2B_REGIMES:
            lam, sp = rg["lam"], rg["shortp"]
            tag = f"lam{lam:g}-sp{sp:g}"
            study = optuna.create_study(
                study_name=f"autotune-p2b-{tag}", storage=storage, load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True))
            cache = {}
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE and "tok_s" in t.user_attrs:
                    cache[(t.params["bmax"], t.params["K"], t.params["W"], t.params["DRAFT_MW"])] = dict(t.user_attrs)

            def objective(trial, _lam=lam, _sp=sp, _cache=cache):
                bmax = trial.suggest_int("bmax", *space["bmax"], step=2)
                K = trial.suggest_int("K", *space["K"])
                W = trial.suggest_int("W", *space["W"])
                mw = trial.suggest_categorical("DRAFT_MW", P2B_ENV_AXES["DRAFT_MW"])
                if K + W <= 8:
                    trial.set_user_attr("pruned", "canonical Nn<=8")
                    raise optuna.TrialPruned()
                ok, need_gb, _ = mem_filter(mp, dp, bmax, K, W)
                if not ok:
                    trial.set_user_attr("pruned", f"memory {need_gb:.1f}GB")
                    raise optuna.TrialPruned()
                key = (bmax, K, W, mw)
                if key in _cache:
                    res = _cache[key]
                else:
                    env_key = tuple(sorted({**fixed_env, "DRAFT_MW": mw, "SHORTP": str(_sp)}.items()))
                    res = server.run(env_key, bmax, K, W, _lam, args.screen_nreq, args.seed)
                    _cache[key] = res
                for k in ("tok_s", "tpot95", "accept", "avgB"):
                    trial.set_user_attr(k, res[k])
                s = score(res, args.tpot_cap)
                print(f"  [{tag}] #{trial.number}: B{bmax} K{K} W{W} mw={mw} -> {res['tok_s']:.1f} "
                      f"tpot {res['tpot95']:.3f} score {s:.1f}", flush=True)
                return s

            done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            todo = max(0, args.regime_trials - done)
            print(f"[p2b {tag}] 기존 {done}, 추가 {todo} trials (screen nreq={args.screen_nreq})", flush=True)
            if todo:
                study.optimize(objective, n_trials=todo)
            # 고충실도 확정: 고유 config 상위 2 → nreq=60 콜드 실행(SHORTP 포함, warm 서버와 독립 검증 겸용)
            comp = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                          key=lambda t: t.value or -1e9, reverse=True)
            seen, top = set(), []
            for t in comp:
                ckey = tuple(sorted(t.params.items()))
                if ckey not in seen:
                    seen.add(ckey)
                    top.append(t)
                if len(top) >= 2:
                    break
            finals = []
            for t in top:
                p = t.params
                eenv = {**fixed_env, "DRAFT_MW": p["DRAFT_MW"], "SHORTP": str(sp)}
                fin, _ = run_bench(mp, p["bmax"], p["K"], p["W"], lam, 60, args.seed, extra_env=eenv)
                finals.append({"params": p, "screen": {k: t.user_attrs.get(k) for k in ("tok_s", "tpot95")},
                               "nreq60": {k: fin.get(k) for k in ("tok_s", "tpot95", "accept", "avgB")}})
                print(f"[p2b {tag}] 확정 {p}: nreq60 {fin.get('tok_s')} tpot {fin.get('tpot95')}", flush=True)
            finals.sort(key=lambda f: (f["nreq60"]["tok_s"] or 0) - max(0.0, (f["nreq60"]["tpot95"] or 9) - args.tpot_cap) * 100, reverse=True)
            # 간이 파레토: throughput 최선 + tpot 최선(다르면) 병기
            by_tpot = min(finals, key=lambda f: f["nreq60"]["tpot95"] or 9)
            table.append({"regime": rg, "best": finals[0],
                          "latency_alt": by_tpot if by_tpot is not finals[0] else None,
                          "candidates": finals})
    finally:
        server.stop()
    return table


# ---------- M2 STEP C: warm-start 훅(레시피 → enqueue_trial) ----------
def recipe_warmstart_configs(recipe_path: str) -> list:
    """recipe.yaml의 regime 승자 config들 → Optuna enqueue용 param dict(중복 제거).
    transfer 경로 실체화: 불일치 장비에서 레시피를 prior로 재튜닝할 때 첫 trial들이 여기서 시작."""
    import yaml
    r = yaml.safe_load(open(recipe_path))
    cfgs, seen = [], set()
    for rg in r["settings"]["policy"]["regimes"]:
        c = rg["config"]
        mw = str((rg.get("backend_env") or {}).get("DRAFT_MW", "0"))
        key = (c["bmax"], c["K"], c["W"], mw)
        if key not in seen:
            seen.add(key)
            cfgs.append({"bmax": c["bmax"], "K": c["K"], "W": c["W"], "DRAFT_MW": mw})
    return cfgs


def enqueue_warmstart(study, cfgs, space, with_mw: bool, phase2_env: bool):
    """탐색 공간 밖 config(예: bmax 상한 초과)는 건너뛰고 정직 보고."""
    n = 0
    for c in cfgs:
        if not (space["bmax"][0] <= c["bmax"] <= space["bmax"][1] and c["bmax"] % 2 == 0
                and space["K"][0] <= c["K"] <= space["K"][1]
                and space["W"][0] <= c["W"] <= space["W"][1]):
            print(f"[warmstart] 공간 밖 → 스킵: {c}", flush=True)
            continue
        p = {"bmax": c["bmax"], "K": c["K"], "W": c["W"]}
        if with_mw:
            p["DRAFT_MW"] = c["DRAFT_MW"]
        if phase2_env:
            p.update({"MTP_Q4": "1", "DRAFT_GEMV": "0", "DRAFT_SEQ": "0"})
        study.enqueue_trial(p, skip_if_exists=True)
        n += 1
    print(f"[warmstart] {n}개 config enqueue(레시피 regime 승자)", flush=True)
    return n


def run_warmstart_ab(mp, dp, args):
    """warm-start 유/무 controlled A/B — 같은 공간·예산·sampler seed·regime(λ, SHORTP), in-memory study 2개.
    목적함수는 결정적(같은 trace seed)이므로 config→결과 캐시를 두 팔이 공유(재실행 절약 — 비교 유효성 불변).
    지표: 두 팔 통합 최고 score의 99% 도달까지 trial 수."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = search_space_from_profiles(mp, dp)
    fixed_env = {"MTP_Q4": "1", "DRAFT_GEMV": "0", "DRAFT_SEQ": "0"}
    cfgs = recipe_warmstart_configs(args.ab_warmstart)
    print(f"[A/B] regime λ={args.lam} SHORTP={args.shortp}, 팔당 {args.trials} trials "
          f"(screen nreq={args.screen_nreq}), 레시피 후보 {len(cfgs)}개", flush=True)
    server = WarmServer(mp)
    cache = {}

    def make_obj():
        def objective(trial):
            bmax = trial.suggest_int("bmax", *space["bmax"], step=2)
            K = trial.suggest_int("K", *space["K"])
            W = trial.suggest_int("W", *space["W"])
            mw = trial.suggest_categorical("DRAFT_MW", P2B_ENV_AXES["DRAFT_MW"])
            if K + W <= 8:
                raise optuna.TrialPruned()
            ok, need_gb, _ = mem_filter(mp, dp, bmax, K, W)
            if not ok:
                raise optuna.TrialPruned()
            key = (bmax, K, W, mw)
            if key not in cache:
                env_key = tuple(sorted({**fixed_env, "DRAFT_MW": mw, "SHORTP": str(args.shortp)}.items()))
                cache[key] = server.run(env_key, bmax, K, W, args.lam, args.screen_nreq, args.seed)
            res = cache[key]
            for k in ("tok_s", "tpot95"):
                trial.set_user_attr(k, res[k])
            return score(res, args.tpot_cap)
        return objective

    arms = {}
    try:
        for name, warm in (("warm", True), ("cold", False)):
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True))
            if warm:
                enqueue_warmstart(study, cfgs, space, with_mw=True, phase2_env=False)
            study.optimize(make_obj(), n_trials=args.trials)
            hist = [(t.number, t.value) for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            arms[name] = {"history": hist, "best": study.best_trial.params,
                          "best_score": study.best_value}
            print(f"[A/B {name}] best {study.best_trial.params} score {study.best_value:.1f}", flush=True)
    finally:
        server.stop()

    target = 0.99 * max(a["best_score"] for a in arms.values())
    print(f"\n[A/B 결과] 목표 = 통합 최고 score의 99% ({target:.1f})")
    for name, a in arms.items():
        reach = next((i + 1 for i, (_, v) in enumerate(a["history"]) if v >= target), None)
        a["trials_to_target"] = reach
        curve = " ".join(f"{v:.1f}" for _, v in a["history"])
        reach_s = str(reach) if reach else "미도달(/%d)" % len(a["history"])
        print(f"  {name}: 도달 trial 수 = {reach_s} | "
              f"best {a['best_score']:.1f} {a['best']} | curve: {curve}")
    return arms


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
    ap.add_argument("--phase2", action="store_true", help="env 축 확장 + warm 서버 + 상위3 검증")
    ap.add_argument("--topk", type=int, default=3, help="phase2: 검증할 상위 후보 수")
    ap.add_argument("--phase2b", action="store_true", help="regime 격자(λ×mix)별 튜닝 + variant 축 + multi-fidelity")
    ap.add_argument("--regime-trials", type=int, default=20, help="phase2b: regime당 스크리닝 trial 수")
    ap.add_argument("--screen-nreq", type=int, default=12, help="phase2b: 저충실도 nreq(in-engine)")
    ap.add_argument("--warmstart", help="recipe.yaml — regime 승자 config를 enqueue_trial로 주입(transfer warm-start)")
    ap.add_argument("--ab-warmstart", help="recipe.yaml — warm-start 유/무 controlled A/B(같은 공간·예산·seed)")
    ap.add_argument("--shortp", type=float, default=0.3, help="A/B regime의 SHORTP(mix)")
    args = ap.parse_args()

    mp = json.load(open(args.model))
    dp = json.load(open(args.device))
    if not os.path.exists(BENCH):
        sys.exit(f"bench 바이너리 없음: {BENCH} (make 먼저)")

    if args.ab_warmstart:
        run_warmstart_ab(mp, dp, args)
        return

    if args.phase2b:
        table = run_phase2b(mp, dp, args)
        art = {"schema": "dgx-serving/autotune-regime-table@1",
               "created": datetime.datetime.now().isoformat(timespec="seconds"),
               "device_profile": dp, "model_profile": mp,
               "objective": {"screen_nreq": args.screen_nreq, "final_nreq": 60, "seed": args.seed,
                             "tpot_cap": args.tpot_cap, "fixed_env": {"MTP_Q4": "1", "DRAFT_GEMV": "0", "DRAFT_SEQ": "0"}},
               "regime_table": table}
        out = os.path.join(HERE, "artifacts", f"regime-table-{mp['name'].replace('/', '_').replace(' ', '')}-{datetime.date.today()}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(art, open(out, "w"), indent=1, ensure_ascii=False)
        print(f"[p2b] regime 테이블: {out}", flush=True)
        for row in table:
            b = row["best"]
            print(f"  {row['regime']} -> {b['params']} : {b['nreq60']['tok_s']} tok/s tpot {b['nreq60']['tpot95']}", flush=True)
        return

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = args.study or (f"autotune-p2-{mp['name'].replace('/', '_').replace(' ', '')}"
                                if args.phase2 else
                                f"autotune-{mp['name'].replace('/', '_').replace(' ', '')}")
    storage = f"sqlite:///{os.path.join(HERE, 'autotune.db')}"
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True,
                                direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.sampler_seed,
                                                                   multivariate=True))
    cache = {}
    # 재개 시 기존 완료 trial을 캐시에 복원(같은 config 재실행 방지)
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and "tok_s" in t.user_attrs:
            cache[(t.params["bmax"], t.params["K"], t.params["W"],
                   params_env_key(t.params))] = dict(t.user_attrs)
    if args.warmstart:
        enqueue_warmstart(study, recipe_warmstart_configs(args.warmstart),
                          search_space_from_profiles(mp, dp), with_mw=False, phase2_env=args.phase2)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[autotune] {study_name}: 기존 완료 {done}, 목표 {args.trials} trials "
          f"(seed={args.seed} λ={args.lam} nreq={args.nreq} tpot_cap={args.tpot_cap} "
          f"phase2={args.phase2})", flush=True)
    server = WarmServer(mp) if args.phase2 else None
    try:
        study.optimize(make_objective(mp, dp, args, cache, server), n_trials=args.trials)
    finally:
        if server:
            server.stop()

    best = study.best_trial
    print(f"\n[autotune] BEST trial#{best.number}: {best.params} score {best.value:.1f} "
          f"tok/s {best.user_attrs.get('tok_s')} tpot {best.user_attrs.get('tpot95')}", flush=True)

    # ---- 검증 프로토콜: 상위 K 후보(고유 config) 각각 — 다른 seed + full ISO(해당 스킴) + nreq=60 확정치 ----
    import optuna.importance
    comp = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    comp.sort(key=lambda t: t.value, reverse=True)
    seen, top = set(), []
    for t in comp:
        ck = tuple(sorted(t.params.items()))
        if ck not in seen:
            seen.add(ck)
            top.append(t)
        if len(top) >= (args.topk if args.phase2 else 1):
            break
    alt_seed = args.seed + 735
    candidates = []
    for rank, t in enumerate(top, 1):
        p = t.params
        eenv = dict(params_env_key(p))
        print(f"[verify #{rank}] {p} — alt-seed({alt_seed}) / nreq60 / full ISO / CYC...", flush=True)
        alt, _ = run_bench(mp, p["bmax"], p["K"], p["W"], args.lam, args.nreq, alt_seed, extra_env=eenv)
        fin, _ = run_bench(mp, p["bmax"], p["K"], p["W"], args.lam, 60, args.seed, extra_env=eenv)
        iso, _ = run_bench(mp, p["bmax"], p["K"], p["W"], args.lam, 12, args.seed, iso=True, extra_env=eenv)
        cyc, _ = run_bench(mp, p["bmax"], p["K"], p["W"], args.lam, 30, args.seed, cyc=True, extra_env=eenv)
        print(f"[verify #{rank}] alt {alt.get('tok_s')} / nreq60 {fin.get('tok_s')} "
              f"(tpot {fin.get('tpot95')}) / {iso.get('iso')}", flush=True)
        candidates.append({
            "rank": rank, "trial": t.number, "params": p, "scheme_env": {"MTP_Q4": "1", **eenv},
            "objective_measured": {k: t.user_attrs.get(k) for k in ("tok_s", "tpot95", "accept", "avgB")},
            "alt_seed": {"seed": alt_seed, **{k: alt.get(k) for k in ("tok_s", "tpot95", "accept")}},
            "nreq60": {k: fin.get(k) for k in ("tok_s", "tpot95", "accept", "avgB")},
            "iso": iso.get("iso"), "iso_pass": iso.get("iso_pass"),
            "cyc": cyc.get("cyc_lines", []),
        })
    try:
        importance = {k: round(v, 4) for k, v in
                      optuna.importance.get_param_importances(study).items()}
    except Exception as e:
        importance = {"error": str(e)}
    print(f"[autotune] 축 중요도: {importance}", flush=True)

    art = {
        "schema": "dgx-serving/autotune-config@2",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "device_profile": dp, "model_profile": mp,
        "objective": {"lam": args.lam, "nreq": args.nreq, "seed": args.seed,
                      "tpot_cap": args.tpot_cap, "metric": "tok_s - 100*max(0,tpot95-cap)",
                      "phase2_axes": list(P2_ENV_AXES) if args.phase2 else [],
                      "excluded_axes": {"compile_time": ["SMAX", "GdnStateT", "MMQ tile"],
                                        "no_knob": ["chunk size", "canonical M boundary", "splitk gate"],
                                        "not_in_spec_path": ["BM16", "SPLITK"],
                                        "known_degenerate": ["DQ4=0 (27b accept collapse)"]}},
        "candidates": candidates,
        "param_importance": importance,
        "trials": [{"n": t.number, "params": t.params, "state": str(t.state),
                    "value": t.value, "attrs": t.user_attrs} for t in study.trials],
    }
    out = os.path.join(HERE, "artifacts",
                       f"{study_name}-{datetime.date.today()}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(art, open(out, "w"), indent=1, ensure_ascii=False)
    ok = candidates and candidates[0]["iso_pass"]
    print(f"[autotune] 아티팩트: {out}  (top1 ISO {'PASS' if ok else 'FAIL — 무효, 차점 검토'})",
          flush=True)


if __name__ == "__main__":
    main()
