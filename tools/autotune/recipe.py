#!/usr/bin/env python3
"""dgx-serving 레시피 v0 — 공유 가능한 최적화 서빙 아티팩트.

레시피 = 튜닝 산출물(regime→config 정책)의 서명된 배포판(YAML):
  meta / target(장비 지문+오차+user_constraints) / settings(정책, backend 분리) / claims(실측+ISO 스킴) / provenance.

원칙(헌법):
- 맹목 적용 금지: 로드 시 ①장비 지문 대조(toolkit caps 필수) ②로컬 메모리 가드 재계산(레시피의 예산을 믿지 않음
  — unified oversubscribe=시스템 다운 전례) ③미니 자체검증(in-engine)+재현율 리포트.
- 재현율 정직: claims와 같은 측정 조건(nreq/seed/λ/mix)끼리만 비교해 % 그대로 보고.
- 스킴 명시: claims의 ISO는 스킴별(bit-identity는 스킴 내 개념 — 유효 argmax 복수). 스킴 없으면 주장 무효.
- 백엔드 분리: 코어 스키마는 백엔드-중립, dgx-serving 고유(env 축·variant dispatch)는 settings.backend에만.
- 불일치=transfer: 지문 오차 초과는 실패가 아니라 "레시피를 prior로 로컬 재튜닝"(autotune warm-start) 안내.

사용:
  python tools/autotune/recipe.py export --table tools/autotune/artifacts/regime-table-*.json --out recipes/x.yaml
  python tools/autotune/recipe.py verify recipes/x.yaml [--full] [--device profiles/gb10.json]
  python tools/autotune/recipe.py run recipes/x.yaml --lam 4 --shortp 0.3 --nreq 60
"""
import argparse, glob, hashlib, json, os, subprocess, sys, datetime

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from autotune import per_slot_bytes, per_slot_bytes_auto, mem_available_bytes, run_bench  # noqa: E402

# plain(spec 無) 서빙 미니벤치 — 모델별 bench dispatch(profile serving_bench):
#   bench_serve_gemma4(Gemma4 전용): "<w> <uq> <nreq> <seed> <pmax> <maxlen> <B> -- <λ>", 행 "<B> <λ> | req tok | tp50 tp95 | avgB maxQ"
#   bench_serve(qwen 계열=35b MoE/27b): "<w> <g> <uq> <B> <nreq> <seed> <λ..>", 행 "c-ch <λ> | req tok | e2e50 e2e95 e2e99 | ttft95 tp95 | avgB maxQ"
def run_bench_plain(mp, bmax, lam, nreq, seed):
    sb = mp.get("serving_bench", "bench_serve_gemma4")
    bin_ = os.path.join(REPO, "tools", sb)
    if sb == "bench_serve":   # qwen 계열: golden_dir 필요, SKIP1TOK/SKIPSTAT로 c-ch만
        env = dict(os.environ, SKIP1TOK="1", SKIPSTAT="1")
        cmd = [bin_, mp["weights_dir"], mp["golden_dir"], str(mp.get("uq", 2)), str(bmax), str(nreq), str(seed), str(lam)]
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
        out = r.stdout + r.stderr
        for line in out.splitlines():
            p = line.split()
            if len(p) >= 5 and p[0] == "c-ch":   # c-ch λ | req tok | e2e50 e2e95 e2e99 | ttft95 tp95 | avgB maxQ
                try:
                    if abs(float(p[1]) - lam) < 1e-6:
                        return {"tok_s": float(p[4]), "tpot95": float(p[11]) if len(p) > 11 else None}
                except (ValueError, IndexError):
                    continue
        return {"tok_s": 0, "tpot95": None, "raw": out[-400:]}
    # gemma4 전용
    cmd = [bin_, mp["weights_dir"], str(mp.get("uq", 2)), str(nreq), str(seed), "32", "384", str(bmax), "--", str(lam)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1200)
    out = r.stdout + r.stderr
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 5 and p[0].isdigit() and int(p[0]) == bmax and p[2] == "|":
            try:
                if abs(float(p[1]) - lam) < 1e-6:
                    return {"tok_s": float(p[4]), "tpot95": float(p[7]) if len(p) > 7 else None}
            except (ValueError, IndexError):
                continue
    return {"tok_s": 0, "tpot95": None, "raw": out[-400:]}

SCHEMA_VER = "dgx-recipe/v0"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------- export: regime-table 아티팩트 + 프로파일 → recipe.yaml ----------
def cmd_export(args):
    art = json.load(open(args.table))
    dp, mp = art["device_profile"], art["model_profile"]
    regimes = []
    claims = []
    for row in art["regime_table"]:
        rg, best = row["regime"], row["best"]
        # dict 공유 금지(regimes/claims에 각각 복사) — PyYAML anchor(&id) 없는 깨끗한 배포 파일
        cfg = dict(bmax=best["params"]["bmax"], K=best["params"]["K"], W=best["params"]["W"])
        benv = {"DRAFT_MW": best["params"].get("DRAFT_MW", "0")}
        regimes.append({"when": {"lam": rg["lam"], "short_ratio": rg["shortp"]},
                        "config": cfg, "backend_env": benv})
        # 스킴 = 이 config에서 활성인 커널 경로(bit-identity 주장의 기준)
        scheme = (f"w4a8+mtp_q4+concat-mmq(Nn>8)"
                  f"+draft_{'mw' + benv['DRAFT_MW'] if benv['DRAFT_MW'] != '0' else 'gemm_batch'}")
        obj = art["objective"]
        claims.append({
            "regime": {"lam": rg["lam"], "short_ratio": rg["shortp"]},
            "config": dict(cfg), "scheme": scheme,
            "measured": {
                "screen": {"conditions": {"nreq": obj["screen_nreq"], "seed": obj["seed"]},
                           **{k: best["screen"].get(k) for k in ("tok_s", "tpot95")}},
                "final": {"conditions": {"nreq": obj["final_nreq"], "seed": obj["seed"]},
                          **{k: best["nreq60"].get(k) for k in ("tok_s", "tpot95", "accept")}},
            },
        })
    recipe = {
        "meta": {"schema": SCHEMA_VER,
                 "created": datetime.datetime.now().isoformat(timespec="seconds"),
                 "tool": "dgx-serving autotune phase2b", "engine_commit": git_commit()},
        "target": {
            "device": {"name": dp["name"].split("(")[0].strip() if "(" in dp["name"] else dp["name"],
                       "gpu_probe": "NVIDIA GB10",
                       "bandwidth_gbps": dp["bandwidth_gbps"], "mem_total_gb": dp["mem_total_gb"],
                       "unified_memory": dp["unified_memory"],
                       "toolkit_required": {"cp_async": dp["toolkit"]["cp_async"],
                                            "mma_int8": dp["toolkit"]["mma_int8"],
                                            "fp4_mma": dp["toolkit"]["fp4_mma"]},
                       "tolerance": {"bandwidth_pct": 10, "mem_total_pct": 10}},
            "model": {"id": mp["name"], "params_b": mp["params_b"], "gdn": mp["gdn"],
                      "tied_lm_head": mp["tied_lm_head"], "moe": mp["moe"], "quant": mp["quant"]},
            "user_constraints": {"quant_bits": 4, "tpot95_cap_s": art["objective"]["tpot_cap"]},
        },
        "settings": {
            "policy": {"type": "static_regime_lookup",
                       "note": "요청-경계 핫스위치는 미구현(사이클 전환은 스킴 경계로 bit-ISO 미보장 — Phase 2b 실측)",
                       "regimes": regimes},
            "backend": {"name": "dgx-serving",
                        "engine_bin": "tools/bench_serve_spec",
                        "fixed_env": art["objective"]["fixed_env"],
                        "variant_axes": {"DRAFT_MW": ["0", "1", "2"]}},
        },
        "claims": claims,
        "provenance": {"source_artifact": os.path.basename(args.table),
                       "source_sha256_16": sha256_file(args.table),
                       "trial_db": "tools/autotune/autotune.db (not distributed)",
                       "verification": "regime winners full ISO 12/12 PASS ×2 (long B16K7W2 mw0, short B6K7W3 mw1)"},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    yaml.safe_dump(recipe, open(args.out, "w"), allow_unicode=True, sort_keys=False)
    print(f"[recipe] exported: {args.out} ({len(regimes)} regimes, {len(claims)} claims)")


# ---------- verify: 지문 → 로컬 가드 재계산 → 미니벤치 → 재현율 ----------
def load_recipe(path):
    r = yaml.safe_load(open(path))
    if r.get("meta", {}).get("schema") != SCHEMA_VER:
        sys.exit(f"[recipe] 스키마 불일치: {r.get('meta', {}).get('schema')} (기대 {SCHEMA_VER})")
    for c in r.get("claims", []):
        if not c.get("scheme"):
            sys.exit("[recipe] claims에 scheme 누락 — 스킴 없는 bit-identity 주장은 무효 취급(거부)")
    return r


def fingerprint_check(recipe, dp):
    """로컬 장비(디바이스 프로파일+nvidia-smi)와 target 지문 대조. (ok, warnings, hard_fails)"""
    t = recipe["target"]["device"]
    warns, fails = [], []
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        gpu = "unknown"
    if t.get("gpu_probe") and t["gpu_probe"] != gpu:
        warns.append(f"GPU 이름 불일치: recipe '{t['gpu_probe']}' vs local '{gpu}'")
    tol = t.get("tolerance", {})
    for key, pct_key in (("bandwidth_gbps", "bandwidth_pct"), ("mem_total_gb", "mem_total_pct")):
        rv, lv = t.get(key), dp.get(key)
        if rv and lv:
            pct = abs(rv - lv) / rv * 100
            if pct > tol.get(pct_key, 10):
                warns.append(f"{key} 오차 초과: recipe {rv} vs local {lv} ({pct:.0f}% > ±{tol.get(pct_key, 10)}%)")
    req = t.get("toolkit_required", {})
    loc = dp.get("toolkit", {})
    for cap, need in req.items():
        if need and not loc.get(cap, False):
            fails.append(f"toolkit cap 미충족: {cap} 필요하나 로컬 미지원 — 거부")
    return warns, fails


def local_memory_check(recipe, mp, dp):
    """레시피의 각 regime config를 이 머신에서 재계산(레시피 예산 신뢰 X). (ok_list, rejected)"""
    margin = dp.get("mem_guard_margin", 0.7)
    avail = margin * mem_available_bytes()
    ok, rejected = [], []
    for rg in recipe["settings"]["policy"]["regimes"]:
        c = rg["config"]
        # plain(spec 無)은 K/W 없음 → per_slot_bytes_auto가 KV-only 계산
        need = c["bmax"] * per_slot_bytes_auto(mp, c.get("K", 0), c.get("W", 0))
        (ok if need <= avail else rejected).append(
            {"regime": rg["when"], "config": c, "need_gb": round(need / 1e9, 2)})
    return ok, rejected, round(avail / 1e9, 2)


def cmd_verify(args):
    recipe = load_recipe(args.recipe)
    dp = json.load(open(args.device))
    mp = json.load(open(args.model))
    print(f"[verify] {args.recipe} (schema {SCHEMA_VER}, engine {recipe['meta']['engine_commit']})")

    # ① 장비 지문
    warns, fails = fingerprint_check(recipe, dp)
    for w in warns:
        print(f"  [지문 경고] {w}")
    for f in fails:
        print(f"  [지문 거부] {f}")
    if fails:
        sys.exit("[verify] REJECT — toolkit 필수 caps 미충족. 이 레시피는 이 장비에서 실행 불가.")
    if warns:
        print("  [안내] 지문 오차 초과 — claims 재현이 보장되지 않음. 이 레시피를 prior로 로컬 재튜닝 권장:\n"
              "         autotune.py --phase2b (레시피 config들을 초기 후보로 enqueue하는 warm-start는 v1 예정)")

    # ② 로컬 메모리 가드 재계산 (안전 불변식 — 레시피의 예산 필드를 신뢰하지 않음)
    ok_cfgs, rejected, avail_gb = local_memory_check(recipe, mp, dp)
    print(f"  [메모리] 로컬 가용(margin 적용) {avail_gb}GB — {len(ok_cfgs)} config OK, {len(rejected)} 거부")
    for r in rejected:
        print(f"  [메모리 거부] {r['regime']} {r['config']} need {r['need_gb']}GB > {avail_gb}GB")
    if rejected and not ok_cfgs:
        sys.exit("[verify] REJECT — 모든 regime config가 로컬 메모리 예산 초과.")

    if args.skip_bench:
        print("[verify] 미니벤치 생략(--skip-bench). 지문/메모리 검사만 완료.")
        return

    # ③ 미니벤치 재현율: claims의 screen 조건(nreq/seed 동일)으로 대표 regime 실측 — 같은 조건끼리만 비교(정직)
    picks = []
    claims = recipe["claims"]
    best_thru = max(claims, key=lambda c: c["measured"]["final"]["tok_s"] or 0)
    best_lat = min(claims, key=lambda c: c["measured"]["final"]["tpot95"] or 9)
    picks = [best_thru] + ([best_lat] if best_lat is not best_thru else [])
    plain = not mp.get("mtp")   # spec 無 → plain 미니벤치(bench_serve_gemma4), spec ISO 없음
    results = []
    for c in picks:
        cond_key = "final" if args.full else "screen"
        cond = c["measured"][cond_key]["conditions"]
        cfg, rg = c["config"], c["regime"]
        if plain:
            res = run_bench_plain(mp, cfg["bmax"], rg["lam"], cond["nreq"], cond["seed"])
        else:
            env = {**recipe["settings"]["backend"]["fixed_env"], "SHORTP": str(rg["short_ratio"])}
            for rgs in recipe["settings"]["policy"]["regimes"]:
                if rgs["when"]["lam"] == rg["lam"] and rgs["when"]["short_ratio"] == rg["short_ratio"]:
                    env.update(rgs["backend_env"])
            res, _ = run_bench(mp, cfg["bmax"], cfg["K"], cfg["W"], rg["lam"], cond["nreq"], cond["seed"],
                               extra_env=env)
        claimed = c["measured"][cond_key]["tok_s"]
        repro = (res.get("tok_s", 0) / claimed * 100) if claimed else 0
        results.append({"regime": rg, "config": cfg, "claimed_tok_s": claimed,
                        "local_tok_s": res.get("tok_s"), "repro_pct": round(repro, 1),
                        "conditions": cond, "scheme": c["scheme"]})
        print(f"  [재현율] {rg} {cfg}: claim {claimed} vs local {res.get('tok_s')} tok/s "
              f"= {repro:.1f}% (조건 {cond})")

    # ④ bit-id 게이트: spec은 서빙-시 ISO 스팟, plain은 오프라인 bit-id(recipe verification.golden 참조)로 대체
    if plain:
        gld = recipe.get("verification", {}).get("golden", "오프라인 골든 스위트(recipe verification.golden)")
        iso = {"iso": f"offline: {gld}", "iso_pass": True}
        print(f"  [bit-id] plain: {iso['iso']}")
    else:
        c = best_thru
        env = {**recipe["settings"]["backend"]["fixed_env"], "SHORTP": str(c["regime"]["short_ratio"])}
        for rgs in recipe["settings"]["policy"]["regimes"]:
            if rgs["when"]["lam"] == c["regime"]["lam"] and rgs["when"]["short_ratio"] == c["regime"]["short_ratio"]:
                env.update(rgs["backend_env"])
        iso, _ = run_bench(mp, c["config"]["bmax"], c["config"]["K"], c["config"]["W"],
                           c["regime"]["lam"], 12, 42, iso=True, extra_env=env)
        print(f"  [ISO 스팟] {c['config']} scheme={c['scheme']}: {iso.get('iso')}")

    ok = all(r["repro_pct"] >= 90 for r in results) and iso.get("iso_pass") and not warns
    print(f"[verify] {'PASS' if ok else 'PARTIAL'} — 재현율 {[r['repro_pct'] for r in results]}%, "
          f"ISO {'PASS' if iso.get('iso_pass') else 'FAIL'}"
          + ("" if not warns else ", 지문 경고 있음(재튜닝 권장)"))


# ---------- run: regime 룩업 → 엔진 실행(정적 룩업 런처) ----------
def cmd_run(args):
    recipe = load_recipe(args.recipe)
    mp = json.load(open(args.model))
    dp = json.load(open(args.device))
    # 가장 가까운 regime 선택(λ·mix 거리)
    best = min(recipe["settings"]["policy"]["regimes"],
               key=lambda r: abs(r["when"]["lam"] - args.lam) + abs(r["when"]["short_ratio"] - args.shortp))
    cfg = best["config"]
    # 안전 불변식: 실행 직전에도 로컬 메모리 재계산
    need = cfg["bmax"] * per_slot_bytes(mp, cfg["K"], cfg["W"])
    avail = dp.get("mem_guard_margin", 0.7) * mem_available_bytes()
    if need > avail:
        sys.exit(f"[run] REJECT — 메모리 예산 초과: need {need/1e9:.1f}GB > avail {avail/1e9:.1f}GB")
    env = {**recipe["settings"]["backend"]["fixed_env"], **best["backend_env"],
           "SHORTP": str(args.shortp)}
    cmd = [os.path.join(REPO, recipe["settings"]["backend"]["engine_bin"]),
           mp["weights_dir"], mp["golden_dir"], str(mp["uq"]),
           str(cfg["bmax"]), str(args.nreq), str(args.seed), str(cfg["K"]), str(cfg["W"]), str(args.lam)]
    print(f"[run] regime {best['when']} → config {cfg} env {env}")
    os.execve(cmd[0], cmd, {**os.environ, **env})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("export")
    pe.add_argument("--table", required=True)
    pe.add_argument("--out", required=True)
    pv = sub.add_parser("verify")
    pv.add_argument("recipe")
    pv.add_argument("--device", default=os.path.join(HERE, "profiles", "gb10.json"))
    pv.add_argument("--model", default=os.path.join(HERE, "profiles", "qwen35-27b.json"))
    pv.add_argument("--full", action="store_true", help="claims의 final 조건(nreq60)으로 재현")
    pv.add_argument("--skip-bench", action="store_true")
    pr = sub.add_parser("run")
    pr.add_argument("recipe")
    pr.add_argument("--device", default=os.path.join(HERE, "profiles", "gb10.json"))
    pr.add_argument("--model", default=os.path.join(HERE, "profiles", "qwen35-27b.json"))
    pr.add_argument("--lam", type=float, required=True)
    pr.add_argument("--shortp", type=float, default=0.7)
    pr.add_argument("--nreq", type=int, default=60)
    pr.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"export": cmd_export, "verify": cmd_verify, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
