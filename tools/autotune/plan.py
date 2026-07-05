#!/usr/bin/env python3
"""M2 Plan 골격 — 모델 해부 → roofline 상한 → 사용자 제약 → 전략 규칙 v0(탐색 계획 자동 생성).

정직 원칙:
- roofline은 **상한**(대역폭 ÷ 토큰당-이동바이트 [× accept]) — 예측이 아님. 실측 달성률을 병기해 오차를 명시.
- 미지원 제약(quant 2/1bit 등)은 실패가 아니라 **정직 리턴**(불가+이유).
- 검증 = ground truth 대조: 해부기→27b 수동 프로파일, 전략→27b가 실측으로 밟은 경로(fANOVA·챕터 이력)와 정합.

사용:
  python tools/autotune/plan.py anatomy --manifest artifacts/qwen35-27b/weights/manifest.json \
      [--compare tools/autotune/profiles/qwen35-27b.json]
  python tools/autotune/plan.py roofline --device profiles/gb10.json --model profiles/qwen35-27b.json
  python tools/autotune/plan.py strategy --device ... --model ... [--constraints '{"quant_bits":4,"tpot95_cap_s":0.35}']
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- 모델 해부기: manifest/HF config → model profile ----------
def anatomize(manifest_path: str) -> dict:
    m = json.load(open(manifest_path))
    is_hf = "model_type" in m or "architectures" in m  # HF config.json
    if is_hf:
        # 멀티모달 래퍼(Gemma4 등): 텍스트 서빙 대상은 text_config — 언랩(유연성 수정 #1, M4)
        outer = m
        if "text_config" in m and isinstance(m["text_config"], dict):
            m = m["text_config"]
        # HF config.json 매핑(Qwen3.5/Gemma4 계열 — 새 계열은 필드 추가로 확장)
        layers = m.get("num_hidden_layers")
        lt = m.get("layer_types") or []
        gdn = any(t in ("linear_attention", "gdn") for t in lt) or m.get("linear_num_value_heads") is not None
        moe = (m.get("num_experts") or 0) > 0 or bool(m.get("enable_moe_block"))
        hd = m.get("head_dim") or (m["hidden_size"] // m["num_attention_heads"]
                                   if m.get("num_attention_heads") else None)
        prof = {
            "name": outer.get("_name_or_path") or outer.get("model_type", "unknown"),
            "num_layers": layers,
            "hidden_size": m.get("hidden_size"),
            "gdn": gdn, "moe": moe,
            "tied_lm_head": bool(m.get("tie_word_embeddings", outer.get("tie_word_embeddings"))),
            "num_kv_heads": m.get("num_key_value_heads"),
            "head_dim": hd,
            "vocab_size": m.get("vocab_size"),
            "params_b": round(estimate_params_hf(m) / 1e9, 1) if m.get("hidden_size") else None,
        }
        # attention 패턴(GDN 없는 계열: sliding/full 혼합 — 서빙 마스크·KV 설계 입력)
        if lt and not gdn:
            nfull = sum(1 for t in lt if t == "full_attention")
            prof["attn_pattern"] = {"full_layers": nfull, "sliding_layers": len(lt) - nfull,
                                    "sliding_window": m.get("sliding_window")}
        # 신규 연산 후보(있을 때만 — 커널 자산 매핑의 입력)
        for k in ("final_logit_softcapping", "attn_logit_softcapping", "hidden_activation",
                  "attention_k_eq_v", "global_head_dim", "num_global_key_value_heads",
                  "rope_parameters", "rope_theta", "partial_rotary_factor"):
            if m.get(k) is not None:
                prof[k] = m[k]
        prof["multimodal_wrapper"] = "text_config" in outer and outer is not m
    else:
        # dgx-serving manifest.json(추출 파이프 산출물)
        lt = m.get("layer_types") or []
        nfull = sum(1 for t in lt if t == "full_attention") if lt else None
        interval = (m["num_layers"] // nfull) if nfull else m.get("full_attention_interval")
        prof = {
            "name": m.get("model", "unknown").split("/")[-1],
            "params_b": round(estimate_params(m) / 1e9),
            "gdn": m.get("linear_num_value_heads") is not None,
            "tied_lm_head": bool(m.get("tie_word_embeddings")),
            "moe": (m.get("num_experts") or 0) > 0,
            "quant": "w4a8" if m.get("qblk") else "fp",
            "num_layers": m["num_layers"],
            "full_attention_interval": interval,
            "num_kv_heads": m["num_kv_heads"],
            "head_dim": m["head_dim"],
            "linear_num_key_heads": m.get("linear_num_key_heads"),
            "linear_num_value_heads": m.get("linear_num_value_heads"),
            "linear_key_head_dim": m.get("linear_key_head_dim"),
            "linear_value_head_dim": m.get("linear_value_head_dim"),
            "linear_conv_kernel": m.get("linear_conv_kernel"),
            "pmax_default": 32,
        }
    return prof


def estimate_params_hf(m: dict) -> float:
    """HF text config 형상으로 파라미터 근사(±10% 추정 — 레이어 타입별 head 구성 반영).
    가정 명시: MLP=gated 3장(gate/up/down), attention_k_eq_v=true면 K/V projection 1장."""
    H, L, V = m["hidden_size"], m["num_hidden_layers"], m["vocab_size"]
    FFN = m.get("intermediate_size", 4 * H)
    NQ = m.get("num_attention_heads")
    lt = m.get("layer_types") or ["full_attention"] * L
    kv_mult = 1 if m.get("attention_k_eq_v") else 2
    attn = 0
    for t in lt:
        if t == "full_attention" and m.get("global_head_dim"):
            hd, nkv = m["global_head_dim"], m.get("num_global_key_value_heads",
                                                   m.get("num_key_value_heads"))
        else:
            hd, nkv = m.get("head_dim", H // NQ), m.get("num_key_value_heads")
        attn += H * NQ * hd + kv_mult * H * nkv * hd + NQ * hd * H  # q + k/v + o
    mlp = L * 3 * H * FFN
    embed = V * H * (1 if m.get("tie_word_embeddings") else 2)
    return attn + mlp + embed


def estimate_params(m: dict) -> float:
    """manifest 형상으로 파라미터 수 추정(대략 — attn/GDN/FFN/embed)."""
    H, L, V = m["hidden_size"], m["num_layers"], m["vocab_size"]
    FFN = m.get("intermediate_size", 4 * H)
    lt = m.get("layer_types") or []
    nfull = sum(1 for t in lt if t == "full_attention") if lt else L // (m.get("full_attention_interval") or 1)
    ngdn = L - nfull
    NQ, NKV, HD = m.get("num_q_heads", H // m["head_dim"]), m["num_kv_heads"], m["head_dim"]
    attn = nfull * (H * NQ * HD * (2 if m.get("attn_output_gate") else 1) + 2 * H * NKV * HD + NQ * HD * H)
    kdim = (m.get("linear_num_key_heads") or 0) * (m.get("linear_key_head_dim") or 0)
    vdim = (m.get("linear_num_value_heads") or 0) * (m.get("linear_value_head_dim") or 0)
    gdn = ngdn * (H * (2 * kdim + 2 * vdim) + vdim * H + H * vdim)  # qkvz 근사+out
    ffn = L * 3 * H * FFN
    embed = V * H * (1 if m.get("tie_word_embeddings") else 2)
    return attn + gdn + ffn + embed


# ---------- MoE 활성 파라미터(decode 1토큰이 실제 이동하는 weight) ----------
def active_params(mp: dict) -> float:
    """MoE decode 활성 파라미터 = lm_head + Σlayer(attn/GDN + router + shared_ffn + top_k·expert_ffn).
    전체 params(모든 expert 상주)와 구분 — decode bytes 지배는 활성분(A3B의 핵심). manifest 형상 근거."""
    H, L, V = mp["hidden_size"], mp["num_layers"], mp["vocab_size"]
    nfull = mp.get("num_full_layers", L // (mp.get("full_attention_interval") or 1))
    ngdn = L - nfull
    NQ, NKV, HD = mp["num_q_heads"], mp["num_kv_heads"], mp["head_dim"]
    attn_per = H * NQ * HD + 2 * H * NKV * HD + NQ * HD * H          # q+kv+o (full 층)
    kdim = mp.get("linear_num_key_heads", 0) * mp.get("linear_key_head_dim", 0)
    vdim = mp.get("linear_num_value_heads", 0) * mp.get("linear_value_head_dim", 0)
    gdn_per = H * (2 * kdim + 2 * vdim) + vdim * H + H * vdim         # qkvz 근사 + out (gdn 층)
    router = H * mp["num_experts"]
    shared = 3 * H * mp["shared_expert_intermediate_size"]
    experts = mp["top_k"] * 3 * H * mp["moe_intermediate_size"]      # ★활성 top_k expert만
    lm_head = V * H                                                  # 매 토큰 logits 읽기
    return lm_head + nfull * attn_per + ngdn * gdn_per + L * (router + shared + experts)


# ---------- roofline 상한 추정기 ----------
def roofline(dp: dict, mp: dict, accept: float = 4.4, batch: int = 12) -> dict:
    """상한 = BW ÷ 토큰당-이동바이트 (×accept for spec). 예측이 아니라 천장 — 달성률로 오차 명시.
    MoE는 활성 파라미터(active_params) 기준 — decode bytes=활성분(전체 상주≠이동)."""
    bw = dp["bandwidth_gbps"] * 1e9
    bytes_w = {"w4a8": 0.5 * 1.0625, "fp16": 2.0, "fp32": 4.0}.get(mp.get("quant", "w4a8"), 0.5 * 1.0625)
    is_moe = bool(mp.get("moe"))
    pb = (active_params(mp) if is_moe else mp["params_b"] * 1e9)
    per_tok = pb * bytes_w
    single = bw / per_tok
    assumptions = {"bandwidth_gbps": dp["bandwidth_gbps"], "quant": mp.get("quant"),
                   "bytes_per_param": round(bytes_w, 4),
                   "note": "weight 이동만 계상(KV/activation/GDN-state 제외) — 상한이며 예측 아님"}
    if is_moe:
        assumptions["moe"] = {"total_params_b": mp["params_b"], "active_params_b": round(pb / 1e9, 2),
                              "top_k": mp["top_k"], "num_experts": mp["num_experts"],
                              "note": "decode bytes=활성분(top_k expert+shared+attn/GDN+lm_head). 전체 상주는 128GB unified."}
    else:
        assumptions["params_b"] = mp["params_b"]
    return {
        "assumptions": assumptions,
        "single_decode_upper_tok_s": round(single, 1),
        "batch_upper_tok_s": {"B": batch, "value": round(single * batch, 1),
                              "note": "weight 1운반/배치 이상적 가정 — 실제는 draft/스케줄/attn per-slot로 하회"},
        "spec_single_upper_tok_s": {"accept": accept, "value": round(single * accept, 1)},
        "spec_serving_upper_tok_s": {"B": batch, "accept": accept, "value": round(single * batch, 1),
                                     "note": "spec 배칭도 weight-이동 상한은 batch와 동일(verify concat이 곱 실현 시)"},
    }


# 27b 실측 ground truth(정합 검증용) — 출처 명시. 갱신 시 여기만 수정.
MEASURED_27B = [
    # (모드, 실측 t/s, roofline 인자(accept, B), 출처)
    ("single non-spec", 11.9, (1.0, 1), "ch9-gemv-roofline(base gemv-MLP)"),
    ("single spec(tree K4W3)", 21.3, (2.27, 1), "ch9(tree+MTP q4, accept 2.27~2.86 하단)"),
    ("batch B16 non-spec", 93.3, (1.0, 16), "ch9-engine-landscape(bf16 GDN state)"),
    ("spec serving B16 K7W2", 51.7, (4.4, 16), "P2b regime long-heavy(avgB≈4.4)"),
]


def validate_27b(dp: dict, mp: dict) -> None:
    """roofline 상한 vs 27b 실측 — 달성률(%)로 오차를 정직하게 명시(상한≠예측의 증거)."""
    bw = dp["bandwidth_gbps"] * 1e9
    per_tok = mp["params_b"] * 1e9 * 0.5 * 1.0625
    single = bw / per_tok
    print(f"{'모드':<28} {'상한(t/s)':>10} {'실측':>8} {'달성률':>8}  출처")
    for mode, meas, (acc, b), src in MEASURED_27B:
        # weight-이동 상한: batch는 ×B(weight 1운반), spec은 ×accept — 단 batch 상한이 이미 accept 흡수를 포함
        upper = single * (b if b > 1 else acc)
        print(f"{mode:<28} {upper:>10.1f} {meas:>8.1f} {meas/upper*100:>7.1f}%  {src}")
    print("→ 달성률 17~63%: 상한은 weight-이동만 계상한 천장이며 예측이 아님. 하회 요인 실측 근거 —"
          "\n  batch: GDN state per-slot 운반+attn(ch9 arc), spec: draft/MTP 오버헤드+accept<천장(P4~7 nsys mmq 59%).")


# ---------- 사용자 제약 ----------
SUPPORTED = {"quant_bits": {4: "w4a8(현행 canonical)"},
             "unsupported_quant": {2: "미지원 — int2 커널/스킴 부재(로드맵 외)",
                                   1: "미지원 — 1bit는 현 아키텍처에서 품질 파탄 예상 + 커널 부재"}}


def check_constraints(c: dict) -> dict:
    out = {"accepted": {}, "rejected": {}}
    qb = c.get("quant_bits")
    if qb is not None:
        if qb in SUPPORTED["quant_bits"]:
            out["accepted"]["quant_bits"] = SUPPORTED["quant_bits"][qb]
        else:
            out["rejected"]["quant_bits"] = SUPPORTED["unsupported_quant"].get(
                qb, f"미지원 quant_bits={qb}")
    for k in ("tpot95_cap_s", "ttft95_cap_s"):
        if k in c:
            metric = k.split("_")[0]
            out["accepted"][k] = (
                f"soft 페널티(기본) 또는 hard SLO 제약(autotune --slo-mode): "
                f"기본은 score = tok/s − 100·max(0, {metric}−cap)(페널티만, cap 초과해도 순위 참여) / "
                f"--slo-mode는 {metric}<=cap 만족 config 중 throughput 최대(위반 config는 -1e6 페널티로 배제)")
    for k in ("fixed_bmax", "fixed_K", "fixed_W"):
        if k in c:
            out["accepted"][k] = "탐색 공간에서 해당 축 고정"
    return out


# ---------- 전략 규칙 v0: 특성 → 우선 축 + 계획 ----------
FANOVA_PRIOR_27B = {"DRAFT_SEQ": 0.33, "bmax": 0.31, "DRAFT_GEMV": 0.19, "MTP_Q4": 0.13, "K": 0.03, "W": 0.01}


def strategy(dp: dict, mp: dict, constraints: dict = None) -> dict:
    traits, axes = [], []
    spec = bool(mp.get("mtp"))   # MTP head 있어야 draft/spec 가능 — 없으면 spec축(K/W/DRAFT/accept) 전부 무의미
    if mp["params_b"] <= 2:
        traits.append("small(<2B)")
        axes.append(("sync/host 오버헤드 축(draft 배칭·디바이스화)", "0.8b 실측: sync가 지배(258 t/s 개선 사례)"))
    else:
        traits.append(f"large {'MoE' if mp.get('moe') else 'dense'}({mp['params_b']}B)")
        if spec:
            axes.append(("bytes/token 축(양자화·MTP spec — 분모 축소)", "27b 실측: MTP q4+spec으로 plain 추월(P5)"))
        else:
            axes.append(("bytes/token 축(양자화만 — MTP 부재로 spec 불가)",
                         "Gemma4: W4A8 weight 4bit 고정, decode matmul-bound(single 9.26=53.3%)"))
    if mp.get("gdn"):
        traits.append("hybrid(GDN+full)")
        axes.append(("GDN state 축(bf16 state·k-split — state 운반 bound)", "챕터9: bf16 state로 batch 1.42→1.02×"))
    if mp.get("moe"):
        traits.append("MoE")
        axes.append(("expert 축(활성 expert 스파시티 — bytes/token↓)", "35b-a3b 대상(미실측 — prior 없음)"))
    if spec and not mp.get("tied_lm_head"):
        axes.append(("draft lm_head 품질 축(dq4 필수 — untied)", "27b P4: dq4=0이 accept 붕괴"))
    if not spec:
        traits.append("plain(MTP 無 — spec 불가)")
        axes.append(("Bmax 축(★주 lever — batch가 weight 적재 amortize)",
                     "Gemma4 이식: W4A8 sb dispatch로 gemm_batch B4포화 돌파 → B16 aggr 76(2.0×)"))
        axes.append(("경로별 양자화 mix 축(프리필 q4·decode W4A8)",
                     "Gemma4 CYC: 프리필 gemm_tiled 75 t/s ≫ W4A8 gemv 12.7 / decode는 W4A8 sb 9.26"))
    cons = check_constraints(constraints or {})
    if spec:
        search_space = {"bmax": "메모리 유도 상한(4..16 step2)", "K": "3..8", "W": "1..5",
                        "env": ["MTP_Q4", "DRAFT_GEMV", "DRAFT_SEQ", "DRAFT_MW"],
                        "canonical_constraint": "K+W>8(concat-mmq)"}
        fanova = {"source": "27b Phase2(120 trial)", "importances": FANOVA_PRIOR_27B,
                  "guidance": "draft 커널 스킴 축 우선 탐색·형상(K/W)은 미세 — 예산 후반 배정"}
    else:
        # ★spec 無 정직 공간 축소: K/W/DRAFT/MTP축 전부 제거. 남는 실질 lever는 Bmax(주)+양자화 mix.
        search_space = {"bmax": "메모리 유도 상한(4..16 step2) — ★주 lever(유일 실측 스케일 축)",
                        "quant_path": "prefill=q4(gemm_tiled)/decode=W4A8(sb) — CYC 유도 고정(스윕 아님)",
                        "env": [],
                        "removed_axes": "K/W/DRAFT_*/MTP_Q4/accept — MTP 부재로 spec 경로 없음(정직 축소)",
                        "scheme_note": "W4A8 sb는 SMAX 버킷 내 bit-id — 버킷 경계(B≥9)/split-K는 스킴 변경"}
        fanova = {"source": "27b(spec) — spec축(DRAFT/K/W/MTP)은 Gemma4에 transfer 불가(공간 밖)",
                  "importances": {"bmax": "주(27b서도 0.31 — 유일 transfer 후보 축)"},
                  "guidance": "spec축 스킵. Bmax를 메모리 상한까지 스윕 + regime(λ×mix)별 최적 Bmax 확정."}
    plan = {
        "traits": traits,
        "spec_capable": spec,
        "priority_axes": [{"axis": a, "evidence": e} for a, e in axes],
        "search_space": search_space,
        "fanova_prior": fanova,
        "budget": {"screen_trials_per_regime": 20, "screen_nreq": 12, "final_nreq": 60,
                   "fidelity": "multi-fidelity(스크리닝→top2 확정) — Phase2b 검증됨"},
        "constraints": cons,
    }
    return plan


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("anatomy")
    pa.add_argument("--manifest", required=True)
    pa.add_argument("--compare")
    pr = sub.add_parser("roofline")
    pr.add_argument("--device", default=os.path.join(HERE, "profiles", "gb10.json"))
    pr.add_argument("--model", default=os.path.join(HERE, "profiles", "qwen35-27b.json"))
    pr.add_argument("--accept", type=float, default=4.4)
    pr.add_argument("--batch", type=int, default=12)
    pr.add_argument("--validate-27b", action="store_true", help="27b 실측 좌표와 정합 대조(달성률 명시)")
    ps = sub.add_parser("strategy")
    ps.add_argument("--device", default=os.path.join(HERE, "profiles", "gb10.json"))
    ps.add_argument("--model", default=os.path.join(HERE, "profiles", "qwen35-27b.json"))
    ps.add_argument("--constraints", default="{}")
    args = ap.parse_args()

    if args.cmd == "anatomy":
        prof = anatomize(args.manifest)
        print(json.dumps(prof, indent=2, ensure_ascii=False))
        if args.compare:
            manual = json.load(open(args.compare))
            diffs = []
            for k, v in manual.items():
                if k not in prof or k in ("name", "weights_dir", "golden_dir", "uq"):
                    continue
                a = prof.get(k)
                if k == "params_b":  # 형상 기반 추정치 — 정확 일치가 아니라 ±10% 정합으로 판정
                    if abs(a - v) / v > 0.10:
                        diffs.append(f"params_b: auto(추정) {a} vs manual {v} (>10%)")
                elif a != v:
                    diffs.append(f"{k}: auto {a} vs manual {v}")
            print("[compare]", "PASS — 수동본 재현" if not diffs else "차이: " + "; ".join(diffs))
            sys.exit(0 if not diffs else 1)
    elif args.cmd == "roofline":
        dp, mp = json.load(open(args.device)), json.load(open(args.model))
        print(json.dumps(roofline(dp, mp, args.accept, args.batch), indent=2, ensure_ascii=False))
        if args.validate_27b:
            validate_27b(dp, mp)
    elif args.cmd == "strategy":
        dp, mp = json.load(open(args.device)), json.load(open(args.model))
        print(json.dumps(strategy(dp, mp, json.loads(args.constraints)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
