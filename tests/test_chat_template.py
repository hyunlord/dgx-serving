#!/usr/bin/env python3
"""템플릿 게이트 — /v1/chat/completions의 정확성 지점 검증.

★위험: messages를 잘못된 템플릿으로 감싸거나(모델 불일치) 손수 concat하면 모델이 이상하게 답한다(조용한
품질 저하 — 속도/버그가 아니라 "품질 오인"). 이 게이트는 serve_api.chat_ids()가:
  (1) HF tokenizer.apply_chat_template와 ID 완전 일치(단일 소스 — 손수 concat 회귀 차단),
  (2) 실제로 템플릿을 적용(naive concat과 다름 — 템플릿 미적용 차단),
  (3) multi-turn(히스토리 포함) 동작,
  (4) chat_template 없는 토크나이저는 명시 거부,
임을 오프라인으로 검증한다. daemon/계산 경로 무관(front 토큰화 계약만).
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from serve_api import chat_ids  # noqa: E402

# 캐시된 instruct 토크나이저(chat_template 보유) 중 로드되는 첫 것으로 검증.
CANDIDATES = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
              "Qwen/Qwen2.5-0.5B-Instruct", "google/gemma-4-E4B"]

def load_tok():
    from transformers import AutoTokenizer
    for name in CANDIDATES:
        try:
            t = AutoTokenizer.from_pretrained(name)
            if getattr(t, "chat_template", None): return name, t
        except Exception:
            continue
    return None, None

def main():
    name, tok = load_tok()
    if tok is None:
        print("[test_chat_template] SKIP — chat_template 보유 토크나이저 캐시 없음(오프라인). 게이트 미실행.")
        return 0
    print(f"[test_chat_template] tokenizer = {name}")

    fails = []
    # (1) 계약: chat_ids == 독립 참조(render→tokenize). ★tokenize=True의 반환형(list vs BatchEncoding dict)
    #     quirk를 우회하려 참조는 "템플릿 문자열 렌더 후 토큰화"로 독립 산출 — 양쪽이 같은 quirk를 공유해
    #     거짓 PASS 나던 버그(dict 키 2개 == 2개)를 차단.
    cases = [
        [{"role":"user","content":"The capital of France is"}],
        [{"role":"system","content":"You are terse."},{"role":"user","content":"2+2?"}],
        # (3) multi-turn(히스토리 포함)
        [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Name a color."}],
    ]
    for i, msgs in enumerate(cases):
        ours = chat_ids(tok, msgs)
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)   # 문자열(사람이 검증 가능)
        ref = tok(rendered, add_special_tokens=False).input_ids                                # 독립 토큰화 경로
        ok = (ours == list(ref))
        # 절대 sanity: 실제 내용이 있으면 토큰이 여러 개여야(degenerate 2-token 붕괴 차단)
        sane = len(ours) >= 5
        print(f"  case{i}(turns={len(msgs)}): our==render→tok {'PASS' if ok else 'FAIL'} · len={len(ours)} {'(sane)' if sane else '(★DEGENERATE)'}")
        if not ok:   fails.append(f"case{i}: {ours[:8]} != {list(ref)[:8]}")
        if not sane: fails.append(f"case{i}: degenerate len={len(ours)} (템플릿 미토큰화 의심)")

    # (2) 템플릿이 실제로 적용됨: wrapping 토큰 추가 → naive concat보다 김(미적용/붕괴 차단)
    msgs = cases[0]
    naive = tok(" ".join(m["content"] for m in msgs), add_special_tokens=False).input_ids
    templ = chat_ids(tok, msgs)
    applied = len(templ) > len(naive)
    print(f"  template-applied(len > naive): {'PASS' if applied else 'FAIL'} (templ {len(templ)} vs naive {len(naive)})")
    if not applied: fails.append(f"template not applied/degenerate (templ {len(templ)} <= naive {len(naive)})")

    # (4) chat_template 없으면 거부
    class _NoTmpl:
        chat_template = None; name_or_path = "dummy-base"
    try:
        chat_ids(_NoTmpl(), cases[0]); fails.append("no-chat_template: 거부 안 함(ValueError 기대)")
        print("  no-template rejection: FAIL (예외 안 남)")
    except ValueError:
        print("  no-template rejection: PASS (ValueError)")

    if fails:
        print("[test_chat_template] FAIL:"); [print("   -", f) for f in fails]; return 1
    print("[test_chat_template] PASS — 템플릿 게이트 통과(ID 일치·적용 확인·multi-turn·거부).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
