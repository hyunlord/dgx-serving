#!/usr/bin/env python3
"""축퇴(degeneration) 마커 스캐너 — 긴 생성 품질 측정(눈이 아닌 수치).

측정 축(모두 측정 가능·주관 배제):
  rep_ngram  : 반복 축퇴. 최다 반복 word 4-gram 횟수 + 후반 25% 4-gram 고유율(uniqueness).
  lang_drift : 언어 드리프트. (한국어 요청 기준) 앞 25% vs 뒤 25% 한글 비율 급락.
  broken_sym : 깨진 기호/LaTeX 나열. 알파넘·공백·상용문장부호 아닌 문자 긴 run + junk 토큰.
  uni_frag   : 유니코드 파편(U+FFFD �) 개수.

degenerate=True 판정: 아래 임계 중 하나라도 초과(측정 기준·조정 가능).
"""
import re, sys, json, collections

def _words(t):
    return re.findall(r"\S+", t)

def rep_ngram(t, n=4):
    w = _words(t)
    if len(w) < n*3:
        return {"max_rep": 0, "tail_uniq": 1.0}
    grams = [" ".join(w[i:i+n]) for i in range(len(w)-n+1)]
    c = collections.Counter(grams)
    max_rep = max(c.values())
    tail = grams[int(len(grams)*0.6):]           # 후반 40%
    tail_uniq = len(set(tail))/max(1, len(tail))
    return {"max_rep": max_rep, "tail_uniq": round(tail_uniq, 3)}

def hangul_ratio(s):
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    h = sum('가' <= ch <= '힣' for ch in letters)
    return h/len(letters)

def lang_drift(t):
    n = len(t)
    if n < 200:
        return {"head_ko": None, "tail_ko": None, "drop": 0.0}
    head = hangul_ratio(t[:n//4]); tail = hangul_ratio(t[3*n//4:])
    return {"head_ko": round(head,3), "tail_ko": round(tail,3), "drop": round(head-tail,3)}

def run_on(t):
    # 주제이탈 run-on 축퇴: 후반부 최장 '문장'(종결부호로 분할) 길이 + 후반 종결부호 밀도.
    n = len(t)
    tail = t[3*n//4:] if n >= 200 else t
    sents = re.split(r"[.!?。\n·•*]|다\.|요\.|니다", tail)
    max_sent = max((len(s.strip()) for s in sents), default=0)
    # 전체 어휘 novelty(고유 단어/전체) — 너무 낮으면 반복
    w = _words(t); uniq = len(set(w))/max(1, len(w))
    return {"max_sent_tail": max_sent, "uniq_word": round(uniq, 3)}

def broken_sym(t):
    # 상용이 아닌 기호가 12+ 연속(정상 수식은 문자/숫자와 섞임 — 순수 기호 긴 run은 깨짐 신호)
    runs = re.findall(r"[^\w\s.,;:!?()\[\]{}'\"/\-+=*<>|%$\\^_&#@~`]{12,}", t)
    junk = len(re.findall(r"EndBlock|\}\}\}\}|\$\$\$|\\quad\s+\\quad\s+\\quad", t))
    # $ 불균형(수식 구분자 홀수 = 깨짐 가능)
    dollar_imbalance = abs(t.count("$$") % 2)
    return {"sym_runs": len(runs), "junk_tok": junk, "dollar_imb": dollar_imbalance}

def analyze(text, lang="ko"):
    r = rep_ngram(text)
    d = lang_drift(text) if lang == "ko" else {"drop":0.0,"head_ko":None,"tail_ko":None}
    b = broken_sym(text)
    ro = run_on(text)
    u = text.count("�")
    # 임계(측정 기준): 4-gram 3회+반복 | 후반 고유율<0.5 | 한글 요청인데 후반 한글비 급락(>0.4) | 기호run/junk | �
    flags = []
    if r["max_rep"] >= 3: flags.append(f"rep4gram×{r['max_rep']}")
    if r["tail_uniq"] < 0.5: flags.append(f"tail_uniq={r['tail_uniq']}")
    if lang == "ko" and d["drop"] is not None and d["drop"] > 0.40 and d["tail_ko"] is not None and d["tail_ko"] < 0.3:
        flags.append(f"lang_drift {d['head_ko']}→{d['tail_ko']}")
    if b["sym_runs"] >= 1: flags.append(f"sym_run×{b['sym_runs']}")
    if b["junk_tok"] >= 1: flags.append(f"junk×{b['junk_tok']}")
    if u >= 1: flags.append(f"repl_char×{u}")
    if ro["max_sent_tail"] >= 220: flags.append(f"run_on={ro['max_sent_tail']}")
    if ro["uniq_word"] < 0.42: flags.append(f"low_uniq={ro['uniq_word']}")
    return {"degenerate": bool(flags), "flags": flags,
            "rep": r, "drift": d, "sym": b, "run_on": ro, "repl_char": u}

def _extract(path):
    d = json.load(open(path))
    ch = d["choices"][0]
    c = ch.get("message", {}).get("content") or ch.get("text", "")
    return c, d.get("usage", {})

if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            c, usage = _extract(p)
            a = analyze(c)
            tag = "DEGEN" if a["degenerate"] else " ok  "
            print(f"[{tag}] {p.split('/')[-1]:16} tok={usage.get('completion_tokens','?'):>4} "
                  f"flags={a['flags']}  rep={a['rep']} drift_drop={a['drift']['drop']}")
        except Exception as e:
            print(f"[ERR ] {p}: {e}")
