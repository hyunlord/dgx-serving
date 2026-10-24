#!/usr/bin/env python3
"""긴 생성 축퇴(degeneration) 게이트 — 실사용 품질 상시 회귀 방지.

배경: rope 문자-붕괴(faa753b)와 별개로, 샘플링 긴 생성서 반복/드리프트 축퇴가 있었음.
  근본 원인 = 반복 페널티가 생성 이력 전체를 복리 누적(repp^N) → 긴 이력서 흔한 토큰 과잉
  억제 → 언어 드리프트/반복 루프. 수정 = serving_online.h 반복 페널티를 최근 윈도우+dedup화
  (REP_WINDOW env·기본 128). 측정: broad grid(3워크로드×rep×seed) rep=1.1 축퇴 3/6 → 0/6.

이 게이트(rope 절대 골든 게이트와 동일 철학):
  (1) 상시: 스캐너 감도 자가증명 — clean은 통과·degenerate 샘플은 반드시 flag(스캐너가 눈 대신).
  (2) 옵션(DEGEN_SERVE_URL 지정 시): 라이브 서버서 800토큰 생성·워크로드별 축퇴율 임계 이하 검증.
      서버 없으면 우아하게 skip(신선 클론·CI 안전) — rope 게이트 골든-자산 skip과 동형.
"""
import os, sys, json, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import degen_scan

# --- (1) 상시: 스캐너 감도 자가증명 (서버 불필요) ---
CLEAN = ("로지스틱 회귀는 분류 문제의 기본 모델입니다. 시그모이드 함수로 확률을 출력하고, "
         "교차 엔트로피 손실을 경사하강법으로 최소화합니다. 규제를 더하면 과적합을 막을 수 있습니다. "
         "다중 클래스는 소프트맥스로 확장합니다. 실무에서는 특성 스케일링과 클래스 불균형에 주의합니다.")
# degenerate 3종(각 마커 축을 자극): 반복 루프 / 언어 드리프트 / 기호·이모지 나열
DEGEN_REP = "어텐션 가중치를 계산합니다. " + "각각은 스스로 위치 정보를 가지며 연결됩니다. " * 5
DEGEN_DRIFT = ("셀프 어텐션은 문장 내 단어 관계를 계산합니다. 쿼리와 키를 비교합니다. "
               + "The core concept is to calculate the attention weights based on similarity. " * 4)
DEGEN_SYM = "대화 예시입니다. " + "🎉🎈🎁🎊💐🏵🌸🌺🌻🌼🍭🍬🍪🍩🍰🧁🍦🍨🍧🍹🥤🍼🍶🍵" * 3

def test_scanner_sensitivity():
    ok = degen_scan.analyze(CLEAN)
    assert not ok["degenerate"], f"clean이 오검출됨: {ok['flags']}"
    for name, txt in [("반복", DEGEN_REP), ("드리프트", DEGEN_DRIFT), ("기호나열", DEGEN_SYM)]:
        a = degen_scan.analyze(txt)
        assert a["degenerate"], f"degenerate({name}) 미검출 — 스캐너 감도 회귀: {a}"
    print("  [감도] clean 통과 · degenerate 3종(반복/드리프트/기호) 모두 flag — 스캐너 유효 PASS")

# --- (2) 옵션: 라이브 서버 800토큰 축퇴율 ---
WORKLOADS = {
    "attn":   "트랜스포머 셀프 어텐션 메커니즘을 예시와 함께 아주 자세히 길게 설명해줘.",
    "code":   "파이썬으로 이진 탐색 트리를 구현하고 삽입·삭제·순회를 주석과 함께 작성해줘.",
    "dialog": "번아웃이 온 친구를 공감하며 위로하고 실천 가능한 조언을 대화체로 길게 해줘.",
}
def _gen(url, prompt, temp, repp, seed, max_tokens=800):
    body = {"model":"gemma4-direct","messages":[{"role":"user","content":prompt}],
            "max_tokens":max_tokens,"temperature":temp,"top_p":0.9,
            "repetition_penalty":repp,"seed":seed}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    ch = d["choices"][0]
    return ch.get("message",{}).get("content") or ch.get("text","")

def test_live_longgen():
    url = os.environ.get("DEGEN_SERVE_URL")
    if not url:
        print("  [라이브] DEGEN_SERVE_URL 미지정 — skip(서버 없이 안전). 예: "
              "DEGEN_SERVE_URL=http://localhost:8081/v1/chat/completions")
        return
    temp = float(os.environ.get("DEGEN_TEMP", "0.7"))
    repp = float(os.environ.get("DEGEN_REP", "1.1"))   # 기본 프리셋(윈도우+dedup서 0/6)
    thresh = int(os.environ.get("DEGEN_MAXFAIL", "0")) # 허용 축퇴 건수(기본 0)
    fails = []
    for wl, prompt in WORKLOADS.items():
        for seed in (42, 7):
            c = _gen(url, prompt, temp, repp, seed)
            a = degen_scan.analyze(c)
            tag = "DEGEN" if a["degenerate"] else "ok"
            print(f"  [라이브] {wl}/seed{seed}: {tag} {a['flags']}")
            if a["degenerate"]:
                fails.append((wl, seed, a["flags"]))
    assert len(fails) <= thresh, (
        f"긴 생성 축퇴 {len(fails)}건 > 허용 {thresh} (temp={temp} rep={repp}): {fails}")
    print(f"  [라이브] 축퇴 {len(fails)}/{len(WORKLOADS)*2} ≤ {thresh} PASS")

if __name__ == "__main__":
    test_scanner_sensitivity()
    test_live_longgen()
    print("[test_longgen_degen] PASS")
