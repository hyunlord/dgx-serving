# q2 measured-first — naive q2 forward 품질 숫자 (설계 0)

> 지난 세션 프레임워크 설계 이탈(YAGNI 위반·measured 0) 교정: **설계 금지, 최소 스크립트로 품질 숫자부터.**
> 지난 study는 weight 재구성 오차(프록시 47%)였음 — 이번은 **실 출력 품질**(forward → perplexity/argmax/생성).
> ★결론: **naive q2 = 완전 붕괴(measured, perplexity 464× 폭등·argmax 10%·생성 gibberish). 정교 기법 필수 or q3가 현실적 바닥.**

## STEP 0 — naive q2 품질 숫자 (PyTorch 시뮬 forward, Qwen3.5-0.8B, group=32)

설계/프레임워크 0. `tools/q2_quality.py` 최소 스크립트: weight를 스킴별 양자화(dequant to fp)→forward. Linear 96개
(attn/mlp proj·embed/norm/lm_head 제외). 0.8b(안전·fits — 27b PyTorch는 54GB 메모리 크래시 위험).

| scheme | bits/w | perplexity | vs fp | argmax 일치(vs fp) | 생성(greedy 20tok) |
|---|---|---|---|---|---|
| fp32 | 16 | 4.95 | — | — | "Paris. The capital of France is Paris..." ✓ |
| q4-g32 | 5 | 5.24 | **1.06×** | **94.8%** | "Paris..." ✓ (사실상 동일·작동) |
| q3-g32 | 4 | 8.61 | 1.74× | 74.0% | "the capital of the country..." (열화되나 coherent) |
| **naive q2-g32** | 3 | **2297.7** | **464×** | **10.4%** | **"a 20000000n00000\nC. with" = 완전 gibberish** |

## STEP A — 숫자가 정한 방향 (measured)

★**naive q2 붕괴 severe**(464× perplexity·argmax 10%·gibberish) — 지난 weight 오차 프록시(47%=5×q4)가 옳았고,
출력 붕괴는 더 극적. **순수 naive 2-bit = 완전 사용 불가**(모델이 랜덤 근사됨). SOTA 2-bit(QuIP#/AQLM류)이
왜 무거운 기법을 쓰는지 measured 확인.

- **q4는 작동**(1.06× ppl·94.8% argmax·생성 동일) = 실용 기준선.
- **q3는 coherent**(1.74× ppl·74% argmax·문장 유지) = **현실적 sub-q4 바닥**(4bit vs q4 5bit=0.8× bytes·decode ~1.25×·품질 중간).
- **naive q2 = 불가** → q2 실용화는 **정교 기법 필수(선택 아님)**: importance-weighted(GPTQ/AWQ)·mixed precision(민감 층 q4)·
  fine group. = 연구 프로젝트(다음 세션·measured-first).

### 판정 + 다음
- **압살(decode 2×) via q2**: naive 불가 확인 → 정교 q2 실용화가 선결(unproven). q3는 modest byte↓(2× 아님).
- **다음 방향(숫자가 정함)**: (a)정교 q2 연구(importance+mixed — naive 464×를 q4급으로 회복 시도, 무거움) or
  (b)q3 실용화(온건·coherent·바로 가능) or (c)대역폭 돌파(2번째 기둥).
- **모델 caveat**: 0.8b(작음=q2에 더 fragile). 27b는 덜 극단일 수 있으나 naive q2는 여전 붕괴 예상(464×라 10× 견뎌도 46×=broken).
- overclaim 없음: naive q2 붕괴가 measured. "설계 0·숫자 먼저"로 방향 확정(정교 필수 or q3).

## 산출물
- `tools/q2_quality.py`(최소 forward 품질 스크립트)·docs/q2-quality-measured.md.
