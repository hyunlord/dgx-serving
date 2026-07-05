# Gemma draft-spec 순 곱 measured 판정 — 접음 (구현 전 성패 숫자·q2 패턴)

> Gemma엔 MTP head 없음 → llama식 별도 draft 모델 speculative가 유일 spec 경로. 단 gemma4_verify_batch(130+줄
> 신규·정확성 위험·2모델 구조)를 짓기 **전에** "순 곱이 있나"를 measured — q2 패턴(큰 구현 전 성패 숫자 하나).
> ★결과: **순 곱 ~1.0×(평균)·chat은 <1 손해 → draft-spec 접음.** 130줄 신규 개발 정당화 안 됨(헛수고 차단).

## 측정 (구현 0 — PyTorch forward만)

- **draft = gemma-4-E2B**(same-family·vocab 262144 정합·L35 H1536 FFN6144, nested MatFormer 10GB). target = 12B.
- **accept rate**(E2B greedy next == 12B greedy next, 같은 prefix, 64토큰·워크로드 3종):

| 워크로드 | accept p |
|---|---|
| chat | 0.438 |
| code | 0.609 |
| prose | 0.500 |
| **평균** | **0.516** |

- **draft 비용 c 실측**(single forward, same context): E2B 35.4 ms / 12B 131.3 ms = **c = 0.270**
  (active-bytes 산술 추정 0.139의 **~2배** — E2B nested가 예상보다 무거움 = 동형 함정의 분모 큼).

## 순 곱 (measured p·c) — 손익분기 미달

순곱 = a/(K·c+1), a=기대 accept 길이=(1−p^(K+1))/(1−p):

| 워크로드 (p) | K3 | K4 | K6 |
|---|---|---|---|
| chat (0.44) | **0.95×** | 0.84× | 0.68× |
| code (0.61) | 1.22× | 1.13× | 0.95× |
| prose (0.50) | 1.04× | 0.93× | 0.76× |
| **평균 (0.52)** | ~1.0× | **0.96×** | 0.79× |

## 판정 — 접음 (measured 헛수고 차단)

- **순 곱 < 1.5×(정당화 문턱)** — 오히려 평균 ~1.0×, chat은 **<1(손해)**. gemma4_verify_batch 130줄 신규 개발
  정당화 안 됨 → **draft-spec 접음**(구현 착수 전 판정 = q2 패턴 성공).
- ★**동형 함정 실현**(measured): draft 비용(c=0.27) × 낮은 accept(p=0.52) = 비용이 절약을 먹음. "bytes 절약 ≠ 속도"·
  q2 혼합정밀·sparsity 예측비용·q3 풀기와 **같은 패턴**(분모 줄이는 비용 > 절약). draft도 매번 도니 비용 제한.
- **왜 낮은가**: E2B(2B eff)가 12B greedy를 절반만 맞힘(작은 모델 한계·같은 family라도)·nested라 forward 무거움
  (c 2배). llama 66.9는 다른 draft(더 작고 target-정합 높은 전용 draft·distilled) 추정 — 우리 E2B는 부적합.

## 남은 여지 (정직 — 전부 문턱 미달)
- 더 작은 draft(270M급): accept↓·비용↓ — same-family 최소가 E2B라 없음·더 작으면 accept 더 낮아 순 곱 여전히 낮음.
- E2B를 우리 엔진 W4A8로(c 절반 0.135): 순 곱 K3 chat 1.13·code 1.45·평균 ~1.2× — 여전히 1.5× 미달.
- code 전용(p=0.61 K3=1.22×): 특정 워크로드 약간 이득이나 범용 <1.5×.

## 결론
draft-spec은 12B에서 measured로 순 곱 부족(~1.0×) → **접음**. 12B plain near-parity(llama plain 0.87×·실사용 가능)로
충분. llama 66.9(spec)는 우리가 못 따라가는 좌표로 정직 유지(다른 draft 자산 필요 — 엔진 밖). 다음 lever = FP4(대기)·
서빙 개선·push. ★q2 패턴 성공: 130줄 신규 개발 전 measured가 헛수고 차단(구현 0으로 판정).
