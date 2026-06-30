# 챕터9 accept lever — n-gram drafting de-risk → NO-GO (MTP보다 약함)

> base(gemv-MLP 75% 포화)·bytes(2bit NO-GO) 막힘 → accept만 열린 lever. n-gram drafting(학습0)으로 accept↑ 시도.
> de-risk(통계, 통합 전): 우리 27B greedy 생성서 n-gram accept 측정. **결과: 모든 콘텐츠서 MTP(2.25~3.0)보다 약함 → NO-GO.**

## 측정 (`tools/gen_dump.cu` 생성 + `tools/probe_ngram.py` 시뮬, 27B fp16s greedy)
spec cycle 시뮬: confirmed t → n-gram(seq[0..t] suffix 최장매칭 → 후속 K=7) → 실제와 consecutive accept. avg=(Σ1+a)/cyc.

| 콘텐츠 | n-gram accept (최선) | hit율 | MTP 비교 |
|--------|---------------------|-------|----------|
| CODE (N=128) | 1.94 (n[1-4], gen) | 30% | **< MTP 3.0** |
| CODE (N=384, 완전 반복 getter/setter) | **1.95** | 34% | **< MTP 3.0** |
| FACT | 1.02 | 2% | < MTP ~1.6 |
| PROSE | 1.03 | 3% | < MTP ~1.8 |

## 왜 약한가 (반복 코드인데도)
생성 텍스트 확인: code는 `def set_X(self, X):\n self._X = X` **구조 반복**이나 **식별자 X(phone/city/country...)가 매 블록
새것** → n-gram이 구조 토큰은 맞추나 **새 식별자 첫 등장서 매번 끊김**(token-identical 반복 필요, 구조 반복 ≠ token 반복).
fact/prose는 비반복 → n-gram ~1.0(거의 0). 메모리 "n-gram 창작 0 accept" 확인 + **code조차 우리 from-scratch 생성선 약함.**
**MTP head(학습됨)는 식별자까지 모델 분포로 예측 → n-gram보다 강(3.0 vs 1.95).**

## 판정 — NO-GO (n-gram, 우리 생성)
- **n-gram 단독 < MTP 모든 콘텐츠** (code 1.95<3.0, fact/prose ~1.0). primary draft 부적합.
- **MTP+n-gram 병행**: n-gram이 잡는 구조 토큰(hit 34% code)은 MTP가 이미 대부분 잡음(강함) → 증분 작음(추정 code 3.0→~3.5,
  fact/prose 0). dual-draft 통합 복잡도 대비 marginal → 통합 안 함. (병행은 추정; n-gram 단독이 strict 하회라 상한 bounded.)
- **90 t/s code 항 ≠ n-gram 도달**: DFlash 90의 accept(4-5)는 (a) 별도 **학습된 draft 모델**(MTP보다 강, 모델 영역) +
  (b) 가능성: **repository-completion 워크로드**(기존 파일 편집 = 줄 token-identical 반복 → 거기선 n-gram 강함, 우리 from-scratch와 다른 조건).

## 정직 좌표 — accept lever는 모델 영역
세 항 측정 종결: ① bw효율 gemv-MLP 75% 포화 ② bytes 2bit 품질 NO-GO ③ **accept = 단일 MTP 2.25~3.0이 우리 천장**
(full-forking NO-GO[STEP6] + n-gram NO-GO[이번]). **accept↑(90의 항)는 더 강한 draft = 별도 draft 모델/MTP 재학습 =
모델·학습 영역**(커널/알고리즘 아님). 우리 커널·무학습 lever는 소진 — decode 21.3 t/s가 현 자산의 정직 천장.
(90 code는 별도 draft 모델 × repository-completion 조건일 개연 — 우리 from-scratch 단일-MTP와 다른 좌표.)

de-risk가 약한 n-gram 통합(MTP보다 나쁨)을 통계로 차단 — STEP6·2bit·FP4와 동형(측정-우선).
