# q3 STEP 0+A — packing·gemv 커널·골든 완결 + ★속도 예비 경고 (measured)

> 지형도(153318a)의 마지막 실 lever q3(decode 이론 +25%) 구현 착수. 이번 세션 = 코어 관문(STEP 0+A):
> packing 설계·CPU 레퍼런스·gemv_q3 커널·scheme-internal 골든. ★결과: **정확성 완전 성공(BIT-ID)** but
> **속도 예비 경고 — 격리 gemv에서 q3가 q4보다 1.2× 느림**(bytes 0.8×인데 로드 비용이 절약 초과). 정직 보고.

## STEP 0 — packing 설계 (bit-plane 분해)

- **포맷**: 그룹(QBLK=32값) = **low 2bit plane 8B(u32×2) + hi 1bit plane 4B(u32×1) = 12B/그룹 + scale 4B = 16B**
  vs q4 20B → **bytes 0.80×** ✓. 풀기 = 시프트 2회+OR(값싼 형태): `q = (lo>>(2j))&3 | ((hi>>j)&1)<<2`.
- 스킴: 8레벨 대칭 `q=clip(round(w/s+3.5),0,7)·s=absmax/3.5·deq=(q-3.5)·s` — **tools/q2_quality.py와 동일**
  (0.8b 품질 measured ppl 1.74×·coherent의 연속성).
- 레이아웃 `[OUT][ngrp][lo0,lo1,hi]`·CPU pack+dequant 레퍼런스(test 내) = scheme-internal 골든의 기준.

## STEP A — gemv_q3 + 골든 (kernels/gemv_q3.cuh)

- 커널 구조 = gemv_q4/gemv_f32와 동일(warp당 1출력·lane 32-stride·warp_reduce) → **누적 순서가 gemv_f32와 동일**
  → CPU-dequant W로 gemv_f32 돌린 결과와 **BIT-IDENTICAL 골든**(rel 게이트보다 강함).
- ★게이트 PASS(test_gemv_q3): **3 dims(비배수 300×256·S=3·27b MLP-up 17408×5120) 전부 bit-diff=0** + 결정성 30회.
  "빠른데 틀림" 차단 — packing/커널 정확성 확정.

## ★속도 예비 (격리, 27b MLP-up S=1, warm) — 정직 경고

| | 시간 | bytes |
|---|---|---|
| gemv_q4 | 305~339us | 20B/그룹 |
| **gemv_q3** | **360~407us (q4의 1.18~1.24×)** | 16B/그룹(0.80×) |

- ★**bytes 0.8×인데 시간 +20%** — 풀기/로드 비용이 절약을 초과(예측비용 함정 동형, 사전 경고했던 바로 그 함정).
- **원인**: q4=그룹당 로드 1개(u8·warp 16B 연속 1트랜잭션). q3 bit-plane=그룹당 **로드 2개**(lo 발산 2주소+hi
  broadcast) = 로드 수 2×·주소 발산. **3bit의 비바이트 정렬이 근본**(어떤 packing이든 값 정보가 byte 경계에 안
  맞음 → 추가 로드/시프트). unroll 4 시도 = 악화(1.24× — 레지스터 압력, MLP_U 스윕과 동일 패턴).
- caveat: fp32-act gemv 격리(q4·q3 동조건=공정). 27b 실 decode는 W4A8(int8 DP4A) — q3 통합은 W3A8 필요하나
  로드 구조 페널티는 동일하게 걸릴 것으로 예상(격리 신호가 이미 음수).

## 판정 + 다음 (stop-report)

- **정확성 자산 확보**: bit-plane packing + gemv_q3 + BIT-ID 골든(커밋 대상) — q3 어떤 후속에도 기반.
- ★**"decode +25%"는 naive 커널로는 역행(-20%)이 measured** — bytes 절약이 자동으로 속도가 아님(이 프로젝트에서
  세 번째 확인: q2 혼합정밀·sparsity 예측비용·이번 q3 풀기 비용).
- **다음 판단 필요(보고)**: (a)로드 재구성 시도(lo/hi 스트림 분리·벡터 로드 — 기대치 낮음, 3bit 비정렬 근본)
  (b)W3A8(int8 DP4A) 통합 후 재판정 (c)q3 소거(격리 신호 존중 — bytes 0.8×로는 로드 마찰 못 이김) →
  27b 20GB 변환·품질·서빙(STEP B~D)은 속도 신호 해소 전 보류(rush 금지).

## 산출물
- kernels/gemv_q3.cuh(bit-plane 커널)·tests/test_gemv_q3.cu(BIT-ID 골든+격리 타이밍).
