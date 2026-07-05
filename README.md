# dgx-serving

**NVIDIA DGX Spark GB10 (sm_121) 전용** 오픈소스 LLM 추론 엔진.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

> ⚠️ **상표 고지**: "DGX", "DGX Spark", "GB10", "Grace Blackwell"은 NVIDIA Corporation의 상표입니다.
> 본 프로젝트는 **독립 오픈소스 프로젝트로 NVIDIA와 무관**하며, 대상 하드웨어를 식별하기 위해서만 명칭을 사용합니다.

> **포지셔닝(정직)**: 이 프로젝트는 "GB10에서 LLM 추론에 무엇이 가능한지의 **측정 지도**"이자 그 위에 세운 서빙 엔진이다.
> 범용 최속 엔진을 표방하지 않는다. **이기는 좌표도 지는 좌표도 measured 그대로** 공개한다 — 신뢰가 곧 제품.

## 무엇인가

- **단일 장비(GB10) × 특정 모델군**에 최적화된 서빙 엔진. GB10의 ~273 GB/s unified LPDDR5X 대역폭과
  128GB unified 메모리를 전제로 설계.
- **모델 독립 공용 커널 라이브러리(`kernels/`) + 모델별 얇은 graph(`models/`)** 구조. 4개 모델
  (Qwen3.5-27B·Gemma4-31B·Qwen3.5-35B-A3B MoE·26B-A4B MoE)이 신규 커널 0으로 서빙됨(아키텍처 실증 → [확장 가이드](./docs/m6-backend-boundary.md)).
- **자동튜닝(베이지안, Optuna TPE)** + **레시피 재현**: 장비×모델별 최적 config를 실측+정확성 게이트(ISO)로
  자동 탐색하고, 검증 가능한 주장이 붙은 `recipe.yaml`로 배포/재현. (vLLM 수동 가이드·llama.cpp 휴리스틱과 차별)
- **MTP-head 내장 speculative decoding** + tree-spec + 연속 배칭 결합(27b).
- **128GB unified로 큰 모델**(27~35B급)을 단일 장비에서 서빙.

## 무엇이 아닌가 (정직)

- **범용 최속 엔진이 아니다.** llama.cpp/vLLM의 성숙한 커널을 raw 속도로 이기는 것이 목표가 아니다(아래 §정직 벤치).
- **모든 GPU용이 아니다.** GB10(sm_121) 전용. probe가 장비를 자동 감지하나 다른 장비에선 튜닝 자산이 transfer만 됨.
- **품질↔속도 트레이드 옵션이 없다.** q4 단일(measured 결론 — q2 붕괴·q3 속도역행 소거, §측정 지형도).

## ★정직 벤치 (measured — 지는 좌표 포함이 원칙)

### 2-way: dgx-serving vs llama.cpp (Gemma4-26B-A4B, 공정 조건, HTTP, warm, ~q4급)

같은 26b MoE·같은 프롬프트/max_tokens·같은 동시성·warm 클럭. HTTP 클라이언트 관측(배포 현실).

| 동시 N | dgx t/s | llama t/s | dgx/llama |
|---|---|---|---|
| 1 | 14.7 | 52.6 | **0.28×** (3.6× 느림) |
| 4 | 48.4 | 65.3 | 0.74× |
| 8 (=Bmax) | 80.6 | 168.3 | **0.48×** (2.1× 느림) |
| 16·32 (초과구독) | ~81 포화 | ~145~163 | 0.50~0.56× |

- **dgx가 진다(measured).** 이기는 raw-속도 좌표는 없다. 숨기지 않는다.
- **왜 지는가(정직)**: 26b는 최신·미최적 MoE(decode graph를 갓 구현·perf 튜닝 0), llama는 수년 성숙한 fused/tiled 커널.
  = **커널 성숙도 갭**(엔진 천장 아님 — 27b dense는 roofline 62.5% 달성). 남은 lever는 per-token 커널 속도(FP4·fused·tiled — 착수 전).
- **배칭은 작동한다(긍정 신호)**: dgx 연속 배칭이 N=1→8에서 near-linear 5.5×(14.7→80.6)로 amortize·초과구독을 큐잉으로 흡수(크래시 없음).
- vLLM 3rd leg = 후속(GB10 구동 자체는 실증됨 — 26b 체크포인트+docker HTTP만 남음, 정직히 범위 밖).

### roofline 달성률 (상한 대비 — 상한이지 예측 아님)

| 모드 | 상한(t/s) | 실측 | 달성률 |
|---|---|---|---|
| 27b single non-spec | 19.0 | 11.9 | 62.5% |
| 27b spec serving (B16 K7W2) | 304.5 | 51.7 | 17.0% |
| gemma4-31b single | 17.4 | 9.28 | 53.3% |
| 35b-a3b MoE single | 165.3 | 51.8 | 29.8% (MoE 구조적 낮음) |

### spec 곱 (27b — 워크로드 조건 명시)

- chain K=3 = **1.40×**(plain 대비)·tree K4W3 = 1.11×·plain 배칭 = 1.21×(같은 Bmax).
- ★**워크로드 민감**: accept 6.99(반복)~3.5(산문) measured — 반복 워크로드에서 지연 크게 유리·산문은 완만.

### 측정 지형도 = 그 자체가 자산 ("이 장비에서 가능한 것의 지도")

대역폭 돌파(2×급) 후보 시드를 **전부 측정으로 판정**(소거도 성과 — 환상 제거):
L2 persist·KV 압축·dense sparsity·spec 극한·prefetch·unified 계층·q2/q3 양자화 = **전부 소거/modest**.
**결론: 273GB/s × 토큰당 bytes가 근본 벽.** 4bit이 이 하드웨어 bytes↓ sweet spot(그 아래는 로드 마찰이 절약 초과).
남은 큰 lever는 전부 엔진 밖(draft 모델·SOTA 2-bit) 또는 toolkit-gated(FP4, CUDA 13.2+ 대기).
→ [`docs/c-bandwidth-seeds-verdict.md`](./docs/c-bandwidth-seeds-verdict.md).

## 방법론 (측정-먼저 문화 — 차별화 포인트)

- **측정 먼저**: 시간만으로 성능 판단 금지 — ncu 지표(sectors/req·occupancy) 동반. "명백한" 오버헤드도 measured 비-지배인 경우 다수.
- **격리 ≠ 엔진**: 격리 벤치 이득이 in-engine에서 감쇠/역전(X-shared·q3 함정 실증). in-engine 검증 필수.
- **scheme-internal 정확성**: bit-identity는 스킴 내 개념(배치 형상·양자화 dispatch가 스킴). fp32-exact가 아님을 명시.
- **정직 좌표**: "추월/delivered"는 measured로 실제 llama 추월(<1.0×) 확인 전까지 금지. overclaim 0.

## 지원 모델 (4모델 — 커널 라이브러리 + 얇은 graph 실증)

| 모델 | 구조 | 서빙 |
|---|---|---|
| Qwen3.5-27B | GDN 혼합 + MTP(spec) | spec 곱 + 연속 배칭 |
| Gemma4-31B | dense(sliding/full·GeGLU·dual-rope·softcap) | plain 연속 배칭 |
| Qwen3.5-35B-A3B | MoE(256 expert top-8) + GDN | fused-MoE plain 배칭 |
| 26B-A4B | 병렬 MoE(dense∥MoE) | plain 배칭 |

## 문서

- [설계 철학·경계·확장 가이드](./docs/m6-backend-boundary.md) — 커널 독립·graph 배선·config-주도, 새 모델 family 추가법.
- [측정 지형도](./docs/c-bandwidth-seeds-verdict.md) — GB10 대역폭 돌파 시드 전수 판정("가능한 것의 지도").
- [다목적 파레토](./docs/m3-multi-objective-pareto.md) — throughput/tpot/메모리/워크로드 축·SLO 제약 튜닝.
- [레시피 스키마](./docs/recipe-schema.md) — 검증 가능한 config 배포판(v0).
- [공개 문서 구조](./docs/m7-public-doc-structure.md) · [정립 노트](./docs/establishment-notes.md) — 설계 철학·정확성 계약·알려진 갭.
- `docs/`의 `ch*`/`02*`/`m4*`/`m5*` 등은 개발 로그(worklog, 정제 전 — 측정 과정의 정직한 기록).

## 알려진 갭 (숨기지 않고 로드맵으로)

- spec 서빙 경계 미정(plain backend 훅 밖 — 별도 추상화 필요) · 26B 서빙 파레토 미측정 ·
  gemma4 sliding KV 링버퍼 TODO(긴 context) · verify 자동화 갭(export/warm-start scheme-class 추상화) ·
  FP4 텐서코어 toolkit-blocked(CUDA 13.2+ 대기).

## 빌드 / 재현

```bash
make            # 전체 빌드 (nvcc -arch=sm_120 — GB10 sm_121 호환). ARCH/NVCC 오버라이드 가능.
make test       # 검증 테스트(레퍼런스 대조 golden)
```

- **전제**: GB10(sm_121) + CUDA 13.0 + nvcc. 다른 장비에선 커널이 빌드될 수 있으나 튜닝 자산은 transfer만 됨.
- 레퍼런스 골든 생성엔 Python(transformers/safetensors/torch) 필요 — HF 캐시는 `HF_HOME`/`HF_HUB_CACHE` 환경변수 존중.
- **가중치는 배포하지 않는다**(모델 라이선스). HuggingFace에서 받아 `tools/extract_weights*.py`로 엔진 포맷 변환.
  각 모델(Qwen/Gemma 등)의 라이선스 준수는 사용자 책임.

## 라이선스

[Apache License 2.0](./LICENSE). 모델 가중치는 포함하지 않으며, 각 모델의 라이선스는 별도로 적용된다.
