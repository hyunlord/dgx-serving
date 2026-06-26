# dgx-serving

NVIDIA DGX Spark **GB10 (sm_121)** 전용 오픈소스 LLM 추론 엔진.

범용 엔진이 아니라 **단일 장비(GB10) × 특정 모델군**에 최적화된 서빙 엔진을 지향한다.
구조는 **모델 독립 공용 커널 라이브러리(`kernels/`) + 모델별 얇은 graph(`models/`)** 이다.

자세한 설계 원칙·검증 규칙은 [`CLAUDE.md`](./CLAUDE.md) 참고.

## 레이아웃

```
kernels/   모델 독립 공용 커널 (.cuh) — rmsnorm, gemv_q4, rope, attention, swiglu, elementwise, reduce
runtime/   ModelConfig, weight 로딩, KV 캐시 (config 기반 범용)
models/    모델별 graph (.cu) — 공용 커널을 배선해 forward 구성 (예: smollm3.cu)
tools/     HF → 엔진 포맷 추출(extract_weights.py), fp32 골든 덤프(make_reference.py)
tests/     graph/커널 출력 vs 레퍼런스 단계별 수치 검증
```

## 빌드

```bash
make            # 전체 빌드 (nvcc -arch=sm_120 -O2)
make test       # 검증 테스트 빌드+실행
make clean
```

## 환경

- CUDA 13.0, `nvcc -arch=sm_120` (GB10 sm_121 호환)
- Python 레퍼런스: `~/github/gb10-cuda-ht/07-engine/.venv` (torch 2.12.1 / transformers 5.12.1 / safetensors 0.8.0)

## 상태

1단계 진행 중: 공용 커널 라이브러리 + SmolLM3 graph, HF fp32 출력과 오차 < 1e-2 일치 목표.
