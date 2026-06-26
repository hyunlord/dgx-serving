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

**1단계 완료 (PASS).** 공용 커널 라이브러리 + SmolLM3 graph. [`docs/01-stage1-blueprint.md`](./docs/01-stage1-blueprint.md).

**2a 완료 (PASS).** Qwen3.5-0.8B 본체 (full attention 6 + GDN/linear attention 18, partial RoPE,
q/k_norm, output gate, gated delta net). fp32 경로는 HF-fp32와 일치(next-token=11751), q4 경로는
golden_q4와 단계별 <1e-4. pre-norm hidden(MTP 입력)도 검증. [`docs/02a-qwen35-blueprint.md`](./docs/02a-qwen35-blueprint.md).
**2b 완료 (PASS).** MTP(Multi-Token Prediction) head — 과거 Python 4회 0% accept를 돌파.
torch 레퍼런스(`tools/mtp_reference.py`) 78.6% accept, CUDA 구현(`tests/test_mtp.cu`)이 부품별 <1e-4로
재현하고 동일 78.6% accept. 핵심: h=post-norm(h_nextn) 확정, prefix attend, partial rope, e=token[P+1].
[`docs/02b-mtp-blueprint.md`](./docs/02b-mtp-blueprint.md).

**2c 완료 (PASS).** Speculative 생성 루프 (chain, greedy, 단일 시퀀스). KV 캐시 3종(full-attn /
MTP 자체 / GDN recurrent state) + incremental decode + draft/verify/accept + GDN 임시승격. 4단계
검증 전부 오차 0 (incremental==prefill, verify==decode, promote==decode, spec on==off greedy 완전 일치).
fp32 기준선에서 **1.17× t/s 가속** (K=4, avg accept 3.76/4=94%) — 대역폭 돌파 첫 완결 실증.
[`docs/02c-speculative-blueprint.md`](./docs/02c-speculative-blueprint.md).

**2d 완료 (성능 진단).** ncu/nsys 기반 gemv 최적화. 클럭 해제로 baseline 1.227×. fp32 gemv-internal(MO+float4)
헤드룸 소진(+2%, occupancy 한계 — X-shared 함정을 이번엔 ncu로 진단). 핵심 진단: gemv 2종(작은 decode=latency-bound→
배칭이 레버 / 큰 weight=read 지배→q4가 레버). 정확성 100% 불변. 다음 레버(q4 / 배칭)는 별도.
[`docs/02d-perf-blueprint.md`](./docs/02d-perf-blueprint.md).

**3a 완료 (대역폭 돌파 챕터1 — q4×spec 지도).** 본체 q4(2.63× plain) + draft lm_head q4(tied 재사용) →
both-q4 spec **191 t/s = fp32 plain의 2.9×**, q4-spec net-POSITIVE 회복. accept 대가 0(lm_head q4가 argmax 견딤),
구현 오차0·mismatch0 유지. 토큰당 바이트 7.8×↓, fp32-환산 유효대역 575 GB/s(물리 273의 2.1×). 단 실 BW 27%
= naive q4 커널이 천장. **llama.cpp 공정 비교**(같은 모델/세션, GDN 런타임 지원 확인): q4 plain 176 vs 266,
q4 spec 191 vs 314 — **1.5~1.6× 뒤짐, 격차는 전적으로 q4 gemv 커널 성숙도**(fp32 경로는 88% 대등).
다음 타깃 = q4 gemv 커널 최적화. [`docs/03a-q4spec-blueprint.md`](./docs/03a-q4spec-blueprint.md).

검증 재현:
```bash
# (레퍼런스 venv 사용) 가중치/골든 생성 — 1회
PY=~/github/gb10-cuda-ht/07-engine/.venv/bin/python
$PY tools/extract_weights.py --model HuggingFaceTB/SmolLM3-3B --out artifacts/smollm3-3b/weights
$PY tools/make_reference.py  --model HuggingFaceTB/SmolLM3-3B --out artifacts/smollm3-3b/golden
make test   # test_build / test_kernels / test_smollm3 모두 PASS
