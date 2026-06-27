# 로드맵 2단계 — vLLM 외부 좌표 STEP 0 (셋업 게이트: 성공)

> 목표: vLLM을 GB10에 올려 Qwen3.5-0.8B로 우리 엔진(B=8 W4A8 980, llama.cpp 추월)과 공정 비교.
> 결과: **셋업 성공 — vLLM 0.17.0이 GB10서 Qwen3.5-0.8B 정상 로드·생성.**

## 정정 기록 (과잉 일반화 → 측정이 기각)
처음 0.15.1에서 막혀 "vLLM이 Qwen3.5/GDN 미지원"으로 일반화했으나 **틀림** — 0.15.1(옛 버전)의 한계지 vLLM의 한계 아님.
**vLLM 0.17.0부터 Qwen3.5 hybrid GDN 네이티브 지원**(`Qwen3_5ForConditionalGeneration`+`Qwen3_5MTP` 등록).
막힌 현상에서 과잉 일반화 → 사용자 정정 + 측정이 기각. (헌법: 끝값 추측 금지, 확인.)

## 셋업 (성공)
- 환경: docker 29.1.3 + nvidia-ctk, `--gpus all` passthrough(GB10 cap 12.1). 디스크 1.4TB.
- 이미지: **`vllm/vllm-openai:v0.17.0`(21.5GB) pull** (0.15.1은 너무 옛 버전). vLLM 0.17.0, torch 2.10.0+cu129,
  arch sm_120(GB10 sm_121 호환). **CUDA/하드웨어 무결.**
- 로드: `LLM(model='Qwen/Qwen3.5-0.8B', hf_overrides={'language_model_only':True}, ...)`.
  arch `Qwen3_5ForConditionalGeneration` resolved, **GDN을 mamba hybrid로 처리**(KV cache 91GB, mamba page size),
  init 26s, 생성 정상("The capital of France is Paris."). multiprocessing은 `if __name__=='__main__':` 가드 필요(스크립트 이슈).
- `--language-model-only`(=`language_model_only` override): Qwen3.5가 멀티모달 VLM 클래스라 text-only 모드로 로드(공식).

## STEP 1 비교 설계 (게이트 — 합의 후 벤치)
### 정밀도 매칭 (핵심 결정)
vLLM은 기본 fp16/bf16(원본 safetensors)로 로드됨. 우리 q4/W4A8(4bit weight)과 공정 비교하려면:
- (A) **fp16 vLLM vs 우리 fp32**: 둘 다 "풀 정밀도"(단 우리 fp32 4byte vs vLLM fp16 2byte — 바이트 다름, 기록).
- (B) **fp8 vLLM(`--quantization fp8`, on-the-fly) vs 우리 W4A8**: 둘 다 "압축"(단 fp8 8bit weight vs 우리 int4 4bit — 차이 기록).
- (C) **int4 vLLM(AWQ/GPTQ) vs 우리 q4/W4A8**: 같은 4bit이나 vLLM이 양자화 체크포인트 필요(AWQ 생성/다운로드 = 추가 작업).
→ 권장: (A)+(B) 먼저(추가 체크포인트 불요), (C)는 필요시. 정밀도 차이 명시 기록.

### 측정 조건 (챕터3 "같이 뽑기")
- 같은 모델(Qwen3.5-0.8B), 같은 클럭(RmProfilingAdminOnly=0), 같은 프롬프트·N=96.
- batch: vLLM은 continuous batching(`--max-num-seqs` B) — 우리 정적 B 스윕(1,8,16)과 total tok/s로 통일.
  vLLM 측정 = `vllm bench throughput` 또는 OpenAI 서버 + 동시요청. llama-batched-bench와 같은 메트릭.
- spec: 우리 MTP spec vs vLLM Qwen3_5MTP — 같은 기법끼리(나중 단계), 우선 plain.

### STEP 1 산출
3-way 표: B 스윕 × {우리 W4A8/q4/fp32, llama.cpp Q4_K_M/F16, vLLM fp16/fp8} total tok/s + per-seq latency + 품질(토큰열).
정직 해석: 어디서 앞서고 뒤지나. vLLM이 높으면 격차 정체(continuous batching/커널/scheduling) → 3단계 방향.
