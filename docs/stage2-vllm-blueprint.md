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

## STEP 1 (A) — vLLM fp16 참고좌표 [완료] + 정직 해석

`vllm bench throughput`, input 16/output 96, CUDA graph on(enforce-eager 빼야 정상속도; eager는 25 tok/s로 1/4).
clean run(random-input-len 16). **total output tok/s:**

| B(max-num-seqs) | vLLM fp16 | 우리 W4A8 | 우리 fp32 | llama.cpp Q4_K_M |
|---|---|---|---|---|
| 1 | 26.3 | 215 | 70 | 277 |
| 8 | 96.6 | 980 | 454 | 912 |
| 16 | 189.8 | 1016 | 650 | 1266 |
| 32 | 364.7 | — | — | — |

- **vLLM fp16이 전 구간 5~10× 느림** (B=8: 우리 W4A8의 1/10, fp32의 1/4.7, llama의 1/9). B=32(364)도 우리 B=16(1016) 못 미침.
- vLLM은 concurrency로 선형 scaling(96→190→365)하나 절대 throughput이 낮음.

### 정직 해석 — 왜 vLLM이 이렇게 느린가 (공정성 caveat)
1. **정밀도 불일치**: vLLM fp16(2byte weight) vs 우리 q4/W4A8(0.5byte). 단 fp16은 우리 fp32(4byte)보다 BW-bound면
   ~2× 빨라야 하는데 오히려 4.7× 느림 → **정밀도로 설명 안 됨, 프레임워크 오버헤드.**
2. **소형 모델 regime**: vLLM은 대형 모델(70B+) 서빙용 — paged attn/continuous batching/scheduling 오버헤드가
   compute에 amortize되는 전제. **0.8B는 compute가 작아 per-step Python/scheduling 오버헤드가 지배** → 낮은 throughput.
   = 우리 "단일 장비 × 특정 모델군 최적화"(CLAUDE.md 정체성)가 빛나는 지점. lean 특화 엔진 vs 범용 프레임워크.
3. (B)fp8/(C)int4도 이 오버헤드 story를 크게 안 바꿈(병목이 weight 바이트가 아니라 per-step 오버헤드) — 단 (C)는 공정성 위해 측정 가치.

### STEP 1 (A) 결론
**소형 GDN 모델(0.8B) GB10 서빙서 우리 특화 엔진이 vLLM 대비 5~10× 빠름.** 단 이는 부분적으로 소형 모델 regime
(vLLM 프레임워크 오버헤드)이지 순수 커널 우위 아님 — 정직히 기록. vLLM의 강점(대형 모델/continuous batching)은
이 비교에 안 드러남. 다음: (C) int4 정합 비교로 "정밀도 통제 후에도 격차 유지되나" 확인(오버헤드 story 검증).
