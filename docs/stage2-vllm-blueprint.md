# 로드맵 2단계 — vLLM 외부 좌표 STEP 0 (셋업 게이트: 블로커 보고)

> 목표: vLLM을 GB10에 올려 Qwen3.5-0.8B로 우리 엔진(B=8 W4A8 980, llama.cpp 추월)과 공정 비교.
> 결과: **셋업 게이트에서 막힘 — vLLM 0.15.1이 Qwen3.5(qwen3_5) 아키텍처 미지원.** (work order가 예상한 블로커.)

## 환경 (전부 OK — 인프라는 준비됨)
- docker 29.1.3 + nvidia-ctk/runtime, `--gpus all` GPU passthrough 동작(컨테이너가 GB10 cap 12.1 인식).
- 디스크 1.4TB 여유. **`vllm/vllm-openai:latest`(20.6GB) 이미 받아져 있음**(pull 불필요).
- 이미지: **vLLM 0.15.1, torch 2.9.1+cu129, arch sm_120**(GB10 sm_121은 sm_120 바이너리 호환 — 우리 엔진도 -arch=sm_120).
  ⇒ **GB10 하드웨어/드라이버/CUDA 호환은 문제 없음.** sm_121 빌드 포크/스크립트 불필요였음.

## 블로커 — vLLM이 Qwen3.5 아키텍처 미지원 (3회 시도, 각 더 깊이)
1. **기본 로드**: 컨테이너 transformers 4.57.6이 `model_type qwen3_5`를 모름(레퍼런스 5.12.1은 앎).
2. **transformers 5.12.1 업그레이드**: HF 파싱 통과 → 이번엔 **vLLM이 `Qwen3_5ForConditionalGeneration` 미지원**
   (vLLM 0.15.1 레지스트리엔 구 이름 `Qwen3NextForCausalLM`만; Qwen3.5는 Qwen3-Next의 신 이름).
3. **arch override → Qwen3NextForCausalLM**: 더 나아갔으나 `AttributeError: 'Qwen3_5Config' object has no
   attribute 'linear_num_key_heads'`. **vLLM 0.15.1 Qwen3Next 코드는 구 qwen3_next config 스키마(`linear_num_key_heads`
   등)를 기대, Qwen3.5는 필드명이 바뀜** → override로도 불가. 같은 아키텍처(GDN+MTP)지만 config 스키마가 갈라짐.

## 정체 + 데이터로서의 의미
- vLLM 0.15.1(최신급)도 **Qwen3.5(qwen3_5) = 갓 나온 GDN+MTP 아키텍처를 아직 미지원**. GB10/CUDA는 문제 없음,
  순전히 **모델 아키텍처 지원 시차**. → **우리 GDN-네이티브 엔진이 차별점**(우리는 Qwen3.5/GDN을 네이티브로 돌림,
  vLLM 가용 버전은 못 돎). work order가 예상한 "그것도 데이터, GDN 네이티브가 차별점" 그대로.
- 공통 모델 부재: 우리 엔진은 Qwen3.5(GDN) 전용 → vLLM이 지원하는 다른 모델(Llama 등)은 우리가 못 돌림 → same-model 3-way 불가.

## 다음 선택지 (게이트)
- (가) **더 최신 vLLM**(post-Qwen3.5 릴리스: nightly/소스 빌드)로 qwen3_5 지원 확인 — 단 미지원 가능성(아키텍처 시차) +
  20GB pull/30분 빌드 비용, 불확실.
- (나) **"vLLM 미지원" 기록으로 마감** — GDN 네이티브를 차별점으로. 외부 좌표는 llama.cpp(추월)로 충분.
- (다) **config 필드 매핑 패치**(qwen3_5→qwen3_next 필드 reverse-map) — 로드돼도 GDN 규약 차이로 오작동/불공정 위험. 비권장.
