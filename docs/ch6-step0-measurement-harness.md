# 챕터6 STEP 0 — 측정 하네스 설계 (3-way 비교 선행조건)

> 방향 재조정: 모델 확장만 쌓고 비교를 미룬 본말전도 교정. 비교가 프로젝트 목적.
> 27B·35B는 지금 3-way 비교 가능(우리 엔진 + vLLM + llama.cpp 다 돌림). 단 공정 비교엔 측정 하네스 선행.
> 순서: ① 측정 하네스(이 문서) → ② 27B·35B 3-way → ③ 보고 → Gemma4/dispatch/q2.

## 왜 지금까지 불공정했나 (교정 대상)
- 짧은 태스크만 잼: `test_spec`(prompt≈15, gen≈40), `test_batch_decode`(N=96 고정). context 스윕 없음.
- **rope 128 상한**: 엔진이 rope를 내부 생성 안 하고 골든의 `rope_cos.bin`/`rope_sin.bin`(파이썬 `maxpos=128`)
  을 읽음 → 긴 context 자체가 불가. 32K 측정하려면 이 의존부터 끊어야.
- 타이밍이 `clock()`(CPU wall) — GPU 비동기/클럭 변동 반영 못 함. cudaEvent 필요.
- 과거 "vLLM 0.8B 26 t/s" 같은 불공정(클럭 미스매치 의심 + 짧은 태스크 + 소형모델 오버헤드).

## 사실 (검증됨)
- Qwen3.5: **max_position_embeddings=262144(256K)**, rope_type='default', theta=1e7, rotary_dim=64.
  → **32K은 native, rope scaling 불필요**. mrope_section은 멀티모달용 — 텍스트는 표준 1D rope로 붕괴.
- KV 캐시: full-attn 레이어만 [max_len, NKV, HD] (GDN은 고정크기 state). 32K KV 메모리:
  27B(16 full, NKV=4, HD=256) ≈ 16·32768·4·256·2·4B ≈ 8.6GB. 35B(10 full, NKV=2) ≈ 2.7GB. GB10 여유.

## 설계

### ① rope 엔진 내부 생성 (선행조건 — 128 의존 제거)
- 신규 커널 `launch_gen_rope(cos, sin, max_len, rotary_dim, theta)`: cos/sin[p,i]=cos(p·theta^(-2i/rot)),
  i<rot/2, 뒤 절반 = 앞 절반(HF cat(freqs,freqs)). 모델 독립(config theta/rotary_dim만).
- **검증**: 생성 cos/sin vs 골든 `rope_cos.bin`(첫 S=5 위치) 오차0 → HF와 일치 확정. 그 뒤 임의 길이(32K) 생성.
- 이걸로 골든 rope 의존 끊김 + 32K 가능. (정확성-우선: 먼저 골든 대조, 그 다음 속도)

### ② 측정 하네스 `tools/bench.cu` (cudaEvent, context 스윕)
- 모델 1개 q4 로드. GPU clock lock 확인(아래 ③).
- context C ∈ {128, 2K, 8K, 32K}, batch B ∈ {1, 8}:
  - prefill C 토큰(합성 토큰 OK — 정확성 이미 증명, 속도만) → **TTFT**(cudaEvent ms).
  - decode N_gen=128 토큰 → **TPOT**(평균 per-token ms) + **throughput**(tok/s).
- 출력 표: model, C, B, TTFT(ms), TPOT(ms), decode tok/s. 메모리/peak도.
- KV max_len = C + N_gen 으로 decode_state 할당(32K+128). decode_state max_len 인자 확인.

### ③ 공정성 셋업 (스크립트 `tools/bench_3way.sh`)
- **GPU clock lock**: `nvidia-smi -lgc <freq>` (+ `-lmc`), 측정 전후 `nvidia-smi -q -d CLOCK` 기록.
- **단독 로드**: 한 번에 엔진 1개·모델 1개만 (메모리/클럭 간섭 방지). 측정 후 언로드.
- 3 엔진 동일 조건: 같은 C/N_gen/B, 같은 모델(우리 q4 ≈ llama.cpp Q4 ≈ vLLM GPTQ-Int4/FP8 참고).
  - 우리: bench.cu (q4 + W4A8 use_q4=1/2)
  - llama.cpp: `llama-bench` -p C -n N_gen (Q4 gguf)
  - vLLM: 벤치 스크립트(GPTQ-Int4), 같은 C/N_gen. (docs/stage2-vllm-blueprint.md 셋업 재사용)

## STEP 0 게이트 — 결정 필요
1. **rope 엔진 생성 + 골든 대조 검증** 방향 동의? (128 의존 제거, 32K native)
2. **측정축 = TTFT + TPOT + throughput, context {128,2K,8K,32K} × B{1,8}, N_gen=128** 동의? (축/값 조정?)
3. **공정성 = clock lock + 단독 로드 + 동일 조건** 동의? clock 고정 주파수는 GB10 기본 부스트로?
4. **이번 범위 = ① rope + ② 우리 엔진 bench(27B·35B)까지. ③ 3-way(vLLM/llama.cpp 실측)는 다음 STEP**
   동의? 아니면 한 번에 3-way까지?
5. 긴 context **정확성**은 어디까지 — 속도가 목적이나 rope-vs-골든(짧은) + 32K no-NaN sanity 정도로 충분?
