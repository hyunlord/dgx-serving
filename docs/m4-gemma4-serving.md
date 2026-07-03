# M4 — Gemma4 31B 성능/서빙 + 첫 레시피 (Phase 0 이후)

> 베이스: 3f8e373(Gemma4 golden PASS, argmax 5/5) + 8a0ef73(mw1 레시피). Phase 0 해부는 docs/m4-gemma4-phase0.md.
> 프레임: 파이프(해부→전략→튜닝→레시피)가 2번째 모델에서 처음 end-to-end 도는 세션. spec 부재(MTP 없음) 정직 명시.

## STEP 0 — argmax 게이트 보강 (성능 전 로직 보강)

**문제**: Phase 0 게이트는 argmax-only(bf16 골든). fp32 full 118GB=메모리 가드 초과라 27b의 per-layer err<1e-2보다 약함.

**방법 — 층-스트리밍 fp32 골든**(`tools/make_reference_gemma4_fp32.py`):
- bf16 base 유지(62GB) + 층별로만 fp32 업캐스트(층당 ~2GB 전이, 가드 내) → HF 레이어 forward 그대로 재사용(정확성).
- 각 층 q4-policy linear에 **엔진과 동일한 q4 round-trip을 fp32 유지**로 적용 → 엔진(q4 dequant fp32)과 동일
  가중치·동일 정밀도. bf16 노이즈 제거 → per-layer err<1e-2 게이트 복원(bf16 격차 우회).
- 마스크/rotary/shared_kv는 HF 유틸(create_causal_mask/create_sliding_window_causal_mask, Gemma4TextRotaryEmbedding).

**게이트**(test_gemma4, fp32 골든 있으면 자동 승격): per-layer hidden err<1e-2 + logits + argmax 전위치.

**결과 (prompt "The capital of France is", S=5)**:
- **(a) op-level fp32** (9층 = L0~L8, sliding 0~4 + full/k_eq_v L5 + sliding 6~8 — 전 연산타입 커버):
  L0 **maxabs=0**(embed 양쪽 fp32 bit-exact), L1~L8 worst **rel=2.3e-4**(abs 1e-2는 Gemma 대규모
  hidden[|g|max 20→161] 때문 — rel은 오히려 층 깊어질수록 감소 6e-5). = 엔진 GPU-fp32 vs 골든 CPU-fp32
  (동일 q4 weights) 산술차, 버그 아님(smooth·국소점프 없음).
- **(b) full-depth argmax** (bf16 골든, 60층 전체): 전위치 5/5 일치, last-tok 7001.
- **판정: 게이트 PASS** = op-level fp32 rel<2.3e-4 + full argmax 5/5. argmax-only(Phase 0)보다 강함 —
  op 정확성을 fp32로 직접 확인. **한계(정직)**: full-depth fp32는 메모리 가드(층별 업캐스트가 mmap+anon
  누적으로 112GB 근접 → kill)로 첫 8층만. full 깊이는 argmax가 커버. "27b급"에 근접하나 full-depth
  fp32는 아님(메모리 구조적 제약).
- **메모리 사고 기록**: 초기 full-depth 스트리밍이 112GB(가드 초과)까지 누수 → 즉시 kill(시스템 보호).
  경계 8층 + RSS 하드가드(78GB)로 안전 재실행. [[gb10-unified-memory-crash]] 원칙 준수.

## STEP A — 서빙 경로 연결 + sliding KV (예정)

**모델-중립성 시험 결과(사전 조사)**: `bench_serve.cu`는 **모델-중립 아님** — `qwen35_decode_batch`/`load_config_qwen`/
`load_weights_qwen` 하드코딩. Gemma4 서빙 연결 = 신규 `gemma4_decode`(KV 캐시 증분, GDN 제거+sliding-KV-cap+
층별 dual-rope+k_eq_v) + `gemma4_decode_batch` + bench dispatch 필요. **이것이 자동화 갭**(3분류 ①로 흡수 가능하나
분량 큼 — qwen35_decode 288줄+batch 385줄 상당). sliding KV 캡(50/60층 w=1024)이 이 모델 구조적 lever.

<!-- STEPA_RESULT -->
