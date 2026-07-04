# dgx-serving — GB10 (sm_121) 전용. nvcc -arch=sm_120 으로 컴파일 (sm_121 바이너리 호환).

NVCC      ?= nvcc
ARCH      ?= sm_120
NVCCFLAGS ?= -arch=$(ARCH) -O2 -std=c++17 -lineinfo
INCLUDES  := -I.

BUILD := build

# 엔진 TU (커널은 header-only라 smollm3.cu 한 곳에서만 인스턴스화됨)
ENGINE_SRCS := models/smollm3.cu runtime/weights.cu

# 테스트 소스 -> 실행파일 (tests/test_xxx.cu -> tests/test_xxx)
# *_27b/*_35b/moe 테스트는 대형모델 artifacts 필요, test_gemv_q4_scale 는 측정 도구 → 기본 스위트 제외(수동).
TEST_SRCS := $(filter-out tests/test_27b.cu tests/test_gdn_27b.cu tests/test_attn_27b.cu tests/test_gemv_q4_scale.cu \
             tests/test_35b.cu tests/test_gdn_35b.cu tests/test_moe_expert.cu tests/test_rope_gen.cu tests/test_chunk_prefill.cu \
             tests/test_gemma4.cu tests/test_gemma4_decode.cu tests/test_gemma4_gen.cu tests/test_gemma4_batch.cu tests/test_gemma4_ragged.cu tests/test_gemma4_prefill_slot.cu tests/test_gemma4_moe.cu tests/test_gemma4_moe_golden.cu tests/test_gemma4_moe_router_iso.cu \
             tests/test_long_ctx.cu tests/test_treeamort.cu tests/test_treespec.cu tests/test_treespec_depth.cu, $(wildcard tests/*.cu))
TEST_BINS := $(TEST_SRCS:.cu=)

HEADERS := $(wildcard kernels/*.cuh) $(wildcard runtime/*.h) $(wildcard models/*.h)

.PHONY: all test clean
all: $(TEST_BINS)

# 통합 테스트: 엔진 graph + weight 로더와 함께 컴파일/링크 (단일 nvcc 호출, 1 TU per source).
tests/test_smollm3: tests/test_smollm3.cu models/smollm3.cu runtime/weights.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_smollm3.cu models/smollm3.cu runtime/weights.cu -o $@

tests/test_qwen35: tests/test_qwen35.cu models/qwen35.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_qwen35.cu models/qwen35.cu runtime/weights_qwen.cu -o $@

# Gemma4 dense 골든 게이트 (대형 artifacts 필요 → 기본 스위트 제외, 수동 빌드)
tests/test_gemma4: tests/test_gemma4.cu models/gemma4.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4.cu models/gemma4.cu runtime/weights_gemma4.cu -o $@

# Gemma4 26B-A4B MoE 스모크 (dense∥MoE prefill graph. 대형 artifacts 필요 → 기본 스위트 제외, 수동 빌드)
tests/test_gemma4_moe: tests/test_gemma4_moe.cu models/gemma4_moe.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4_moe.cu models/gemma4_moe.cu runtime/weights_gemma4.cu -o $@

# Gemma4 26B-A4B MoE 골든 게이트 (op-level rel + argmax + ★router idx/weight. 대형 artifacts+골든 필요 → 수동)
tests/test_gemma4_moe_golden: tests/test_gemma4_moe_golden.cu models/gemma4_moe.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4_moe_golden.cu models/gemma4_moe.cu runtime/weights_gemma4.cu -o $@

# 26B decode 게이트: gemma4_moe_decode(single)+gemma4_moe_decode_batch(batch) == forward(엔진 정합·연속성·batch 격리).
#   gemma4_decode.cu=alloc_g4_decode_state / gemma4_batch.cu=gemma4_moe_decode_batch·alloc_g4_batch_state.
tests/test_gemma4_moe_decode: tests/test_gemma4_moe_decode.cu models/gemma4_moe.cu models/gemma4_decode.cu models/gemma4_batch.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4_moe_decode.cu models/gemma4_moe.cu models/gemma4_decode.cu models/gemma4_batch.cu runtime/weights_gemma4.cu -o $@

# Gemma4 26B-A4B router ISOLATED 검증 (동일 rin → 엔진 router == HF. 소형 텐서만, standalone)
tests/test_gemma4_moe_router_iso: tests/test_gemma4_moe_router_iso.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4_moe_router_iso.cu -o $@

tests/test_gemma4_decode: tests/test_gemma4_decode.cu models/gemma4.cu models/gemma4_decode.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4_decode.cu models/gemma4.cu models/gemma4_decode.cu runtime/weights_gemma4.cu -o $@

tests/test_gemma4_gen: tests/test_gemma4_gen.cu models/gemma4_decode.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4_gen.cu models/gemma4_decode.cu runtime/weights_gemma4.cu -o $@

tests/test_gemma4_batch: tests/test_gemma4_batch.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4_batch.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu -o $@

tests/test_decode: tests/test_decode.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_decode.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

tests/test_verify: tests/test_verify.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_verify.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

tests/test_promote: tests/test_promote.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_promote.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

tests/test_spec: tests/test_spec.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_spec.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

tests/test_treefork: tests/test_treefork.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_treefork.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

tests/test_treeverify: tests/test_treeverify.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_treeverify.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

tests/test_treepromote: tests/test_treepromote.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_treepromote.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

tests/test_treespec_e2e: tests/test_treespec_e2e.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_treespec_e2e.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

# 수동(27B 필요, 느림): tree 1-forward amortize 속도 de-risk. 사용법: ./tests/test_treeamort [weights] [golden] [bq4]
tests/test_treeamort: tests/test_treeamort.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_treeamort.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

# 수동(27B, 느림): STEP6.0a per-depth rank de-risk. 사용법: ./tests/test_treespec_depth [w] [g] [N] [bq4] [maxdepth]
tests/test_treespec_depth: tests/test_treespec_depth.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_treespec_depth.cu models/qwen35.cu models/qwen35_decode.cu runtime/weights_qwen.cu -o $@

tests/test_batch_decode: tests/test_batch_decode.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_batch_decode.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

tests/test_cbatch: tests/test_cbatch.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_cbatch.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

# 챕터10 Phase1: 2-요청 batch+spec 결합 mism=0. batch verify(qwen35_verify_batch)==단독 spec. 사용법: ./tests/test_batchspec2 [w] [g] [N] [K] [W] [uq]
tests/test_batchspec2: tests/test_batchspec2.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_batchspec2.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

# production scheduler: admission guard(KV예산)+slot 재사용+요청 큐. 사용법: ./tests/test_cbatch_prod [w] [g] [uq] [Bmax] [maxlen]
tests/test_cbatch_prod: tests/test_cbatch_prod.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_cbatch_prod.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

# 실부하 서빙 벤치(Poisson 스트림+가변길이+warm-up+정적대비+부하스윕). 사용법: ./tools/bench_serve [w] [g] [uq] [Bmax] [nreq] [seed] [λ...]
tools/bench_serve: tools/bench_serve.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tools/bench_serve.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

# M4 라운드2: Gemma4 full 서빙 스모크(serving_core.h 모델-무지 코어 + Gemma4Backend). 결정성+완료+누수 게이트.
#   사용법: ./tools/serve_gemma4 [w] [g] [uq] [Bmax] [nreq] [seed] [λ]
tools/serve_gemma4: tools/serve_gemma4.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tools/serve_gemma4.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu -o $@

# M5 Phase 0: Gemma4 plain 서빙 daemon(영속 token-ID in/out, greedy). Python front(serve_api.py)이 HTTP/토큰화 담당.
tools/serve_daemon_gemma4: tools/serve_daemon_gemma4.cu models/gemma4_decode.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tools/serve_daemon_gemma4.cu models/gemma4_decode.cu runtime/weights_gemma4.cu -o $@

# M5 온라인 연속 배칭 daemon(모델-중립 루프 serving_online.h + Gemma4OnlineBackend). batch primitive 재사용.
tools/serve_daemon_gemma4_online: tools/serve_daemon_gemma4_online.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tools/serve_daemon_gemma4_online.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu -o $@ -lpthread

# M5 온라인 daemon — Qwen3.5 계열(27b·35b-a3b MoE plain, greedy). 같은 모델-중립 루프(serving_online.h) + QwenOnlineBackend.
tools/serve_daemon_qwen_online: tools/serve_daemon_qwen_online.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tools/serve_daemon_qwen_online.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@ -lpthread

# M5 STEP C: 온라인 spec daemon — Qwen3.5 27b spec(MTP draft→verify→accept, 태그드 연속 배칭). run_spec_continuous 실시간 짝.
tools/serve_daemon_qwen_spec_online: tools/serve_daemon_qwen_spec_online.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tools/serve_daemon_qwen_spec_online.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@ -lpthread

# M4 STEP B: Gemma4 성능 벤치(single/batch decode t/s + prefill, roofline 대조). 사용법: ./tools/bench_gemma4 <w> [uq] [ctx] [ngen] [B..]
tools/bench_gemma4: tools/bench_gemma4.cu models/gemma4_decode.cu models/gemma4_batch.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tools/bench_gemma4.cu models/gemma4_decode.cu models/gemma4_batch.cu runtime/weights_gemma4.cu -o $@

# M4 STEP C: Gemma4 plain 서빙 튜닝 스윕(λ×Bmax, dummy rope, 메모리 필터). 사용법: ./tools/bench_serve_gemma4 <w> [uq] [nreq] [seed] [pmax] [maxlen] [B..] -- [λ..]
tools/bench_serve_gemma4: tools/bench_serve_gemma4.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tools/bench_serve_gemma4.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu -o $@

# 챕터10 축2: spec-serving 벤치(연속배칭+슬롯별 spec 사이클, ragged verify_batch). ISO=1 iso게이트, TIME=1 곱 B버킷.
#   사용법: ./tools/bench_serve_spec [w] [g] [uq] [Bmax] [nreq] [seed] [K] [W] [λ...]
tools/bench_serve_spec: tools/bench_serve_spec.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tools/bench_serve_spec.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

# SMAX 8→16 iso 게이트: batch decode(B=12 SB16, B=16 gemm_batch) == single. 사용법: ./tests/test_sb16_iso [w] [g] [uq]
tests/test_sb16_iso: tests/test_sb16_iso.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_sb16_iso.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

# chunked prefill mism=0: prefill_batch_slot(청크) == 1토큰/step prefill. 사용법: ./tests/test_chunkprefill [w] [g] [uq] [P] [N] [chunk]
tests/test_chunkprefill: tests/test_chunkprefill.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_chunkprefill.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

# long-rope pos>=128: ext 테이블(gen_rope_long.py 먼저)로 P>128 chunked==1토큰 prefill. 사용법: ./tests/test_longrope [w] [g] [uq] [P] [N] [chunk]
tests/test_longrope: tests/test_longrope.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_longrope.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

tests/test_engine: tests/test_engine.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_engine.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

# 단일 TU 테스트 (header-only 커널만 사용).
tests/%: tests/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

# 빌드 + 모든 테스트 실행
test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "=== run $$t ==="; $$t || exit 1; done

clean:
	rm -f $(TEST_BINS)
	rm -rf $(BUILD)

tests/test_27b: tests/test_27b.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_27b.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

# 수동(27B 필요, 느림): W4A8 청크 자기-일관성/결정성. 사용법: ./tests/test_long_ctx <weights> [C] [use_q4]
tests/test_long_ctx: tests/test_long_ctx.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_long_ctx.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

tests/test_35b: tests/test_35b.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_35b.cu models/qwen35.cu models/qwen35_decode.cu models/qwen35_batch.cu runtime/weights_qwen.cu -o $@

tests/test_gemma4_ragged: tests/test_gemma4_ragged.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4_ragged.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu -o $@

# STEP A chunked prefill 게이트: gemma4_prefill_batch_slot(어댑터) == c1tok bit-id/scheme-internal 검증.
tests/test_gemma4_prefill_slot: tests/test_gemma4_prefill_slot.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_gemma4_prefill_slot.cu models/gemma4_batch.cu models/gemma4_decode.cu runtime/weights_gemma4.cu -o $@
