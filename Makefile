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
