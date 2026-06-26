# dgx-serving — GB10 (sm_121) 전용. nvcc -arch=sm_120 으로 컴파일 (sm_121 바이너리 호환).

NVCC      ?= nvcc
ARCH      ?= sm_120
NVCCFLAGS ?= -arch=$(ARCH) -O2 -std=c++17 -lineinfo
INCLUDES  := -I.

BUILD := build

# 엔진 TU (커널은 header-only라 smollm3.cu 한 곳에서만 인스턴스화됨)
ENGINE_SRCS := models/smollm3.cu runtime/weights.cu

# 테스트 소스 -> 실행파일 (tests/test_xxx.cu -> tests/test_xxx)
TEST_SRCS := $(wildcard tests/*.cu)
TEST_BINS := $(TEST_SRCS:.cu=)

HEADERS := $(wildcard kernels/*.cuh) $(wildcard runtime/*.h) $(wildcard models/*.h)

.PHONY: all test clean
all: $(TEST_BINS)

# 통합 테스트: 엔진 graph + weight 로더와 함께 컴파일/링크 (단일 nvcc 호출, 1 TU per source).
tests/test_smollm3: tests/test_smollm3.cu models/smollm3.cu runtime/weights.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_smollm3.cu models/smollm3.cu runtime/weights.cu -o $@

tests/test_qwen35: tests/test_qwen35.cu models/qwen35.cu runtime/weights_qwen.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/test_qwen35.cu models/qwen35.cu runtime/weights_qwen.cu -o $@

# 단일 TU 테스트 (header-only 커널만 사용).
tests/%: tests/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

# 빌드 + 모든 테스트 실행
test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "=== run $$t ==="; $$t || exit 1; done

clean:
	rm -f $(TEST_BINS)
	rm -rf $(BUILD)
