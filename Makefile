# dgx-serving — GB10 (sm_121) 전용. nvcc -arch=sm_120 으로 컴파일 (sm_121 바이너리 호환).

NVCC      ?= nvcc
ARCH      ?= sm_120
NVCCFLAGS ?= -arch=$(ARCH) -O2 -std=c++17 -lineinfo
INCLUDES  := -I.

BUILD := build

# 테스트 소스 -> 실행파일 (tests/test_xxx.cu -> tests/test_xxx)
TEST_SRCS := $(wildcard tests/*.cu)
TEST_BINS := $(TEST_SRCS:.cu=)

.PHONY: all test clean
all: $(TEST_BINS)

# 각 테스트는 헤더 온리 커널(kernels/*.cuh)과 config(runtime/*.h)에 의존.
tests/%: tests/%.cu $(wildcard kernels/*.cuh) $(wildcard runtime/*.h)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

# 빌드 + 모든 테스트 실행
test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "=== run $$t ==="; $$t || exit 1; done

clean:
	rm -f $(TEST_BINS)
	rm -rf $(BUILD)
