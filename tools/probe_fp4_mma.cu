// Standalone instruction probe; no engine integration or performance claims.
// PROBE_VARIANT: 1=NVFP4 K64, 2=MXFP4 K64, 3=byte-padded FP4 K32.
// Syntax/encoding: https://docs.nvidia.com/cuda/parallel-thread-execution/
// Fragment mapping cross-checked against gemm_warp{,_mxf4}.cu at:
// https://github.com/secYOUre/nvfp4bench/tree/df8e8f469b0df929d7c5ba58ff26c5dfdd4e920c
// Mapping is independently expressed below; the three exact checks test it.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef PROBE_VARIANT
#define PROBE_VARIANT 1
#endif
#if PROBE_VARIANT == 1
#define FP4_INSTRUCTION "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
constexpr int kDepth = 64;
constexpr const char* kVariant = "NVFP4_4X_ue4m3";
#elif PROBE_VARIANT == 2
#define FP4_INSTRUCTION "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
constexpr int kDepth = 64;
constexpr const char* kVariant = "MXFP4_2X_ue8m0";
#elif PROBE_VARIANT == 3
#define FP4_INSTRUCTION "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
constexpr int kDepth = 32;
constexpr const char* kVariant = "byte_padded_FP4_1X_ue8m0";
#else
#error PROBE_VARIANT must be 1, 2, or 3
#endif

#define CUDA_CHECK(call) do { \
  const cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
    std::exit(2); \
  } \
} while (0)

__device__ __forceinline__ uint32_t repeated_code(unsigned code) {
#if PROBE_VARIANT == 3
  // E2M1 occupies bits 5:2, NOT the low nibble, in an 8-bit container.
  return (code << 2) * 0x01010101u;
#else
  return code * 0x11111111u;
#endif
}

__device__ __forceinline__ uint32_t scale_word(bool twice) {
#if PROBE_VARIANT == 1
  return (twice ? 0x40u : 0x38u) * 0x01010101u;
#elif PROBE_VARIANT == 2
  return (twice ? 0x80u : 0x7fu) * 0x00000101u;
#else
  return twice ? 0x80u : 0x7fu;
#endif
}

__device__ __forceinline__ void fp4_mma(float (&d)[4], const uint32_t (&a)[4],
                                       const uint32_t (&b)[2], uint32_t sa, uint32_t sb) {
  const uint16_t zero = 0;
  asm volatile(FP4_INSTRUCTION
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
    "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
      "r"(sa), "h"(zero), "h"(zero), "r"(sb), "h"(zero), "h"(zero));
}

__device__ __forceinline__ void int8_mma(int (&d)[4]) {
  const uint32_t ones = 0x01010101u;
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};\n"
    : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(ones));
}

__global__ void check_one_mma(float* output, int stage) {
  const int group = threadIdx.x / 4;
  const int pair = threadIdx.x % 4;
  const unsigned even = stage == 0 ? 2u : unsigned((2 * group) % 4 + 1);
  const unsigned odd = stage == 0 ? 2u : unsigned((2 * group + 1) % 4 + 1);
  const unsigned col = stage == 0 ? 2u : unsigned(group % 4 + 1);
  const uint32_t a[4] = {repeated_code(even), repeated_code(odd),
                         repeated_code(even), repeated_code(odd)};
  const uint32_t b[2] = {repeated_code(col), repeated_code(col)};
  float d[4] = {};
  fp4_mma(d, a, b, scale_word(stage == 2), scale_word(false));
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    output[(2 * group + i / 2) * 8 + 2 * pair + i % 2] = d[i];
}

constexpr int kAccumulators = 16;
constexpr int kIterations = 700;
constexpr int kThreads = 128;

template<bool Int8>
__global__ void throughput(float* output) {
  float fp[kAccumulators][4] = {};
  int si[kAccumulators][4] = {};
  // Distinct initial values prevent ptxas from merging identical MMA chains.
  #pragma unroll
  for (int i = 0; i < kAccumulators; ++i) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      fp[i][j] = float(4 * i + j + 1);
      si[i][j] = 4 * i + j + 1;
    }
  }
  const uint32_t word = repeated_code(2);
  const uint32_t a[4] = {word, word, word, word};
  const uint32_t b[2] = {word, word};
  // No memory operations inside this loop. Inspect SASS for register spills.
  #pragma unroll 1
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    #pragma unroll
    for (int i = 0; i < kAccumulators; ++i) {
      if (Int8) int8_mma(si[i]);
      else fp4_mma(fp[i], a, b, scale_word(false), scale_word(false));
    }
  }
  float sum = 0;
  #pragma unroll
  for (int i = 0; i < kAccumulators; ++i)
    #pragma unroll
    for (int j = 0; j < 4; ++j) sum += Int8 ? float(si[i][j]) : fp[i][j];
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

static uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

static bool correctness(float* device) {
  bool passed = true;
  for (int stage = 0; stage < 3; ++stage) {
    CUDA_CHECK(cudaMemset(device, 0xff, 128 * sizeof(float)));
    check_one_mma<<<1, 32>>>(device, stage);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float values[128];
    CUDA_CHECK(cudaMemcpy(values, device, sizeof(values), cudaMemcpyDeviceToHost));
    int mismatches = 0;
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 8; ++col) {
        const float a = stage == 0 ? 1.f : float(row % 4 + 1) * .5f;
        const float b = stage == 0 ? 1.f : float(col % 4 + 1) * .5f;
        const float expected = float(kDepth) * a * b * (stage == 2 ? 2.f : 1.f);
        const float actual = values[row * 8 + col];
        if (bits(expected) != bits(actual)) {
          if (mismatches++ == 0)
            std::printf("stage=%d first_mismatch row=%d col=%d expected=%.9g actual=%.9g expected_bits=%08x actual_bits=%08x\n",
                        stage + 1, row, col, expected, actual, bits(expected), bits(actual));
        }
      }
    }
    std::printf("stage=%d result=%s mismatches=%d elements=128 K=%d\n",
                stage + 1, mismatches ? "FAIL" : "PASS", mismatches, kDepth);
    passed = passed && mismatches == 0;
  }
  return passed;
}

template<bool Int8>
static double benchmark(float* device, int blocks, bool warm_only) {
  throughput<Int8><<<blocks, kThreads>>>(device);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (warm_only) {
    for (int i = 0; i < 200; ++i) throughput<Int8><<<blocks, kThreads>>>(device);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  double best = 0;
  const double operations = double(blocks) * (kThreads / 32) * kIterations *
                            kAccumulators * 2 * 16 * 8 * (Int8 ? 32 : kDepth);
  for (int sample = 0; sample < 5; ++sample) {
    CUDA_CHECK(cudaEventRecord(start));
    throughput<Int8><<<blocks, kThreads>>>(device);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    const double tops = operations / (double(milliseconds) * 1e9);
    best = std::max(best, tops);
    std::printf("instruction=%s sample=%d milliseconds=%.9g operations=%.17g teraops_per_second=%.17g\n",
                Int8 ? "INT8" : kVariant, sample + 1, double(milliseconds), operations, tops);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  return best;
}

int main(int argc, char** argv) {
  if (argc != 2 || (std::strcmp(argv[1], "--check") &&
                   std::strcmp(argv[1], "--warm") && std::strcmp(argv[1], "--bench"))) {
    std::fprintf(stderr, "usage: %s --check|--warm|--bench\n", argv[0]);
    return 2;
  }
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  const int blocks = properties.multiProcessorCount * 8;
  const size_t bytes = size_t(blocks) * kThreads * sizeof(float);
  if (bytes > 64u * 1024u * 1024u || bytes < 128 * sizeof(float)) {
    std::fprintf(stderr, "allocation guard failed: bytes=%zu\n", bytes);
    return 2;
  }
  std::printf("variant=%s K=%d device=%s sm=%d.%d multiprocessors=%d allocated_bytes=%zu blocks=%d threads=%d accumulators=%d iterations=%d\n",
              kVariant, kDepth, properties.name, properties.major, properties.minor,
              properties.multiProcessorCount, bytes, blocks, kThreads, kAccumulators, kIterations);
  float* device;
  CUDA_CHECK(cudaMalloc(&device, bytes));
  if (!correctness(device)) {
    CUDA_CHECK(cudaFree(device));
    return 1;
  }
  if (!std::strcmp(argv[1], "--warm")) {
    benchmark<false>(device, blocks, true);
    benchmark<true>(device, blocks, true);
    std::puts("warm_complete=1");
  } else if (!std::strcmp(argv[1], "--bench")) {
    const double fp4 = benchmark<false>(device, blocks, false);
    const double int8 = benchmark<true>(device, blocks, false);
    std::printf("best_FP4_teraops=%.17g best_INT8_teraops=%.17g FP4_INT8_ratio=%.17g scope=instruction_only_not_GEMM\n",
                fp4, int8, fp4 / int8);
  }
  CUDA_CHECK(cudaFree(device));
  return 0;
}
