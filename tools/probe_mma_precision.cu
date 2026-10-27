// Register-only instruction probe adapted from probe_fp4_mma.cu (476ba77).
// Throughput is an instruction upper bound, not an attention/GEMM speedup.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do { \
  const cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
    std::exit(2); \
  } \
} while (0)

enum class Kind { F16F32, F16F16, BF16F32, INT8S32 };
constexpr int kChains = 16;
constexpr int kIterations = 700;
constexpr int kThreads = 128;

static const char* name(Kind kind) {
  switch (kind) {
    case Kind::F16F32: return "f16_f32";
    case Kind::F16F16: return "f16_f16";
    case Kind::BF16F32: return "bf16_f32";
    case Kind::INT8S32: return "int8_s32";
  }
  return "invalid";
}

template<Kind K> struct Accumulator {
  float f[4];
  int s[4];
  uint32_t h[2];

  __device__ __forceinline__ void init(int chain, bool zero) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int value = zero ? 0 : 4 * chain + j + 1;
      if constexpr (K == Kind::INT8S32) s[j] = value;
      else if constexpr (K != Kind::F16F16) f[j] = float(value);
    }
    if constexpr (K == Kind::F16F16) {
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float lo = zero ? 0.f : float(4 * chain + 2 * j + 1);
        const float hi = zero ? 0.f : float(4 * chain + 2 * j + 2);
        h[j] = uint32_t(__half_as_ushort(__float2half_rn(lo))) |
               (uint32_t(__half_as_ushort(__float2half_rn(hi))) << 16);
      }
    }
  }

  __device__ __forceinline__ void step() {
    if constexpr (K == Kind::F16F32) {
      const uint32_t ones = 0x3c003c00u;
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};\n"
        : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3]) : "r"(ones));
    } else if constexpr (K == Kind::BF16F32) {
      const uint32_t ones = 0x3f803f80u;
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};\n"
        : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3]) : "r"(ones));
    } else if constexpr (K == Kind::F16F16) {
      const uint32_t ones = 0x3c003c00u;
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%2,%2,%2}, {%2,%2}, {%0,%1};\n"
        : "+r"(h[0]), "+r"(h[1]) : "r"(ones));
    } else {
      const uint32_t ones = 0x01010101u;
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};\n"
        : "+r"(s[0]), "+r"(s[1]), "+r"(s[2]), "+r"(s[3]) : "r"(ones));
    }
  }

  __device__ __forceinline__ float value(int j) const {
    if constexpr (K == Kind::INT8S32) return float(s[j]);
    else if constexpr (K == Kind::F16F16)
      return __half2float(__ushort_as_half(static_cast<unsigned short>(h[j / 2] >> (16 * (j % 2)))));
    else return f[j];
  }
};

template<Kind K> __global__ void check_mma_precision(float* output) {
  Accumulator<K> a;
  a.init(0, true);
  a.step();
  #pragma unroll
  for (int j = 0; j < 4; ++j) output[4 * threadIdx.x + j] = a.value(j);
}

template<Kind K> __global__ void throughput_mma_precision(float* output) {
  Accumulator<K> a[kChains];
  // Distinct seeds retain independent chains; inspect SASS for spills/merging.
  #pragma unroll
  for (int i = 0; i < kChains; ++i) a[i].init(i, false);
  #pragma unroll 1
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    #pragma unroll
    for (int i = 0; i < kChains; ++i) a[i].step();
  }
  float sum = 0.f;
  #pragma unroll
  for (int i = 0; i < kChains; ++i) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) sum += a[i].value(j);
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

static double epoch_seconds() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static void marker(const char* phase, const char* edge, Kind kind, int pair) {
  std::printf("interval=%s edge=%s instruction=%s pair=%d epoch_seconds=%.17g\n",
              phase, edge, name(kind), pair, epoch_seconds());
  std::fflush(stdout);
}

template<Kind K> static bool check(float* device) {
  check_mma_precision<K><<<1, 32>>>(device);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float values[128];
  CUDA_CHECK(cudaMemcpy(values, device, sizeof(values), cudaMemcpyDeviceToHost));
  const float expected = K == Kind::INT8S32 ? 32.f : 16.f;
  int mismatches = 0;
  for (int i = 0; i < 128; ++i) {
    if (values[i] != expected) {
      if (mismatches++ == 0)
        std::printf("first_mismatch index=%d expected=%.9g actual=%.9g\n", i, expected, values[i]);
    }
  }
  std::printf("check instruction=%s result=%s mismatches=%d elements=128\n",
              name(K), mismatches ? "FAIL" : "PASS", mismatches);
  return mismatches == 0;
}

template<Kind K> static bool run(float* device, int blocks, int pair, bool check_only) {
  if (!check<K>(device)) return false;
  if (check_only) return true;
  CUDA_CHECK(cudaDeviceSynchronize());
  marker("warm", "start", K, pair);
  const auto warm_start = std::chrono::steady_clock::now();
  size_t warm_launches = 0;
  double seconds;
  do {
    for (int i = 0; i < 100; ++i) throughput_mma_precision<K><<<blocks, kThreads>>>(device);
    warm_launches += 100;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_start).count();
  } while (seconds < 10.0);
  marker("warm", "end", K, pair);
  std::printf("warm instruction=%s pair=%d seconds=%.17g reps=%zu\n", name(K), pair, seconds, warm_launches);

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaDeviceSynchronize());
  marker("measure", "start", K, pair);
  CUDA_CHECK(cudaEventRecord(start));
  size_t reps = 0;
  float ms = 0.f;
  // One event interval includes stream-idle host gaps between 100-launch chunks.
  do {
    for (int i = 0; i < 100; ++i) throughput_mma_precision<K><<<blocks, kThreads>>>(device);
    reps += 100;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  } while (ms < 1000.f);
  marker("measure", "end", K, pair);
  const int depth = K == Kind::INT8S32 ? 32 : 16;
  const double ops = double(blocks) * (kThreads / 32) * kIterations * kChains * 2 * 16 * 8 * depth * double(reps);
  const double teraops = ops / (double(ms) * 1e9);
  std::vector<float> values(size_t(blocks) * kThreads);
  CUDA_CHECK(cudaMemcpy(values.data(), device, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
  bool bounded = true;
  for (float value : values) bounded = bounded && std::isfinite(value) && value > 0.f && value <= 64.f * 65504.f;
  std::printf("result instruction=%s pair=%d reps=%zu cuda_event_ms=%.9g operations=%.17g teraops_per_second=%.17g unit=%s checksum=%.9g bounded=%s\n",
              name(K), pair, reps, double(ms), ops, teraops,
              K == Kind::INT8S32 ? "TOPS" : "TFLOPS", values[0], bounded ? "PASS" : "FAIL");
  std::fflush(stdout);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  return bounded;
}

int main(int argc, char** argv) {
  Kind kind = Kind::F16F32;
  bool found = false;
  if (argc >= 2) {
    for (Kind candidate : {Kind::F16F32, Kind::F16F16, Kind::BF16F32, Kind::INT8S32}) {
      if (!std::strcmp(argv[1], name(candidate))) { kind = candidate; found = true; }
    }
  }
  const bool check_only = argc >= 3 && !std::strcmp(argv[2], "--check");
  const bool bench = argc >= 3 && !std::strcmp(argv[2], "--bench");
  char* end = nullptr;
  const long pair = argc == 4 ? std::strtol(argv[3], &end, 10) : 1;
  if (!found || (!check_only && !bench) || (argc != 3 && argc != 4) ||
      (argc == 4 && (end == argv[3] || *end)) || pair < 1 || pair > 5) {
    std::fprintf(stderr, "usage: %s f16_f32|f16_f16|bf16_f32|int8_s32 --check|--bench [pair 1..5]\n", argv[0]);
    return 2;
  }
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  const int blocks = properties.multiProcessorCount * 8;
  const size_t bytes = size_t(blocks) * kThreads * sizeof(float);
  if (bytes < 128 * sizeof(float) || bytes > 64u * 1024u * 1024u) return 2;
  std::printf("instruction=%s device=%s sm=%d.%d blocks=%d threads=%d chains=%d iterations=%d allocation_bytes=%zu\n",
              name(kind), properties.name, properties.major, properties.minor,
              blocks, kThreads, kChains, kIterations, bytes);
  float* device;
  CUDA_CHECK(cudaMalloc(&device, bytes));
  bool passed = false;
  switch (kind) {
    case Kind::F16F32: passed = run<Kind::F16F32>(device, blocks, int(pair), check_only); break;
    case Kind::F16F16: passed = run<Kind::F16F16>(device, blocks, int(pair), check_only); break;
    case Kind::BF16F32: passed = run<Kind::BF16F32>(device, blocks, int(pair), check_only); break;
    case Kind::INT8S32: passed = run<Kind::INT8S32>(device, blocks, int(pair), check_only); break;
  }
  CUDA_CHECK(cudaFree(device));
  return passed ? 0 : 1;
}
