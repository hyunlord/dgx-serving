// Standalone LUT2-v0 upper-bound probe; no engine/model changes.
// Build: nvcc -std=c++17 -O2 -arch=sm_120 -o <artifact>/probe_gemv_lut2 tools/probe_gemv_lut2.cu
// Run check first. Then warm, check SM clock externally, then bench. All modes are serialized by the caller.
// profile emits exactly one selected kernel for ncu after the separate correctness gate has passed.
// No model-quality comparison: LUT2 and q4 contain independent synthetic weights.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "../kernels/gemv_w4a8.cuh"

static void cuda_check(cudaError_t error, const char* call) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(error));
}
#define CUDA_CHECK(call) cuda_check((call), #call)
constexpr size_t BUFFER_CAP = 2000000000ULL;
constexpr uint32_t SEED = 20261024U;
constexpr int THREADS = 256;
constexpr int TRIALS = 5;
constexpr int REPEATS = 100;

// One warp per output row; one uint32 index load = 16 weights per lane.
// Row padding to a multiple of four indices keeps every uint32 load aligned even for the edge case.
template<bool Packed>
__global__ void lut2_gemv(const uint8_t* indices, const __half* scales,
                          const __half* codebook, const __half* x, float* y,
                          int K, int N, int index_stride, int scale_stride) {
    __shared__ __align__(8) uint16_t table[1024];
    extern __shared__ __half shared_x[];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        table[i] = __half_as_ushort(codebook[i]);
    if constexpr (Packed) {
        // Coalesce activation global loads without changing lane ownership or accumulation order.
        for (int i = threadIdx.x; i < K; i += blockDim.x) shared_x[i] = x[i];
    }
    __syncthreads();
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (row >= N) return;
    float sum = 0.0f;
    for (int base = lane * 16; base < K; base += 32 * 16) {
        const uint32_t packed_indices = *reinterpret_cast<const uint32_t*>(
            indices + static_cast<size_t>(row) * index_stride + base / 4);
        const float scale = __half2float(scales[static_cast<size_t>(row) * scale_stride + base / 32]);
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const int entry = (packed_indices >> (8 * group)) & 255;
            float w[4];
            if constexpr (Packed) {
                // One 64-bit shared load contains two half2 values; inspect SASS to confirm the load width.
                const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(table + entry * 4));
                unsigned long long bits;
                asm volatile("ld.shared.u64 %0, [%1];" : "=l"(bits) : "r"(address));
                const __half2 first = __halves2half2(
                    __ushort_as_half(static_cast<unsigned short>(bits)),
                    __ushort_as_half(static_cast<unsigned short>(bits >> 16)));
                const __half2 second = __halves2half2(
                    __ushort_as_half(static_cast<unsigned short>(bits >> 32)),
                    __ushort_as_half(static_cast<unsigned short>(bits >> 48)));
                const float2 first_float = __half22float2(first), second_float = __half22float2(second);
                w[0] = first_float.x; w[1] = first_float.y;
                w[2] = second_float.x; w[3] = second_float.y;
            } else {
                const volatile uint16_t* scalar_table = table;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    w[j] = __half2float(__ushort_as_half(scalar_table[entry * 4 + j]));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = base + group * 4 + j;
                if (k < K) {
                    if constexpr (Packed) sum = fmaf(w[j], __half2float(shared_x[k]) * scale, sum);
                    else sum = fmaf(w[j], __half2float(x[k]) * scale, sum);
                }
            }
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

struct Shape {
    int K, N;
    const char* name;
};
static Shape shape_from(const std::string& name) {
    if (name == "up") return {5120, 17408, "up"};
    if (name == "down") return {17408, 5120, "down"};
    if (name == "edge") return {513, 37, "edge"};
    throw std::runtime_error("shape must be up, down, or edge");
}
enum class Variant { Scalar, Packed, Q4 };
static Variant variant_from(const std::string& name) {
    if (name == "scalar") return Variant::Scalar;
    if (name == "packed") return Variant::Packed;
    if (name == "q4") return Variant::Q4;
    throw std::runtime_error("variant must be scalar, packed, or q4");
}
static const char* variant_name(Variant variant) {
    switch (variant) {
        case Variant::Scalar: return "lut2_scalar";
        case Variant::Packed: return "lut2_packed";
        case Variant::Q4: return "q4_mlp_auto";
    }
    throw std::runtime_error("invalid variant");
}
static size_t mem_available() {
    std::ifstream input("/proc/meminfo");
    std::string key, unit;
    size_t value;
    while (input >> key >> value >> unit)
        if (key == "MemAvailable:") return value * 1024;
    throw std::runtime_error("cannot read /proc/meminfo MemAvailable; refusing allocation");
}

template<typename T> struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;
    void allocate(size_t n) { count = n; CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void copy(const T* source, size_t n, size_t offset = 0) {
        CUDA_CHECK(cudaMemcpy(ptr + offset, source, n * sizeof(T), cudaMemcpyHostToDevice));
    }
};
struct Dataset {
    Shape shape;
    int index_stride, scale_stride, lut_copies, q4_copies, q4_scale_type;
    size_t lut_index_bytes, lut_scale_count, q4_weight_bytes, q4_scale_count;
    size_t lut_matrix_bytes, q4_matrix_bytes, gpu_bytes, host_bytes;
    std::vector<uint8_t> indices, q4_weights;
    std::vector<__half> scales, codebook, x, q4_scales_half;
    std::vector<int8_t> xq;
    std::vector<float> xs, q4_scales_float;
    DeviceBuffer<uint8_t> d_indices, d_q4_weights;
    DeviceBuffer<__half> d_scales, d_codebook, d_x, d_q4_scales_half;
    DeviceBuffer<int8_t> d_xq;
    DeviceBuffer<float> d_xs, d_q4_scales_float, d_y;

    Dataset(Shape s, bool ring, int scale_type, int l2_bytes, bool exact=false) : shape(s), q4_scale_type(scale_type) {
        index_stride = ((s.K + 15) / 16) * 4;
        scale_stride = (s.K + 31) / 32;
        lut_index_bytes = static_cast<size_t>(s.N) * index_stride;
        lut_scale_count = static_cast<size_t>(s.N) * scale_stride;
        q4_weight_bytes = s.K % 32 == 0 ? static_cast<size_t>(s.N) * s.K / 2 : 0;
        q4_scale_count = s.K % 32 == 0 ? static_cast<size_t>(s.N) * s.K / 32 : 0;
        lut_matrix_bytes = lut_index_bytes + lut_scale_count * sizeof(__half);
        q4_matrix_bytes = q4_weight_bytes + q4_scale_count * (scale_type ? sizeof(__half) : sizeof(float));
        const size_t target = std::max<size_t>(96ULL * 1024 * 1024, static_cast<size_t>(l2_bytes) * 4);
        lut_copies = ring ? static_cast<int>((target + lut_matrix_bytes - 1) / lut_matrix_bytes) : 1;
        q4_copies = q4_matrix_bytes ? (ring ? static_cast<int>((target + q4_matrix_bytes - 1) / q4_matrix_bytes) : 1) : 0;
        const size_t common = 2048 + s.K * sizeof(__half) + (q4_matrix_bytes ? s.K + scale_stride * sizeof(float) : 0);
        gpu_bytes = lut_copies * lut_matrix_bytes + q4_copies * q4_matrix_bytes + common + s.N * sizeof(float);
        // Includes CPU reference/output scratch and one host matrix per format; no full-ring host replicas.
        host_bytes = lut_matrix_bytes + q4_matrix_bytes + common + s.N * (3 * sizeof(float) + 3 * sizeof(double));
        const size_t available = mem_available();
        std::printf("MEMORY shape=%s gpu_buffer_bytes=%zu host_buffer_bytes_upper=%zu total_buffer_bytes_upper=%zu mem_available_bytes=%zu cap_bytes=%zu\n",
                    s.name, gpu_bytes, host_bytes, gpu_bytes + host_bytes, available, BUFFER_CAP);
        if (gpu_bytes + host_bytes > BUFFER_CAP || gpu_bytes + host_bytes > available * 7 / 10)
            throw std::runtime_error("memory budget exceeded before allocation");
        std::printf("FORMAT shape=%s K=%d N=%d seed=%u lut_index_bytes=%zu lut_scale_bytes=%zu codebook_bytes=2048 lut_matrix_bytes=%zu q4_matrix_bytes=%zu q4_scale_dtype=%s lut_ring=%d lut_ring_bytes=%zu q4_ring=%d q4_ring_bytes=%zu l2_bytes=%d ring_target_bytes=%zu\n",
                    s.name,s.K,s.N,SEED,lut_index_bytes,lut_scale_count*2,lut_matrix_bytes,q4_matrix_bytes,
                    scale_type ? "fp16" : "fp32",lut_copies,lut_copies*lut_matrix_bytes,q4_copies,q4_copies*q4_matrix_bytes,l2_bytes,target);
        std::fflush(stdout);
        std::mt19937 random(SEED);
        auto signed_value = [&]() { return (static_cast<int>(random() % 65537) - 32768) / 32768.0f; };
        auto scale_value = [&]() { return 0.125f + (random() % 32769) / 32768.0f; };
        codebook.resize(1024); x.resize(s.K); indices.resize(lut_index_bytes); scales.resize(lut_scale_count);
        for (auto& value : codebook) value = __float2half(signed_value());
        for (auto& value : x) value = __float2half(signed_value());
        for (auto& value : indices) value = static_cast<uint8_t>(random() & 255);
        for (auto& value : scales) value = __float2half(scale_value());
        if(exact) {
            constexpr float weights[]={-2,-1,-0.5f,0,0.5f,1,2};
            constexpr float activations[]={-1,0,1,2};
            std::mt19937 exact_random(SEED);
            for(auto& value:codebook)value=__float2half(weights[exact_random()%7]);
            for(auto& value:x)value=__float2half(activations[exact_random()%4]);
            for(auto& value:scales)value=__float2half(1.0f);
        }
        d_indices.allocate(lut_copies * lut_index_bytes); d_scales.allocate(lut_copies * lut_scale_count);
        d_codebook.allocate(codebook.size()); d_x.allocate(x.size()); d_y.allocate(s.N);
        d_codebook.copy(codebook.data(),codebook.size()); d_x.copy(x.data(),x.size());
        for (int copy=0;copy<lut_copies;++copy) {
            d_indices.copy(indices.data(),indices.size(),copy*lut_index_bytes);
            d_scales.copy(scales.data(),scales.size(),copy*lut_scale_count);
        }
        if (!q4_matrix_bytes) return;
        q4_weights.resize(q4_weight_bytes); xq.resize(s.K); xs.resize(scale_stride);
        for (auto& value:q4_weights) value=static_cast<uint8_t>(random() & 255);
        if (scale_type) { q4_scales_half.resize(q4_scale_count); for(auto& v:q4_scales_half)v=__float2half(scale_value()); }
        else { q4_scales_float.resize(q4_scale_count); for(auto& v:q4_scales_float)v=scale_value(); }
        for(int block=0;block<scale_stride;++block) {
            float maximum=0;
            for(int i=0;i<32;++i)maximum=std::max(maximum,std::fabs(__half2float(x[block*32+i])));
            xs[block]=maximum==0 ? 1.0f : maximum/127.0f;
            for(int i=0;i<32;++i)xq[block*32+i]=static_cast<int8_t>(std::lround(__half2float(x[block*32+i])/xs[block]));
        }
        d_q4_weights.allocate(q4_copies*q4_weight_bytes); d_xq.allocate(xq.size()); d_xs.allocate(xs.size());
        d_xq.copy(xq.data(),xq.size()); d_xs.copy(xs.data(),xs.size());
        if(scale_type)d_q4_scales_half.allocate(q4_copies*q4_scale_count);
        else d_q4_scales_float.allocate(q4_copies*q4_scale_count);
        for(int copy=0;copy<q4_copies;++copy) {
            d_q4_weights.copy(q4_weights.data(),q4_weights.size(),copy*q4_weight_bytes);
            if(scale_type)d_q4_scales_half.copy(q4_scales_half.data(),q4_scale_count,copy*q4_scale_count);
            else d_q4_scales_float.copy(q4_scales_float.data(),q4_scale_count,copy*q4_scale_count);
        }
    }
    void launch(Variant variant, int iteration) {
        if(variant==Variant::Q4) {
            if(!q4_matrix_bytes)throw std::runtime_error("engine q4 launcher requires K divisible by 32");
            const int copy=iteration%q4_copies;
            const void* weight_scales=q4_scale_type ? static_cast<const void*>(d_q4_scales_half.ptr+copy*q4_scale_count)
                                                   : static_cast<const void*>(d_q4_scales_float.ptr+copy*q4_scale_count);
            launch_gemv_w4a8_mlp_auto(d_xq.ptr,d_xs.ptr,nullptr,d_q4_weights.ptr+copy*q4_weight_bytes,
                                    weight_scales,d_y.ptr,shape.K,shape.N,1,q4_scale_type);
        } else {
            const int copy=iteration%lut_copies;
            const int grid=(shape.N+THREADS/32-1)/(THREADS/32);
            if(variant==Variant::Scalar)lut2_gemv<false><<<grid,THREADS>>>(d_indices.ptr+copy*lut_index_bytes,
                d_scales.ptr+copy*lut_scale_count,d_codebook.ptr,d_x.ptr,d_y.ptr,shape.K,shape.N,index_stride,scale_stride);
            else lut2_gemv<true><<<grid,THREADS,shape.K*sizeof(__half)>>>(d_indices.ptr+copy*lut_index_bytes,
                d_scales.ptr+copy*lut_scale_count,d_codebook.ptr,d_x.ptr,d_y.ptr,shape.K,shape.N,index_stride,scale_stride);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<float> output() {
        std::vector<float> result(shape.N);
        CUDA_CHECK(cudaMemcpy(result.data(),d_y.ptr,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
        return result;
    }
};

static long long epoch_ns();
static uint32_t float_bits(float value) {
    uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));return bits;
}
static void check(Dataset& data, bool exact) {
    std::vector<double> reference(data.shape.N,0.0),sum_absolute(data.shape.N,0.0);
    double sum_squares=0;
    for(int row=0;row<data.shape.N;++row) {
        for(int k=0;k<data.shape.K;++k) {
            const int entry=data.indices[static_cast<size_t>(row)*data.index_stride+k/4];
            const double product=static_cast<double>(__half2float(data.codebook[entry*4+k%4]))*
                __half2float(data.scales[static_cast<size_t>(row)*data.scale_stride+k/32])*__half2float(data.x[k]);
            reference[row]+=product;sum_absolute[row]+=std::fabs(product);
        }
        sum_squares+=reference[row]*reference[row];
    }
    const double rms=std::sqrt(sum_squares/data.shape.N);
    for(Variant variant:{Variant::Scalar,Variant::Packed}) {
        const long long begin=epoch_ns();
        data.launch(variant,0);const auto expected_bits=data.output();
        for(int repeat=0;repeat<30;++repeat) {
            data.launch(variant,repeat);const auto observed=data.output();
            if(std::memcmp(expected_bits.data(),observed.data(),observed.size()*sizeof(float))!=0)
                throw std::runtime_error("B-3 bitwise determinism gate failed");
        }
        const long long end=epoch_ns();
        std::vector<double> normalized(data.shape.N);
        double max_absolute=0;int first_failure=-1;int expected_zero=0,actual_negative_zero=0;
        for(int row=0;row<data.shape.N;++row) {
            const float expected=static_cast<float>(reference[row]);
            const double absolute=std::fabs(static_cast<double>(expected_bits[row])-reference[row]);
            normalized[row]=sum_absolute[row]==0 ? (absolute==0 ? 0 : std::numeric_limits<double>::infinity())
                : absolute/sum_absolute[row];
            max_absolute=std::max(max_absolute,absolute);
            if(expected==0)++expected_zero;
            if(expected_bits[row]==0 && std::signbit(expected_bits[row]))++actual_negative_zero;
            const bool pass=std::isfinite(expected_bits[row]) && (exact ? float_bits(expected)==float_bits(expected_bits[row])
                                                                         : normalized[row]<1e-5);
            if(!pass && first_failure<0)first_failure=row;
        }
        std::sort(normalized.begin(),normalized.end());
        // Median averages the middle two for even N; p99 is nearest-rank sorted[ceil(0.99*N)-1].
        const size_t middle=normalized.size()/2;
        const double median=normalized.size()%2 ? normalized[middle] : (normalized[middle-1]+normalized[middle])/2.0;
        const double p99=normalized[static_cast<size_t>(std::ceil(0.99*data.shape.N))-1];
        const double max_over_rms=rms==0 ? (max_absolute==0?0:std::numeric_limits<double>::infinity()) : max_absolute/rms;
        std::printf("CHECK_%s %s shape=%s variant=%s outputs=%d normalized_max=%.17g normalized_median=%.17g normalized_p99=%.17g median_method=middle_mean p99_method=nearest_rank max_absolute=%.17g reference_rms=%.17g max_absolute_over_rms=%.17g deterministic_repeats=30 expected_zero_count=%d actual_negative_zero_count=%d signzero=literal_bits begin_epoch_ns=%lld end_epoch_ns=%lld\n",
                    exact?"EXACT":"ROUND",first_failure<0?"PASS":"FAIL",data.shape.name,variant_name(variant),data.shape.N,
                    normalized.back(),median,p99,max_absolute,rms,max_over_rms,expected_zero,actual_negative_zero,begin,end);
        if(first_failure>=0) {
            const int row=first_failure;
            std::printf("FIRST_MISMATCH row=%d reference_double=%.17g expected_float=%.9g actual=%.9g expected_bits=0x%08x actual_bits=0x%08x sum_absolute=%.17g\n",
                        row,reference[row],static_cast<float>(reference[row]),expected_bits[row],
                        float_bits(static_cast<float>(reference[row])),float_bits(expected_bits[row]),sum_absolute[row]);
            throw std::runtime_error(exact ? "B-3a exact bit gate failed" : "B-3b normalized max error gate failed");
        }
    }
}
static long long epoch_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
static int warm(Dataset& data, Variant variant) {
    const auto start=std::chrono::steady_clock::now();
    const long long begin=epoch_ns();
    int launches=0;
    do {
        for(int i=0;i<REPEATS;++i)data.launch(variant,launches++);
        CUDA_CHECK(cudaDeviceSynchronize());
    } while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<10.0);
    std::printf("WARM shape=%s variant=%s launches=%d begin_epoch_ns=%lld end_epoch_ns=%lld elapsed_s=%.17g\n",
                data.shape.name,variant_name(variant),launches,begin,epoch_ns(),
                std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    std::fflush(stdout);
    return launches;
}
static double timed_trial(Dataset& data, Variant variant, int trial, const char* comparison) {
    const int warm_launches=warm(data,variant);
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    // Host bounds include inter-batch synchronization; event sums exclude those gaps.
    const long long begin=epoch_ns();
    int launches=0,batches=0;
    double total_ms=0;
    do {
        CUDA_CHECK(cudaEventRecord(start));
        for(int iteration=0;iteration<REPEATS;++iteration)
            data.launch(variant,warm_launches+launches++);
        CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
        float batch_ms=0; CUDA_CHECK(cudaEventElapsedTime(&batch_ms,start,stop));
        total_ms+=static_cast<double>(batch_ms);
        ++batches;
    } while(total_ms<1000.0);
    const long long end=epoch_ns();
    const double us=total_ms*1000.0/launches;
    // Logical unique bytes, not actual DRAM traffic; ncu supplies actual traffic.
    const size_t read_bytes=variant==Variant::Q4 ? data.q4_matrix_bytes+data.shape.K+data.scale_stride*sizeof(float)
        : data.lut_matrix_bytes+2048+data.shape.K*sizeof(__half);
    std::printf("TRIAL shape=%s comparison=%s variant=%s trial=%d iterations=%d total_launches=%d batches=%d batch_launches=100 event_ms=%.17g mean_us=%.17g read_bytes=%zu effective_GBps=%.17g begin_epoch_ns=%lld end_epoch_ns=%lld\n",
                data.shape.name,comparison,variant_name(variant),trial+1,launches,launches,batches,total_ms,us,read_bytes,read_bytes/(us*1000.0),begin,end);
    std::fflush(stdout);
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return us;
}
static void benchmark(Dataset& data, Variant variant) {
    double best=std::numeric_limits<double>::infinity();
    for(int trial=0;trial<TRIALS;++trial)best=std::min(best,timed_trial(data,variant,trial,"standalone"));
    std::printf("BEST shape=%s variant=%s trials=5 batch_launches=100 minimum_event_ms=1000 mean_us=%.17g\n",data.shape.name,variant_name(variant),best);
}
static void paired_benchmark(Dataset& data, Variant variant) {
    double best_lut=std::numeric_limits<double>::infinity(),best_q4=best_lut;
    for(int pair=0;pair<TRIALS;++pair) {
        const double lut=timed_trial(data,variant,pair,variant_name(variant));
        const double q4=timed_trial(data,Variant::Q4,pair,variant_name(variant));
        best_lut=std::min(best_lut,lut);best_q4=std::min(best_q4,q4);
        std::printf("PAIR shape=%s variant=%s pair=%d lut_mean_us=%.17g q4_mean_us=%.17g q4_over_lut=%.17g\n",
                    data.shape.name,variant_name(variant),pair+1,lut,q4,q4/lut);
        std::fflush(stdout);
    }
    std::printf("PAIRED_BEST shape=%s variant=%s pairs=5 batch_launches=100 minimum_event_ms=1000 lut_mean_us=%.17g q4_mean_us=%.17g\n",
                data.shape.name,variant_name(variant),best_lut,best_q4);
}
static void usage() {
    std::puts("probe_gemv_lut2 <check|check-exact|check-round|warm|bench|profile> [up|down|edge|all] [scalar|packed|q4|all] [fp32|fp16]\n"
              "check: exact then original random; check-exact: bit equality; check-round: abs_error/sum_abs_products <1e-5. Fixed seed20261024;30 bitwise repeats.\n"
              "warm/bench default: up/down, all variants; check MUST pass first. profile requires one shape/variant.\n"
              "bench packed: packed,q4 alternating 5 pairs; packed stages unchanged fp16 activations in shared memory.\n"
              "bench all: scalar,q4 alternating 5 pairs, then packed,q4 alternating 5 pairs; every trial warms >=10s, then repeats 100-launch batches until summed CUDA event time >=1000ms.\n"
              "TRIAL epoch bounds are host envelopes. External telemetry median is N/A if no sample lies inside.\n"
              "q4 defaults to fp32 weight scales (5 bits/weight); fp16 = 4.5 bits. LUT2 = 2.5 bits + 2048-byte codebook.\n"
              "Timing excludes allocation, transfers and q4 activation quantization. No model-quality equivalence is claimed.\n"
              "Before each run caller must check free -h, resident GPU apps, and start gpumon/apps logs.\n"
              "Condition: ComfyUI resident (idle, 3.2GB); benchmark validity is decided from external monitor logs.");
}
int main(int argc,char** argv) {
    try {
        if(argc<2 || std::string(argv[1])=="--help") {usage();return argc<2?2:0;}
        const std::string mode=argv[1], shape_arg=argc>2?argv[2]:"all", variant_arg=argc>3?argv[3]:"all";
        const std::string scale_arg=argc>4?argv[4]:"fp32";
        const bool checking=mode=="check" || mode=="check-exact" || mode=="check-round";
        if(argc>5 || (!checking && mode!="warm" && mode!="bench" && mode!="profile"))throw std::runtime_error("invalid arguments");
        if(scale_arg!="fp32" && scale_arg!="fp16")throw std::runtime_error("scale must be fp32 or fp16");
        const char* mlp_u=std::getenv("MLP_U");
        if(mlp_u && std::string(mlp_u)!="2")throw std::runtime_error("MLP_U must be unset or 2; refusing altered comparison");
        if(mode=="profile" && (shape_arg=="all" || variant_arg=="all"))throw std::runtime_error("profile requires one shape and one variant");
        std::vector<Shape> shapes;
        if(shape_arg=="all") {shapes={shape_from("up"),shape_from("down")};if(checking)shapes.push_back(shape_from("edge"));}
        else shapes.push_back(shape_from(shape_arg));
        std::vector<Variant> variants=variant_arg=="all" ? std::vector<Variant>{Variant::Scalar,Variant::Packed,Variant::Q4}
                                                       : std::vector<Variant>{variant_from(variant_arg)};
        cudaDeviceProp properties{}; CUDA_CHECK(cudaGetDeviceProperties(&properties,0));
        std::printf("DEVICE name=%s cc=%d.%d l2_bytes=%d mode=%s q4_scale=%s MLP_U=2 seed=%u\n",
                    properties.name,properties.major,properties.minor,properties.l2CacheSize,mode.c_str(),scale_arg.c_str(),SEED);
        for(const Shape shape:shapes) {
            if(shape.K%32 && !checking)throw std::runtime_error("edge shape is correctness-only");
            if(checking) {
                if(mode!="check-round") {
                    Dataset exact_data(shape,false,scale_arg=="fp16",properties.l2CacheSize,true);
                    check(exact_data,true);
                }
                if(mode!="check-exact") {
                    Dataset random_data(shape,false,scale_arg=="fp16",properties.l2CacheSize);
                    check(random_data,false);
                }
                continue;
            }
            Dataset data(shape,true,scale_arg=="fp16",properties.l2CacheSize);
            if(mode=="bench" && variant_arg=="all") {
                paired_benchmark(data,Variant::Scalar);
                paired_benchmark(data,Variant::Packed);
            }
            else if(mode=="bench" && variant_arg=="packed")paired_benchmark(data,Variant::Packed);
            else for(Variant variant:variants) {
                if(mode=="bench")benchmark(data,variant);
                else if(mode=="warm")warm(data,variant);
                else {
                    const int count=mode=="profile"?1:REPEATS;
                    for(int i=0;i<count;++i)data.launch(variant,i);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    std::printf("%s_DONE shape=%s variant=%s launches=%d\n",mode.c_str(),shape.name,variant_name(variant),count);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::puts("PROBE PASS");return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"PROBE FAIL: %s\n",error.what());return 1;
    }
}
