#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../kernels/attention_flash.cuh"
#ifndef R09_CAPTURE_ONLY
#include "../kernels/attention.cuh"
#endif

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA_ERROR line=%d %s\n", __LINE__, cudaGetErrorString(e)); exit(2); } } while (0)

static void require(bool condition, const char *message) {
    if (!condition) { fprintf(stderr, "INPUT_ERROR %s\n", message); exit(2); }
}

template<typename T> struct Device {
    T *data = nullptr;
    explicit Device(size_t count) { CK(cudaMalloc(&data, count * sizeof(T))); }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    void put(const std::vector<T> &values) {
        CK(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

static float random_value(uint32_t &state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
}

static std::vector<float> download(const float *device, size_t count) {
    std::vector<float> values(count);
    CK(cudaMemcpy(values.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost));
    return values;
}

static double epoch() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

struct Shape { int hd, nq, nkv, sq, base; };
struct Inputs {
    std::vector<float> q, k, v;
    std::vector<float> rounded_q, rounded_k, rounded_v;
    std::vector<__nv_bfloat16> bf_k, bf_v;
    explicit Inputs(const Shape &s) {
        q.resize(static_cast<size_t>(s.sq) * s.nq * s.hd);
        k.resize(static_cast<size_t>(s.base + s.sq) * s.nkv * s.hd);
        v.resize(k.size());
        uint32_t seed = 909u + s.hd * 17u + s.nkv * 31u + s.sq * 43u + s.base * 59u;
        for (auto *values : {&q, &k, &v}) for (float &value : *values) value = random_value(seed);
        rounded_q.resize(q.size()); rounded_k.resize(k.size()); rounded_v.resize(v.size());
        bf_k.resize(k.size()); bf_v.resize(v.size());
        for (size_t i = 0; i < q.size(); ++i) rounded_q[i] = __bfloat162float(__float2bfloat16(q[i]));
        for (size_t i = 0; i < k.size(); ++i) {
            bf_k[i] = __float2bfloat16(k[i]); bf_v[i] = __float2bfloat16(v[i]);
            rounded_k[i] = __bfloat162float(bf_k[i]); rounded_v[i] = __bfloat162float(bf_v[i]);
        }
    }
};

static void write_binary(FILE *file, const void *data, size_t size) {
    require(fwrite(data, 1, size, file) == size, "capture write failed");
}

// This mode calls only the original API and can be compiled against the pre-change header.
static int capture_defaults(const std::string &path) {
    require(!std::filesystem::exists(path), "capture output already exists");
    FILE *file = fopen(path.c_str(), "wb"); require(file != nullptr, "cannot create capture output");
    const uint32_t magic[] = {0x52303946u, 1u}; write_binary(file, magic, sizeof(magic));
    int count = 0;
    // Largest HD first also works with the old header's HD-unaware scratch capacity.
    for (int hd : {256, 128}) {
        for (int sq : {1, 7, 16, 37}) {
            for (int base : {0, 300, 777}) {
                const Shape s{hd, 16, 8, sq, base}; Inputs input(s);
                Device<float> q(input.q.size()), out(input.q.size());
                Device<__nv_bfloat16> k(input.k.size()), v(input.v.size());
                q.put(input.q); k.put(input.bf_k); v.put(input.bf_v);
                launch_flash_attn(q.data, k.data, v.data, out.data, sq, 16, 8, hd, base);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                const auto values = download(out.data, input.q.size());
                const uint32_t header[] = {static_cast<uint32_t>(hd), 16, 8,
                    static_cast<uint32_t>(sq), static_cast<uint32_t>(base), 0};
                write_binary(file, header, sizeof(header));
                write_binary(file, values.data(), values.size() * sizeof(float));
                printf("CAPTURE_DEFAULT hd=%d sq=%d base=%d tree=0 values=%zu\n", hd, sq, base, values.size());
                ++count;
            }
        }
        const Shape s{hd, 16, 1, 4, 300}; Inputs input(s);
        Device<float> q(input.q.size()), out(input.q.size());
        Device<__nv_bfloat16> k(input.k.size()), v(input.v.size());
        Device<unsigned> ancestor(4);
        q.put(input.q); k.put(input.bf_k); v.put(input.bf_v);
        for (const std::vector<unsigned> mask : {std::vector<unsigned>{1, 3, 7, 15}, {1, 3, 5, 11}}) {
            ancestor.put(mask);
            launch_flash_attn(q.data, k.data, v.data, out.data, 4, 16, 1, hd, 300, 0, ancestor.data);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            const auto values = download(out.data, input.q.size());
            const uint32_t header[] = {static_cast<uint32_t>(hd), 16, 1, 4, 300, mask[2]};
            write_binary(file, header, sizeof(header));
            write_binary(file, values.data(), values.size() * sizeof(float));
            printf("CAPTURE_DEFAULT hd=%d sq=4 base=300 tree=%u values=%zu\n", hd, mask[2], values.size());
            ++count;
        }
    }
    require(fclose(file) == 0, "capture close failed");
    printf("RESULT capture PASS cases=%d file=%s\n", count, path.c_str());
    return 0;
}

#ifndef R09_CAPTURE_ONLY
// Each worker computes complete rows in a fixed scalar summation order. No CPU work overlaps GPU calls.
static std::vector<double> reference(const Inputs &input, const Shape &s, float requested_scale,
                                     int window, int threads) {
    const double scale = requested_scale > 0 ? requested_scale : 1.0 / std::sqrt(static_cast<double>(s.hd));
    std::vector<double> result(input.q.size());
    std::atomic<int> next{0};
    auto worker = [&]() {
        std::vector<double> scores(s.base + s.sq), accumulator(s.hd);
        for (int row = next.fetch_add(1); row < s.sq * s.nq; row = next.fetch_add(1)) {
            const int qi = row / s.nq, head = row % s.nq, kv = head / (s.nq / s.nkv);
            const int hi = s.base + qi + 1;
            const int lo = window > 0 ? std::max(0, hi - window) : 0;
            const float *q = input.rounded_q.data() + static_cast<size_t>(row) * s.hd;
            double maximum = -std::numeric_limits<double>::infinity();
            for (int j = lo; j < hi; ++j) {
                const float *k = input.rounded_k.data() + (static_cast<size_t>(j) * s.nkv + kv) * s.hd;
                double dot = 0;
                for (int d = 0; d < s.hd; ++d) dot += static_cast<double>(q[d]) * k[d];
                scores[j] = dot * scale; maximum = std::max(maximum, scores[j]);
            }
            double sum = 0;
            for (int j = lo; j < hi; ++j) { scores[j] = std::exp(scores[j] - maximum); sum += scores[j]; }
            std::fill(accumulator.begin(), accumulator.end(), 0.0);
            for (int j = lo; j < hi; ++j) {
                const float *v = input.rounded_v.data() + (static_cast<size_t>(j) * s.nkv + kv) * s.hd;
                for (int d = 0; d < s.hd; ++d) accumulator[d] += scores[j] * v[d];
            }
            for (int d = 0; d < s.hd; ++d) result[static_cast<size_t>(row) * s.hd + d] = accumulator[d] / sum;
        }
    };
    std::vector<std::thread> workers;
    for (int i = 0; i < threads; ++i) workers.emplace_back(worker);
    for (auto &thread : workers) thread.join();
    return result;
}

struct Metrics { double maximum = 0, mean = 0, maxabs = 0, denominator = 0; size_t worst = 0; bool finite = true; };
template<typename T> static Metrics metrics(const std::vector<float> &actual, const std::vector<T> &ref) {
    require(actual.size() == ref.size(), "comparison shape mismatch"); Metrics m;
    for (T value : ref) { m.finite = m.finite && std::isfinite(value); m.denominator = std::max(m.denominator, std::abs(static_cast<double>(value))); }
    double sum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        m.finite = m.finite && std::isfinite(actual[i]);
        const double error = std::abs(static_cast<double>(actual[i]) - ref[i]);
        sum += error;
        if (error > m.maxabs) { m.maxabs = error; m.worst = i; }
    }
    if (m.denominator == 0) {
        m.maximum = m.mean = m.maxabs == 0 ? 0 : std::numeric_limits<double>::infinity();
    } else { m.maximum = m.maxabs / m.denominator; m.mean = sum / actual.size() / m.denominator; }
    return m;
}

static int execute_case(int index, const Shape &s, const Inputs &input, const std::vector<double> &ref,
                        float scale, int window, bool fp32_kv, const char *label) {
    Device<float> q(input.q.size()), k(input.k.size()), v(input.v.size()), out(input.q.size()), old_out(input.q.size());
    Device<__nv_bfloat16> bk(input.k.size()), bv(input.v.size());
    q.put(input.q); k.put(input.k); v.put(input.v); bk.put(input.bf_k); bv.put(input.bf_v);
    auto launch = [&]() {
        if (fp32_kv) launch_flash_attn(q.data, k.data, v.data, out.data, s.sq, s.nq, s.nkv, s.hd, s.base, 0, nullptr, scale, window);
        else launch_flash_attn(q.data, bk.data, bv.data, out.data, s.sq, s.nq, s.nkv, s.hd, s.base, 0, nullptr, scale, window);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    };
    CK(cudaDeviceSynchronize()); const double start = epoch(); launch();
    const auto actual = download(out.data, input.q.size()); const Metrics error = metrics(actual, ref);
    bool deterministic = true; int bad_repeat = -1; size_t bad_index = 0;
    for (int repeat = 1; repeat < 30; ++repeat) {
        launch(); const auto next = download(out.data, input.q.size());
        if (memcmp(actual.data(), next.data(), actual.size() * sizeof(float)) != 0) {
            deterministic = false; bad_repeat = repeat;
            while (bad_index < actual.size() && memcmp(&actual[bad_index], &next[bad_index], sizeof(float)) == 0) ++bad_index;
            break;
        }
    }
    const double end = epoch();
    launch_attn_decode(q.data, k.data, v.data, old_out.data, s.sq, s.nq, s.nkv, s.hd, s.base, 0, scale, window);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const Metrics old_error = metrics(actual, download(old_out.data, input.q.size()));
    const bool pass = error.finite && error.maximum < 1e-2 && error.mean < 1e-3 && deterministic;
    printf("A4_INTERVAL case=%d start=%.17g end=%.17g scope=flash_30_runs_and_output_copies\n", index, start, end);
    printf("A4_CASE index=%d label=%s hd=%d nq=%d nkv=%d sq=%d base=%d window=%d scale=%.9g kv=%s max_error=%.17g mean_error=%.17g maxabs=%.17g ref_maxabs=%.17g finite=%d deterministic=%d runs=%d old_fp32_max_error=%.17g old_fp32_mean_error=%.17g result=%s\n",
           index, label, s.hd, s.nq, s.nkv, s.sq, s.base, window, scale, fp32_kv ? "float" : "bf16",
           error.maximum, error.mean, error.maxabs, error.denominator, error.finite, deterministic,
           deterministic ? 30 : bad_repeat + 1, old_error.maximum, old_error.mean, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("A4_FAILURE index=%d flat=%zu query=%zu head=%zu dimension=%zu expected=%.17g actual=%.17g determinism_repeat=%d determinism_flat=%zu\n",
               index, error.worst, error.worst / (s.nq * s.hd), error.worst / s.hd % s.nq,
               error.worst % s.hd, ref[error.worst], static_cast<double>(actual[error.worst]), bad_repeat, bad_index);
    }
    fflush(stdout); return pass ? 0 : 1;
}

static int positive(const char *text, bool zero = false) {
    char *end = nullptr; long value = strtol(text, &end, 10);
    require(*text && end && *end == 0 && value >= (zero ? 0 : 1) && value <= 100000, "invalid integer argument");
    return static_cast<int>(value);
}

static int accuracy(int first, int limit, int threads) {
    int index = 0, executed = 0;
    // Alternate HD at identical/decreasing scratch needs before the complete Cartesian sweep.
    if (first == 0) {
        for (const Shape s : {Shape{256,16,8,37,300}, Shape{512,16,8,37,300},
                              Shape{256,16,8,16,0}, Shape{512,16,8,16,0}}) {
            Inputs input(s); auto ref = reference(input, s, 0.0625f, 256, threads);
            if (execute_case(-1, s, input, ref, 0.0625f, 256, true, "scratch_transition")) return 1;
        }
    }
    for (int sq : {16,37,512,1024}) for (int base : {0,300}) for (int nkv : {8,1}) {
        for (int window : {0,1024,256}) for (float scale : {0.0f,0.0625f}) for (int hd : {256,512}) {
            const Shape s{hd,16,nkv,sq,base};
            const bool selected = index + 1 >= first && index < first + limit;
            std::vector<double> ref;
            if (selected) {
                Inputs input(s);
                printf("A4_REFERENCE first_case=%d hd=%d sq=%d base=%d nkv=%d window=%d scale=%.9g phase=CPU_only\n", index, hd, sq, base, nkv, window, scale);
                ref = reference(input, s, scale, window, threads);
                for (bool fp32 : {true,false}) {
                    if (index >= first && index < first + limit) {
                        if (execute_case(index, s, input, ref, scale, window, fp32, "cartesian")) return 1;
                        ++executed;
                    }
                    ++index;
                }
            } else index += 2;
        }
    }
    require(first < index && executed > 0, "empty case range");
    printf("RESULT A4 PASS executed=%d total_cartesian=%d first=%d limit=%d threads=%d full_sweep=%d\n", executed, index, first, limit, threads, first == 0 && executed == index);
    return 0;
}
#endif

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc == 3 && std::string(argv[1]) == "--capture-default") return capture_defaults(argv[2]);
#ifdef R09_CAPTURE_ONLY
    fprintf(stderr, "usage: %s --capture-default NEW_OUTPUT.bin\n", argv[0]); return 2;
#else
    int first = 0, count = 384, threads = 8;
    for (int i = 1; i < argc; i += 2) {
        require(i + 1 < argc, "argument value missing");
        const std::string key = argv[i];
        if (key == "--first") first = positive(argv[i + 1], true);
        else if (key == "--count") count = positive(argv[i + 1]);
        else if (key == "--threads") { threads = positive(argv[i + 1]); require(threads <= 8, "CPU thread limit is eight"); }
        else { fprintf(stderr, "unknown argument %s\n", key.c_str()); return 2; }
    }
    return accuracy(first, count, threads);
#endif
}
