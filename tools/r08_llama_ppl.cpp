// R08 measurement-only masked next-token PPL. Tokenization is deliberately absent.
// Build against the pinned llama.cpp public API; do not copy engine/model code.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static double nll(const float * logits, int32_t vocab, int32_t target) {
    if (!logits || vocab <= 0 || target < 0 || target >= vocab) {
        throw std::runtime_error("Invalid logits or target");
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (int32_t i = 0; i < vocab; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("Nonfinite logits");
        maximum = std::max(maximum, static_cast<double>(logits[i]));
    }
    double sum = 0;
    for (int32_t i = 0; i < vocab; ++i) sum += std::exp(static_cast<double>(logits[i]) - maximum);
    return maximum + std::log(sum) - static_cast<double>(logits[target]);
}

#ifdef R08_NLL_SELF_TEST
int main() {
    const float flat[] = {0, 0, 0};
    const float extreme[] = {1000, 999, -1000};
    const double a = nll(flat, 3, 2), b = nll(extreme, 3, 1);
    if (std::abs(a - std::log(3.0)) > 1e-12 || std::abs(b - (1 + std::log1p(std::exp(-1.0)))) > 1e-12) return 1;
    bool rejected = false;
    try { nll(flat, 3, 3); } catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) return 1;
    std::cout << std::setprecision(17) << "{\"flat_nll\":" << a << ",\"extreme_nll\":" << b << ",\"invalid_target_rejected\":true}\n";
}
#else
#include "llama.h"
#include "ggml-backend.h"

template <typename T> static std::vector<T> read_array(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open input: " + path);
    const auto size = file.tellg();
    if (size <= 0 || static_cast<uint64_t>(size) % sizeof(T) != 0 || size > 16 * 1024 * 1024) {
        throw std::runtime_error("Invalid or excessive input size: " + path);
    }
    std::vector<T> values(static_cast<size_t>(size) / sizeof(T));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(values.data()), size)) throw std::runtime_error("Short read: " + path);
    return values;
}

static std::string quote(const std::string & value) {
    std::string output = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { output += '\\'; output += static_cast<char>(c); }
        else if (c < 0x20) throw std::runtime_error("Control character in path");
        else output += static_cast<char>(c);
    }
    return output + '"';
}

static double epoch() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

struct Batch {
    llama_batch value;
    explicit Batch(int32_t length) : value(llama_batch_init(length, 0, 1)) {
        if (!value.token || !value.pos || !value.n_seq_id || !value.seq_id || !value.logits) {
            throw std::runtime_error("Batch allocation failed");
        }
    }
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
};

int main(int argc, char ** argv) {
    try {
        std::string weights, token_path, length_path, mask_path;
        for (int i = 1; i < argc; ++i) {
            if (i + 1 >= argc) throw std::runtime_error("Every option requires a value");
            const std::string flag = argv[i++];
            const std::string value = argv[i];
            if (flag == "--weights") { if (!weights.empty()) throw std::runtime_error("Duplicate weights"); weights = value; }
            else if (flag == "--tokens") { if (!token_path.empty()) throw std::runtime_error("Duplicate tokens"); token_path = value; }
            else if (flag == "--lengths") { if (!length_path.empty()) throw std::runtime_error("Duplicate lengths"); length_path = value; }
            else if (flag == "--mask") { if (!mask_path.empty()) throw std::runtime_error("Duplicate mask"); mask_path = value; }
            else throw std::runtime_error("Unknown option: " + flag);
        }
        if (weights.empty() || token_path.empty() || length_path.empty() || mask_path.empty()) {
            throw std::runtime_error("Required: --weights model.gguf --tokens ids.bin --lengths lengths.bin --mask mask.bin");
        }
        const uint32_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1) throw std::runtime_error("Little-endian host required");
        const auto tokens = read_array<int32_t>(token_path);
        const auto lengths = read_array<int32_t>(length_path);
        const auto mask = read_array<uint8_t>(mask_path);
        if (mask.size() != tokens.size()) throw std::runtime_error("Mask/token size mismatch");
        size_t offset = 0, expected_targets = 0;
        for (int32_t length : lengths) {
            if (length < 2 || length > 1024 || offset + static_cast<size_t>(length) > tokens.size()) {
                throw std::runtime_error("Invalid conversation length");
            }
            if (mask[offset] != 0) throw std::runtime_error("First-token mask must be zero");
            size_t scored = 0;
            for (int32_t j = 0; j < length; ++j) {
                if (mask[offset + j] > 1 || tokens[offset + j] < 0) throw std::runtime_error("Invalid mask/token");
                scored += mask[offset + j];
            }
            if (scored == 0) throw std::runtime_error("Conversation has no scored targets");
            expected_targets += scored;
            offset += static_cast<size_t>(length);
        }
        if (offset != tokens.size()) throw std::runtime_error("Lengths/token size mismatch");
        std::cout << std::setprecision(17);
        std::cerr << "R08_PPL_CONFIG ctx=1024 batch=1024 ubatch=512 n_gpu_layers=99 FA=on KV=f16 threads=8 logits=all no_tokenization=1\n";
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(weights.c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("Model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 512; cp.n_seq_max = 1;
        cp.n_threads = 8; cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("Context creation failed");
        const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (int32_t token : tokens) if (token >= vocab) throw std::runtime_error("Token outside vocabulary");
        Batch batch(1024);
        offset = 0;
        double total = 0;
        size_t count = 0;
        std::vector<double> means;
        for (size_t window = 0; window < lengths.size(); ++window) {
            const int32_t length = lengths[window];
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            batch.value.n_tokens = length;
            for (int32_t t = 0; t < length; ++t) {
                batch.value.token[t] = tokens[offset + t];
                batch.value.pos[t] = t;
                batch.value.n_seq_id[t] = 1;
                batch.value.seq_id[t][0] = 0;
                batch.value.logits[t] = 1;
            }
            llama_synchronize(ctx.get());
            const double start = epoch();
            const int32_t rc = llama_decode(ctx.get(), batch.value);
            if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
            llama_synchronize(ctx.get());
            const double end = epoch();
            std::cout << "PPL_INTERVAL window=" << window << " start=" << start << " end=" << end << " scope=independent_forward\n";
            double subtotal = 0;
            size_t n = 0;
            for (int32_t t = 0; t + 1 < length; ++t) {
                if (mask[offset + t + 1] == 0) continue;
                subtotal += nll(llama_get_logits_ith(ctx.get(), t), vocab, tokens[offset + t + 1]);
                ++n;
            }
            means.push_back(subtotal / static_cast<double>(n));
            std::cout << "{\"window\":" << window << ",\"length\":" << length << ",\"ntok\":" << n
                      << ",\"nll_sum\":" << subtotal << ",\"mean_nll\":" << means.back() << "}\n" << std::flush;
            total += subtotal; count += n; offset += static_cast<size_t>(length);
        }
        if (count != expected_targets) throw std::runtime_error("Scored target count mismatch");
        const double mean = total / static_cast<double>(count);
        std::cout << "{\"model\":" << quote(weights) << ",\"weights\":" << quote(weights)
                  << ",\"path\":\"LLAMA_QAT_Q4_0_FA_ON\",\"scale_dtype\":\"GGUF\",\"nwin\":" << lengths.size()
                  << ",\"ntok\":" << count << ",\"mean_nll\":" << mean << ",\"ppl\":" << std::exp(mean)
                  << ",\"per_window_nll\":[";
        for (size_t i = 0; i < means.size(); ++i) { if (i) std::cout << ','; std::cout << means[i]; }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "R08_PPL_ERROR " << error.what() << '\n';
        return 1;
    }
}
#endif
