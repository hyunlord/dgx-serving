// R18 llama.cpp reference: ggml_mul_mat(q4_0 weight [K x N], f32 activations [K x n]) on the CUDA backend at the
// 12B dense linear shapes, n in {1, 2, 4, 8}. Uses only the public ggml API of an existing llama.cpp build (no
// llama.cpp source is copied). One graph holds `copies` independent mul_mats on distinct weight copies sharing
// the activations (copies >= 3 x L2, like tools/r18_gemv_bench.cu), so each op reads its weight from DRAM; the
// time per op therefore includes llama's own per-op activation quantization (q8_1), as in real decode.
// Per (shape, n, pair): one >= 1 s interval of repeated graph computes after a 10 s warm-up per shape.
//   R18_L shape=KxN n= pair= us_per_op= graphs= copies= t0_ms= t1_ms=
// Build (DGX): g++ -O2 -std=c++17 tools/r18_llama_mm.cpp -I$L/ggml/include -L$L/build/bin -lggml -lggml-base
//              -lggml-cuda -Wl,-rpath,$L/build/bin -o artifacts/r18/r18_llama_mm      (L = llama.cpp checkout)
// Usage: r18_llama_mm <n list comma> <pairs> [shapes KxN,...]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

struct Shape { int K, N; const char *role; };
static const Shape SHAPES[] = {
    {3840, 4096, "sliding_q"}, {3840, 2048, "sliding_kv"}, {4096, 3840, "sliding_o"},
    {3840, 8192, "full_q"}, {3840, 512, "full_k"}, {8192, 3840, "full_o"},
    {3840, 15360, "gate_up"}, {15360, 3840, "down"}, {3840, 262144, "lm_head"},
};
static std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> v; size_t p = 0;
    while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) v.push_back(s.substr(p, q - p)); p = q + 1; }
    return v;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 3) { fprintf(stderr, "usage: %s <n list> <pairs> [shapes]\n", argv[0]); return 1; }
    std::vector<int> ns; for (auto &s : split(argv[1])) ns.push_back(atoi(s.c_str()));
    const int pairs = atoi(argv[2]);
    std::vector<std::string> filt; if (argc > 3) filt = split(argv[3]);
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "cuda backend init failed\n"); return 1; }
    for (const Shape &sh : SHAPES) {
        char name[32]; snprintf(name, sizeof name, "%dx%d", sh.K, sh.N);
        if (!filt.empty()) { bool ok = false; for (auto &f : filt) ok |= f == name; if (!ok) continue; }
        const size_t wbytes = (size_t)sh.K / 32 * 18 * sh.N;   // q4_0: fp16 d + 16 bytes per 32 weights
        int copies = (int)((3 * (24u << 20) + wbytes - 1) / wbytes); if (copies < 1) copies = 1;
        printf("R18_LSHAPE shape=%s role=%s copies=%d weight_bytes=%zu\n", name, sh.role, copies, wbytes);
        // weights: one context/buffer for all copies, filled with random bytes; fp16 d forced to a small value
        ggml_init_params wp = { ggml_tensor_overhead() * (copies + 4), nullptr, true };
        ggml_context *wctx = ggml_init(wp);
        std::vector<ggml_tensor *> W;
        for (int c = 0; c < copies; c++) W.push_back(ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, sh.K, sh.N));
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        {
            std::vector<uint8_t> h(wbytes); uint32_t r = 12345;
            for (int c = 0; c < copies; c++) {
                for (size_t i = 0; i < wbytes; i++) { r = r * 1664525u + 1013904223u; h[i] = (uint8_t)(r >> 24); }
                for (size_t b = 0; b < wbytes; b += 18) { const uint16_t d = 0x1c00 + (h[b + 2] & 0x3ff) % 0x200; memcpy(&h[b], &d, 2); }  // ~0.004..0.012
                ggml_backend_tensor_set(W[c], h.data(), 0, wbytes);
            }
        }
        for (int n : ns) {
            ggml_init_params gp = { ggml_tensor_overhead() * (2 * copies + 8) + ggml_graph_overhead_custom(4 * copies + 16, false), nullptr, true };
            ggml_context *gctx = ggml_init(gp);
            ggml_tensor *x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, sh.K, n);
            ggml_cgraph *g = ggml_new_graph_custom(gctx, 4 * copies + 16, false);
            for (int c = 0; c < copies; c++) ggml_build_forward_expand(g, ggml_mul_mat(gctx, W[c], x));
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            ggml_gallocr_alloc_graph(ga, g);
            { std::vector<float> hx((size_t)sh.K * n); uint32_t r = 7;
              for (auto &v : hx) { r = r * 1664525u + 1013904223u; v = ((float)(r >> 8) / 16777216.f - 0.5f) * 2.f; }
              ggml_backend_tensor_set(x, hx.data(), 0, hx.size() * 4); }
            auto run = [&](double target_ms, long &graphs, double &t0, double &t1) {
                long k = 1;
                for (;;) {
                    ggml_backend_synchronize(be); t0 = now_ms();
                    for (long i = 0; i < k; i++) ggml_backend_graph_compute(be, g);
                    ggml_backend_synchronize(be); t1 = now_ms();
                    if (t1 - t0 >= target_ms) break;
                    k = (long)(k * target_ms * 1.1 / ((t1 - t0) > 0.01 ? (t1 - t0) : 0.01)) + 1;
                }
                graphs = k; return (t1 - t0) * 1000.0 / ((double)k * copies);
            };
            long gr; double t0, t1;
            if (n == ns.front()) { const double wend = now_ms() + 10000.0; while (now_ms() < wend) run(500.0, gr, t0, t1); }
            else run(1000.0, gr, t0, t1);
            for (int p = 1; p <= pairs; p++) {
                const double us = run(1000.0, gr, t0, t1);
                printf("R18_L shape=%s role=%s n=%d pair=%d us_per_op=%.4f graphs=%ld copies=%d t0_ms=%.1f t1_ms=%.1f\n",
                       name, sh.role, n, p, us, gr, copies, t0, t1);
            }
            ggml_gallocr_free(ga); ggml_free(gctx);
        }
        ggml_backend_buffer_free(wbuf); ggml_free(wctx);
    }
    ggml_backend_free(be);
    return 0;
}
