// R18 isolated multi-column W4A8 GEMV harness (12B dense linear shapes, synthetic weights of real size).
//   Families ("fam"): cur = current engine kernels (S=1 plain gemv_w4a8, S>=2 gemv_w4a8_sb).
//   Further families are added by kernels/gemv_mc.cuh (R18 PART B) through fam_launch().
// Modes:
//   time <fams> <S list> <pairs>   per shape: 10 s warm-up once, then for each pair, S and family
//                                  (family order alternates per pair) a >= 1 s interval of back-to-back
//                                  launches cycling weight copies (copies >= 3 x L2 so the weights are not
//                                  L2 resident; L2 is never flushed explicitly). Prints
//                                  R18_T shape=KxN S= fam= pair= us= iters= t0_ms= t1_ms= (epoch ms)
//   ncu <fams> <S list>            one launch per (shape, S, family), in this order (for ncu -k filters)
//   gate <fams>                    B-2 gates per family: (1) every column of S=2..8 == the same column
//                                  computed with S=1 (same family), bit for bit; (2) S=1 vs current plain
//                                  gemv: max e = |d| / sum|terms|; (3) 30 repeats bit identical.
// Usage: r18_gemv_bench <mode> <fams comma> [S list comma] [pairs] [shape filter KxN,...]
#include "runtime/knobs.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../runtime/config.h"
#include "../kernels/gemv_w4a8.cuh"
#if __has_include("../kernels/gemv_mc.cuh")
#include "../kernels/gemv_mc.cuh"
#define R18_HAVE_MC 1
#endif

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

struct Shape { int K, N; const char *role; };
static const Shape SHAPES[] = {
    {3840, 4096, "sliding_q"}, {3840, 2048, "sliding_kv"}, {4096, 3840, "sliding_o"},
    {3840, 8192, "full_q"}, {3840, 512, "full_k"}, {8192, 3840, "full_o"},
    {3840, 15360, "gate_up"}, {15360, 3840, "down"}, {3840, 262144, "lm_head"},
};

static void fam_launch(const std::string &fam, const int8_t *xq, const float *xs, const uint8_t *Wq, const __half *sw,
                       float *Y, int IN, int OUT, int S) {
    if (fam == "cur") {
        if (S == 1) launch_gemv_w4a8(xq, xs, nullptr, Wq, sw, Y, IN, OUT, 1, 1);
        else        launch_gemv_w4a8_sb(xq, xs, Wq, sw, Y, IN, OUT, S, 1);
        return;
    }
    if (fam == "plain") { launch_gemv_w4a8(xq, xs, nullptr, Wq, sw, Y, IN, OUT, S, 1); return; }
#ifdef R18_HAVE_MC
    if (gemv_mc_launch(fam.c_str(), xq, xs, Wq, sw, Y, IN, OUT, S)) return;
#endif
    fprintf(stderr, "unknown family %s\n", fam.c_str()); exit(1);
}

__global__ void fill_u8(uint8_t *p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = (uint8_t)h;
    }
}
__global__ void fill_scale(__half *p, size_t n, uint32_t seed) {   // positive scales in [0.002, 0.03)
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2246822519u ^ seed; h ^= h >> 16; h *= 2654435761u; h ^= h >> 13;
        p[i] = __float2half(0.002f + 0.028f * (float)(h & 0xFFFF) / 65536.f);
    }
}

static std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> v; size_t p = 0;
    while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) v.push_back(s.substr(p, q - p)); p = q + 1; }
    return v;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count(); }

struct Bufs {
    int K, N, copies; size_t wb, sb;
    std::vector<uint8_t *> W; std::vector<__half *> Sc;
    int8_t *xq; float *xs; float *Y;
};
static Bufs alloc_shape(const Shape &sh, int smax, int copies_override) {
    Bufs b; b.K = sh.K; b.N = sh.N; b.wb = (size_t)sh.K * sh.N / 2; b.sb = (size_t)(sh.K / QBLK) * sh.N * 2;
    const size_t l2 = 24u << 20;
    b.copies = copies_override > 0 ? copies_override : (int)((3 * l2 + b.wb + b.sb - 1) / (b.wb + b.sb));
    if (b.copies < 1) b.copies = 1;
    for (int c = 0; c < b.copies; c++) {
        uint8_t *w; __half *s; CK(cudaMalloc(&w, b.wb)); CK(cudaMalloc(&s, b.sb));
        fill_u8<<<1024, 256>>>(w, b.wb, 0x9e3779b9u * (c + 1)); fill_scale<<<1024, 256>>>(s, b.sb / 2, 0x85ebca6bu * (c + 1));
        b.W.push_back(w); b.Sc.push_back(s);
    }
    CK(cudaMalloc(&b.xq, (size_t)smax * sh.K)); CK(cudaMalloc(&b.xs, (size_t)smax * (sh.K / QBLK) * 4));
    CK(cudaMalloc(&b.Y, (size_t)smax * sh.N * 4));
    fill_u8<<<256, 256>>>((uint8_t *)b.xq, (size_t)smax * sh.K, 0x27d4eb2fu);
    std::vector<float> hx((size_t)smax * (sh.K / QBLK));
    uint32_t r = 7; for (auto &v : hx) { r = r * 1664525u + 1013904223u; v = 0.001f + 0.02f * (float)(r >> 8) / 16777216.f; }
    CK(cudaMemcpy(b.xs, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    return b;
}
static void free_shape(Bufs &b) {
    for (auto p : b.W) cudaFree(p); for (auto p : b.Sc) cudaFree(p);
    cudaFree(b.xq); cudaFree(b.xs); cudaFree(b.Y);
}

// back-to-back launches for >= target_ms; returns mean us per launch
static double timed(const std::string &fam, Bufs &b, int S, double target_ms, long &iters, double &t0, double &t1) {
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    long n = 64; double ms = 0;
    for (;;) {   // calibrate: grow n until one batch is >= target
        CK(cudaDeviceSynchronize()); t0 = now_ms(); CK(cudaEventRecord(e0));
        for (long i = 0; i < n; i++) { int c = (int)(i % b.copies); fam_launch(fam, b.xq, b.xs, b.W[c], b.Sc[c], b.Y, b.K, b.N, S); }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); t1 = now_ms();
        float f; CK(cudaEventElapsedTime(&f, e0, e1)); ms = f;
        if (ms >= target_ms) break;
        n = (long)(n * (target_ms * 1.1 / (ms > 0.01 ? ms : 0.01))) + 1;
    }
    CK(cudaGetLastError());
    cudaEventDestroy(e0); cudaEventDestroy(e1); iters = n;
    return ms * 1000.0 / n;
}

int main(int argc, char **argv) {
    knobs_init("r18_gemv_bench");
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 3) { fprintf(stderr, "usage: %s time|ncu|gate <fams> [S list] [pairs] [shapes KxN,...]\n", argv[0]); return 1; }
    const std::string mode = argv[1];
    const auto fams = split(argv[2]);
    std::vector<int> Ss; for (auto &s : split(argc > 3 ? argv[3] : "1,2,4,8")) Ss.push_back(atoi(s.c_str()));
    const int pairs = argc > 4 ? atoi(argv[4]) : 5;
    std::vector<std::string> filt; if (argc > 5) filt = split(argv[5]);
    int smax = 8; for (int s : Ss) if (s > smax) smax = s;
    for (const Shape &sh : SHAPES) {
        char name[32]; snprintf(name, sizeof name, "%dx%d", sh.K, sh.N);
        if (!filt.empty()) { bool ok = false; for (auto &f : filt) ok |= f == name; if (!ok) continue; }
        if (mode == "time") {
            Bufs b = alloc_shape(sh, smax, 0);
            long it; double t0, t1; double wend = now_ms() + 10000.0;
            while (now_ms() < wend) timed(fams[0], b, Ss.back(), 500.0, it, t0, t1);   // 10 s warm-up
            printf("R18_SHAPE shape=%s role=%s copies=%d weight_bytes=%zu scale_bytes=%zu\n", name, sh.role, b.copies, b.wb, b.sb);
            for (int p = 1; p <= pairs; p++)
                for (int S : Ss)
                    for (size_t fi = 0; fi < fams.size(); fi++) {
                        const auto &fam = fams[(p % 2 == 1) ? fi : fams.size() - 1 - fi];
                        const double us = timed(fam, b, S, 1000.0, it, t0, t1);
                        printf("R18_T shape=%s role=%s S=%d fam=%s pair=%d us=%.4f iters=%ld t0_ms=%.1f t1_ms=%.1f\n",
                               name, sh.role, S, fam.c_str(), p, us, it, t0, t1);
                    }
            free_shape(b);
        } else if (mode == "ncu") {
            Bufs b = alloc_shape(sh, smax, 1);
            for (int S : Ss) for (auto &fam : fams) {
                printf("R18_NCU shape=%s role=%s S=%d fam=%s\n", name, sh.role, S, fam.c_str());
                fam_launch(fam, b.xq, b.xs, b.W[0], b.Sc[0], b.Y, b.K, b.N, S); CK(cudaDeviceSynchronize());
            }
            free_shape(b);
        } else if (mode == "gate") {
            Bufs b = alloc_shape(sh, 8, 1);
            const size_t N = sh.N;
            std::vector<float> ref((size_t)8 * N), y((size_t)8 * N), plain((size_t)8 * N), tmp(N);
            // plain reference per column (current S=1 kernel), and sum|terms| bound per output from host
            for (int s = 0; s < 8; s++) {
                launch_gemv_w4a8(b.xq + (size_t)s * sh.K, b.xs + (size_t)s * (sh.K / QBLK), nullptr, b.W[0], b.Sc[0], b.Y, sh.K, sh.N, 1, 1);
                CK(cudaMemcpy(plain.data() + s * N, b.Y, N * 4, cudaMemcpyDeviceToHost));
            }
            // host sum|terms| (term = xs*sw*dot per 32-block) for e = |d| / sum|terms|
            std::vector<uint8_t> hw(b.wb); std::vector<__half> hs(b.sb / 2); std::vector<int8_t> hx((size_t)8 * sh.K);
            std::vector<float> hxs((size_t)8 * (sh.K / QBLK));
            CK(cudaMemcpy(hw.data(), b.W[0], b.wb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hs.data(), b.Sc[0], b.sb, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hx.data(), b.xq, hx.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hxs.data(), b.xs, hxs.size() * 4, cudaMemcpyDeviceToHost));
            const int nb = sh.K / QBLK;
            auto absterms = [&](int s, size_t o) {
                double a = 0;
                for (int k = 0; k < nb; k++) {
                    int d = 0;
                    for (int i = 0; i < QBLK; i++) {
                        const size_t e = (size_t)o * sh.K + k * QBLK + i; const uint8_t by = hw[e / 2];
                        const int q = ((e & 1) ? (by >> 4) : (by & 0xF)) - 8; d += q * hx[(size_t)s * sh.K + k * QBLK + i];
                    }
                    a += fabs((double)hxs[(size_t)s * nb + k] * (double)__half2float(hs[(size_t)o * nb + k]) * d);
                }
                return a;
            };
            for (auto &fam : fams) {
                // (1) column bit-identity: S=1 per column vs S=2..8
                for (int s = 0; s < 8; s++) {
                    fam_launch(fam, b.xq + (size_t)s * sh.K, b.xs + (size_t)s * nb, b.W[0], b.Sc[0], b.Y, sh.K, sh.N, 1);
                    CK(cudaMemcpy(ref.data() + s * N, b.Y, N * 4, cudaMemcpyDeviceToHost));
                }
                long bad = 0; int first_S = -1, first_s = -1; long first_o = -1;
                for (int S = 2; S <= 8; S++) {
                    fam_launch(fam, b.xq, b.xs, b.W[0], b.Sc[0], b.Y, sh.K, sh.N, S);
                    CK(cudaMemcpy(y.data(), b.Y, (size_t)S * N * 4, cudaMemcpyDeviceToHost));
                    for (int s = 0; s < S; s++) for (size_t o = 0; o < N; o++)
                        if (memcmp(&y[s * N + o], &ref[s * N + o], 4)) { if (!bad) { first_S = S; first_s = s; first_o = (long)o; } bad++; }
                }
                // (2) S=1 of the family vs current plain gemv: normalized error on a sample of outputs
                double emax = 0; long nsamp = 0; size_t step = N > 4096 ? N / 4096 : 1;
                for (int s = 0; s < 8; s += 3) for (size_t o = 0; o < N; o += step) {
                    const double d = fabs((double)ref[s * N + o] - (double)plain[s * N + o]);
                    if (d == 0) { nsamp++; continue; }
                    const double e = d / absterms(s, o); if (e > emax) emax = e; nsamp++;
                }
                long bitdiff_plain = 0; for (size_t i = 0; i < (size_t)8 * N; i++) bitdiff_plain += memcmp(&ref[i], &plain[i], 4) != 0;
                // (3) determinism: 30 repeats at S=1, 4, 8
                long nondet = 0;
                for (int S : {1, 4, 8}) {
                    fam_launch(fam, b.xq, b.xs, b.W[0], b.Sc[0], b.Y, sh.K, sh.N, S);
                    std::vector<float> first((size_t)S * N); CK(cudaMemcpy(first.data(), b.Y, first.size() * 4, cudaMemcpyDeviceToHost));
                    for (int r = 1; r < 30; r++) {
                        fam_launch(fam, b.xq, b.xs, b.W[0], b.Sc[0], b.Y, sh.K, sh.N, S);
                        CK(cudaMemcpy(y.data(), b.Y, first.size() * 4, cudaMemcpyDeviceToHost));
                        nondet += memcmp(y.data(), first.data(), first.size() * 4) != 0;
                    }
                }
                printf("R18_GATE shape=%s role=%s fam=%s col_bitdiff=%ld first=S%d/col%d/row%ld plain_bitdiff=%ld/%zu emax=%.3e samples=%ld nondet=%ld %s\n",
                       name, sh.role, fam.c_str(), bad, first_S, first_s, first_o, bitdiff_plain, (size_t)8 * N, emax, nsamp, nondet,
                       (bad == 0 && emax < 1e-5 && nondet == 0) ? "PASS" : "FAIL");
            }
            free_shape(b);
        }
    }
    return 0;
}
