// R15 gates 1-2 for kernels/attention_decode_split.cuh (manual; GPU, no model files).
//   Gate 1: split kernel vs fp64 CPU reference on the same fp32 KV. Error per output element
//     e = |o - o_ref| / sum_j p_j |V_j,d| (p = fp64 softmax); require max e < 1e-5.
//     KV lengths d in {1,31,32,33,1023,1024,1025,2048,8192,32768}; sliding (HD256, 16q/8kv, window 1024)
//     and full (HD512, 16q/1kv, no window) layers; scale 1.0 (Gemma attn_scaling).
//   Gate 2: verify rows (S=2..16, consecutive positions, shared KV) and batch rows (W=1..12, own KV per
//     slot, ragged positions) must be bitwise equal to the S=1 result at the same position and KV.
// Usage: test_attn_decode_split [G4_ATTN_CHUNK env selects the chunk size]
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/attention_decode_split.cuh"

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Rng { uint64_t s; float next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return ((uint32_t)((s * 2685821657736338717ull) >> 40) / 16777216.0f) * 2.f - 1.f; } };
struct Layer { const char *name; int HD, NQ, NKV, window; };
static const Layer LAYERS[2] = {{"sliding", 256, 16, 8, 1024}, {"full", 512, 16, 1, 0}};

template<class T> static T *dev(const std::vector<T> &h) {
    T *p; ck(cudaMalloc(&p, h.size() * sizeof(T))); ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
static std::vector<float> run(const Layer &L, const float *dq, const float *dk, const float *dv, int rows, int base,
                              int pos_step, const int *dpos, int max_pos, long kv_stride, int max_len) {
    float *dout; ck(cudaMalloc(&dout, (size_t)rows * L.NQ * L.HD * 4));
    attn_split_reserve(rows, L.NQ, max_len, L.HD);
    launch_attn_decode_split(dq, dk, dv, dout, rows, L.NQ, L.NKV, L.HD, base, pos_step, dpos, max_pos, kv_stride,
                             attn_split_ns_cap(max_len), 0, 1.0f, L.window);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    std::vector<float> o((size_t)rows * L.NQ * L.HD);
    ck(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost)); ck(cudaFree(dout));
    return o;
}
static uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }

int main() try {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    printf("ATTN_SPLIT_GATE chunk=%d\n", attn_split_chunk());
    int fails = 0;
    // Gate 1
    for (const Layer &L : LAYERS) {
        const int maxd = 32768, G = L.NQ / L.NKV;
        Rng rng{0x5eed1500ull + (uint64_t)L.HD};
        std::vector<float> k((size_t)maxd * L.NKV * L.HD), v(k.size()), q((size_t)L.NQ * L.HD);
        for (auto &x : k) x = rng.next(); for (auto &x : v) x = rng.next();
        for (auto &x : q) x = rng.next() * 0.25f;
        float *dk = dev(k), *dv = dev(v), *dq = dev(q);
        for (int d : {1, 31, 32, 33, 1023, 1024, 1025, 2048, 8192, 32768}) {
            const int pos = d - 1;
            auto o = run(L, dq, dk, dv, 1, pos, 1, nullptr, pos, 0, d);
            const int lo = (L.window > 0 && pos - L.window + 1 > 0) ? pos - L.window + 1 : 0;
            double worst = 0;
            for (int h = 0; h < L.NQ; h++) {
                const int kvh = h / G;
                std::vector<double> s(pos + 1 - lo); double m = -1e300;
                for (int j = lo; j <= pos; j++) {
                    double acc = 0; for (int t = 0; t < L.HD; t++) acc += (double)q[(size_t)h * L.HD + t] * k[((size_t)j * L.NKV + kvh) * L.HD + t];
                    s[j - lo] = acc; m = std::max(m, acc);
                }
                double z = 0; for (auto &x : s) { x = std::exp(x - m); z += x; }
                for (int t = 0; t < L.HD; t++) {
                    double ref = 0, den = 0;
                    for (int j = lo; j <= pos; j++) { double vv = v[((size_t)j * L.NKV + kvh) * L.HD + t];
                        ref += s[j - lo] / z * vv; den += s[j - lo] / z * std::fabs(vv); }
                    const double e = std::fabs((double)o[(size_t)h * L.HD + t] - ref) / den;
                    if (!(e == e) || !std::isfinite(o[(size_t)h * L.HD + t])) worst = INFINITY; else worst = std::max(worst, e);
                }
            }
            const bool ok = worst < 1e-5;
            fails += !ok;
            printf("GATE1 %s %s d=%d max_normalized_error=%.3e\n", ok ? "PASS" : "FAIL", L.name, d, worst);
        }
        // Gate 2 verify rows: S=2..16 consecutive positions vs S=1 at each position (shared KV).
        for (int base : {1000, 1017, 1023, 2040, 8190}) for (int S = 2; S <= 16; S++) {
            std::vector<float> qs((size_t)S * L.NQ * L.HD); for (auto &x : qs) x = rng.next() * 0.25f;
            float *dqs = dev(qs);
            const int max_len = base + 16 + 1;
            auto multi = run(L, dqs, dk, dv, S, base, 1, nullptr, base + S - 1, 0, max_len);
            for (int r = 0; r < S; r++) {
                auto single = run(L, dqs + (size_t)r * L.NQ * L.HD, dk, dv, 1, base + r, 1, nullptr, base + r, 0, max_len);
                for (size_t i = 0; i < single.size(); i++) if (bits(single[i]) != bits(multi[(size_t)r * L.NQ * L.HD + i])) {
                    printf("GATE2 FAIL verify %s base=%d S=%d row=%d elem=%zu single=%.9g multi=%.9g\n", L.name, base, S, r, i,
                           single[i], multi[(size_t)r * L.NQ * L.HD + i]); fails++; break; }
            }
            ck(cudaFree(dqs));
        }
        printf("GATE2 verify %s done\n", L.name);
        ck(cudaFree(dk)); ck(cudaFree(dv)); ck(cudaFree(dq));
        // Gate 2 batch rows: W slots with their own KV and ragged positions vs S=1 on that slot.
        const int cap = 4096; const long stride = (long)cap * L.NKV * L.HD;
        std::vector<float> bk((size_t)12 * stride), bv(bk.size()); for (auto &x : bk) x = rng.next(); for (auto &x : bv) x = rng.next();
        float *dbk = dev(bk), *dbv = dev(bv);
        for (int W = 1; W <= 12; W++) {
            std::vector<int> pos(W); int mx = 0;
            for (int b = 0; b < W; b++) { pos[b] = (b * 977 + W * 131 + 63) % (cap - 1); mx = std::max(mx, pos[b]); }
            std::vector<float> qs((size_t)W * L.NQ * L.HD); for (auto &x : qs) x = rng.next() * 0.25f;
            float *dqs = dev(qs); int *dpos = dev(pos);
            auto multi = run(L, dqs, dbk, dbv, W, 0, 0, dpos, mx, stride, cap);
            for (int b = 0; b < W; b++) {
                auto single = run(L, dqs + (size_t)b * L.NQ * L.HD, dbk + (size_t)b * stride, dbv + (size_t)b * stride, 1, pos[b], 1,
                                  nullptr, pos[b], 0, cap);
                for (size_t i = 0; i < single.size(); i++) if (bits(single[i]) != bits(multi[(size_t)b * L.NQ * L.HD + i])) {
                    printf("GATE2 FAIL batch %s W=%d slot=%d pos=%d elem=%zu\n", L.name, W, b, pos[b], i); fails++; break; }
            }
            ck(cudaFree(dqs)); ck(cudaFree(dpos));
        }
        printf("GATE2 batch %s done\n", L.name);
        ck(cudaFree(dbk)); ck(cudaFree(dbv));
    }
    printf("[test_attn_decode_split] %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
} catch (const std::exception &e) { fprintf(stderr, "[test_attn_decode_split] FAIL: %s\n", e.what()); return 1; }
