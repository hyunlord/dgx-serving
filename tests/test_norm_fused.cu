// R16 PART B isolated gates for kernels/norm_fused.cuh (no model weights needed).
//   1. accuracy: rmsnorm_par vs fp64 CPU, normalized output max relative error < 1e-6 (hh 3840/256/512);
//      the existing one-block rmsnorm_kernel is measured the same way for reference.
//   2. row identity: a row's output does not depend on how many rows are launched with it
//      (1..16 rows: covers verify S <= 16 and batch W <= 12, which differ only in the row count) — bitwise vs 1 row.
//   3. fusion equivalence (bitwise): each fused kernel == rmsnorm_par followed by the existing kernels
//      (rope_kernel / rope_partial_pos / rope_partial_batch, add_kernel, scale_kernel, kv_store f16/bf16).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/rmsnorm.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/kv_store.cuh"
#include "../kernels/norm_fused.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static int fails = 0;
static void check(bool ok, const char *what) { printf("%s %s\n", ok ? "PASS" : "FAIL", what); if (!ok) fails++; }
template <class T> static T *up(const std::vector<T> &h) { T *d; CK(cudaMalloc(&d, h.size() * sizeof(T))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> static std::vector<T> down(const T *d, size_t n) { std::vector<T> h(n); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
template <class T> static bool same(const std::vector<T> &a, const std::vector<T> &b) { return a.size() == b.size() && !memcmp(a.data(), b.data(), a.size() * sizeof(T)); }
static std::vector<float> rnd(size_t n, unsigned seed, float scale) {
    std::mt19937 g(seed); std::normal_distribution<float> d(0.f, scale); std::vector<float> v(n); for (auto &x : v) x = d(g); return v;
}

int main() {
    const float eps = 1e-6f;
    // ---- 1. accuracy vs fp64 ----
    for (int hh : {3840, 256, 512}) {
        const int rows = 64;
        auto x = rnd((size_t)rows * hh, 7 + hh, 3.f);
        for (int r = 0; r < rows; r++) x[(size_t)r * hh + r % hh] *= 40.f;   // a few outliers per row
        std::vector<float> one(hh, 1.f);
        float *dx = up(x), *dw = up(one), *dy; CK(cudaMalloc(&dy, x.size() * 4));
        double worst_par = 0, worst_old = 0;
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 0) launch_rmsnorm_par(dx, dw, dy, rows, hh, eps); else launch_rmsnorm(dx, dw, dy, rows, hh, eps);
            auto y = down(dy, x.size());
            for (int r = 0; r < rows; r++) {
                double ss = 0; for (int i = 0; i < hh; i++) ss += (double)x[(size_t)r * hh + i] * x[(size_t)r * hh + i];
                const double inv = 1.0 / std::sqrt(ss / hh + eps);
                for (int i = 0; i < hh; i++) {
                    const double ref = x[(size_t)r * hh + i] * inv;
                    if (std::fabs(ref) < 1e-3) continue;
                    const double e = std::fabs(y[(size_t)r * hh + i] - ref) / std::fabs(ref);
                    double &w = pass == 0 ? worst_par : worst_old; if (e > w) w = e;
                }
            }
        }
        printf("  hh=%d threads=%d max rel err: par %.3e  old(one block, 256 threads) %.3e\n", hh, norm_par_threads(hh), worst_par, worst_old);
        char msg[96]; snprintf(msg, sizeof msg, "accuracy hh=%d (par < 1e-6)", hh); check(worst_par < 1e-6, msg);
        cudaFree(dx); cudaFree(dw); cudaFree(dy);
    }

    // ---- 2. row identity: S=1 vs S rows and W batch rows ----
    for (int hh : {3840, 256, 512}) {
        auto base = rnd(hh, 11 + hh, 2.f), w = rnd(hh, 13 + hh, 1.f);
        float *dw = up(w);
        std::vector<float> ref;
        bool ok = true;
        for (int S = 1; S <= 16; S++) {
            std::vector<float> x((size_t)S * hh);
            for (int r = 0; r < S; r++) for (int i = 0; i < hh; i++) x[(size_t)r * hh + i] = r == S - 1 ? base[i] : base[(i + r + 1) % hh] * 0.5f;
            float *dx = up(x), *dy; CK(cudaMalloc(&dy, x.size() * 4));
            launch_rmsnorm_par(dx, dw, dy, S, hh, eps);
            auto y = down(dy, x.size());
            std::vector<float> last(y.end() - hh, y.end());
            if (S == 1) ref = last; else ok = ok && same(last, ref);
            cudaFree(dx); cudaFree(dy);
        }
        char msg[96]; snprintf(msg, sizeof msg, "row identity hh=%d (last row of 1..16-row launches == 1-row launch)", hh); check(ok, msg);
        cudaFree(dw);
    }

    // ---- 3. fusion equivalence ----
    const int NQ = 16;
    for (int hd : {256, 512}) {
        const int nkv = hd == 512 ? 1 : 8, maxpos = 4200;
        std::vector<float> cosv((size_t)maxpos * hd), sinv((size_t)maxpos * hd);
        for (int p = 0; p < maxpos; p++) for (int i = 0; i < hd / 2; i++) {
            const float a = p * powf(10000.f, -2.f * i / hd);
            cosv[(size_t)p * hd + i] = cosv[(size_t)p * hd + hd / 2 + i] = cosf(a);
            sinv[(size_t)p * hd + i] = sinv[(size_t)p * hd + hd / 2 + i] = sinf(a);
        }
        float *dc = up(cosv), *ds = up(sinv);
        auto w = rnd(hd, 3 + hd, 1.f); float *dw = up(w);
        bool q_dec = true, q_batch = true, q_pos = true, kv16 = true, kvb16 = true, v_ok = true;
        for (int S : {1, 5, 16}) {
            const int rows = S * NQ, base = 4100;
            auto x = rnd((size_t)rows * hd, 17 + S + hd, 2.f);
            float *dx = up(x), *ya, *yb; CK(cudaMalloc(&ya, x.size() * 4)); CK(cudaMalloc(&yb, x.size() * 4));
            // decode/verify/prefill: positions base + s
            launch_rmsnorm_par(dx, dw, ya, rows, hd, eps);
            launch_rope(ya, dc + (size_t)base * hd, ds + (size_t)base * hd, S, NQ, hd);
            launch_norm_rope_store<float>(dx, dw, yb, rows, NQ, hd, eps, dc, ds, base, 1, nullptr, (long)NQ * hd, 0, true);
            q_dec = q_dec && same(down(ya, x.size()), down(yb, x.size()));
            // same-length batch: every slot at base
            launch_rmsnorm_par(dx, dw, ya, rows, hd, eps);
            launch_rope_partial_batch(ya, dc + (size_t)base * hd, ds + (size_t)base * hd, S, NQ, hd, hd);
            launch_norm_rope_store<float>(dx, dw, yb, rows, NQ, hd, eps, dc, ds, base, 0, nullptr, (long)NQ * hd, 0, true);
            q_batch = q_batch && same(down(ya, x.size()), down(yb, x.size()));
            // ragged / concat: per-row positions
            std::vector<int> pos(S); for (int s = 0; s < S; s++) pos[s] = (s * 977 + 13) % maxpos;
            int *dp = up(pos);
            launch_rmsnorm_par(dx, dw, ya, rows, hd, eps);
            launch_rope_partial_pos(ya, dc, ds, dp, S, NQ, hd, hd);
            launch_norm_rope_store<float>(dx, dw, yb, rows, NQ, hd, eps, dc, ds, 0, 0, dp, (long)NQ * hd, 0, true);
            q_pos = q_pos && same(down(ya, x.size()), down(yb, x.size()));
            // K into a 16-bit cache slice (decode layout: slot stride 0, row stride nkv*hd) vs par+rope+kv_store
            const int krows = S * nkv; const size_t kn = (size_t)krows * hd;
            for (int dt : {1, 2}) {
                void *ca, *cb; CK(cudaMalloc(&ca, (size_t)(base + S) * nkv * hd * 2)); CK(cudaMalloc(&cb, (size_t)(base + S) * nkv * hd * 2));
                CK(cudaMemset(ca, 0, (size_t)(base + S) * nkv * hd * 2)); CK(cudaMemset(cb, 0, (size_t)(base + S) * nkv * hd * 2));
                g_kv_dtype_override = dt;
                launch_rmsnorm_par(dx, dw, ya, krows, hd, eps);
                launch_rope(ya, dc + (size_t)base * hd, ds + (size_t)base * hd, S, nkv, hd);
                kv_store(ya, kv_at((float *)ca, (size_t)base * nkv * hd), kn, 0, 0);
                if (dt == 1) launch_norm_rope_store<__half>(dx, dw, (__half *)cb, krows, nkv, hd, eps, dc, ds, base, 1, nullptr, 0, (long)nkv * hd, true);
                else launch_norm_rope_store<__nv_bfloat16>(dx, dw, (__nv_bfloat16 *)cb, krows, nkv, hd, eps, dc, ds, base, 1, nullptr, 0, (long)nkv * hd, true);
                const size_t n16 = (size_t)(base + S) * nkv * hd;
                const bool eq = same(down((uint16_t *)ca, n16), down((uint16_t *)cb, n16));
                if (dt == 1) kv16 = kv16 && eq; else kvb16 = kvb16 && eq;
                // V: no scale, no RoPE
                launch_rmsnorm_par(dx, nullptr, ya, krows, hd, eps);
                kv_store(ya, kv_at((float *)ca, (size_t)base * nkv * hd), kn, 1, 0);
                if (dt == 1) launch_norm_rope_store<__half>(dx, nullptr, (__half *)cb, krows, nkv, hd, eps, dc, ds, base, 1, nullptr, 0, (long)nkv * hd, false);
                else launch_norm_rope_store<__nv_bfloat16>(dx, nullptr, (__nv_bfloat16 *)cb, krows, nkv, hd, eps, dc, ds, base, 1, nullptr, 0, (long)nkv * hd, false);
                v_ok = v_ok && same(down((uint16_t *)ca, n16), down((uint16_t *)cb, n16));
                g_kv_dtype_override = -1;
                cudaFree(ca); cudaFree(cb);
            }
            cudaFree(dx); cudaFree(ya); cudaFree(yb); cudaFree(dp);
        }
        char msg[128];
        snprintf(msg, sizeof msg, "hd=%d q norm+RoPE == par+rope_kernel (S=1,5,16)", hd); check(q_dec, msg);
        snprintf(msg, sizeof msg, "hd=%d q norm+RoPE == par+rope_partial_batch", hd); check(q_batch, msg);
        snprintf(msg, sizeof msg, "hd=%d q norm+RoPE == par+rope_partial_pos", hd); check(q_pos, msg);
        snprintf(msg, sizeof msg, "hd=%d K norm+RoPE+store == par+rope+kv_store (f16)", hd); check(kv16, msg);
        snprintf(msg, sizeof msg, "hd=%d K norm+RoPE+store == par+rope+kv_store (bf16)", hd); check(kvb16, msg);
        snprintf(msg, sizeof msg, "hd=%d V norm+store == par(no scale)+kv_store (f16, bf16)", hd); check(v_ok, msg);
        cudaFree(dc); cudaFree(ds); cudaFree(dw);
    }
    // post-norm + residual (+ scale) (+ next pre-norm)
    {
        const int H = 3840;
        auto w1 = rnd(H, 21, 1.f), w2 = rnd(H, 22, 1.f); float *d1 = up(w1), *d2 = up(w2);
        bool ok = true;
        for (int S : {1, 7, 12}) for (float scale : {1.0f, 0.83f}) {
            auto in = rnd((size_t)S * H, 31 + S, 3.f), res = rnd((size_t)S * H, 41 + S, 5.f);
            float *din = up(in), *dres = up(res), *tmp, *oa, *na, *ob, *nb;
            for (float **p : {&tmp, &oa, &na, &ob, &nb}) CK(cudaMalloc(p, in.size() * 4));
            launch_rmsnorm_par(din, d1, tmp, S, H, eps);
            launch_add(oa, dres, tmp, S * H);
            if (scale != 1.0f) launch_scale(oa, oa, scale, (size_t)S * H);
            launch_rmsnorm_par(oa, d2, na, S, H, eps);
            launch_post_norm_residual(din, d1, dres, ob, scale, d2, nb, S, H, eps);
            ok = ok && same(down(oa, in.size()), down(ob, in.size())) && same(down(na, in.size()), down(nb, in.size()));
            for (float *p : {din, dres, tmp, oa, na, ob, nb}) cudaFree(p);
        }
        check(ok, "post-norm+residual(+scale)+next norm == par+add+scale+par (S=1,7,12)");
        cudaFree(d1); cudaFree(d2);
    }
    printf("[test_norm_fused] %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
