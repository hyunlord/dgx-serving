// Prefill flash attention v2 isolated gate (R21 G1/G3 + kernel-level row-composition check).
//   Shapes: Gemma 4 12B sliding (hd256, q16/kv8, window 1024) and full (hd512, q16/kv1).
//   Sq in {1,6,15,16,17,512,1024,2048}, KV length in {Sq, Sq+1024, Sq+8192} (base = 0, 1024, 8192).
//   KV is stored as f16 (engine default). The CPU fp64 reference uses the values the kernel multiplies
//   (Q rounded to bf16, K/V f16 rounded to bf16, as R09), over sampled query tokens for large Sq.
//   Checks per case: (1) normalized max error = max|out-ref| / max|ref| < 1e-3 (pre-registered G1),
//   (2) 30 launches bit-identical (G3), (3) the last rows launched alone (base+s1, Sq-s1 rows) equal the
//   same rows of the full launch bit for bit (row composition independence, the kernel half of G2).
//   The v1 kernel's error on the same inputs is printed for reference only.
//   Usage: ./tests/test_flash_v2 [quick]   (quick: Sq <= 512, base <= 1024)
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "../kernels/attention_flash.cuh"
#include "../kernels/attention_flash_v2.cuh"

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA_ERROR line=%d %s\n", __LINE__, cudaGetErrorString(e)); exit(2); } } while (0)

static float rnd(uint32_t &s) { s = s * 1664525u + 1013904223u; return (float)(s >> 8) / 8388608.0f - 1.0f; }
static float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

struct Shape { const char *name; int hd, nq, nkv, window; };

int main(int argc, char **argv) {
    const bool quick = argc > 1 && strcmp(argv[1], "quick") == 0;
    const Shape shapes[] = {{"sliding", 256, 16, 8, 1024}, {"full", 512, 16, 1, 0}};
    const int sqs[] = {1, 6, 15, 16, 17, 512, 1024, 2048}, extra[] = {0, 1024, 8192};
    const int threads = std::max(1u, std::thread::hardware_concurrency());
    int fails = 0, cases = 0; double worst = 0;
    for (const Shape &sh : shapes) for (int sq : sqs) for (int base : extra) {
        if (quick && (sq > 512 || base > 1024)) continue;
        const int hd = sh.hd, nq = sh.nq, nkv = sh.nkv, kvlen = base + sq;
        const size_t nqe = (size_t)sq * nq * hd, nke = (size_t)kvlen * nkv * hd;
        std::vector<float> q(nqe), qr(nqe), kr(nke), vr(nke);
        std::vector<__half> k(nke), v(nke);
        uint32_t seed = 2101u + hd * 17u + sq * 43u + base * 59u;
        for (size_t i = 0; i < nqe; i++) { q[i] = rnd(seed); qr[i] = bf(q[i]); }
        for (size_t i = 0; i < nke; i++) {
            k[i] = __float2half(rnd(seed)); v[i] = __float2half(rnd(seed));
            kr[i] = bf(__half2float(k[i])); vr[i] = bf(__half2float(v[i]));
        }
        float *dq, *dout, *dtail, *dold; __half *dk, *dv;
        CK(cudaMalloc(&dq, nqe * 4)); CK(cudaMalloc(&dout, nqe * 4)); CK(cudaMalloc(&dtail, nqe * 4)); CK(cudaMalloc(&dold, nqe * 4));
        CK(cudaMalloc(&dk, nke * 2)); CK(cudaMalloc(&dv, nke * 2));
        CK(cudaMemcpy(dq, q.data(), nqe * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, k.data(), nke * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv, v.data(), nke * 2, cudaMemcpyHostToDevice));
        auto launch = [&] { launch_flash_attn_v2<__half>(dq, dk, dv, dout, sq, nq, nkv, hd, base, 0, 0.f, sh.window); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); };
        launch();
        std::vector<float> out(nqe), next(nqe), tail(nqe), old(nqe);
        CK(cudaMemcpy(out.data(), dout, nqe * 4, cudaMemcpyDeviceToHost));
        bool det = true; int bad_rep = -1;
        for (int r = 1; r < 30 && det; r++) {
            launch(); CK(cudaMemcpy(next.data(), dout, nqe * 4, cudaMemcpyDeviceToHost));
            if (memcmp(out.data(), next.data(), nqe * 4)) { det = false; bad_rep = r; }
        }
        // Row composition: the last rows alone (queries s1.., positions base+s1..) must equal the full launch's rows.
        const int s1 = sq > 6 ? sq - 6 : (sq > 1 ? sq - 1 : 0);
        bool comp = true; long comp_row = -1;
        if (s1 > 0) {
            launch_flash_attn_v2<__half>(dq + (size_t)s1 * nq * hd, dk, dv, dtail, sq - s1, nq, nkv, hd, base + s1, 0, 0.f, sh.window);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(tail.data(), dtail, (size_t)(sq - s1) * nq * hd * 4, cudaMemcpyDeviceToHost));
            const float *a = out.data() + (size_t)s1 * nq * hd;
            for (size_t i = 0; i < (size_t)(sq - s1) * nq * hd; i++) if (memcmp(&a[i], &tail[i], 4)) { comp = false; comp_row = (long)(s1 + i / (nq * hd)); break; }
        }
        launch_flash_attn<__half>(dq, dk, dv, dold, sq, nq, nkv, hd, base, 0, nullptr, 0.f, sh.window);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(old.data(), dold, nqe * 4, cudaMemcpyDeviceToHost));
        // Sampled query tokens: all for sq <= 64; else first/last, middle, window edges, 24 pseudo-random.
        std::vector<int> toks;
        if (sq <= 64) for (int i = 0; i < sq; i++) toks.push_back(i);
        else {
            for (int i : {0, 1, 2, 15, 16, 17, sq / 2, sq - 3, sq - 2, sq - 1}) toks.push_back(i);
            for (int e : {1023, 1024, 1025}) if (e - base >= 0 && e - base < sq) toks.push_back(e - base);
            uint32_t s2 = 77u + sq; for (int i = 0; i < 24; i++) { s2 = s2 * 1664525u + 1013904223u; toks.push_back((int)(s2 % (unsigned)sq)); }
            std::sort(toks.begin(), toks.end()); toks.erase(std::unique(toks.begin(), toks.end()), toks.end());
        }
        const double scale = 1.0 / std::sqrt((double)hd);
        std::vector<double> ref(toks.size() * nq * hd);
        std::atomic<int> nextrow{0};
        auto worker = [&] {
            std::vector<double> sc(kvlen), acc(hd);
            for (int idx = nextrow.fetch_add(1); idx < (int)(toks.size() * nq); idx = nextrow.fetch_add(1)) {
                const int ti = idx / nq, h = idx % nq, qi = toks[ti], kvh = h / (nq / nkv), qpos = base + qi;
                const int lo = sh.window > 0 ? std::max(0, qpos - sh.window + 1) : 0;
                const float *qq = qr.data() + ((size_t)qi * nq + h) * hd; double mx = -1e300;
                for (int j = lo; j <= qpos; j++) { const float *kk = kr.data() + ((size_t)j * nkv + kvh) * hd; double d = 0;
                    for (int c = 0; c < hd; c++) d += (double)qq[c] * kk[c]; sc[j] = d * scale; mx = std::max(mx, sc[j]); }
                double sum = 0; for (int j = lo; j <= qpos; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int j = lo; j <= qpos; j++) { const float *vv = vr.data() + ((size_t)j * nkv + kvh) * hd; for (int c = 0; c < hd; c++) acc[c] += sc[j] * vv[c]; }
                for (int c = 0; c < hd; c++) ref[(size_t)idx * hd + c] = acc[c] / sum;
            }
        };
        std::vector<std::thread> pool; for (int i = 0; i < threads; i++) pool.emplace_back(worker); for (auto &t : pool) t.join();
        double den = 0, mx = 0, sum = 0, oldmx = 0; size_t n = 0, wi = 0; bool finite = true;
        for (size_t i = 0; i < ref.size(); i++) den = std::max(den, std::abs(ref[i]));
        for (size_t ti = 0; ti < toks.size(); ti++) for (int h = 0; h < nq; h++) for (int c = 0; c < hd; c++) {
            const size_t ri = (ti * nq + h) * hd + c, oi = ((size_t)toks[ti] * nq + h) * hd + c;
            finite = finite && std::isfinite(out[oi]);
            const double e = std::abs((double)out[oi] - ref[ri]); sum += e; n++;
            if (e > mx) { mx = e; wi = oi; }
            oldmx = std::max(oldmx, std::abs((double)old[oi] - ref[ri]));
        }
        const double nmax = mx / den, nmean = sum / n / den, oldn = oldmx / den;
        const bool pass = finite && nmax < 1e-3 && det && comp;
        worst = std::max(worst, nmax); cases++; fails += !pass;
        printf("V2_CASE shape=%s hd=%d sq=%d base=%d kvlen=%d rows_checked=%zu max_err=%.6g mean_err=%.6g v1_max_err=%.6g det30=%d comp=%d%s %s\n",
               sh.name, hd, sq, base, kvlen, toks.size() * nq, nmax, nmean, oldn, det, comp, s1 ? "" : "(n/a)", pass ? "PASS" : "FAIL");
        if (!pass) printf("V2_FAIL worst_query=%zu head=%zu dim=%zu det_repeat=%d comp_first_row=%ld\n", wi / (nq * hd), wi / hd % nq, wi % hd, bad_rep, comp_row);
        fflush(stdout);
        cudaFree(dq); cudaFree(dout); cudaFree(dtail); cudaFree(dold); cudaFree(dk); cudaFree(dv);
    }
    printf("%s cases=%d failures=%d worst_max_err=%.6g\n", fails ? "FLASH_V2_FAIL" : "FLASH_V2_PASS", cases, fails, worst);
    return fails ? 1 : 0;
}
