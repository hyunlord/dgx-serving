// R21 long-context NLL through the prefill path (G5 auxiliary metric): prefills one long window in prefill mode
//   (gemma4_decode prefill=1, amort 1, all-position logits) in fixed chunks, and reports mean NLL for targets in
//   [1,1023], [1024,4095], [4096,N-1] and overall (NLL as tools/r15_long_nll.cu: double log-sum-exp over the
//   softcapped logits). Prefill is chunk-invariant (tests/test_gemma4_prefill_chunk), so the chunk size only
//   bounds logits memory. Non-finite rows are counted (must be 0).
//   Usage: r21_long_prefill_nll <weights> <tokens.bin int32> [N=all] [chunk=1024]   (env: G4_MMQ_PREFILL, MMQ256, G4_FLASH_V2, ...)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/knobs.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

static double nll(const float *row, int V, int target) {
    double m = -INFINITY; for (int i = 0; i < V; i++) m = std::max(m, (double)row[i]);
    double s = 0; for (int i = 0; i < V; i++) s += std::exp((double)row[i] - m);
    return m + std::log(s) - (double)row[target];
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <weights> <tokens.bin> [N] [chunk=1024]\n", argv[0]); return 2; }
    knobs_init("r21_long_prefill_nll");
    const std::string WD = argv[1];
    std::vector<int> tok;
    { FILE *f = fopen(argv[2], "rb"); if (!f) return 2; int x; while (fread(&x, 4, 1, f) == 1) tok.push_back(x); fclose(f); }
    const int N = argc > 3 ? std::min(atoi(argv[3]), (int)tok.size()) : (int)tok.size(), CH = argc > 4 ? atoi(argv[4]) : 1024;
    ModelConfig c = load_config_gemma4((WD + "/manifest.json").c_str());
    const int V = c.vocab_size, hs = c.head_dim, hf = c.global_head_dim ? c.global_head_dim : hs;
    float *cf, *sf, *cs, *ss;
    CK(cudaMalloc(&cf, (size_t)N * hf * 4)); CK(cudaMalloc(&sf, (size_t)N * hf * 4)); CK(cudaMalloc(&cs, (size_t)N * hs * 4)); CK(cudaMalloc(&ss, (size_t)N * hs * 4));
    if (gemma4_gen_rope_tables((WD + "/manifest.json").c_str(), N, hf, cf, sf, hs, cs, ss) != 0) return 1;
    G4Weights w = load_weights_gemma4(WD.c_str(), c, 1, 0);
    G4DecodeState st = alloc_g4_decode_state(c, N);
    int *ids; float *logits; CK(cudaMalloc(&ids, (size_t)N * 4)); CK(cudaMalloc(&logits, (size_t)CH * V * 4));
    CK(cudaMemcpy(ids, tok.data(), (size_t)N * 4, cudaMemcpyHostToDevice));
    std::vector<float> rows((size_t)CH * V);
    const int edges[4] = {1, 1024, 4096, N};
    double sum[3] = {0, 0, 0}; long cnt[3] = {0, 0, 0}; long nonfinite = 0;
    reset_g4_decode_state(st);
    for (int o = 0; o + 1 < N; o += CH) {
        const int n = std::min(CH, N - 1 - o);   // positions o..o+n-1 predict targets o+1..o+n
        gemma4_decode(c, w, st, ids + o, n, cf, sf, cs, ss, logits, 1, 2, 0, 1, nullptr, /*prefill*/1);
        CK(cudaMemcpy(rows.data(), logits, (size_t)n * V * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            const float *row = rows.data() + (size_t)i * V; bool finite = true;
            for (int j = 0; j < V; j++) finite = finite && std::isfinite(row[j]);
            if (!finite) { nonfinite++; continue; }
            const int t = o + i + 1; const double v = nll(row, V, tok[t]);
            for (int b = 0; b < 3; b++) if (t >= edges[b] && t < edges[b + 1]) { sum[b] += v; cnt[b]++; }
        }
    }
    double all = sum[0] + sum[1] + sum[2]; long call = cnt[0] + cnt[1] + cnt[2];
    printf("{\"tokens\":%d,\"chunk\":%d,\"nonfinite_rows\":%ld,\"mean_nll_all\":%.17g,\"targets_all\":%ld", N, CH, nonfinite, all / call, call);
    const char *names[3] = {"1_1023", "1024_4095", "4096_end"};
    for (int b = 0; b < 3; b++) printf(",\"mean_nll_%s\":%.17g,\"targets_%s\":%ld", names[b], cnt[b] ? sum[b] / cnt[b] : NAN, names[b], cnt[b]);
    printf("}\n");
    return nonfinite ? 3 : 0;
}
