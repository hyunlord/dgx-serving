// R15 long-context teacher-forced decode NLL by target position bucket (auxiliary quality metric).
//   Feeds every token of one window through S=1 gemma4_decode (eager, amort 0, the decode attention path),
//   scores target p+1 from the logits of step p immediately (one logits row; NLL as in tools/ppl_gemma4.cu:
//   double log-sum-exp over softcapped logits), and reports mean NLL for targets in [1,1023], [1024,4095],
//   [4096,N-1] and overall. Also counts non-finite logits rows (must be 0).
//   Usage: r15_long_nll <weights> <tokens.bin int32> [N=all]   (env as for ppl_gemma4: G4_MMQ_PREFILL, MMQ256)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/config.h"
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
    if (argc < 3) { fprintf(stderr, "usage: %s <weights> <tokens.bin> [N]\n", argv[0]); return 2; }
    std::string WD = argv[1];
    FILE *f = fopen(argv[2], "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<int> tok(bytes / 4); if (fread(tok.data(), 4, tok.size(), f) != tok.size()) return 1; fclose(f);
    const int N = argc > 3 ? atoi(argv[3]) : (int)tok.size();
    if (N < 2 || N > (int)tok.size()) { fprintf(stderr, "bad N\n"); return 1; }
    ModelConfig c = load_config_gemma4((WD + "/manifest.json").c_str());
    const int V = c.vocab_size, hs = c.head_dim, hf = c.global_head_dim ? c.global_head_dim : hs;
    float *cf, *sf, *cs, *ss;
    CK(cudaMalloc(&cf, (size_t)N * hf * 4)); CK(cudaMalloc(&sf, (size_t)N * hf * 4));
    CK(cudaMalloc(&cs, (size_t)N * hs * 4)); CK(cudaMalloc(&ss, (size_t)N * hs * 4));
    if (gemma4_gen_rope_tables((WD + "/manifest.json").c_str(), N, hf, cf, sf, hs, cs, ss) != 0) return 1;
    G4Weights w = load_weights_gemma4(WD.c_str(), c, 1, 0);
    G4DecodeState st = alloc_g4_decode_state(c, N);
    int *ids; float *logits; CK(cudaMalloc(&ids, (size_t)N * 4)); CK(cudaMalloc(&logits, (size_t)V * 4));
    CK(cudaMemcpy(ids, tok.data(), (size_t)N * 4, cudaMemcpyHostToDevice));
    std::vector<float> row(V);
    const int edges[4] = {1, 1024, 4096, N};
    double sum[3] = {0, 0, 0}; long cnt[3] = {0, 0, 0}; long nonfinite = 0;
    for (int p = 0; p + 1 < N; p++) {
        gemma4_decode(c, w, st, ids + p, 1, cf, sf, cs, ss, logits, 1, 2, 0, 0, nullptr);
        CK(cudaMemcpy(row.data(), logits, (size_t)V * 4, cudaMemcpyDeviceToHost));
        bool finite = true; for (float x : row) finite = finite && std::isfinite(x);
        if (!finite) { nonfinite++; continue; }
        const int t = p + 1; const double v = nll(row.data(), V, tok[t]);
        for (int b = 0; b < 3; b++) if (t >= edges[b] && t < edges[b + 1]) { sum[b] += v; cnt[b]++; }
        if ((t & 1023) == 0) fprintf(stderr, "R15_LONG_PROGRESS target=%d\n", t);
    }
    double all = sum[0] + sum[1] + sum[2]; long call = cnt[0] + cnt[1] + cnt[2];
    printf("{\"tokens\":%d,\"nonfinite_rows\":%ld,\"mean_nll_all\":%.17g,\"targets_all\":%ld", N, nonfinite, all / call, call);
    const char *names[3] = {"1_1023", "1024_4095", "4096_end"};
    for (int b = 0; b < 3; b++) printf(",\"mean_nll_%s\":%.17g,\"targets_%s\":%ld", names[b], cnt[b] ? sum[b] / cnt[b] : NAN, names[b], cnt[b]);
    printf("}\n");
    return nonfinite ? 3 : 0;
}
