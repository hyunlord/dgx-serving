// R16 knob-registry gate probe (dense Gemma 4, uq=2): writes byte-comparable outputs so the same
// env can be run on two builds and compared with cmp.
//   <out>.prefill1024_last.f32  last-row logits of a 1024-token prefill (golden prompt repeated)
//   <out>.decode128.f32         logits of 128 greedy S=1 decode steps after that prefill
//   <out>.decode128.i32         the 128 greedy tokens
// Usage: r16_knob_gate <weights> <golden dir with token_ids.bin> <out_prefix>
// Uses only gemma4_decode (no registry headers), so it builds unchanged on main and on the branch.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> static T *alloc(size_t n) { T *p; ck(cudaMalloc(&p, n * sizeof(T))); return p; }
static std::vector<int> read_ids(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("missing tokens: " + path);
    fseek(f, 0, SEEK_END); long bytes = ftell(f); rewind(f);
    if (bytes <= 0 || bytes % 4) throw std::runtime_error("invalid token file");
    std::vector<int> ids(bytes / 4);
    if (fread(ids.data(), 4, ids.size(), f) != ids.size()) throw std::runtime_error("token read failed");
    fclose(f); return ids;
}
template<class T> static void write_new(const std::string &path, const T *p, size_t n) {
    FILE *f = fopen(path.c_str(), "wbx");
    if (!f) throw std::runtime_error("output exists or cannot be created: " + path);
    if (fwrite(p, sizeof(T), n, f) != n || fclose(f)) throw std::runtime_error("write failed: " + path);
}

int main(int argc, char **argv) try {
    if (argc != 4) { fprintf(stderr, "usage: %s <weights> <golden dir> <out_prefix>\n", argv[0]); return 1; }
    const std::string wd = argv[1], out = argv[3], manifest = wd + "/manifest.json";
    ModelConfig c = load_config_gemma4(manifest.c_str());
    if (c.num_experts) throw std::runtime_error("dense model required");
    const int V = c.vocab_size, H = c.hidden_size, hd_s = c.head_dim, hd_f = c.global_head_dim ? c.global_head_dim : hd_s;
    const int P = 1024, N = 128, ML = P + N + 8;
    auto prompt = read_ids(std::string(argv[2]) + "/token_ids.bin");
    std::vector<int> ids(P);
    for (int j = 0; j < P; j++) ids[j] = prompt[j % prompt.size()];
    float *cf = alloc<float>((size_t)ML * hd_f), *sf = alloc<float>((size_t)ML * hd_f);
    float *cs = alloc<float>((size_t)ML * hd_s), *ss = alloc<float>((size_t)ML * hd_s);
    if (gemma4_gen_rope_tables(manifest.c_str(), ML, hd_f, cf, sf, hd_s, cs, ss)) throw std::runtime_error("RoPE generation failed");
    G4Weights w = load_weights_gemma4(wd.c_str(), c, 1, 0);
    int *dt = alloc<int>(P); float *lg = alloc<float>(V); float *pn = alloc<float>((size_t)P * H);
    G4DecodeState st = alloc_g4_decode_state(c, ML);
    ck(cudaMemcpy(dt, ids.data(), (size_t)P * 4, cudaMemcpyHostToDevice));
    gemma4_decode(c, w, st, dt, P, cf, sf, cs, ss, lg, 1, 2, 1, 1, pn);
    std::vector<float> h(V), all((size_t)N * V);
    ck(cudaMemcpy(h.data(), lg, (size_t)V * 4, cudaMemcpyDeviceToHost));
    write_new(out + ".prefill1024_last.f32", h.data(), h.size());
    std::vector<int> toks(N);
    int tok = (int)(std::max_element(h.begin(), h.end()) - h.begin());
    for (int t = 0; t < N; t++) {
        ck(cudaMemcpy(dt, &tok, 4, cudaMemcpyHostToDevice));
        gemma4_decode(c, w, st, dt, 1, cf, sf, cs, ss, lg, 1, 2, 1, 0, pn);
        ck(cudaMemcpy(all.data() + (size_t)t * V, lg, (size_t)V * 4, cudaMemcpyDeviceToHost));
        const float *row = all.data() + (size_t)t * V;
        for (int v = 0; v < V; v++) if (!std::isfinite(row[v])) throw std::runtime_error("non-finite logits");
        tok = (int)(std::max_element(row, row + V) - row);
        toks[t] = tok;
    }
    write_new(out + ".decode128.f32", all.data(), all.size());
    write_new(out + ".decode128.i32", toks.data(), toks.size());
    printf("R16_KNOB_GATE_DONE P=%d N=%d first_tokens=%d,%d,%d last=%d\n", P, N, toks[0], toks[1], toks[2], toks[N - 1]);
    return 0;
} catch (const std::exception &e) { fprintf(stderr, "r16_knob_gate: %s\n", e.what()); return 1; }
