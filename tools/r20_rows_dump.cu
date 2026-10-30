// R20 bit-exactness probe (E2, E3 and PART A-3(d)): runs the multi-row Gemma 4 entry points on one prompt and prints
// an FNV-1a 64 hash per section, so two builds (main vs a branch) can be compared line by line; with a dump prefix
// it also writes the raw bytes. Sections:
//   verify_concat S=<n>   1 slot, Nb = n (2..8) rows after a prefill, logits [n, V]        (spec serving verify)
//   verify_concat ragged  2 slots, Nb = {3, 5}                                            (concat grouping)
//   verify_batch S=<n>    per-slot canonical path (gemma4_decode amort=0, plain gemv rows)  (A-3(b))
//   batch W=<w>           gemma4_decode_batch, W slots (w = 2..8), 4 steps, logits [W, V] per step
//   prefill P=<p>         gemma4_decode(Sq = p, last_only, amort = 1) last logits + whole KV (P = 512, 1024)
//   chunk 512+512         same prompt prefilled as two chunks: last logits + KV; also compared in-process to P=1024
//   chunk 1024+6 / 1030   A-3(d): a chunk below 16 tokens vs one 1030-token prefill (compared in-process)
// KV buffers are zeroed after allocation so the whole buffer is comparable. RoPE tables come from the manifest.
// Usage: r20_rows_dump <weights> <prompt ids .bin (>= 1100 tokens)> [dump prefix]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"
#include "../kernels/kv_store.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string PFX;
static uint64_t fnv(const void *p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t *b = (const uint8_t *)p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; } return h;
}
static std::vector<uint8_t> d2h(const void *d, size_t n) { std::vector<uint8_t> h(n); CK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost)); return h; }
static void emit(const std::string &name, const std::vector<uint8_t> &h) {
    printf("R20_HASH %-28s bytes=%zu fnv=%016llx\n", name.c_str(), h.size(), (unsigned long long)fnv(h.data(), h.size()));
    if (!PFX.empty()) { std::string f = PFX + "." + name; for (auto &ch : f) if (ch == ' ' || ch == '=' || ch == '+') ch = '_';
        FILE *o = fopen(f.c_str(), "wb"); fwrite(h.data(), 1, h.size(), o); fclose(o); }
}
static std::vector<int> rii(const std::string &p) {
    FILE *f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long b = ftell(f); rewind(f); std::vector<int> v(b / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1); fclose(f); return v;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 3) { fprintf(stderr, "usage: %s <weights> <prompt.bin> [dump prefix]\n", argv[0]); return 1; }
    const std::string W = argv[1]; const std::vector<int> ids = rii(argv[2]); if (argc > 3) PFX = argv[3];
    if (ids.size() < 1100) { fprintf(stderr, "prompt needs >= 1100 tokens (has %zu)\n", ids.size()); return 1; }
    ModelConfig c = load_config_gemma4((W + "/manifest.json").c_str());
    const int V = c.vocab_size, hd_f = c.global_head_dim ? c.global_head_dim : c.head_dim, hd_s = c.head_dim, uq = 2;
    const int ML = 1056;
    float *cf, *sf, *cs, *ss;
    CK(cudaMalloc(&cf, (size_t)ML * hd_f * 4)); CK(cudaMalloc(&sf, (size_t)ML * hd_f * 4));
    CK(cudaMalloc(&cs, (size_t)ML * hd_s * 4)); CK(cudaMalloc(&ss, (size_t)ML * hd_s * 4));
    if (gemma4_gen_rope_tables((W + "/manifest.json").c_str(), ML, hd_f, cf, sf, hd_s, cs, ss)) { fprintf(stderr, "rope\n"); return 1; }
    G4Weights w = load_weights_gemma4(W.c_str(), c, 1, 0);
    int *d_tok; CK(cudaMalloc(&d_tok, (size_t)ML * 4));
    float *d_lg; CK(cudaMalloc(&d_lg, (size_t)16 * V * 4));
    auto kv_bytes = [&](int l) { const bool full = c.layer_is_full[l]; const int nkv = full ? (c.num_global_kv_heads ? c.num_global_kv_heads : c.num_kv_heads) : c.num_kv_heads;
                                 return (size_t)nkv * (full ? hd_f : hd_s) * g4_kv_elem(); };
    auto zero_single = [&](G4DecodeState &st) { for (int l = 0; l < c.num_layers; l++) { CK(cudaMemset(st.Kc[l], 0, ML * kv_bytes(l))); CK(cudaMemset(st.Vc[l], 0, ML * kv_bytes(l))); } };
    auto kv_single = [&](G4DecodeState &st) { std::vector<uint8_t> all; for (int l = 0; l < c.num_layers; l++) { auto k = d2h(st.Kc[l], ML * kv_bytes(l)), v = d2h(st.Vc[l], ML * kv_bytes(l));
                                 all.insert(all.end(), k.begin(), k.end()); all.insert(all.end(), v.begin(), v.end()); } return all; };
    auto prefill = [&](G4DecodeState &st, int off, int n) {
        CK(cudaMemcpy(d_tok, ids.data() + off, (size_t)n * 4, cudaMemcpyHostToDevice));
        gemma4_decode(c, w, st, d_tok, n, cf, sf, cs, ss, d_lg, 1, uq, 1, 1, nullptr);
        return d2h(d_lg, (size_t)V * 4);
    };
    // ---- prefill and chunk sections ----
    std::vector<uint8_t> lg1024, kv1024;
    for (int P : {512, 1024}) {
        G4DecodeState st = alloc_g4_decode_state(c, ML); zero_single(st);
        auto lg = prefill(st, 0, P); auto kv = kv_single(st);
        emit("prefill P=" + std::to_string(P) + " logits", lg); emit("prefill P=" + std::to_string(P) + " kv", kv);
        if (P == 1024) { lg1024 = lg; kv1024 = kv; }
        free_g4_decode_state(st);
    }
    {
        G4DecodeState st = alloc_g4_decode_state(c, ML); zero_single(st);
        prefill(st, 0, 512); auto lg = prefill(st, 512, 512); auto kv = kv_single(st);
        emit("chunk 512+512 logits", lg); emit("chunk 512+512 kv", kv);
        printf("R20_CHECK chunk512+512==P1024 logits=%d kv=%d\n", lg == lg1024, kv == kv1024);
        free_g4_decode_state(st);
    }
    {
        G4DecodeState a = alloc_g4_decode_state(c, ML); zero_single(a);
        auto la = prefill(a, 0, 1030); auto ka = kv_single(a);
        G4DecodeState b = alloc_g4_decode_state(c, ML); zero_single(b);
        prefill(b, 0, 1024); auto lb = prefill(b, 1024, 6); auto kb = kv_single(b);
        emit("prefill P=1030 logits", la); emit("chunk 1024+6 logits", lb);
        long nd = 0; for (size_t i = 0; i < la.size(); i += 4) nd += memcmp(&la[i], &lb[i], 4) != 0;
        long kd = 0; for (size_t i = 0; i < ka.size(); i++) kd += ka[i] != kb[i];
        printf("R20_CHECK chunk1024+6==P1030 logits=%d logit_words_differ=%ld kv=%d kv_bytes_differ=%ld\n", la == lb, nd, ka == kb, kd);
        free_g4_decode_state(a); free_g4_decode_state(b);
    }
    // ---- verify (concat and per-slot) ----
    const int P0 = 300;
    auto batch_prefill = [&](G4BatchState &bs, int slot, int off) {
        CK(cudaMemcpy(d_tok, ids.data() + off, (size_t)P0 * 4, cudaMemcpyHostToDevice));
        gemma4_prefill_batch_slot(c, w, bs, slot, d_tok, P0, cf, sf, cs, ss, d_lg, uq);
    };
    for (int mode = 0; mode < 2; mode++) {
        for (int S = 2; S <= 8; S++) {
            G4BatchState bs = alloc_g4_batch_state(c, 2, ML); batch_prefill(bs, 0, 0);
            int slot = 0, Nb = S, pos = P0;
            CK(cudaMemcpy(d_tok, ids.data() + P0, (size_t)S * 4, cudaMemcpyHostToDevice));
            if (mode == 0) gemma4_verify_concat(c, w, bs, 1, &slot, &Nb, &pos, d_tok, cf, sf, cs, ss, d_lg, uq);
            else           gemma4_verify_batch(c, w, bs, 1, &slot, &Nb, &pos, d_tok, cf, sf, cs, ss, d_lg, uq);
            CK(cudaDeviceSynchronize());
            emit(std::string(mode == 0 ? "verify_concat" : "verify_batch") + " S=" + std::to_string(S), d2h(d_lg, (size_t)S * V * 4));
            free_g4_batch_state(bs);
        }
    }
    {
        G4BatchState bs = alloc_g4_batch_state(c, 2, ML); batch_prefill(bs, 0, 0); batch_prefill(bs, 1, 400);
        int slots[2] = {0, 1}, Nb[2] = {3, 5}, pos[2] = {P0, P0};
        std::vector<int> cat(ids.begin() + P0, ids.begin() + P0 + 3); cat.insert(cat.end(), ids.begin() + 700, ids.begin() + 705);
        CK(cudaMemcpy(d_tok, cat.data(), cat.size() * 4, cudaMemcpyHostToDevice));
        gemma4_verify_concat(c, w, bs, 2, slots, Nb, pos, d_tok, cf, sf, cs, ss, d_lg, uq); CK(cudaDeviceSynchronize());
        emit("verify_concat ragged", d2h(d_lg, (size_t)8 * V * 4));
        free_g4_batch_state(bs);
    }
    // ---- batch decode W = 2..8 ----
    for (int Wn = 2; Wn <= 8; Wn++) {
        G4BatchState bs = alloc_g4_batch_state(c, Wn, ML);
        for (int b = 0; b < Wn; b++) batch_prefill(bs, b, 0);   // same-length slots (base = st.n_past)
        bs.n_past = P0;
        std::vector<uint8_t> all;
        for (int step = 0; step < 4; step++) {
            std::vector<int> t(Wn); for (int b = 0; b < Wn; b++) t[b] = ids[P0 + 50 * b + step];
            CK(cudaMemcpy(d_tok, t.data(), (size_t)Wn * 4, cudaMemcpyHostToDevice));
            gemma4_decode_batch(c, w, bs, d_tok, cf, sf, cs, ss, d_lg, 1, uq); CK(cudaDeviceSynchronize());
            auto h = d2h(d_lg, (size_t)Wn * V * 4); all.insert(all.end(), h.begin(), h.end());
        }
        emit("batch W=" + std::to_string(Wn), all);
        free_g4_batch_state(bs);
    }
    printf("R20_DONE\n");
    return 0;
}
