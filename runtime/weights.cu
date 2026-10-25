#include "weights.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <cuda_runtime.h>

#define WCK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static void *load_dev(const std::string &path, size_t bytes) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "못 엶: %s\n", path.c_str()); exit(1); }
    void *host = malloc(bytes);
    if (fread(host, 1, bytes, f) != bytes) { fprintf(stderr, "크기 불일치: %s (%zu bytes)\n", path.c_str(), bytes); exit(1); }
    int advice_error = posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
    if (advice_error) fprintf(stderr, "warning: posix_fadvise %s: %s\n", path.c_str(), std::strerror(advice_error));
    fclose(f);
    void *dev; WCK(cudaMalloc(&dev, bytes)); WCK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    free(host); return dev;
}
static float   *load_f32(const std::string &p, size_t n) { return (float *)load_dev(p, n * sizeof(float)); }
static uint8_t *load_u8 (const std::string &p, size_t n) { return (uint8_t *)load_dev(p, n); }

EngineWeights load_weights(const char *dir, const ModelConfig &c) {
    std::string R = std::string(dir) + "/";
    const size_t H = c.hidden_size, FFN = c.intermediate_size, V = c.vocab_size;
    const size_t qdim = (size_t)c.num_q_heads * c.head_dim;
    const size_t kvdim = (size_t)c.num_kv_heads * c.head_dim;
    const int Q = QBLK;

    EngineWeights w{}; w.L = c.num_layers;
    w.embed      = load_f32(R + "w_embed.bin", V * H);
    w.final_norm = load_f32(R + "w_final_norm.bin", H);
    w.lm_head    = load_u8 (R + "w_lm_head_q4.bin", V * H / 2);
    w.lm_head_s  = load_f32(R + "w_lm_head_scale.bin", V * H / Q);

    w.layers = (LayerW *)calloc(c.num_layers, sizeof(LayerW));
    for (int i = 0; i < c.num_layers; i++) {
        std::string p = R + "L" + std::to_string(i) + "_";
        LayerW &L = w.layers[i];
        L.input_ln = load_f32(p + "input_ln.bin", H);
        L.post_ln  = load_f32(p + "post_ln.bin", H);
        // q4 (q4.bin = OUT*IN/2 bytes), scale (OUT*IN/QBLK floats)
        L.q = load_u8(p + "q_q4.bin", qdim * H / 2);   L.q_s = load_f32(p + "q_scale.bin", qdim * H / Q);
        L.k = load_u8(p + "k_q4.bin", kvdim * H / 2);  L.k_s = load_f32(p + "k_scale.bin", kvdim * H / Q);
        L.v = load_u8(p + "v_q4.bin", kvdim * H / 2);  L.v_s = load_f32(p + "v_scale.bin", kvdim * H / Q);
        L.o = load_u8(p + "o_q4.bin", H * qdim / 2);   L.o_s = load_f32(p + "o_scale.bin", H * qdim / Q);
        L.gate = load_u8(p + "gate_q4.bin", FFN * H / 2); L.gate_s = load_f32(p + "gate_scale.bin", FFN * H / Q);
        L.up   = load_u8(p + "up_q4.bin", FFN * H / 2);   L.up_s   = load_f32(p + "up_scale.bin", FFN * H / Q);
        L.down = load_u8(p + "down_q4.bin", H * FFN / 2); L.down_s = load_f32(p + "down_scale.bin", H * FFN / Q);
    }
    return w;
}

void free_weights(EngineWeights &w) {
    auto F = [](void *p) { if (p) cudaFree(p); };
    F(w.embed); F(w.final_norm); F(w.lm_head); F(w.lm_head_s);
    for (int i = 0; i < w.L; i++) {
        LayerW &L = w.layers[i];
        F(L.input_ln); F(L.post_ln);
        F(L.q); F(L.k); F(L.v); F(L.o); F(L.gate); F(L.up); F(L.down);
        F(L.q_s); F(L.k_s); F(L.v_s); F(L.o_s); F(L.gate_s); F(L.up_s); F(L.down_s);
    }
    free(w.layers); w.layers = nullptr; w.L = 0;
}
