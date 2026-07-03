#include "weights_gemma4.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#define WCK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static void *load_dev_g4(const std::string &path, size_t bytes) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "못 엶: %s\n", path.c_str()); exit(1); }
    void *host = malloc(bytes);
    if (fread(host, 1, bytes, f) != bytes) { fprintf(stderr, "크기 불일치: %s (%zu)\n", path.c_str(), bytes); exit(1); }
    fclose(f);
    void *dev; WCK(cudaMalloc(&dev, bytes)); WCK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    free(host); return dev;
}

namespace {
std::string R; int QB, LQ4, LF32, SD;
float *f32(const std::string &n, size_t cnt) { return (float *)load_dev_g4(R + n + ".bin", cnt * 4); }
float host_scalar(const std::string &n) {   // 1-float bin → host (layer_scalar)
    FILE *f = fopen((R + n + ".bin").c_str(), "rb");
    if (!f) { fprintf(stderr, "못 엶: %s\n", (R + n).c_str()); exit(1); }
    float v; if (fread(&v, 4, 1, f) != 1) { fprintf(stderr, "스칼라 read err: %s\n", n.c_str()); exit(1); }
    fclose(f); return v;
}
QMat mat(const std::string &n, size_t OUT, size_t IN) {
    QMat m;
    if (LQ4) {
        m.wq = (uint8_t *)load_dev_g4(R + n + "_q4.bin", OUT * IN / 2);
        m.sc = (void *)load_dev_g4(R + n + "_scale.bin", (OUT * IN / QB) * (SD == 1 ? 2 : 4));
    }
    if (LF32) m.wf = (float *)load_dev_g4(R + n + "_f32.bin", OUT * IN * 4);
    return m;
}
}

G4Weights load_weights_gemma4(const char *dir, const ModelConfig &c, int load_q4, int load_fp32) {
    R = std::string(dir) + "/"; QB = QBLK; LQ4 = load_q4; LF32 = load_fp32; SD = c.scale_dtype;
    const size_t H = c.hidden_size, FFN = c.intermediate_size, V = c.vocab_size;

    G4Weights w{}; w.L = c.num_layers; w.have_q4 = load_q4; w.have_fp32 = load_fp32;
    w.embed      = f32("w_embed", V * H);
    w.final_norm = f32("w_final_norm", H);
    w.lm_head    = mat("w_lm_head", V, H);

    w.layers = (G4Layer *)calloc(c.num_layers, sizeof(G4Layer));
    for (int i = 0; i < c.num_layers; i++) {
        G4Layer &L = w.layers[i];
        std::string p = "L" + std::to_string(i) + "_";
        L.is_full = c.layer_is_full ? c.layer_is_full[i] : 1;
        // 층 타입별 head 구성 (config 주도 — full: global_head_dim/num_global_kv_heads)
        const size_t hd  = (L.is_full && c.global_head_dim)     ? c.global_head_dim     : c.head_dim;
        const size_t nkv = (L.is_full && c.num_global_kv_heads) ? c.num_global_kv_heads : c.num_kv_heads;
        const size_t qdim = (size_t)c.num_q_heads * hd, kvdim = nkv * hd;

        L.input_ln     = f32(p + "input_layernorm", H);
        L.post_attn_ln = f32(p + "post_attention_layernorm", H);
        L.pre_ffn_ln   = f32(p + "pre_feedforward_layernorm", H);
        L.post_ffn_ln  = f32(p + "post_feedforward_layernorm", H);
        L.layer_scalar = host_scalar(p + "layer_scalar");

        L.q = mat(p + "self_attn.q_proj", qdim, H);
        L.k = mat(p + "self_attn.k_proj", kvdim, H);
        if (!(L.is_full && c.attention_k_eq_v))
            L.v = mat(p + "self_attn.v_proj", kvdim, H);   // full+k_eq_v 층은 v_proj 없음(가중치 자체 부재)
        L.o = mat(p + "self_attn.o_proj", H, qdim);
        L.q_norm = f32(p + "self_attn.q_norm", hd);
        L.k_norm = f32(p + "self_attn.k_norm", hd);

        L.gate = mat(p + "mlp.gate_proj", FFN, H);
        L.up   = mat(p + "mlp.up_proj",   FFN, H);
        L.down = mat(p + "mlp.down_proj", H, FFN);
    }
    return w;
}

void free_weights_gemma4(G4Weights &w) {
    auto F = [](void *p) { if (p) cudaFree(p); };
    auto FM = [&](QMat &m) { F(m.wq); F(m.sc); F(m.wf); };
    F(w.embed); F(w.final_norm); FM(w.lm_head);
    for (int i = 0; i < w.L; i++) {
        G4Layer &L = w.layers[i];
        F(L.input_ln); F(L.post_attn_ln); F(L.pre_ffn_ln); F(L.post_ffn_ln);
        FM(L.q); FM(L.k); FM(L.v); FM(L.o); F(L.q_norm); F(L.k_norm);
        FM(L.gate); FM(L.up); FM(L.down);
    }
    free(w.layers); w.layers = nullptr; w.L = 0;
}
