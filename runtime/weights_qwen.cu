#include "weights_qwen.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#define WCK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static void *load_dev(const std::string &path, size_t bytes) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "못 엶: %s\n", path.c_str()); exit(1); }
    void *host = malloc(bytes);
    if (fread(host, 1, bytes, f) != bytes) { fprintf(stderr, "크기 불일치: %s (%zu)\n", path.c_str(), bytes); exit(1); }
    fclose(f);
    void *dev; WCK(cudaMalloc(&dev, bytes)); WCK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    free(host); return dev;
}

namespace {
std::string R;
int QB;
float *f32(const std::string &n, size_t cnt) { return (float *)load_dev(R + n + ".bin", cnt * 4); }
// q4 weight [OUT,IN]: q4 = OUT*IN/2 bytes, scale = OUT*IN/QB floats
void q4(const std::string &n, size_t OUT, size_t IN, uint8_t **wq, float **sc) {
    *wq = (uint8_t *)load_dev(R + n + "_q4.bin", OUT * IN / 2);            // uint8: count==bytes
    *sc = (float *)load_dev(R + n + "_scale.bin", (OUT * IN / QB) * 4);   // float: ×4 bytes
}
}

QwenWeights load_weights_qwen(const char *dir, const ModelConfig &c) {
    R = std::string(dir) + "/";
    QB = QBLK;
    const size_t H = c.hidden_size, FFN = c.intermediate_size, V = c.vocab_size, HD = c.head_dim;
    const size_t qdim2 = (size_t)c.num_q_heads * HD * (c.attn_output_gate ? 2 : 1);
    const size_t kvdim = (size_t)c.num_kv_heads * HD;
    const size_t adim = (size_t)c.num_q_heads * HD;            // o_proj IN
    const size_t conv_dim = cfg_gdn_conv_dim(&c);
    const size_t vdim = cfg_gdn_value_dim(&c);
    const size_t Hv = c.gdn_head_v_dim;
    const size_t nvh = c.gdn_num_v_heads;

    QwenWeights w{}; w.L = c.num_layers;
    w.embed      = f32("w_embed", V * H);
    w.final_norm = f32("w_final_norm", H);
    w.lm_head    = (uint8_t *)load_dev(R + "w_lm_head_q4.bin", V * H / 2);
    w.lm_head_s  = (float *)load_dev(R + "w_lm_head_scale.bin", (V * H / QB) * 4);

    w.layers = (QLayer *)calloc(c.num_layers, sizeof(QLayer));
    for (int i = 0; i < c.num_layers; i++) {
        QLayer &L = w.layers[i];
        std::string p = "L" + std::to_string(i) + "_";
        L.is_full = c.layer_is_full ? c.layer_is_full[i] : 1;
        L.input_ln = f32(p + "input_layernorm", H);
        L.post_ln  = f32(p + "post_attention_layernorm", H);
        // MLP (both types)
        q4(p + "mlp.gate_proj", FFN, H, &L.gate, &L.gate_s);
        q4(p + "mlp.up_proj",   FFN, H, &L.up,   &L.up_s);
        q4(p + "mlp.down_proj", H, FFN, &L.down, &L.down_s);
        if (L.is_full) {
            q4(p + "self_attn.q_proj", qdim2, H, &L.q, &L.q_s);
            q4(p + "self_attn.k_proj", kvdim, H, &L.k, &L.k_s);
            q4(p + "self_attn.v_proj", kvdim, H, &L.v, &L.v_s);
            q4(p + "self_attn.o_proj", H, adim, &L.o, &L.o_s);
            L.q_norm = f32(p + "self_attn.q_norm", HD);
            L.k_norm = f32(p + "self_attn.k_norm", HD);
        } else {
            q4(p + "linear_attn.in_proj_qkv", conv_dim, H, &L.in_qkv, &L.in_qkv_s);
            q4(p + "linear_attn.in_proj_z",   vdim, H, &L.in_z, &L.in_z_s);
            q4(p + "linear_attn.out_proj",    H, vdim, &L.out_p, &L.out_s);
            L.in_a    = f32(p + "linear_attn.in_proj_a", (size_t)nvh * H);   // [16,1024]
            L.in_b    = f32(p + "linear_attn.in_proj_b", (size_t)nvh * H);
            L.conv1d  = f32(p + "linear_attn.conv1d", conv_dim * c.gdn_conv_kernel);
            L.A_log   = f32(p + "linear_attn.A_log", nvh);
            L.dt_bias = f32(p + "linear_attn.dt_bias", nvh);
            L.gdn_norm= f32(p + "linear_attn.norm", Hv);
        }
    }
    return w;
}

void free_weights_qwen(QwenWeights &w) {
    auto F = [](void *p) { if (p) cudaFree(p); };
    F(w.embed); F(w.final_norm); F(w.lm_head); F(w.lm_head_s);
    for (int i = 0; i < w.L; i++) {
        QLayer &L = w.layers[i];
        F(L.input_ln); F(L.post_ln);
        F(L.gate); F(L.up); F(L.down); F(L.gate_s); F(L.up_s); F(L.down_s);
        F(L.q); F(L.k); F(L.v); F(L.o); F(L.q_s); F(L.k_s); F(L.v_s); F(L.o_s);
        F(L.q_norm); F(L.k_norm);
        F(L.in_qkv); F(L.in_z); F(L.out_p); F(L.in_qkv_s); F(L.in_z_s); F(L.out_s);
        F(L.in_a); F(L.in_b); F(L.conv1d); F(L.A_log); F(L.dt_bias); F(L.gdn_norm);
    }
    free(w.layers); w.layers = nullptr; w.L = 0;
}
