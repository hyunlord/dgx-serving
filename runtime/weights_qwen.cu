#include "weights_qwen.h"
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
    if (fread(host, 1, bytes, f) != bytes) { fprintf(stderr, "크기 불일치: %s (%zu)\n", path.c_str(), bytes); exit(1); }
    int advice_error = posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
    if (advice_error) fprintf(stderr, "warning: posix_fadvise %s: %s\n", path.c_str(), std::strerror(advice_error));
    fclose(f);
    void *dev; WCK(cudaMalloc(&dev, bytes)); WCK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    free(host); return dev;
}

namespace {
std::string R; int QB; int LQ4, LF32; int SD;
float *f32(const std::string &n, size_t cnt) { return (float *)load_dev(R + n + ".bin", cnt * 4); }
// matmul weight [OUT,IN]: q4 경로(wq + sc) / fp32 경로(wf), 플래그에 따라 로드.
QMat mat(const std::string &n, size_t OUT, size_t IN) {
    QMat m;
    if (LQ4) {
        m.wq = (uint8_t *)load_dev(R + n + "_q4.bin", OUT * IN / 2);
        m.sc = (void *)load_dev(R + n + "_scale.bin", (OUT * IN / QB) * (SD == 1 ? 2 : 4));
    }
    if (LF32) m.wf = (float *)load_dev(R + n + "_f32.bin", OUT * IN * 4);
    return m;
}
}

QwenWeights load_weights_qwen(const char *dir, const ModelConfig &c, int load_q4, int load_fp32) {
    R = std::string(dir) + "/"; QB = QBLK; LQ4 = load_q4; LF32 = load_fp32; SD = c.scale_dtype;
    const size_t H = c.hidden_size, FFN = c.intermediate_size, V = c.vocab_size, HD = c.head_dim;
    const size_t qdim2 = (size_t)c.num_q_heads * HD * (c.attn_output_gate ? 2 : 1);
    const size_t kvdim = (size_t)c.num_kv_heads * HD;
    const size_t adim = (size_t)c.num_q_heads * HD;
    const size_t conv_dim = cfg_gdn_conv_dim(&c), vdim = cfg_gdn_value_dim(&c);
    const size_t Hv = c.gdn_head_v_dim, nvh = c.gdn_num_v_heads;

    QwenWeights w{}; w.L = c.num_layers; w.have_q4 = load_q4; w.have_fp32 = load_fp32;
    w.embed      = f32("w_embed", V * H);
    w.final_norm = f32("w_final_norm", H);
    w.lm_head    = mat("w_lm_head", V, H);

    w.layers = (QLayer *)calloc(c.num_layers, sizeof(QLayer));
    for (int i = 0; i < c.num_layers; i++) {
        QLayer &L = w.layers[i];
        std::string p = "L" + std::to_string(i) + "_";
        L.is_full = c.layer_is_full ? c.layer_is_full[i] : 1;
        L.input_ln = f32(p + "input_layernorm", H);
        L.post_ln  = f32(p + "post_attention_layernorm", H);
        if (c.num_experts > 0) {                          // MoE MLP (batched experts + router + shared)
            const size_t E = c.num_experts, MI = c.moe_intermediate_size, SI = c.shared_expert_intermediate_size;
            L.experts_gate_up = mat(p + "mlp.experts.gate_up_proj", E * 2 * MI, H);   // [E,2*inter,H] flat
            L.experts_down    = mat(p + "mlp.experts.down_proj",    E * H, MI);        // [E,H,inter] flat
            L.router          = f32(p + "mlp.gate", E * H);                            // [E,H] fp32
            L.shared_gate     = mat(p + "mlp.shared_expert.gate_proj", SI, H);
            L.shared_up       = mat(p + "mlp.shared_expert.up_proj",   SI, H);
            L.shared_down     = mat(p + "mlp.shared_expert.down_proj", H, SI);
            L.shared_gate_w   = f32(p + "mlp.shared_expert_gate", H);                  // [1,H] fp32
        } else {                                          // dense MLP
            L.gate = mat(p + "mlp.gate_proj", FFN, H);
            L.up   = mat(p + "mlp.up_proj",   FFN, H);
            L.down = mat(p + "mlp.down_proj", H, FFN);
        }
        if (L.is_full) {
            L.q = mat(p + "self_attn.q_proj", qdim2, H);
            L.k = mat(p + "self_attn.k_proj", kvdim, H);
            L.v = mat(p + "self_attn.v_proj", kvdim, H);
            L.o = mat(p + "self_attn.o_proj", H, adim);
            L.q_norm = f32(p + "self_attn.q_norm", HD);
            L.k_norm = f32(p + "self_attn.k_norm", HD);
        } else {
            L.in_qkv = mat(p + "linear_attn.in_proj_qkv", conv_dim, H);
            L.in_z   = mat(p + "linear_attn.in_proj_z", vdim, H);
            L.out_p  = mat(p + "linear_attn.out_proj", H, vdim);
            L.in_a    = f32(p + "linear_attn.in_proj_a", (size_t)nvh * H);
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
    auto FM = [&](QMat &m) { F(m.wq); F(m.sc); F(m.wf); };
    F(w.embed); F(w.final_norm); FM(w.lm_head);
    for (int i = 0; i < w.L; i++) {
        QLayer &L = w.layers[i];
        F(L.input_ln); F(L.post_ln);
        FM(L.gate); FM(L.up); FM(L.down);
        FM(L.q); FM(L.k); FM(L.v); FM(L.o); F(L.q_norm); F(L.k_norm);
        FM(L.in_qkv); FM(L.in_z); FM(L.out_p);
        F(L.in_a); F(L.in_b); F(L.conv1d); F(L.A_log); F(L.dt_bias); F(L.gdn_norm);
        FM(L.experts_gate_up); FM(L.experts_down); F(L.router);
        FM(L.shared_gate); FM(L.shared_up); FM(L.shared_down); F(L.shared_gate_w);
    }
    free(w.layers); w.layers = nullptr; w.L = 0;
}
