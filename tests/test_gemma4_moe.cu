// Gemma4 26B-A4B MoE 스모크 (세션1: 빌드+로드+forward 크래시無+logits 유한만. 골든 정확성은 다음 세션).
//   ★골든/HF reference 대조 아님. rope는 엔진 내부 생성(launch_gen_rope, rot=hd 전체 — rotary_dim 미반영).
//   목표: 18GB q4 로드 + dense∥MoE prefill(5토큰) 실행 시 크래시 없음 + logits에 NaN/Inf 없음.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../models/gemma4_moe.h"
#include "../kernels/rope.cuh"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

int main(int argc, char **argv) {
    std::string WD = argc > 1 ? argv[1] : "artifacts/gemma4-26b-a4b/weights";
    int uq = argc > 2 ? atoi(argv[2]) : 1;   // 1=q4(활성화 fp32) / 2=W4A8. experts는 moe_fused q4-weight 공통.
    ModelConfig c = load_config_gemma4((WD + "/manifest.json").c_str());
    const int H = c.hidden_size, V = c.vocab_size, L = c.num_layers;
    const int hd_s = c.head_dim, hd_f = c.global_head_dim ? c.global_head_dim : hd_s;
    int nf = 0; for (int i = 0; i < L; i++) nf += c.layer_is_full[i];
    printf("[cfg] H=%d L=%d(full=%d/sliding=%d w=%d) NQ=%d hd_s=%d/hd_f=%d NKV_s=%d/NKV_f=%d V=%d k_eq_v=%d\n"
           "      E=%d topk=%d MI(moe)=%d FFN(dense)=%d parallel_dense=%d router_gemma=%d softcap=%.1f embed_scale=%.2f\n",
           H, L, nf, L - nf, c.sliding_window, c.num_q_heads, hd_s, hd_f, c.num_kv_heads, c.num_global_kv_heads, V,
           c.attention_k_eq_v, c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.intermediate_size,
           c.moe_parallel_dense, c.router_gemma, c.final_logit_softcapping, c.embed_scale);

    // 짧은 프롬프트(5토큰) — 시스템 크래시 방지(full fp32 전개 없음). 임의 유효 토큰(골든 대조 아님).
    std::vector<int> ids = {2, 651, 1596, 573, 1001};
    int S = (int)ids.size();
    printf("[in] S=%d tokens=", S); for (int t : ids) printf("%d ", t); printf("\n");
    int *d_ids; TCK(cudaMalloc(&d_ids, S * 4)); TCK(cudaMemcpy(d_ids, ids.data(), S * 4, cudaMemcpyHostToDevice));

    // rope cos/sin 엔진 내부 생성(골든 불필요). full: [S,hd_f] θ=1e6 / sliding: [S,hd_s] θ=1e4.
    float *d_cf, *d_sf, *d_cs, *d_ss;
    TCK(cudaMalloc(&d_cf, (size_t)S * hd_f * 4)); TCK(cudaMalloc(&d_sf, (size_t)S * hd_f * 4));
    TCK(cudaMalloc(&d_cs, (size_t)S * hd_s * 4)); TCK(cudaMalloc(&d_ss, (size_t)S * hd_s * 4));
    launch_gen_rope(d_cf, d_sf, S, hd_f, 1e6f);
    launch_gen_rope(d_cs, d_ss, S, hd_s, 1e4f);

    printf("[load] weights (q4) ~18GB ...\n");
    G4Weights w = load_weights_gemma4(WD.c_str(), c, /*q4*/1, /*fp32*/0);

    float *d_logits; TCK(cudaMalloc(&d_logits, (size_t)S * V * 4));
    printf("[fwd] gemma4_moe_forward use_q4=%d (%s) ...\n", uq, uq == 2 ? "W4A8" : "q4");
    gemma4_moe_forward(c, w, d_ids, S, d_cf, d_sf, d_cs, d_ss, d_logits, nullptr, uq);

    std::vector<float> logits((size_t)S * V);
    TCK(cudaMemcpy(logits.data(), d_logits, logits.size() * 4, cudaMemcpyDeviceToHost));

    // 유한성 + argmax (골든 대조 아님 — 크래시無 + NaN/Inf 없음만 확인)
    size_t nnan = 0, ninf = 0; double amax = 0;
    for (size_t i = 0; i < logits.size(); i++) {
        float v = logits[i];
        if (std::isnan(v)) nnan++; else if (std::isinf(v)) ninf++; else amax = fmax(amax, fabs((double)v));
    }
    auto am = [&](int pos) { int b = 0; float bv = logits[(size_t)pos * V]; for (int j = 1; j < V; j++) { float x = logits[(size_t)pos * V + j]; if (x > bv) { bv = x; b = j; } } return b; };
    printf("[out] argmax/pos:"); for (int p = 0; p < S; p++) printf(" %d", am(p)); printf("\n");
    printf("[finite] NaN=%zu Inf=%zu maxabs=%.4f (softcap=%.0f → |logit|<=%.0f 기대)\n",
           nnan, ninf, amax, c.final_logit_softcapping, c.final_logit_softcapping);
    bool ok = (nnan == 0 && ninf == 0);
    printf("[test_gemma4_moe] %s (26B-A4B MoE 로드+dense∥MoE prefill: 크래시無 + logits 유한. 정확성 게이트=다음 세션 골든)\n",
           ok ? "PASS" : "FAIL");

    free_weights_gemma4(w); free_config_gemma4(c);
    return ok ? 0 : 1;
}
