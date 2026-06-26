// 통합 검증: SmolLM3 graph 출력 vs HF 골든.
//   (a) 레이어별 입력 hidden state 단계 대조 (어디서 갈라지는지 특정)
//   (b) 최종 logits 대조.
// 기준 골든 = golden_q4 (엔진과 동일 q4 가중치 → 커널/graph 버그만 격리, 헌법 검증전략 A).
// fp32 골든과의 비교는 정보용(quant budget).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest.h"
#include "../runtime/weights.h"
#include "../models/smollm3.h"

#define TCK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static std::vector<float> read_f32(const std::string &p, size_t n) {
    FILE *f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "못 엶 %s\n", p.c_str()); exit(1); }
    std::vector<float> v(n); if (fread(v.data(), 4, n, f) != n) { fprintf(stderr, "크기 %s\n", p.c_str()); exit(1); } fclose(f); return v;
}
static std::vector<int> read_i32(const std::string &p) {
    FILE *f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "못 엶 %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<int> v(b / 4); if (fread(v.data(), 4, v.size(), f) != v.size()) { exit(1); } fclose(f); return v;
}
static double maxabs(const std::vector<float> &a, const std::vector<float> &b) {
    double e = 0; for (size_t i = 0; i < a.size(); i++) e = fmax(e, fabs((double)a[i] - b[i])); return e;
}

int main(int argc, char **argv) {
    std::string wdir = argc > 1 ? argv[1] : "artifacts/smollm3-3b/weights";
    std::string gdir = argc > 2 ? argv[2] : "artifacts/smollm3-3b/golden";

    ModelConfig c = load_config((wdir + "/manifest.json").c_str());
    const int H = c.hidden_size, V = c.vocab_size, L = c.num_layers, HD = c.head_dim;
    printf("[cfg] H=%d FFN=%d L=%d NQ=%d NKV=%d HD=%d V=%d eps=%.0e rope_layers=%d/%d\n",
           H, c.intermediate_size, L, c.num_q_heads, c.num_kv_heads, HD, V, c.rms_norm_eps,
           [&]{int s=0;for(int i=0;i<L;i++)s+=c.use_rope[i];return s;}(), L);

    std::vector<int> ids = read_i32(gdir + "/token_ids.bin");
    int S = (int)ids.size();
    printf("[in] S=%d tokens=", S); for (int t : ids) printf("%d ", t); printf("\n");

    std::vector<float> cos = read_f32(gdir + "/rope_cos.bin", (size_t)S * HD);
    std::vector<float> sin = read_f32(gdir + "/rope_sin.bin", (size_t)S * HD);

    int *d_ids; TCK(cudaMalloc(&d_ids, S * 4)); TCK(cudaMemcpy(d_ids, ids.data(), S * 4, cudaMemcpyHostToDevice));
    float *d_cos, *d_sin;
    TCK(cudaMalloc(&d_cos, cos.size() * 4)); TCK(cudaMemcpy(d_cos, cos.data(), cos.size() * 4, cudaMemcpyHostToDevice));
    TCK(cudaMalloc(&d_sin, sin.size() * 4)); TCK(cudaMemcpy(d_sin, sin.data(), sin.size() * 4, cudaMemcpyHostToDevice));

    printf("[load] weights from %s ...\n", wdir.c_str());
    EngineWeights w = load_weights(wdir.c_str(), c);

    const size_t SH = (size_t)S * H;
    float *d_logits, *d_hidden;
    TCK(cudaMalloc(&d_logits, (size_t)S * V * 4));
    TCK(cudaMalloc(&d_hidden, (size_t)(L + 1) * SH * 4));

    printf("[run] forward...\n");
    smollm3_forward(c, w, d_ids, S, d_cos, d_sin, d_logits, d_hidden);

    std::vector<float> hidden((size_t)(L + 1) * SH);
    TCK(cudaMemcpy(hidden.data(), d_hidden, hidden.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<float> logits((size_t)S * V);
    TCK(cudaMemcpy(logits.data(), d_logits, logits.size() * 4, cudaMemcpyDeviceToHost));

    // ---- (a) 레이어별 단계 대조 (vs golden_q4) ----
    printf("\n=== per-layer hidden state: engine vs golden_q4 (max abs err) ===\n");
    double worst = 0; int worst_l = -1;
    for (int l = 0; l <= L; l++) {
        std::vector<float> g = read_f32(gdir + "/q4_hs_" + std::to_string(l) + ".bin", SH);
        std::vector<float> e(hidden.begin() + (size_t)l * SH, hidden.begin() + (size_t)(l + 1) * SH);
        double err = maxabs(e, g);
        if (err > worst) { worst = err; worst_l = l; }
        if (l <= 2 || l >= L - 1 || err > 0.05)
            printf("  hs[%2d] err=%.4e\n", l, err);
    }
    printf("  → worst per-layer hidden err = %.4e at layer %d\n", worst, worst_l);

    // ---- (b) 최종 logits 대조 ----
    std::vector<float> glog_q4   = read_f32(gdir + "/q4_logits.bin",   (size_t)S * V);
    std::vector<float> glog_fp32 = read_f32(gdir + "/fp32_logits.bin", (size_t)S * V);
    double err_q4 = maxabs(logits, glog_q4);
    double err_fp32 = maxabs(logits, glog_fp32);
    printf("\n=== final logits ===\n");
    printf("  engine vs golden_q4   max abs err = %.4e  (정확성 기준)\n", err_q4);
    printf("  engine vs golden_fp32 max abs err = %.4e  (quant budget 포함)\n", err_fp32);

    // argmax (마지막 위치 = next-token 예측) 대조
    auto argmax_at = [&](const std::vector<float> &lg, int pos) {
        int best = 0; float bv = lg[(size_t)pos * V];
        for (int j = 1; j < V; j++) { float x = lg[(size_t)pos * V + j]; if (x > bv) { bv = x; best = j; } }
        return best;
    };
    printf("  argmax per pos: engine=[");
    bool am_match_q4 = true, am_match_fp32 = true;
    for (int p = 0; p < S; p++) {
        int ae = argmax_at(logits, p), aq = argmax_at(glog_q4, p), af = argmax_at(glog_fp32, p);
        printf("%d ", ae); if (ae != aq) am_match_q4 = false; if (ae != af) am_match_fp32 = false;
    }
    printf("]\n  argmax match: vs golden_q4=%s, vs golden_fp32=%s\n",
           am_match_q4 ? "ALL" : "partial", am_match_fp32 ? "ALL" : "partial");
    int last_e = argmax_at(logits, S - 1), last_f = argmax_at(glog_fp32, S - 1);
    printf("  last-pos next-token: engine=%d  golden_fp32=%d  %s\n",
           last_e, last_f, last_e == last_f ? "MATCH" : "DIFF");

    free_weights(w); free_config(c);
    cudaFree(d_ids); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_logits); cudaFree(d_hidden);

    // 통과 기준: graph 수학(vs golden_q4) logits < 1e-2 AND 레이어별 발산 없음(< 1e-2) AND 마지막토큰 fp32 일치
    bool pass = (err_q4 < 1e-2) && (worst < 1e-2) && (last_e == last_f);
    printf("\n[test_smollm3] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
