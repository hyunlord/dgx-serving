#pragma once
// MoE MLP 블록 (qwen3_5_moe): prefill(qwen35.cu)·decode(qwen35_decode.cu) 공유.
//   router top-k → 선택 expert SwiGLU 가중합 + shared expert(sigmoid 게이트). 백본/MLP 커널 재사용,
//   신규는 graph 배선 + router_topk 1개. 정확성-우선 naive(토큰×expert 루프). expert q4 = 배치 offset 인덱싱.
//   n2[S,H] (post_attention_layernorm 출력) → out[S,H].
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/weights_qwen.h"
#include "../kernels/gemv_q4.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/router_topk.cuh"

#define MOEK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static inline float *moe_dmalloc(size_t n) { float *p; MOEK(cudaMalloc(&p, n * 4)); return p; }

static inline void moe_mlp_block(const ModelConfig &c, const QLayer &L, const float *n2, float *out, int S) {
    const int H=c.hidden_size, E=c.num_experts, K=c.num_experts_per_tok;
    const int MI=c.moe_intermediate_size, SI=c.shared_expert_intermediate_size;
    const int sdtype=c.scale_dtype;
    const size_t gu_w=(size_t)2*MI*H, dn_w=(size_t)H*MI;   // expert당 weight 수
    const size_t sc_elem_sz = (sdtype==1) ? 2 : 4;  // scale 원소 크기 (fp16=2, fp32=4)
    float *rlog=moe_dmalloc((size_t)S*E), *dw=moe_dmalloc((size_t)S*K);
    int *didx; MOEK(cudaMalloc(&didx,(size_t)S*K*4));
    float *gu=moe_dmalloc(2*MI), *hh=moe_dmalloc(MI), *eo=moe_dmalloc(H);
    // router: gemv_f32(fp32 weight) → top-k
    launch_gemv_f32(n2, L.router, rlog, H, E, S);
    launch_router_topk(rlog, didx, dw, S, E, K);
    std::vector<int> hidx((size_t)S*K); std::vector<float> hw((size_t)S*K);
    MOEK(cudaMemcpy(hidx.data(),didx,(size_t)S*K*4,cudaMemcpyDeviceToHost));
    MOEK(cudaMemcpy(hw.data(),dw,(size_t)S*K*4,cudaMemcpyDeviceToHost));
    MOEK(cudaMemset(out,0,(size_t)S*H*4));
    for (int s=0;s<S;s++) for (int k=0;k<K;k++) {                  // naive 토큰×expert
        int e=hidx[(size_t)s*K+k]; float wk=hw[(size_t)s*K+k];
        launch_gemv_q4(n2+(size_t)s*H, L.experts_gate_up.wq+e*gu_w/2, (const char*)L.experts_gate_up.sc+e*(gu_w/QBLK)*sc_elem_sz, gu, H, 2*MI, 1, sdtype);
        launch_swiglu(gu, gu+MI, hh, (size_t)MI);                  // silu(gate)*up
        launch_gemv_q4(hh, L.experts_down.wq+e*dn_w/2, (const char*)L.experts_down.sc+e*(dn_w/QBLK)*sc_elem_sz, eo, MI, H, 1, sdtype);
        launch_axpy_acc(out+(size_t)s*H, wk, eo, H);              // out[s] += wk*eo
    }
    // shared expert (항상): SwiGLU → sigmoid(H→1) 게이트로 스케일 후 누적
    float *sg=moe_dmalloc((size_t)S*SI), *su=moe_dmalloc((size_t)S*SI), *sh=moe_dmalloc((size_t)S*SI);
    float *so=moe_dmalloc((size_t)S*H), *glog=moe_dmalloc((size_t)S);
    launch_gemv_q4(n2, L.shared_gate.wq, L.shared_gate.sc, sg, H, SI, S, sdtype);
    launch_gemv_q4(n2, L.shared_up.wq,   L.shared_up.sc,   su, H, SI, S, sdtype);
    launch_swiglu(sg, su, sh, (size_t)S*SI);
    launch_gemv_q4(sh, L.shared_down.wq, L.shared_down.sc, so, SI, H, S, sdtype);
    launch_gemv_f32(n2, L.shared_gate_w, glog, H, 1, S);          // [S,1] logit
    launch_moe_shared_add(out, so, glog, S, H);                   // out[s] += sigmoid(glog[s])*so[s]
    for (float *p : {rlog,dw,gu,hh,eo,sg,su,sh,so,glog}) cudaFree(p);
    cudaFree(didx);
}
