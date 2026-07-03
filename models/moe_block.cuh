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
#include "../kernels/moe_fused.cuh"   // #2 device-fused MoE(grouped gemv, D2H·host loop 제거). env MOE_FUSED=1로 opt-in.

#define MOEK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static inline float *moe_dmalloc(size_t n) { float *p; MOEK(cudaMalloc(&p, n * 4)); return p; }

// ★persistent scratch (nsys: 호출당 malloc/free의 cudaFree가 API 41.8% — 동기화 직렬화). grow-on-demand 재사용
//   (기존 qwen35_batch g_bxq 패턴). 각 버퍼는 매 호출 커널 출력으로 완전 덮어써진 뒤 읽힘(누산기 out은 memset 0)
//   → 재사용해도 수학 불변(골든/bit-id 게이트로 증명). static inline라 TU별(prefill/decode/batch) 자체 버퍼(정확).
static inline float *moe_grow(float *&p, size_t &cap, size_t n) {
    if (cap < n) { if (p) cudaFree(p); MOEK(cudaMalloc(&p, n * 4)); cap = n; } return p;
}

static inline void moe_mlp_block(const ModelConfig &c, const QLayer &L, const float *n2, float *out, int S) {
    const int H=c.hidden_size, E=c.num_experts, K=c.num_experts_per_tok;
    const int MI=c.moe_intermediate_size, SI=c.shared_expert_intermediate_size;
    const int sdtype=c.scale_dtype;
    const size_t gu_w=(size_t)2*MI*H, dn_w=(size_t)H*MI;   // expert당 weight 수
    const size_t sc_elem_sz = (sdtype==1) ? 2 : 4;  // scale 원소 크기 (fp16=2, fp32=4)
    static float *rlog=nullptr,*dw=nullptr,*gu=nullptr,*hh=nullptr,*eo=nullptr,*sg=nullptr,*su=nullptr,*sh=nullptr,*so=nullptr,*glog=nullptr; static int *didx=nullptr;
    static float *fgu=nullptr,*fhid=nullptr,*fsout=nullptr;   // #2 fused scratch (grouped)
    static size_t c_rlog=0,c_dw=0,c_gu=0,c_hh=0,c_eo=0,c_sg=0,c_su=0,c_sh=0,c_so=0,c_glog=0,c_didx=0,c_fgu=0,c_fhid=0,c_fsout=0;
    moe_grow(rlog,c_rlog,(size_t)S*E); moe_grow(dw,c_dw,(size_t)S*K);
    if(c_didx<(size_t)S*K){ if(didx)cudaFree(didx); MOEK(cudaMalloc(&didx,(size_t)S*K*4)); c_didx=(size_t)S*K; }
    // router: gemv_f32(fp32 weight) → top-k (didx/dw device 상주 — fused는 D2H 안 함)
    launch_gemv_f32(n2, L.router, rlog, H, E, S);
    launch_router_topk(rlog, didx, dw, S, E, K);
    static int g_fused = getenv("MOE_FUSED") ? atoi(getenv("MOE_FUSED")) : 1;   // ★default fused(bit-id+전축 개선). MOE_FUSED=0=naive 폴백
    if (g_fused) {
        // #2 device-fused: grouped gemv 4커널(D2H·host loop 제거). reduce가 out을 덮어씀(memset 불요).
        long nslot=(long)S*K;
        moe_grow(fgu,c_fgu,nslot*(size_t)2*MI); moe_grow(fhid,c_fhid,nslot*(size_t)MI); moe_grow(fsout,c_fsout,nslot*(size_t)H);
        launch_moe_fused_gu(n2, L.experts_gate_up.wq, L.experts_gate_up.sc, didx, fgu, H, 2*MI, K, nslot, (long)gu_w, sdtype);
        launch_moe_fused_swiglu(fgu, fhid, MI, nslot);
        launch_moe_fused_down(fhid, L.experts_down.wq, L.experts_down.sc, didx, dw, fsout, MI, H, K, nslot, (long)dn_w, sdtype);
        launch_moe_fused_reduce(fsout, out, H, K, S);
    } else {
        // naive(폴백): D2H + 호스트 토큰×expert 루프
        moe_grow(gu,c_gu,2*MI); moe_grow(hh,c_hh,MI); moe_grow(eo,c_eo,H);
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
    }
    // shared expert (항상): SwiGLU → sigmoid(H→1) 게이트로 스케일 후 누적
    moe_grow(sg,c_sg,(size_t)S*SI); moe_grow(su,c_su,(size_t)S*SI); moe_grow(sh,c_sh,(size_t)S*SI);
    moe_grow(so,c_so,(size_t)S*H); moe_grow(glog,c_glog,(size_t)S);
    launch_gemv_q4(n2, L.shared_gate.wq, L.shared_gate.sc, sg, H, SI, S, sdtype);
    launch_gemv_q4(n2, L.shared_up.wq,   L.shared_up.sc,   su, H, SI, S, sdtype);
    launch_swiglu(sg, su, sh, (size_t)S*SI);
    launch_gemv_q4(sh, L.shared_down.wq, L.shared_down.sc, so, SI, H, S, sdtype);
    launch_gemv_f32(n2, L.shared_gate_w, glog, H, 1, S);          // [S,1] logit
    launch_moe_shared_add(out, so, glog, S, H);                   // out[s] += sigmoid(glog[s])*so[s]
    // ★free 없음 — persistent scratch(다음 호출 재사용). cudaFree 41.8% API 오버헤드 제거.
}
