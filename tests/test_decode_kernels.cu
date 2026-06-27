// 2c STEP 1 토대: decode/verify 신규 커널을 2a prefill 커널과 동치 검증 (격리).
//   (1) attn_decode(cache attend) == attn_kernel(prefill causal)  — 동일 Q/K/V
//   (2) gdn_step(Sin=0) == gdn_delta_rule
//   (3) gdn_step 연속성: 0→n1 진행 후 그 state에서 이어가면 통짜와 동일 (decode 핵심)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/attention.cuh"
#include "../kernels/gdn.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static int g_fail=0;
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static std::vector<float> dn(const float*d,size_t n){std::vector<float>h(n);CK(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost));return h;}
static void chk(const char*n,double e,double tol){printf("  %-26s max abs err=%.3e (tol %.0e) %s\n",n,e,tol,e<=tol?"OK":"FAIL");if(e>tol)g_fail=1;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(){
    unsigned seed=2024;

    // ===== (1) attn_decode == attn_kernel (prefill causal), 동일 Q/K/V =====
    {
        int S=6,NQ=4,NKV=2,HD=128;
        std::vector<float> Q(S*NQ*HD),K(S*NKV*HD),Vv(S*NKV*HD);
        for(auto&v:Q)v=frand(seed)*0.2f; for(auto&v:K)v=frand(seed)*0.2f; for(auto&v:Vv)v=frand(seed);
        float*dQ=up(Q),*dK=up(K),*dV=up(Vv),*dO1,*dO2; CK(cudaMalloc(&dO1,(size_t)S*NQ*HD*4)); CK(cudaMalloc(&dO2,(size_t)S*NQ*HD*4));
        launch_attn(dQ,dK,dV,dO1,S,NQ,NKV,HD);                 // prefill (ref)
        launch_attn_decode(dQ,dK,dV,dO2,S,NQ,NKV,HD,0);        // decode: query i attends 0..i
        CK(cudaDeviceSynchronize());
        chk("attn_decode==prefill", mae(dn(dO1,(size_t)S*NQ*HD),dn(dO2,(size_t)S*NQ*HD)), 1e-4);
        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO1);cudaFree(dO2);
    }

    // ===== (2)(3) gdn_step vs gdn_delta_rule + 연속성 =====
    {
        int S=6,NH=4,D=128;
        std::vector<float> Q(S*NH*D),K(S*NH*D),Vv(S*NH*D),G(S*NH),Bt(S*NH);
        for(auto&v:Q)v=frand(seed)*0.2f; for(auto&v:K)v=frand(seed)*0.2f; for(auto&v:Vv)v=frand(seed);
        for(auto&v:G)v=-fabsf(frand(seed))*0.1f; for(auto&v:Bt)v=0.5f+0.4f*frand(seed);
        float*dQ=up(Q),*dK=up(K),*dV=up(Vv),*dG=up(G),*dB=up(Bt);
        float*dref; CK(cudaMalloc(&dref,(size_t)S*NH*D*4));
        launch_gdn_delta_rule(dQ,dK,dV,dG,dB,dref,S,NH,D,NH);     // ref
        // (2) gdn_step Sin=0, 전체
        float*dO; CK(cudaMalloc(&dO,(size_t)S*NH*D*4));
        float*dSnap; CK(cudaMalloc(&dSnap,(size_t)S*NH*D*D*4));
        launch_gdn_step(dQ,dK,dV,dG,dB,nullptr,dSnap,dO,S,NH,D,NH);
        CK(cudaDeviceSynchronize());
        chk("gdn_step(Sin=0)==delta", mae(dn(dref,(size_t)S*NH*D),dn(dO,(size_t)S*NH*D)), 1e-4);

        // (3) 연속성: 0..n1 진행 → snap[n1-1] 에서 n1..S-1 이어가기 == 통짜
        int n1=2, n2=S-n1;
        float*dSnap1; CK(cudaMalloc(&dSnap1,(size_t)n1*NH*D*D*4));
        float*dO1; CK(cudaMalloc(&dO1,(size_t)n1*NH*D*4));
        launch_gdn_step(dQ,dK,dV,dG,dB,nullptr,dSnap1,dO1,n1,NH,D,NH);   // 첫 n1
        float*Sin = dSnap1 + (size_t)(n1-1)*NH*D*D;                   // state after n1
        // 이어갈 입력 = 토큰 n1..S-1 (offset)
        float*dSnap2; CK(cudaMalloc(&dSnap2,(size_t)n2*NH*D*D*4));
        float*dO2; CK(cudaMalloc(&dO2,(size_t)n2*NH*D*4));
        launch_gdn_step(dQ+(size_t)n1*NH*D, dK+(size_t)n1*NH*D, dV+(size_t)n1*NH*D,
                        dG+(size_t)n1*NH, dB+(size_t)n1*NH, Sin, dSnap2, dO2, n2, NH, D, NH);
        CK(cudaDeviceSynchronize());
        // O2 == ref[n1..]
        std::vector<float> refv=dn(dref,(size_t)S*NH*D), o2=dn(dO2,(size_t)n2*NH*D);
        std::vector<float> reftail(refv.begin()+(size_t)n1*NH*D, refv.end());
        chk("gdn_step continuation", mae(reftail,o2), 1e-4);
        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dG);cudaFree(dB);cudaFree(dref);cudaFree(dO);cudaFree(dSnap);
        cudaFree(dSnap1);cudaFree(dO1);cudaFree(dSnap2);cudaFree(dO2);
    }

    // ===== (4) conv1d decode 연속성: prev context 로 청크 쪼개도 통짜와 동일 =====
    {
        int S=7,C=64,K=4;
        std::vector<float> in(S*C),w(C*K);
        for(auto&v:in)v=frand(seed); for(auto&v:w)v=frand(seed)*0.3f;
        float*din=up(in),*dw=up(w),*dref; CK(cudaMalloc(&dref,(size_t)S*C*4));
        launch_gdn_conv1d_silu(din,dw,dref,S,C,K);             // 통짜 ref
        // 청크: 0..n1-1 (prev=NULL), n1..S-1 (prev=직전 K-1 토큰 = in[n1-3..n1-1])
        int n1=4,n2=S-n1;
        float*dO1; CK(cudaMalloc(&dO1,(size_t)n1*C*4));
        launch_gdn_conv1d_silu_decode(din,nullptr,dw,dO1,n1,C,K);
        std::vector<float> prev((K-1)*C);
        for(int j=0;j<K-1;j++) for(int c=0;c<C;c++) prev[j*C+c]=in[(n1-(K-1)+j)*C+c];
        float*dprev=up(prev),*dO2; CK(cudaMalloc(&dO2,(size_t)n2*C*4));
        launch_gdn_conv1d_silu_decode(din+(size_t)n1*C,dprev,dw,dO2,n2,C,K);
        CK(cudaDeviceSynchronize());
        std::vector<float> refv=dn(dref,(size_t)S*C), o2=dn(dO2,(size_t)n2*C);
        std::vector<float> reftail(refv.begin()+(size_t)n1*C,refv.end());
        chk("conv1d decode continuation", mae(reftail,o2), 1e-4);
        cudaFree(din);cudaFree(dw);cudaFree(dref);cudaFree(dO1);cudaFree(dprev);cudaFree(dO2);
    }

    printf(g_fail?"[test_decode_kernels] FAIL\n":"[test_decode_kernels] PASS\n");
    return g_fail;
}
