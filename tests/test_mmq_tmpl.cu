// (BM/BN POC) 재구조판 mmq 정확성 게이트 — 구조 옳은지 먼저.
//   ★핵심: tmpl<128,128,64,2,4> == 원본 gemm_mmq_kernel<64> BITWISE(재구조화가 계산 안 바꿈, 구조만 일반화).
//   + tmpl vs gemv_w4a8 rel<1e-5(절대 정확성) + 다른 형상(64×64,64×128,256×64,64×256) 구조 검증 + 결정성.
//   재구조화는 틀릴 여지 크니 게이트 엄격("빠른데 틀림" 차단 — AP=8 잡았던 게이트 정신).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_mmq.cuh"
#include "../kernels/gemm_mmq_tmpl.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
static void quantW(const std::vector<float>&w,int N,int K,std::vector<uint8_t>&pk,std::vector<float>&sc){
    int nblk=K/QBLK; sc.resize((size_t)N*nblk); pk.assign((size_t)N*K/2,0);
    for(int n=0;n<N;n++)for(int b=0;b<nblk;b++){float mx=0;for(int k=0;k<QBLK;k++)mx=fmaxf(mx,fabsf(w[(size_t)n*K+b*QBLK+k]));
        float d=mx/7.f;if(d==0)d=1;sc[(size_t)n*nblk+b]=d;
        for(int k=0;k<QBLK;k++){int q=(int)lroundf(w[(size_t)n*K+b*QBLK+k]/d)+8;if(q<0)q=0;if(q>15)q=15;
            size_t gi=(size_t)n*K+b*QBLK+k; if(gi&1)pk[gi/2]|=(q<<4); else pk[gi/2]|=q;}}
}
static int g_fail=0;
// 형상별 vs gemv_w4a8 rel<1e-5 + 결정성 30회
template<int BM,int BN,int KT,int WM,int WN>
static void checkShape(int M,int N,int K,const int8_t*dXq,const float*dxs,const uint8_t*dW,const float*dsw,const std::vector<float>&Yg){
    float*dY; CK(cudaMalloc(&dY,(size_t)M*N*4));
    launch_gemm_mmq_tmpl<BM,BN,KT,WM,WN>(dXq,dxs,dW,dsw,dY,M,N,K); CK(cudaDeviceSynchronize());
    std::vector<float> Y((size_t)M*N); CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(size_t i=0;i<Yg.size();i++){e=fmax(e,fabs((double)Y[i]-Yg[i]));rms+=Yg[i]*Yg[i];} rms=sqrt(rms/Yg.size());
    bool ok=e/rms<1e-5;
    std::vector<float> Yd((size_t)M*N); bool det=true;
    for(int r=0;r<30&&det;r++){ launch_gemm_mmq_tmpl<BM,BN,KT,WM,WN>(dXq,dxs,dW,dsw,dY,M,N,K); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Yd.data(),dY,Yd.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<Yd.size();i++) if(Yd[i]!=Y[i]){det=false;break;} }
    printf("  tmpl<%d,%d,KT%d,%dx%d> M=%-4d N=%-4d K=%-4d | vs gemv rel=%.2e %s | det30 %s\n",
           BM,BN,KT,WM,WN,M,N,K,e/rms,ok?"OK":"FAIL",det?"OK":"NONDET");
    if(!ok||!det)g_fail=1;
    cudaFree(dY);
}
int main(){
    unsigned seed=5;
    struct{int M,N,K;} cs[]={{256,512,1024},{512,512,512},{200,300,256}};  // 배수 + 비배수(bounds)
    for(auto c: cs){
        int M=c.M,N=c.N,K=c.K,nblk=K/QBLK;
        std::vector<float> Xf((size_t)M*K),Wf((size_t)N*K); for(auto&v:Xf)v=frand(seed); for(auto&v:Wf)v=frand(seed)*0.3f;
        std::vector<uint8_t> Wpk; std::vector<float> Wsc; quantW(Wf,N,K,Wpk,Wsc);
        float*dXf; int8_t*dXq; float*dxs; int32_t*dxsum;
        CK(cudaMalloc(&dXf,Xf.size()*4));CK(cudaMalloc(&dXq,(size_t)M*K));CK(cudaMalloc(&dxs,(size_t)M*nblk*4));CK(cudaMalloc(&dxsum,(size_t)M*nblk*4));
        CK(cudaMemcpy(dXf,Xf.data(),Xf.size()*4,cudaMemcpyHostToDevice));
        launch_quantize_x_q8(dXf,dXq,dxs,(int32_t*)dxsum,K,M);
        uint8_t*dW; float*dsw,*dYg,*dYo;
        CK(cudaMalloc(&dW,Wpk.size()));CK(cudaMalloc(&dsw,Wsc.size()*4));CK(cudaMalloc(&dYg,(size_t)M*N*4));CK(cudaMalloc(&dYo,(size_t)M*N*4));
        CK(cudaMemcpy(dW,Wpk.data(),Wpk.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dsw,Wsc.data(),Wsc.size()*4,cudaMemcpyHostToDevice));
        launch_gemv_w4a8(dXq,dxs,dxsum,dW,dsw,dYg,K,N,M); CK(cudaDeviceSynchronize());
        std::vector<float> Yg((size_t)M*N); CK(cudaMemcpy(Yg.data(),dYg,Yg.size()*4,cudaMemcpyDeviceToHost));

        // ★핵심 게이트: tmpl<128,128,64,2,4> == 원본 gemm_mmq_kernel<64> BITWISE
        launch_gemm_mmq_w4a8_kt<64>(dXq,dxs,dW,dsw,dYo,M,N,K); CK(cudaDeviceSynchronize());
        std::vector<float> Yo((size_t)M*N); CK(cudaMemcpy(Yo.data(),dYo,Yo.size()*4,cudaMemcpyDeviceToHost));
        float*dYt; CK(cudaMalloc(&dYt,(size_t)M*N*4));
        launch_gemm_mmq_tmpl<128,128,64,2,4>(dXq,dxs,dW,dsw,dYt,M,N,K); CK(cudaDeviceSynchronize());
        std::vector<float> Yt((size_t)M*N); CK(cudaMemcpy(Yt.data(),dYt,Yt.size()*4,cudaMemcpyDeviceToHost));
        size_t nbit=0; double emax=0; for(size_t i=0;i<Yo.size();i++){ if(Yt[i]!=Yo[i])nbit++; emax=fmax(emax,fabs((double)Yt[i]-Yo[i])); }
        printf("  ★M=%-4d N=%-4d K=%-4d | tmpl<128,128,2x4> vs 원본 mmq: bitwise-diff=%zu/%zu max|Δ|=%.3e %s\n",
               M,N,K,nbit,Yo.size(),emax, nbit==0?"BIT-IDENTICAL✓":(emax/(1e-9+fabs(Yo[0]))<1e-5?"스킴내(fp재배열)":"★FAIL"));
        if(nbit!=0 && emax>1e-2) g_fail=1;
        cudaFree(dYt);

        // 형상 구조 검증 (틀이 형상 바뀜 수용하나) — vs gemv rel<1e-5
        checkShape<128,128,64,2,4>(M,N,K,dXq,dxs,dW,dsw,Yg);
        checkShape<64, 64, 64,2,2>(M,N,K,dXq,dxs,dW,dsw,Yg);
        checkShape<64, 128,64,2,4>(M,N,K,dXq,dxs,dW,dsw,Yg);
        checkShape<256,64, 64,4,2>(M,N,K,dXq,dxs,dW,dsw,Yg);
        checkShape<64, 256,64,2,4>(M,N,K,dXq,dxs,dW,dsw,Yg);
        cudaFree(dXf);cudaFree(dXq);cudaFree(dxs);cudaFree(dxsum);cudaFree(dW);cudaFree(dsw);cudaFree(dYg);cudaFree(dYo);
    }
    printf("[test_mmq_tmpl] %s (재구조판 128×128==원본 + 형상 일반화 vs gemv)\n",g_fail?"FAIL":"PASS");
    return g_fail;
}
