// 챕터7 STEP 1.2: software tiled q4 GEMM 정확성. gemm_tiled_q4 == gemv_q4 == CPU double ref.
//   M(Sq)·N·K 비배수 포함(bounds 검증). 작은 M(안전). 타일 누적순서가 gemv와 달라 ~fp32 오차(헌법 <1e-2).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/gemv_q4.cuh"
#include "../kernels/gemm_tiled.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
static void quant(const std::vector<float>&w,std::vector<uint8_t>&packed,std::vector<float>&scale){
    int n=w.size(),nblk=n/QBLK; scale.resize(nblk); packed.assign(n/2,0);
    for(int b=0;b<nblk;b++){ float mx=0; for(int i=0;i<QBLK;i++)mx=fmaxf(mx,fabsf(w[b*QBLK+i]));
        float sc=mx/7.f; if(sc==0)sc=1; scale[b]=sc;
        for(int i=0;i<QBLK;i++){ int q=(int)lroundf(w[b*QBLK+i]/sc)+8; if(q<0)q=0;if(q>15)q=15;
            int gi=b*QBLK+i; if(gi&1)packed[gi/2]|=(q<<4); else packed[gi/2]|=q; } }
}
static inline float deq(uint8_t pk,int hi,float sc){return (((hi?(pk>>4):(pk&0x0F))-8))*sc;}

int main(){
    unsigned seed=11; int g_fail=0;
    struct{int M,N,K;} cases[]={{64,256,512},{100,256,512},{256,128,256},{129,130,256},{200,512,1024}};
    for(auto cs: cases){
        int M=cs.M,N=cs.N,K=cs.K;
        std::vector<float> W((size_t)N*K), X((size_t)M*K);
        for(auto&v:W)v=frand(seed)*0.3f; for(auto&v:X)v=frand(seed);
        std::vector<uint8_t> packed; std::vector<float> scale; quant(W,packed,scale);
        // CPU double ref: Y[m,n] = Σ_k X[m,k]*deq(W[n,k])
        std::vector<double> ref((size_t)M*N);
        for(int m=0;m<M;m++) for(int n=0;n<N;n++){ double a=0;
            for(int k=0;k<K;k++){ size_t gi=(size_t)n*K+k; a+=(double)X[(size_t)m*K+k]*deq(packed[gi/2],gi&1,scale[gi/QBLK]); }
            ref[(size_t)m*N+n]=a; }
        float *dX,*dYt,*dYg,*dsc; uint8_t*dWq;
        CK(cudaMalloc(&dX,X.size()*4)); CK(cudaMalloc(&dYt,(size_t)M*N*4)); CK(cudaMalloc(&dYg,(size_t)M*N*4));
        CK(cudaMalloc(&dWq,packed.size())); CK(cudaMalloc(&dsc,scale.size()*4));
        CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWq,packed.data(),packed.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsc,scale.data(),scale.size()*4,cudaMemcpyHostToDevice));
        launch_gemm_tiled_q4(dX,dWq,dsc,dYt,M,N,K);     // 신규 tiled
        launch_gemv_q4(dX,dWq,dsc,dYg,K,N,M);           // 기존 기준 (IN=K, OUT=N, S=M)
        CK(cudaDeviceSynchronize());
        std::vector<float> Yt((size_t)M*N),Yg((size_t)M*N);
        CK(cudaMemcpy(Yt.data(),dYt,Yt.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Yg.data(),dYg,Yg.size()*4,cudaMemcpyDeviceToHost));
        double et_ref=0,et_gemv=0,rms=0;
        for(size_t i=0;i<Yt.size();i++){ et_ref=fmax(et_ref,fabs(Yt[i]-ref[i])); et_gemv=fmax(et_gemv,fabs((double)Yt[i]-Yg[i])); rms+=ref[i]*ref[i]; }
        rms=sqrt(rms/Yt.size());
        bool ok=(et_ref/rms<1e-5)&&(et_gemv/rms<1e-5);
        printf("  M=%-4d N=%-4d K=%-5d | tiled vs double=%.2e vs gemv=%.2e (rel, rms=%.1e) %s\n",
               M,N,K,et_ref,et_gemv,rms,ok?"OK":"FAIL"); if(!ok)g_fail=1;
        cudaFree(dX);cudaFree(dYt);cudaFree(dYg);cudaFree(dWq);cudaFree(dsc);
    }
    printf("[test_gemm_tiled] %s (K-slice 타일 GEMM == gemv == double ref)\n",g_fail?"FAIL":"PASS");
    return g_fail;
}
