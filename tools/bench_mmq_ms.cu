// (파이프 POC) multi-stage 벤치 — ms<BM,BN,KT,WM,WN,STAGES> 게이트+타이밍.
//   컴파일: -DSB_BM=256 -DSB_BN=64 -DSB_KT=64 -DSB_WM=4 -DSB_WN=2 -DSB_STAGES=3
//   ★게이트: (1)STAGES==2면 ms==tmpl BITWISE(깊이 일반화가 계산 불변) (2)ms vs gemv rel<1e-5 (3)결정성. "빠른데 틀림" 차단.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_mmq_tmpl.cuh"
#include "../kernels/gemm_mmq_ms.cuh"
#ifndef SB_BM
#define SB_BM 256
#define SB_BN 64
#define SB_KT 64
#define SB_WM 4
#define SB_WN 2
#define SB_STAGES 3
#endif
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
static void quantW(const std::vector<float>&w,int N,int K,std::vector<uint8_t>&pk,std::vector<float>&sc){
    int nblk=K/QBLK; sc.resize((size_t)N*nblk); pk.assign((size_t)N*K/2,0);
    for(int n=0;n<N;n++)for(int b=0;b<nblk;b++){float mx=0;for(int k=0;k<QBLK;k++)mx=fmaxf(mx,fabsf(w[(size_t)n*K+b*QBLK+k]));
        float d=mx/7.f;if(d==0)d=1;sc[(size_t)n*nblk+b]=d;
        for(int k=0;k<QBLK;k++){int q=(int)lroundf(w[(size_t)n*K+b*QBLK+k]/d)+8;if(q<0)q=0;if(q>15)q=15;
            size_t gi=(size_t)n*K+b*QBLK+k; if(gi&1)pk[gi/2]|=(q<<4); else pk[gi/2]|=q;}}
}
int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):512, N=argc>2?atoi(argv[2]):17408, K=argc>3?atoi(argv[3]):5120, R=argc>4?atoi(argv[4]):100;
    int nblk=K/QBLK; unsigned seed=7;
    std::vector<float> Xf((size_t)M*K),Wf((size_t)N*K); for(auto&v:Xf)v=frand(seed); for(auto&v:Wf)v=frand(seed)*0.3f;
    std::vector<uint8_t> Wpk; std::vector<float> Wsc; quantW(Wf,N,K,Wpk,Wsc);
    float*dXf; int8_t*dXq; float*dxs; int32_t*dxsum;
    CK(cudaMalloc(&dXf,Xf.size()*4));CK(cudaMalloc(&dXq,(size_t)M*K));CK(cudaMalloc(&dxs,(size_t)M*nblk*4));CK(cudaMalloc(&dxsum,(size_t)M*nblk*4));
    CK(cudaMemcpy(dXf,Xf.data(),Xf.size()*4,cudaMemcpyHostToDevice));
    launch_quantize_x_q8(dXf,dXq,dxs,(int32_t*)dxsum,K,M);
    uint8_t*dW; float*dsw,*dYg,*dYm,*dYt;
    CK(cudaMalloc(&dW,Wpk.size()));CK(cudaMalloc(&dsw,Wsc.size()*4));CK(cudaMalloc(&dYg,(size_t)M*N*4));CK(cudaMalloc(&dYm,(size_t)M*N*4));CK(cudaMalloc(&dYt,(size_t)M*N*4));
    CK(cudaMemcpy(dW,Wpk.data(),Wpk.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dsw,Wsc.data(),Wsc.size()*4,cudaMemcpyHostToDevice));
    launch_gemv_w4a8(dXq,dxs,dxsum,dW,dsw,dYg,K,N,M); CK(cudaDeviceSynchronize());
    std::vector<float> Yg((size_t)M*N); CK(cudaMemcpy(Yg.data(),dYg,Yg.size()*4,cudaMemcpyDeviceToHost));
    launch_gemm_mmq_ms<SB_BM,SB_BN,SB_KT,SB_WM,SB_WN,SB_STAGES>(dXq,dxs,dW,dsw,dYm,M,N,K);
    cudaError_t le=cudaGetLastError(); if(le!=cudaSuccess){ printf("LAUNCHFAIL BM=%d BN=%d KT=%d %dx%d S=%d : %s\n",SB_BM,SB_BN,SB_KT,SB_WM,SB_WN,SB_STAGES,cudaGetErrorString(le)); return 3; }
    CK(cudaDeviceSynchronize());
    std::vector<float> Ym((size_t)M*N); CK(cudaMemcpy(Ym.data(),dYm,Ym.size()*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(size_t i=0;i<Yg.size();i++){e=fmax(e,fabs((double)Ym[i]-Yg[i]));rms+=Yg[i]*Yg[i];} rms=sqrt(rms/Yg.size());
    double rel=e/rms; bool ok=rel<1e-5;
    // STAGES==2면 tmpl과 bit-id 대조(깊이 일반화가 계산 불변 검증)
    const char* bitcmp="(STAGES>2:tmpl 대조 생략)"; size_t nbit=999;
    if(SB_STAGES==2){ launch_gemm_mmq_tmpl<SB_BM,SB_BN,SB_KT,SB_WM,SB_WN>(dXq,dxs,dW,dsw,dYt,M,N,K); CK(cudaDeviceSynchronize());
        std::vector<float> Yt((size_t)M*N); CK(cudaMemcpy(Yt.data(),dYt,Yt.size()*4,cudaMemcpyDeviceToHost));
        nbit=0; for(size_t i=0;i<Ym.size();i++) if(Ym[i]!=Yt[i])nbit++; bitcmp = nbit==0?"BIT-ID(vs tmpl)✓":"★DIFF(vs tmpl)"; }
    bool det=true; std::vector<float> Yd((size_t)M*N);
    for(int r=0;r<30&&det;r++){ launch_gemm_mmq_ms<SB_BM,SB_BN,SB_KT,SB_WM,SB_WN,SB_STAGES>(dXq,dxs,dW,dsw,dYm,M,N,K); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Yd.data(),dYm,Yd.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<Yd.size();i++) if(Yd[i]!=Ym[i]){det=false;break;} }
    for(int w=0;w<20;w++) launch_gemm_mmq_ms<SB_BM,SB_BN,SB_KT,SB_WM,SB_WN,SB_STAGES>(dXq,dxs,dW,dsw,dYm,M,N,K); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b); float ms=0;
    cudaEventRecord(a); for(int r=0;r<R;r++) launch_gemm_mmq_ms<SB_BM,SB_BN,SB_KT,SB_WM,SB_WN,SB_STAGES>(dXq,dxs,dW,dsw,dYm,M,N,K); cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms,a,b); double us=ms*1000.0/R; double gflops=2.0*M*N*K/(us*1e-6)/1e9;
    printf("RESULT BM=%d BN=%d KT=%d WARP=%dx%d STAGES=%d | time_us=%.1f GFLOPs=%.0f | rel=%.2e gate=%s det=%s %s\n",
           SB_BM,SB_BN,SB_KT,SB_WM,SB_WN,SB_STAGES,us,gflops,rel,ok?"PASS":"FAIL",det?"OK":"NONDET",bitcmp);
    return (ok&&det&&(SB_STAGES!=2||nbit==0))?0:1;
}
