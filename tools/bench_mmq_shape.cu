// (BM/BN POC) 형상 스윕 벤치 — tmpl<BM,BN,KT,WM,WN> 형상별 컴파일→게이트→warm 실측.
//   컴파일: nvcc ... -DSB_BM=64 -DSB_BN=64 -DSB_KT=128 -DSB_WM=2 -DSB_WN=2
//   ★게이트: tmpl vs gemv_w4a8 rel<1e-5 + 결정성(빠른데 틀림 차단). warm 50 rep 자체워밍. 출력 파싱형.
#include "runtime/knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_mmq_tmpl.cuh"
#ifndef SB_BM
#define SB_BM 128
#define SB_BN 128
#define SB_KT 64
#define SB_WM 2
#define SB_WN 4
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
    knobs_init("bench_mmq_shape");
    int M=argc>1?atoi(argv[1]):512, N=argc>2?atoi(argv[2]):17408, K=argc>3?atoi(argv[3]):5120, R=argc>4?atoi(argv[4]):100;
    int nblk=K/QBLK; unsigned seed=7;
    std::vector<float> Xf((size_t)M*K),Wf((size_t)N*K); for(auto&v:Xf)v=frand(seed); for(auto&v:Wf)v=frand(seed)*0.3f;
    std::vector<uint8_t> Wpk; std::vector<float> Wsc; quantW(Wf,N,K,Wpk,Wsc);
    float*dXf; int8_t*dXq; float*dxs; int32_t*dxsum;
    CK(cudaMalloc(&dXf,Xf.size()*4));CK(cudaMalloc(&dXq,(size_t)M*K));CK(cudaMalloc(&dxs,(size_t)M*nblk*4));CK(cudaMalloc(&dxsum,(size_t)M*nblk*4));
    CK(cudaMemcpy(dXf,Xf.data(),Xf.size()*4,cudaMemcpyHostToDevice));
    launch_quantize_x_q8(dXf,dXq,dxs,(int32_t*)dxsum,K,M);
    uint8_t*dW; float*dsw,*dYg,*dYm;
    CK(cudaMalloc(&dW,Wpk.size()));CK(cudaMalloc(&dsw,Wsc.size()*4));CK(cudaMalloc(&dYg,(size_t)M*N*4));CK(cudaMalloc(&dYm,(size_t)M*N*4));
    CK(cudaMemcpy(dW,Wpk.data(),Wpk.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dsw,Wsc.data(),Wsc.size()*4,cudaMemcpyHostToDevice));
    launch_gemv_w4a8(dXq,dxs,dxsum,dW,dsw,dYg,K,N,M); CK(cudaDeviceSynchronize());
    std::vector<float> Yg((size_t)M*N); CK(cudaMemcpy(Yg.data(),dYg,Yg.size()*4,cudaMemcpyDeviceToHost));
    launch_gemm_mmq_tmpl<SB_BM,SB_BN,SB_KT,SB_WM,SB_WN>(dXq,dxs,dW,dsw,dYm,M,N,K);
    cudaError_t le=cudaGetLastError(); if(le!=cudaSuccess){ printf("LAUNCHFAIL BM=%d BN=%d KT=%d %dx%d : %s\n",SB_BM,SB_BN,SB_KT,SB_WM,SB_WN,cudaGetErrorString(le)); return 3; }
    CK(cudaDeviceSynchronize());
    std::vector<float> Ym((size_t)M*N); CK(cudaMemcpy(Ym.data(),dYm,Ym.size()*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(size_t i=0;i<Yg.size();i++){e=fmax(e,fabs((double)Ym[i]-Yg[i]));rms+=Yg[i]*Yg[i];} rms=sqrt(rms/Yg.size());
    double rel=e/rms; bool ok=rel<1e-5;
    bool det=true; std::vector<float> Yd((size_t)M*N);
    for(int r=0;r<30&&det;r++){ launch_gemm_mmq_tmpl<SB_BM,SB_BN,SB_KT,SB_WM,SB_WN>(dXq,dxs,dW,dsw,dYm,M,N,K); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Yd.data(),dYm,Yd.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<Yd.size();i++) if(Yd[i]!=Ym[i]){det=false;break;} }
    for(int w=0;w<20;w++) launch_gemm_mmq_tmpl<SB_BM,SB_BN,SB_KT,SB_WM,SB_WN>(dXq,dxs,dW,dsw,dYm,M,N,K); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b); float ms=0;
    cudaEventRecord(a); for(int r=0;r<R;r++) launch_gemm_mmq_tmpl<SB_BM,SB_BN,SB_KT,SB_WM,SB_WN>(dXq,dxs,dW,dsw,dYm,M,N,K); cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms,a,b); double us=ms*1000.0/R; double gflops=2.0*M*N*K/(us*1e-6)/1e9;
    printf("RESULT BM=%d BN=%d KT=%d WARP=%dx%d | time_us=%.1f GFLOPs=%.0f | rel=%.2e gate=%s det=%s\n",
           SB_BM,SB_BN,SB_KT,SB_WM,SB_WN,us,gflops,rel,ok?"PASS":"FAIL",det?"OK":"NONDET");
    return ok&&det?0:1;
}
