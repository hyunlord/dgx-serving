// STEP 2.0 decode lever 진단: gemv_w4a8(decode 91%) 대역폭 limiter 규명.
//   decode=memory-bound weight 적재. 현 52% 효율(vs llama 73%) 원인 = DRAM% / stall / occupancy / sectors-req?
//   shape = decode(M=1) 대표: 기본 FFN gate(K=5120 N=17408=최대 가중치 44.6MB). argv[1]=K argv[2]=N.
//   이론 하한: weight bytes(N*K/2 int4) / 273GBps. 효율 = 이론/실측.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/gemv_w4a8.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static unsigned sd=99u; static int ir(int lo,int hi){sd=sd*1664525u+1013904223u;return lo+(int)((sd>>9)%(unsigned)(hi-lo+1));}
int main(int argc,char**argv){
    const int M=1, K=argc>1?atoi(argv[1]):5120, N=argc>2?atoi(argv[2]):17408;   // decode gemv: M=1, FFN gate 기본
    const int nblk=K/QBLK, K2=K/2;
    double wbytes=(double)N*K2;                          // int4 weight bytes (지배 트래픽)
    double ideal_us = wbytes/273e9*1e6;                  // 273 GB/s 하한
    printf("[gemv_w4a8] M=%d K=%d N=%d | weight=%.1fMB, 이론하한@273GBps=%.1fus\n",M,K,N,wbytes/1e6,ideal_us);
    std::vector<int8_t> hX((size_t)M*K); for(auto&x:hX)x=(int8_t)ir(-40,40);
    std::vector<uint8_t> hW((size_t)N*K2); for(auto&x:hW)x=(uint8_t)ir(0,255);
    std::vector<float> hXs((size_t)M*nblk),hSw((size_t)N*nblk); for(auto&x:hXs)x=0.03f; for(auto&x:hSw)x=0.02f;
    int8_t*dX;uint8_t*dW;float*dXs,*dSw,*dY;
    CK(cudaMalloc(&dX,hX.size()));CK(cudaMalloc(&dW,hW.size()));CK(cudaMalloc(&dXs,hXs.size()*4));
    CK(cudaMalloc(&dSw,hSw.size()*4));CK(cudaMalloc(&dY,(size_t)M*N*4));
    CK(cudaMemcpy(dX,hX.data(),hX.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW,hW.data(),hW.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dXs,hXs.data(),hXs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSw,hSw.data(),hSw.size()*4,cudaMemcpyHostToDevice));
    for(int w=0;w<10;w++) launch_gemv_w4a8(dX,dXs,nullptr,dW,dSw,dY,K,N,M); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);int R=200;float t=0;
    cudaEventRecord(a); for(int r=0;r<R;r++) launch_gemv_w4a8(dX,dXs,nullptr,dW,dSw,dY,K,N,M); cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&t,a,b);
    double us=t/R*1000.0, gbps=wbytes/(us*1e-6)/1e9;
    printf("[gemv_w4a8] 실측 %.1fus | 유효대역폭 %.1f GB/s (%.0f%% of 273) | 이론대비 %.2fx\n", us, gbps, gbps/273*100, us/ideal_us);
    long warps=(long)M*N, grid=(warps*32+255)/256;
    printf("[gemv_w4a8] grid=%ld blocks (warps=%ld=M*N, 1warp/output) — full GPU? (~48 SM, 2CTA=96)\n", grid, warps);
    return 0;
}
