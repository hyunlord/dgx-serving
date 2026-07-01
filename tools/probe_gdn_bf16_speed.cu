// 챕터9 bf16 GDN state 속도 de-risk: gdn_step_batch_kernel_t<128,ST> fp32 vs bf16 state 커널 타이밍.
//   state 50MB(≫캐시)라 격리≈in-engine(DRAM-bound, matmul 격리 오도와 다름). 27B 형상 B16 NH48 NHk16 D128.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../kernels/gdn.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
int main(int argc,char**argv){
    int B=argc>1?atoi(argv[1]):16, NH=48, NHk=16, D=128, N=argc>2?atoi(argv[2]):200;
    size_t stsz=(size_t)B*NH*D*D, qk=(size_t)B*NHk*D, vo=(size_t)B*NH*D, gb=(size_t)B*NH;
    printf("[gdn bf16 speed] B=%d NH=%d D=%d N=%d | state %.1fMB(fp32)/%.1fMB(bf16) read+write\n",
           B,NH,D,N, stsz*4/1e6*2, stsz*2/1e6*2);
    float *Q,*K,*V,*G,*Bt,*O,*Sf,*Sf2; __nv_bfloat16 *Sb,*Sb2;
    CK(cudaMalloc(&Q,qk*4));CK(cudaMalloc(&K,qk*4));CK(cudaMalloc(&V,vo*4));CK(cudaMalloc(&G,gb*4));CK(cudaMalloc(&Bt,gb*4));CK(cudaMalloc(&O,vo*4));
    CK(cudaMalloc(&Sf,stsz*4));CK(cudaMalloc(&Sf2,stsz*4));CK(cudaMalloc(&Sb,stsz*2));CK(cudaMalloc(&Sb2,stsz*2));
    CK(cudaMemset(Q,0,qk*4));CK(cudaMemset(K,0,qk*4));CK(cudaMemset(V,0,vo*4));CK(cudaMemset(G,0,gb*4));CK(cudaMemset(Bt,0,gb*4));
    CK(cudaMemset(Sf,0,stsz*4));CK(cudaMemset(Sb,0,stsz*2));
    dim3 grid(NH,B); size_t shmem=(size_t)(2*D+32)*4;
    cudaEvent_t t0,t1; cudaEventCreate(&t0);cudaEventCreate(&t1);
    // warmup
    for(int i=0;i<10;i++) gdn_step_batch_kernel_t<128,float><<<grid,D,shmem>>>(Q,K,V,G,Bt,Sf,Sf2,O,NH,NHk);
    for(int i=0;i<10;i++) gdn_step_batch_kernel_t<128,__nv_bfloat16><<<grid,D,shmem>>>(Q,K,V,G,Bt,Sb,Sb2,O,NH,NHk);
    CK(cudaDeviceSynchronize());
    auto bw=[&](float ms,int bytes){ return (double)N*stsz*bytes*2/1e9/(ms/1e3); }; // GB/s (read+write)
    // fp32 state
    float best_f=1e30f; for(int r=0;r<3;r++){ cudaEventRecord(t0); for(int i=0;i<N;i++) gdn_step_batch_kernel_t<128,float><<<grid,D,shmem>>>(Q,K,V,G,Bt,Sf,Sf2,O,NH,NHk);
        cudaEventRecord(t1);CK(cudaEventSynchronize(t1)); float ms;cudaEventElapsedTime(&ms,t0,t1); if(ms<best_f)best_f=ms; }
    // bf16 state
    float best_b=1e30f; for(int r=0;r<3;r++){ cudaEventRecord(t0); for(int i=0;i<N;i++) gdn_step_batch_kernel_t<128,__nv_bfloat16><<<grid,D,shmem>>>(Q,K,V,G,Bt,Sb,Sb2,O,NH,NHk);
        cudaEventRecord(t1);CK(cudaEventSynchronize(t1)); float ms;cudaEventElapsedTime(&ms,t0,t1); if(ms<best_b)best_b=ms; }
    printf("  fp32 state: %.1f us/call, %.0f GB/s\n", best_f/N*1000, bw(best_f,4));
    printf("  bf16 state: %.1f us/call, %.0f GB/s\n", best_b/N*1000, bw(best_b,2));
    printf("  speedup: %.2fx (bf16 vs fp32 state)\n", best_f/best_b);
    return 0;
}
