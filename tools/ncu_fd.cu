// ncu: Flash-Decoding 패스1 GPU 채움 확인. decode Sq=1, base=8192(긴 KV → split多). Waves 0.17→?
#include <cstdio>
#include <cuda_runtime.h>
#include "../kernels/attention_flash.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}}while(0)
int main(){
    const int HD=256, NQ=24, NKV=4, base=8192; int Lc=base+1;
    float *Q,*K,*V,*O;
    CK(cudaMalloc(&Q,(size_t)NQ*HD*4)); CK(cudaMalloc(&O,(size_t)NQ*HD*4));
    CK(cudaMalloc(&K,(size_t)Lc*NKV*HD*4)); CK(cudaMalloc(&V,(size_t)Lc*NKV*HD*4));
    CK(cudaMemset(Q,0,(size_t)NQ*HD*4)); CK(cudaMemset(K,0,(size_t)Lc*NKV*HD*4)); CK(cudaMemset(V,0,(size_t)Lc*NKV*HD*4));
    for(int i=0;i<3;i++) launch_flash_decode<HD>(Q,K,V,O,NQ,NKV,base);
    CK(cudaDeviceSynchronize());
    printf("Flash-Decode ncu done (NQ=%d base=%d)\n",NQ,base);
    return 0;
}
