// ncu: B(flash_attn2bs) prefill(Sq=256) / decode(Sq=1) 병목. 인자 argv[1]=Sq, argv[2]=base.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../kernels/attention_flash.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}}while(0)
int main(int argc,char**argv){
    int Sq=argc>1?atoi(argv[1]):256, base=argc>2?atoi(argv[2]):2048;
    const int HD=256,NQ=24,NKV=4; int Lc=base+Sq;
    float *Q,*K,*V,*O;
    CK(cudaMalloc(&Q,(size_t)Sq*NQ*HD*4)); CK(cudaMalloc(&O,(size_t)Sq*NQ*HD*4));
    CK(cudaMalloc(&K,(size_t)Lc*NKV*HD*4)); CK(cudaMalloc(&V,(size_t)Lc*NKV*HD*4));
    CK(cudaMemset(Q,0,(size_t)Sq*NQ*HD*4)); CK(cudaMemset(K,0,(size_t)Lc*NKV*HD*4)); CK(cudaMemset(V,0,(size_t)Lc*NKV*HD*4));
    for(int i=0;i<3;i++) launch_flash_attn2bs<256,16,8>(Q,K,V,O,Sq,NQ,NKV,base);
    CK(cudaDeviceSynchronize());
    printf("B ncu done (Sq=%d base=%d)\n",Sq,base);
    return 0;
}
