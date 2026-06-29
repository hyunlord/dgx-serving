// ncu 마이크로벤치: FA.2 커널(flash_attn2b_kernel) 병목 측정. 실 prefill dim.
//   Sq=256(청크), HD=256, NQ=24/NKV=4(GQA), base=2048. occupancy/util/tensor-core active 확인용.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/attention_flash.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}}while(0)
int main(){
    const int Sq=256, HD=256, NQ=24, NKV=4, base=2048; int Lc=base+Sq;
    float *Q,*K,*V,*O;
    CK(cudaMalloc(&Q,(size_t)Sq*NQ*HD*4)); CK(cudaMalloc(&O,(size_t)Sq*NQ*HD*4));
    CK(cudaMalloc(&K,(size_t)Lc*NKV*HD*4)); CK(cudaMalloc(&V,(size_t)Lc*NKV*HD*4));
    CK(cudaMemset(Q,0,(size_t)Sq*NQ*HD*4)); CK(cudaMemset(K,0,(size_t)Lc*NKV*HD*4)); CK(cudaMemset(V,0,(size_t)Lc*NKV*HD*4));
    for(int i=0;i<3;i++) launch_flash_attn2b_t<256,16,8>(Q,K,V,O,Sq,NQ,NKV,base);  // warmup + ncu 대상
    CK(cudaDeviceSynchronize());
    printf("FA.2 ncu bench done (Sq=%d HD=%d NQ=%d NKV=%d base=%d)\n",Sq,HD,NQ,NKV,base);
    return 0;
}
