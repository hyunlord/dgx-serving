// ncu: tensor-core MMA W4A8 GEMM 실dim 프로파일. int8 X + q4 W (양자화는 호스트 더미).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/gemm_mma.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(){
    const int M=256,N=17408,K=5120,nblk=K/QBLK;   // 27B MLP up, prefill 256
    std::vector<int8_t> Xq((size_t)M*K,3); std::vector<float> xs((size_t)M*nblk,0.01f),sw((size_t)N*nblk,0.02f);
    std::vector<uint8_t> Wq((size_t)N*K/2,0x42);
    int8_t*dXq; float*dxs,*dsw,*dY; uint8_t*dW;
    CK(cudaMalloc(&dXq,Xq.size()));CK(cudaMalloc(&dxs,xs.size()*4));CK(cudaMalloc(&dsw,sw.size()*4));CK(cudaMalloc(&dW,Wq.size()));CK(cudaMalloc(&dY,(size_t)M*N*4));
    CK(cudaMemcpy(dXq,Xq.data(),Xq.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dxs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsw,sw.data(),sw.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dW,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    launch_gemm_mma_w4a8(dXq,dxs,dW,dsw,dY,M,N,K); CK(cudaDeviceSynchronize());
    printf("done\n"); return 0;
}
