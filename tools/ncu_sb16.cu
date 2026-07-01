// batch compute/memory-bound de-risk: gemv_w4a8_sb<16>(B=16 DP4A) 단독 커널 → ncu SpeedOfLight(Compute% vs Memory%)
//   + stall 이유. Compute 포화=MMA 이득(①), Memory 포화=MMA 무관(③). 27B MLP-ish shape.
//   사용법: ncu --section SpeedOfLight --section WarpStateStats ./ncu_sb16 [IN] [OUT] [S]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/gemv_w4a8.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
int main(int argc,char**argv){
    int IN=argc>1?atoi(argv[1]):5120, OUT=argc>2?atoi(argv[2]):5120, S=argc>3?atoi(argv[3]):16;
    std::vector<int8_t> xq((size_t)S*IN); for(auto&v:xq)v=(int8_t)((rand()%255)-127);
    std::vector<float> xs((size_t)S*(IN/QBLK),0.01f);
    std::vector<uint8_t> Wq((size_t)OUT*(IN/2)); for(auto&v:Wq)v=(uint8_t)(rand()%256);
    std::vector<__half> sw((size_t)OUT*(IN/QBLK)); for(auto&v:sw)v=__float2half(0.01f);
    int8_t*dxq;float*dxs;uint8_t*dwq;__half*dsw;float*dY;
    CK(cudaMalloc(&dxq,xq.size()));CK(cudaMemcpy(dxq,xq.data(),xq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dxs,xs.size()*4));CK(cudaMemcpy(dxs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dwq,Wq.size()));CK(cudaMemcpy(dwq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsw,sw.size()*2));CK(cudaMemcpy(dsw,sw.data(),sw.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dY,(size_t)S*OUT*4));
    for(int i=0;i<3;i++) launch_gemv_w4a8_sb(dxq,dxs,dwq,dsw,dY,IN,OUT,S,1);  // warmup
    CK(cudaDeviceSynchronize());
    launch_gemv_w4a8_sb(dxq,dxs,dwq,dsw,dY,IN,OUT,S,1);  // ncu 대상 1회
    CK(cudaDeviceSynchronize());
    printf("[ncu_sb16] IN=%d OUT=%d S=%d gemv_w4a8_sb 실행 완료\n",IN,OUT,S);
    return 0;
}
