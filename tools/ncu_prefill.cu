// ncu 마이크로벤치: 실 모델 dim(IN=H, OUT=FFN)서 gemv_q4(weight S회 read) vs 16-타일 GEMM(S/16회).
//   weight 글로벌 로드 트래픽 16× 감소 확인용. ncu -k 로 두 커널 l1tex 비교. (실행 자체는 작은 S=256, 안전.)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/gemv_q4.cuh"
#include "../kernels/gemm_batch.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(){
    const int IN=5120, OUT=17408, S=256;   // 27B MLP up dim, prefill 256 토큰
    std::vector<uint8_t> wq((size_t)OUT*IN/2, 0x42); std::vector<float> sc((size_t)OUT*IN/QBLK, 0.01f), X((size_t)S*IN, 0.1f);
    uint8_t*dwq; float*dsc,*dX,*dYg,*dYt;
    CK(cudaMalloc(&dwq,wq.size())); CK(cudaMalloc(&dsc,sc.size()*4)); CK(cudaMalloc(&dX,X.size()*4));
    CK(cudaMalloc(&dYg,(size_t)S*OUT*4)); CK(cudaMalloc(&dYt,(size_t)S*OUT*4));
    CK(cudaMemcpy(dwq,wq.data(),wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsc,sc.data(),sc.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));
    launch_gemv_q4(dX,dwq,dsc,dYg,IN,OUT,S);            // 기준: weight S회 read
    launch_gemm_q4_tiled(dX,dwq,dsc,dYt,IN,OUT,S);      // 16-타일 gemm_batch: weight S/16회
    // mw 8-타일 (멀티워프, X 접근 다름 — X-bound 완화 시험)
    for(int t=0;t+8<=S;t+=8) launch_gemm_mw_q4(dX+(size_t)t*IN,dwq,dsc,dYt+(size_t)t*OUT,IN,OUT,8);
    CK(cudaDeviceSynchronize());
    printf("done (IN=%d OUT=%d S=%d)\n",IN,OUT,S);
    return 0;
}
