// mlp<2> vs sb 커널 S=1 bit-exact 판별: solo decode(mlp)와 batch decode(sb)가 진짜 동일 출력인가.
//   random int8 활성화 + int4 weight + scale → 두 커널 실행, Y bit-for-bit 대조. (골든 대조, 추측 금지.)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/gemv_w4a8.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

int main(int argc,char**argv){
    int IN = argc>1?atoi(argv[1]):3072;   // MLP up dim 예: 0.8b intermediate
    int OUT= argc>2?atoi(argv[2]):1024;
    int seed=argc>3?atoi(argv[3]):1234;
    srand(seed);
    int S=1;
    std::vector<int8_t> xq((size_t)S*IN); for(auto&v:xq) v=(int8_t)((rand()%255)-127);
    std::vector<float> xs((size_t)S*(IN/QBLK)); for(auto&v:xs) v=((rand()%1000)/1000.f)*0.05f+0.001f;
    std::vector<uint8_t> Wq((size_t)OUT*(IN/2)); for(auto&v:Wq) v=(uint8_t)(rand()%256);
    std::vector<float> sw((size_t)OUT*(IN/QBLK)); for(auto&v:sw) v=((rand()%1000)/1000.f)*0.02f+0.0005f;
    int8_t*d_xq;float*d_xs;uint8_t*d_wq;float*d_sw;float*d_ymlp;float*d_ysb;
    CK(cudaMalloc(&d_xq,xq.size()));CK(cudaMemcpy(d_xq,xq.data(),xq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xs,xs.size()*4));CK(cudaMemcpy(d_xs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_wq,Wq.size()));CK(cudaMemcpy(d_wq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_sw,sw.size()*4));CK(cudaMemcpy(d_sw,sw.data(),sw.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_ymlp,(size_t)S*OUT*4));CK(cudaMalloc(&d_ysb,(size_t)S*OUT*4));
    // solo decode 경로: mlp<2>
    launch_gemv_w4a8_mlp<2>(d_xq,d_xs,nullptr,d_wq,d_sw,d_ymlp,IN,OUT,S,0);
    // batch decode 경로: sb (S=1)
    launch_gemv_w4a8_sb(d_xq,d_xs,d_wq,d_sw,d_ysb,IN,OUT,S,0);
    CK(cudaDeviceSynchronize());
    std::vector<float> ym(OUT),ysb(OUT);
    CK(cudaMemcpy(ym.data(),d_ymlp,OUT*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ysb.data(),d_ysb,OUT*4,cudaMemcpyDeviceToHost));
    int bitmm=0,argm_m=0,argm_s=0; float maxabs=0; double sad=0;
    float bvm=ym[0],bvs=ysb[0];
    for(int o=0;o<OUT;o++){
        if(memcmp(&ym[o],&ysb[o],4)!=0) bitmm++;
        float d=fabsf(ym[o]-ysb[o]); if(d>maxabs)maxabs=d; sad+=d;
        if(ym[o]>bvm){bvm=ym[o];argm_m=o;} if(ysb[o]>bvs){bvs=ysb[o];argm_s=o;}
    }
    printf("[probe mlp<2> vs sb] IN=%d OUT=%d S=1 seed=%d\n",IN,OUT,seed);
    printf("  bit-mismatch %d/%d, max|Δ|=%.3e, meanAD=%.3e, argmax mlp=%d sb=%d %s\n",
           bitmm,OUT,maxabs,sad/OUT,argm_m,argm_s, argm_m==argm_s?"(argmax 일치)":"(★ARGMAX FLIP)");
    printf("  → %s\n", bitmm==0? "BIT-IDENTICAL (decode 커널 원인 아님)":"커널 상이(decode 커널이 solo/batch 갈림 근본)");
    return bitmm==0?0:1;
}
