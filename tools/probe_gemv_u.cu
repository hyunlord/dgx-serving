// decode gemv register-ILP U-sweep: gemv_w4a8_mlp<U> U∈{1,2,4,8} at B=1 (register 여유). llama Priority-1(4-8 체인).
//   챕터9 "U=2 최적" 재확인 — U=4/8이 더 빠르면 decode gap(1.16×) lever. bit-exact(누적순서 동일).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/gemv_w4a8.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
int main(int argc,char**argv){
    int IN=argc>1?atoi(argv[1]):5120, OUT=argc>2?atoi(argv[2]):5120; int S=1;
    srand(7);
    std::vector<int8_t> xq((size_t)S*IN); for(auto&v:xq)v=(int8_t)((rand()%255)-127);
    std::vector<float> xs((size_t)S*(IN/QBLK)); for(auto&v:xs)v=((rand()%1000)/1000.f)*0.05f+0.001f;
    std::vector<uint8_t> Wq((size_t)OUT*(IN/2)); for(auto&v:Wq)v=(uint8_t)(rand()%256);
    std::vector<__half> sw((size_t)OUT*(IN/QBLK)); for(auto&v:sw)v=__float2half(((rand()%1000)/1000.f)*0.02f+0.0005f);
    int8_t*dxq;float*dxs;uint8_t*dwq;__half*dsw;float*dY,*dref;
    CK(cudaMalloc(&dxq,xq.size()));CK(cudaMemcpy(dxq,xq.data(),xq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dxs,xs.size()*4));CK(cudaMemcpy(dxs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dwq,Wq.size()));CK(cudaMemcpy(dwq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsw,sw.size()*2));CK(cudaMemcpy(dsw,sw.data(),sw.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dY,(size_t)OUT*4));CK(cudaMalloc(&dref,(size_t)OUT*4));
    cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
    launch_gemv_w4a8_mlp<1>(dxq,dxs,nullptr,dwq,dsw,dref,IN,OUT,S,1); CK(cudaDeviceSynchronize());
    std::vector<float> ref(OUT); CK(cudaMemcpy(ref.data(),dref,OUT*4,cudaMemcpyDeviceToHost));
    printf("[probe_gemv_u] IN=%d OUT=%d S=1 (decode register-ILP U-sweep)\n",IN,OUT);
    auto run=[&](auto tag,int U,void(*fn)(const int8_t*,const float*,const int32_t*,const uint8_t*,const void*,float*,int,int,int,int,cudaStream_t)){
        for(int i=0;i<30;i++)fn(dxq,dxs,nullptr,dwq,dsw,dY,IN,OUT,S,1,0); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));for(int i=0;i<500;i++)fn(dxq,dxs,nullptr,dwq,dsw,dY,IN,OUT,S,1,0);CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));
        float t;CK(cudaEventElapsedTime(&t,e0,e1));
        std::vector<float> y(OUT);CK(cudaMemcpy(y.data(),dY,OUT*4,cudaMemcpyDeviceToHost));
        int mm=0;for(int i=0;i<OUT;i++)if(memcmp(&y[i],&ref[i],4))mm++;
        printf("  U=%-2d %.1fus  bit-exact vs U=1: %s\n",U,t/500*1000,mm?"★DIFF":"OK");
    };
    run("",1,&launch_gemv_w4a8_mlp<1>); run("",2,&launch_gemv_w4a8_mlp<2>);
    run("",4,&launch_gemv_w4a8_mlp<4>); run("",8,&launch_gemv_w4a8_mlp<8>);
    printf("  → 가장 빠른 U가 decode ILP 최적(챕터9 U=2 재확인/갱신)\n");
    return 0;
}
