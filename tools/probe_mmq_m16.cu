// big-tile 실현성 grounding: gemm_mmq(BM=128 MMA 더블버퍼) at M=16 vs SB(DP4A) — 타이밍 + (ncu long_scoreboard).
//   목적: big-tile 더블버퍼가 M=16서 load-latency 숨기나(long_scoreboard 낮음?). 낮으면 함정=empty-row 낭비뿐(BM=16이 해결).
//   높으면 big-tile도 latency 못 숨김(설계 우려). + gemm_mmq==sb bit-exact(둘 다 정수 누적).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_mmq.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
int main(int argc,char**argv){
    int IN=argc>1?atoi(argv[1]):5120, OUT=argc>2?atoi(argv[2]):5120, S=argc>3?atoi(argv[3]):16;
    int which=argc>4?atoi(argv[4]):0;  // 0=both+timing, 1=mmq only(ncu), 2=sb only(ncu)
    srand(7);
    std::vector<int8_t> xq((size_t)S*IN); for(auto&v:xq)v=(int8_t)((rand()%255)-127);
    std::vector<float> xs((size_t)S*(IN/QBLK)); for(auto&v:xs)v=((rand()%1000)/1000.f)*0.05f+0.001f;
    std::vector<uint8_t> Wq((size_t)OUT*(IN/2)); for(auto&v:Wq)v=(uint8_t)(rand()%256);
    std::vector<__half> sw((size_t)OUT*(IN/QBLK)); for(auto&v:sw)v=__float2half(((rand()%1000)/1000.f)*0.02f+0.0005f);
    int8_t*dxq;float*dxs;uint8_t*dwq;__half*dsw;float*dYm,*dYs;
    CK(cudaMalloc(&dxq,xq.size()));CK(cudaMemcpy(dxq,xq.data(),xq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dxs,xs.size()*4));CK(cudaMemcpy(dxs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dwq,Wq.size()));CK(cudaMemcpy(dwq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsw,sw.size()*2));CK(cudaMemcpy(dsw,sw.data(),sw.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dYm,(size_t)S*OUT*4));CK(cudaMalloc(&dYs,(size_t)S*OUT*4));
    auto mmq=[&](){ launch_gemm_mmq_w4a8(dxq,dxs,dwq,dsw,dYm,S,OUT,IN,1); };
    auto sb =[&](){ launch_gemv_w4a8_sb(dxq,dxs,dwq,dsw,dYs,IN,OUT,S,1); };
    if(which!=2) for(int i=0;i<3;i++)mmq();
    if(which!=1) for(int i=0;i<3;i++)sb();
    CK(cudaDeviceSynchronize());
    if(which==1){ mmq(); CK(cudaDeviceSynchronize()); printf("[mmq M=%d] ncu 대상\n",S); return 0; }
    if(which==2){ sb();  CK(cudaDeviceSynchronize()); printf("[sb M=%d] ncu 대상\n",S); return 0; }
    // bit-exact
    mmq(); sb(); CK(cudaDeviceSynchronize());
    std::vector<float> ym((size_t)S*OUT),ys((size_t)S*OUT);
    CK(cudaMemcpy(ym.data(),dYm,ym.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(ys.data(),dYs,ys.size()*4,cudaMemcpyDeviceToHost));
    int mm=0;double mx=0;for(size_t i=0;i<ym.size();i++){if(memcmp(&ym[i],&ys[i],4))mm++;double d=fabs(ym[i]-ys[i]);if(d>mx)mx=d;}
    printf("[probe_mmq_m16] IN=%d OUT=%d M=%d\n  gemm_mmq vs sb bit-exact: %d/%zu max|Δ|=%.3e %s\n",IN,OUT,S,mm,ym.size(),mx,mm?"(정수누적 동일해야)":"(bit-identical ✓)");
    // 타이밍
    cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
    for(int i=0;i<20;i++)mmq();CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));for(int i=0;i<200;i++)mmq();CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));float tm;CK(cudaEventElapsedTime(&tm,e0,e1));
    for(int i=0;i<20;i++)sb();CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));for(int i=0;i<200;i++)sb();CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));float ts;CK(cudaEventElapsedTime(&ts,e0,e1));
    printf("  타이밍(200 avg): gemm_mmq(BM=128 empty)=%.1fus  sb(DP4A)=%.1fus  → mmq %.2f× %s\n",tm/200*1000,ts/200*1000,ts/tm,tm<ts?"빠름(empty에도)":"느림(MMQ 함정=empty 낭비)");
    return 0;
}
