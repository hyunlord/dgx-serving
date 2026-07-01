// SMAX 8→16 de-risk: SB<8> vs SB<16> occupancy(register-limited, CUDA API=결정적) + bit-exact + B=2~16 타이밍.
//   판정: ① occupancy 유지 → 확장 순이득 ② 하락 → B≤8 회귀 확인. + SB<16>가 gemm_batch(B×reload) 이기나(B=12,16).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_batch.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

static void occ(const char*nm, const void*fn, int block){
    cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a,fn));
    int nb=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,fn,block,0));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    int warpsSM = nb*(block/32); double occv=(double)warpsSM/(p.maxThreadsPerMultiProcessor/32);
    printf("  %-22s regs=%-4d smem=%-5zu blocks/SM=%-3d warps/SM=%-3d occ=%.0f%%\n",
           nm,a.numRegs,a.sharedSizeBytes,nb,warpsSM,occv*100);
}

int main(int argc,char**argv){
    int IN=argc>1?atoi(argv[1]):4096, OUT=argc>2?atoi(argv[2]):4096, seed=argc>3?atoi(argv[3]):7;
    int block=256; srand(seed);
    printf("[probe_sb16] IN=%d OUT=%d (scale fp16=__half, 엔진 기본)\n",IN,OUT);
    printf("=== occupancy (register-limited, CUDA API 결정적) ===\n");
    occ("SB<8>  __half",(const void*)gemv_w4a8_sb_kernel<__half,8>,block);
    occ("SB<16> __half",(const void*)gemv_w4a8_sb_kernel<__half,16>,block);
    occ("gemv mlp<2>",(const void*)gemv_w4a8_mlp_kernel<__half,2>,block);

    // 데이터: X[16*IN] int8, xs[16*IN/QBLK], W[OUT*IN/2], sw[OUT*IN/QBLK] fp16
    int Smax=16;
    std::vector<int8_t> xq((size_t)Smax*IN); for(auto&v:xq)v=(int8_t)((rand()%255)-127);
    std::vector<float> xs((size_t)Smax*(IN/QBLK)); for(auto&v:xs)v=((rand()%1000)/1000.f)*0.05f+0.001f;
    std::vector<uint8_t> Wq((size_t)OUT*(IN/2)); for(auto&v:Wq)v=(uint8_t)(rand()%256);
    std::vector<__half> sw((size_t)OUT*(IN/QBLK)); for(auto&v:sw)v=__float2half(((rand()%1000)/1000.f)*0.02f+0.0005f);
    int8_t*dxq;float*dxs;uint8_t*dwq;__half*dsw;float*dY8,*dY16,*dYg;
    CK(cudaMalloc(&dxq,xq.size()));CK(cudaMemcpy(dxq,xq.data(),xq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dxs,xs.size()*4));CK(cudaMemcpy(dxs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dwq,Wq.size()));CK(cudaMemcpy(dwq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsw,sw.size()*2));CK(cudaMemcpy(dsw,sw.data(),sw.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dY8,(size_t)Smax*OUT*4));CK(cudaMalloc(&dY16,(size_t)Smax*OUT*4));CK(cudaMalloc(&dYg,(size_t)Smax*OUT*4));
    auto runSB=[&](int SMAXt,int S,float*Y){ long grid=((long)OUT*32+block-1)/block;
        if(SMAXt==8) gemv_w4a8_sb_kernel<__half,8><<<(unsigned)grid,block>>>(dxq,dxs,dwq,dsw,Y,IN,OUT,S);
        else         gemv_w4a8_sb_kernel<__half,16><<<(unsigned)grid,block>>>(dxq,dxs,dwq,dsw,Y,IN,OUT,S); };

    printf("=== bit-exact ===\n");
    int fail=0;
    for(int S=1;S<=16;S++){
        runSB(16,S,dY16); if(S<=8) runSB(8,S,dY8);
        launch_gemm_batch_w4a8(dxq,dxs,dwq,dsw,dYg,IN,OUT,S,1);  // scale_dtype=1(fp16)
        CK(cudaDeviceSynchronize());
        std::vector<float> y16((size_t)S*OUT),yg((size_t)S*OUT); std::vector<float> y8;
        CK(cudaMemcpy(y16.data(),dY16,y16.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yg.data(),dYg,yg.size()*4,cudaMemcpyDeviceToHost));
        if(S<=8){y8.resize((size_t)S*OUT);CK(cudaMemcpy(y8.data(),dY8,y8.size()*4,cudaMemcpyDeviceToHost));}
        int mm16g=0,mm168=0; for(size_t i=0;i<y16.size();i++){ if(memcmp(&y16[i],&yg[i],4))mm16g++; if(S<=8&&memcmp(&y16[i],&y8[i],4))mm168++; }
        if(mm16g||mm168){ printf("  S=%2d: SB16 vs gemm_batch mm=%d, SB16 vs SB8 mm=%d ★\n",S,mm16g,mm168); fail++; }
    }
    printf("  → %s (SB16==gemm_batch 모든 S, SB16==SB8 S≤8)\n", fail?"FAIL":"BIT-IDENTICAL");

    printf("=== 타이밍 (us, %d iter 평균; SB16 vs 현행[S≤8 SB8 / S>8 gemm_batch]) ===\n",200);
    cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
    for(int S:{2,4,8,12,16}){
        // 현행 경로
        auto cur=[&](){ if(S<=8) runSB(8,S,dY8); else launch_gemm_batch_w4a8(dxq,dxs,dwq,dsw,dYg,IN,OUT,S,1); };
        auto sb16=[&](){ runSB(16,S,dY16); };
        for(int i=0;i<20;i++){cur();} CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0)); for(int i=0;i<200;i++)cur(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float tc;CK(cudaEventElapsedTime(&tc,e0,e1));
        for(int i=0;i<20;i++){sb16();} CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0)); for(int i=0;i<200;i++)sb16(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ts;CK(cudaEventElapsedTime(&ts,e0,e1));
        const char*cn=S<=8?"SB8":"gemm_batch";
        printf("  S=%2d: 현행(%-10s)=%.1fus  SB16=%.1fus  → SB16 %.2f× %s\n",
               S,cn,tc/200*1000,ts/200*1000, tc/ts, ts<tc?"빠름":"느림");
    }
    return fail?1:0;
}
