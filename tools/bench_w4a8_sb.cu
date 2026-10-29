// 챕터8 STEP D de-risk: 소-batch(Sq 2~8) W4A8 GEMV 대역폭·정확성 마이크로벤치.
//   verify Sq=K가 현재 MMQ(prefill 128-tile)로 가서 소-Sq에 행 대부분 비워 대역폭 반토막(측정: 79 GB/s).
//   gemv_w4a8_sb = warp 1개가 출력 o 1개, weight 1회 적재·S 컬럼 DP4A → 대역폭 최적(목표 ~163 GB/s).
//   3-way 비교: naive-gemv(launch_gemv_w4a8, S× weight 재적재) vs MMQ(현 경로) vs SB(신규).
//   정확성: SB == naive-gemv (같은 누적 순서 → bit-동일 기대).
//   사용: ./bench_w4a8_sb [IN] [OUT] [scale_dtype=1]   (S=1..8 스윕)
#include "runtime/knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../runtime/config.h"            // QBLK
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_mmq.cuh"
#include "../kernels/gemm_batch.cuh"   // 기존 batch decode W4A8 커널 (qwen35_decode_batch가 사용) — SB와 비교

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

static float ev(cudaEvent_t a, cudaEvent_t b){ float ms; cudaEventElapsedTime(&ms,a,b); return ms; }

// 단순 LCG (Math.random 불요, 결정적 재현)
static uint32_t rs=12345; static uint32_t rnd(){ rs=rs*1664525u+1013904223u; return rs; }

int main(int argc,char**argv){
    knobs_init("bench_w4a8_sb");
    setvbuf(stdout,NULL,_IOLBF,0);
    int IN  = argc>1?atoi(argv[1]):5120;     // 27B FFN gate: H=5120 → FFN=17408
    int OUT = argc>2?atoi(argv[2]):17408;
    int sdt = argc>3?atoi(argv[3]):1;        // 1=fp16 scale(27B), 0=fp32
    const int SMAX=8, ITERS=50;
    printf("# W4A8 소-batch de-risk: IN=%d OUT=%d scale_dtype=%d (weight=%.1f MB)\n",
           IN,OUT,sdt,(double)OUT*IN/2/1e6);
    if(IN%QBLK){fprintf(stderr,"IN must be QBLK(%d) 배수\n",QBLK);return 1;}

    // ---- 호스트 랜덤 데이터 ----
    size_t wbytes=(size_t)OUT*IN/2, nsc=(size_t)OUT*(IN/QBLK);
    std::vector<uint8_t> hW(wbytes); for(auto&b:hW) b=(uint8_t)(rnd()&0xFF);
    std::vector<float> hScF(nsc); for(auto&v:hScF) v=0.005f+(rnd()%1000)*1e-5f;   // 양수 scale
    std::vector<__half> hScH(nsc); for(size_t i=0;i<nsc;i++) hScH[i]=__float2half(hScF[i]);
    std::vector<int8_t> hX((size_t)SMAX*IN); for(auto&b:hX) b=(int8_t)((rnd()%255)-127);
    std::vector<float> hXs((size_t)SMAX*(IN/QBLK)); for(auto&v:hXs) v=0.01f+(rnd()%1000)*2e-5f;

    // ---- 디바이스 ----
    uint8_t *dW; CK(cudaMalloc(&dW,wbytes)); CK(cudaMemcpy(dW,hW.data(),wbytes,cudaMemcpyHostToDevice));
    void *dSc; CK(cudaMalloc(&dSc,nsc*(sdt?2:4)));
    if(sdt) CK(cudaMemcpy(dSc,hScH.data(),nsc*2,cudaMemcpyHostToDevice));
    else    CK(cudaMemcpy(dSc,hScF.data(),nsc*4,cudaMemcpyHostToDevice));
    int8_t *dX; CK(cudaMalloc(&dX,(size_t)SMAX*IN)); CK(cudaMemcpy(dX,hX.data(),(size_t)SMAX*IN,cudaMemcpyHostToDevice));
    float *dXs; CK(cudaMalloc(&dXs,(size_t)SMAX*(IN/QBLK)*4)); CK(cudaMemcpy(dXs,hXs.data(),(size_t)SMAX*(IN/QBLK)*4,cudaMemcpyHostToDevice));
    float *dYg,*dYm,*dYsb,*dYgb; CK(cudaMalloc(&dYg,(size_t)SMAX*OUT*4)); CK(cudaMalloc(&dYm,(size_t)SMAX*OUT*4)); CK(cudaMalloc(&dYsb,(size_t)SMAX*OUT*4)); CK(cudaMalloc(&dYgb,(size_t)SMAX*OUT*4));

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    double wMB=(double)OUT*IN/2/1e9;   // GB

    printf("# GB/s 높을수록 좋음. naive-gemv=S× weight 재적재, MMQ=prefill 128-tile, gemm_batch=기존 batch decode, SB=신규 소-batch\n");
    for(int S : {1,2,3,4,5,6,8}){
        // warmup
        launch_gemv_w4a8(dX,dXs,nullptr,dW,dSc,dYg,IN,OUT,S,sdt);
        if(S>1) launch_gemm_mmq_w4a8(dX,dXs,dW,dSc,dYm,S,OUT,IN,sdt);
        launch_gemv_w4a8_sb(dX,dXs,dW,dSc,dYsb,IN,OUT,S,sdt);
        if(S>1) launch_gemm_batch_w4a8(dX,dXs,dW,dSc,dYgb,IN,OUT,S,sdt);
        CK(cudaDeviceSynchronize());

        // naive gemv (S× weight 재적재)
        CK(cudaEventRecord(t0)); for(int i=0;i<ITERS;i++) launch_gemv_w4a8(dX,dXs,nullptr,dW,dSc,dYg,IN,OUT,S,sdt);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); float tg=ev(t0,t1)/ITERS;
        // MMQ (현 verify 경로)
        float tm=-1;
        if(S>1){ CK(cudaEventRecord(t0)); for(int i=0;i<ITERS;i++) launch_gemm_mmq_w4a8(dX,dXs,dW,dSc,dYm,S,OUT,IN,sdt);
                 CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); tm=ev(t0,t1)/ITERS; }
        // SB (신규)
        CK(cudaEventRecord(t0)); for(int i=0;i<ITERS;i++) launch_gemv_w4a8_sb(dX,dXs,dW,dSc,dYsb,IN,OUT,S,sdt);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); float tsb=ev(t0,t1)/ITERS;
        // gemm_batch (기존 batch decode 경로)
        float tgb=-1;
        if(S>1){ CK(cudaEventRecord(t0)); for(int i=0;i<ITERS;i++) launch_gemm_batch_w4a8(dX,dXs,dW,dSc,dYgb,IN,OUT,S,sdt);
                 CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); tgb=ev(t0,t1)/ITERS; }

        // 정확성: SB vs naive-gemv (bit-exact 기대)
        std::vector<float> Yg((size_t)S*OUT), Ysb((size_t)S*OUT);
        CK(cudaMemcpy(Yg.data(),dYg,(size_t)S*OUT*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Ysb.data(),dYsb,(size_t)S*OUT*4,cudaMemcpyDeviceToHost));
        double maxae=0,maxrel=0; for(size_t i=0;i<(size_t)S*OUT;i++){ double ae=fabs(Yg[i]-Ysb[i]); maxae=fmax(maxae,ae);
            double d=fabs(Yg[i]); if(d>1e-6) maxrel=fmax(maxrel,ae/d); }

        auto gbs=[&](float t){ return t<0?0.0:wMB/(t/1e3); };   // GB/s
        printf("  %-3d | gemv %5.0f | MMQ %5.0f | gemm_batch %5.0f | SB %5.0f GB/s | SB==gemv ae=%.1e\n",
               S, gbs(tg), gbs(tm), gbs(tgb), gbs(tsb), maxae);
    }
    printf("# 해석: SB GB/s가 S 무관 평탄(~plain Sq=1 대역폭)이면 amortize 성공. MMQ는 소-S서 낮음.\n");
    return 0;
}
