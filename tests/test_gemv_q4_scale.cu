// 챕터5 STEP 3+: q4 GEMV 누적오차의 reduction-dim 스케일 법칙 (35B/q2 예측 토대).
//   27B 통합 hidden 잔차 ~17%의 원인 = q4 GEMV GPU-fp32 누적이 CPU(golden) 순서와 다름.
//   여기선 동일 q4 weight·입력으로 (a)GPU-fp32 vs double-ref(누적오차 절대치), (b)GPU vs CPU-fp32-순차
//   (순서차 = engine-vs-golden 프록시)를 IN(reduction dim) 스윕으로 측정 → ε(IN) 법칙.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/gemv_q4.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
// per-block-32 대칭 int4 양자화 (extract 정책과 동일: scale=maxabs/7, q=round(w/scale)+8)
static void quant(const std::vector<float>&w,std::vector<uint8_t>&packed,std::vector<float>&scale){
    int n=w.size(),nblk=n/QBLK; scale.resize(nblk); packed.assign(n/2,0);
    for(int b=0;b<nblk;b++){ float mx=0; for(int i=0;i<QBLK;i++)mx=fmaxf(mx,fabsf(w[b*QBLK+i]));
        float sc=mx/7.f; if(sc==0)sc=1; scale[b]=sc;
        for(int i=0;i<QBLK;i++){ int q=(int)lroundf(w[b*QBLK+i]/sc)+8; if(q<0)q=0; if(q>15)q=15;
            int gi=b*QBLK+i; if(gi&1)packed[gi/2]|=(q<<4); else packed[gi/2]|=q; } }
}
static inline float deq(uint8_t p,int hi,float sc){return (((hi?(p>>4):(p&0x0F))-8))*sc;}

int main(){
    unsigned seed=7; const int OUT=256, S=1;
    int INs[]={512,1024,2048,5120,6144,10240,17408};  // 0.8B~27B reduction dim 범위
    printf("[gemv_q4_scale] OUT=%d S=%d | per-GEMV 누적오차 vs reduction dim(IN)\n",OUT,S);
    printf("  %-7s %-12s %-14s %-12s %-12s\n","IN","|y|rms","gpu-vs-dbl(abs)","rel(abs)","gpu-vs-cpufp32(순서차)");
    for(int IN: INs){
        std::vector<float> Wf((size_t)OUT*IN), X(IN);
        for(auto&v:Wf)v=frand(seed)*0.3f; for(auto&v:X)v=frand(seed);
        std::vector<uint8_t> packed; std::vector<float> scale; quant(Wf,packed,scale);
        // double ref + CPU fp32 순차 ref (동일 dequant weight)
        std::vector<double> refd(OUT); std::vector<float> reff(OUT);
        for(int o=0;o<OUT;o++){ double ad=0; float af=0;
            for(int i=0;i<IN;i++){ int gi=o*IN+i; float wq=deq(packed[gi/2],gi&1,scale[gi/QBLK]);
                ad+=(double)X[i]*wq; af+=X[i]*wq; }
            refd[o]=ad; reff[o]=af; }
        float *dX,*dY,*dsc; uint8_t*dWq;
        CK(cudaMalloc(&dX,X.size()*4));CK(cudaMalloc(&dY,OUT*4));CK(cudaMalloc(&dWq,packed.size()));CK(cudaMalloc(&dsc,scale.size()*4));
        CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWq,packed.data(),packed.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsc,scale.data(),scale.size()*4,cudaMemcpyHostToDevice));
        launch_gemv_q4(dX,dWq,dsc,dY,IN,OUT,S); CK(cudaDeviceSynchronize());
        std::vector<float> Y(OUT); CK(cudaMemcpy(Y.data(),dY,OUT*4,cudaMemcpyDeviceToHost));
        double egd=0,egf=0,sq=0; for(int o=0;o<OUT;o++){ egd=fmax(egd,fabs(Y[o]-refd[o]));
            egf=fmax(egf,fabs((double)Y[o]-reff[o])); sq+=refd[o]*refd[o]; }
        double rms=sqrt(sq/OUT);
        printf("  %-7d %-12.4e %-14.3e %-12.2e %-12.3e\n",IN,rms,egd,egd/rms,egf);
        cudaFree(dX);cudaFree(dY);cudaFree(dWq);cudaFree(dsc);
    }
    printf("[test_gemv_q4_scale] (측정 전용, pass/fail 없음)\n");
    return 0;
}
