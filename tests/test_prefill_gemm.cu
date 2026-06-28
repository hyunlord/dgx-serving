// 챕터6 STEP 1: prefill 16-타일 q4 GEMM == gemv_q4 (오차0). weight-reuse가 결과 안 바꿈.
//   mm_q4_tiled 로직(16-타일 + 나머지 8/4/2/1 분해) 복제 → gemv_q4 기준과 대조. 작은 Sq(안전).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/gemv_q4.cuh"
#include "../kernels/gemm_batch.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
static void quant(const std::vector<float>&w,std::vector<uint8_t>&packed,std::vector<float>&scale){
    int n=w.size(),nblk=n/QBLK; scale.resize(nblk); packed.assign(n/2,0);
    for(int b=0;b<nblk;b++){ float mx=0; for(int i=0;i<QBLK;i++)mx=fmaxf(mx,fabsf(w[b*QBLK+i]));
        float sc=mx/7.f; if(sc==0)sc=1; scale[b]=sc;
        for(int i=0;i<QBLK;i++){ int q=(int)lroundf(w[b*QBLK+i]/sc)+8; if(q<0)q=0;if(q>15)q=15;
            int gi=b*QBLK+i; if(gi&1)packed[gi/2]|=(q<<4); else packed[gi/2]|=q; } }
}
// mm_q4_tiled 복제 (qwen35.cu와 동일 로직)
static void tiled(const float*X,const uint8_t*wq,const float*sc,float*Y,int IN,int OUT,int S){
    if(S==1){launch_gemv_q4(X,wq,sc,Y,IN,OUT,1);return;}
    int t=0;
    for(;t+16<=S;t+=16) launch_gemm_batch_q4(X+(size_t)t*IN,wq,sc,Y+(size_t)t*OUT,IN,OUT,16);
    int r=S-t;
    for(int b=8;b>=1;b>>=1) while(r>=b){ launch_gemm_batch_q4(X+(size_t)t*IN,wq,sc,Y+(size_t)t*OUT,IN,OUT,b); t+=b; r-=b; }
}
int main(){
    unsigned seed=7; int g_fail=0;
    for(int S : {16,37,64,129}){           // B=16 타일 + 나머지(5,0,1) 경로 커버
        int IN=1024, OUT=512;
        std::vector<float> W((size_t)OUT*IN), X((size_t)S*IN);
        for(auto&v:W)v=frand(seed)*0.3f; for(auto&v:X)v=frand(seed);
        std::vector<uint8_t> packed; std::vector<float> scale; quant(W,packed,scale);
        float *dX,*dYg,*dYt; uint8_t*dWq; float*dsc;
        CK(cudaMalloc(&dX,X.size()*4)); CK(cudaMalloc(&dYg,(size_t)S*OUT*4)); CK(cudaMalloc(&dYt,(size_t)S*OUT*4));
        CK(cudaMalloc(&dWq,packed.size())); CK(cudaMalloc(&dsc,scale.size()*4));
        CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWq,packed.data(),packed.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsc,scale.data(),scale.size()*4,cudaMemcpyHostToDevice));
        launch_gemv_q4(dX,dWq,dsc,dYg,IN,OUT,S);          // 기준
        tiled(dX,dWq,dsc,dYt,IN,OUT,S);                   // 16-타일 GEMM
        CK(cudaDeviceSynchronize());
        std::vector<float> Yg((size_t)S*OUT),Yt((size_t)S*OUT);
        CK(cudaMemcpy(Yg.data(),dYg,Yg.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Yt.data(),dYt,Yt.size()*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<Yg.size();i++) e=fmax(e,fabs((double)Yg[i]-Yt[i]));
        bool ok=e<1e-4; printf("  S=%-4d tiled-GEMM vs gemv_q4: max err=%.3e %s\n",S,e,ok?"OK":"FAIL"); if(!ok)g_fail=1;
        cudaFree(dX);cudaFree(dYg);cudaFree(dYt);cudaFree(dWq);cudaFree(dsc);
    }
    printf("[test_prefill_gemm] %s (16-타일 weight-reuse == gemv, 오차0)\n",g_fail?"FAIL":"PASS");
    return g_fail;
}
