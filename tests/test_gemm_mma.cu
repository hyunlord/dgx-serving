// 챕터7 STEP B.2 + C.2: MMA W4A8 GEMM == gemv_w4a8 오차0, BM 스윕(64/128/256) 전부 동일.
//   같은 양자화 operand(X int8, W int4)에 gemm_mma_w4a8 vs gemv_w4a8. 둘 다 int8 dot×scale → 오차0 기대.
//   C.2: BM 템플릿(weight 재read↓, M/BM회) — 정확성·결정성이 BM 무관 보존됨을 확인. 비배수 bounds 포함.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_mma.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
static void quantW(const std::vector<float>&w,int N,int K,std::vector<uint8_t>&pk,std::vector<float>&sc){
    int nblk=K/QBLK; sc.resize((size_t)N*nblk); pk.assign((size_t)N*K/2,0);
    for(int n=0;n<N;n++)for(int b=0;b<nblk;b++){float mx=0;for(int k=0;k<QBLK;k++)mx=fmaxf(mx,fabsf(w[(size_t)n*K+b*QBLK+k]));
        float d=mx/7.f;if(d==0)d=1;sc[(size_t)n*nblk+b]=d;
        for(int k=0;k<QBLK;k++){int q=(int)lroundf(w[(size_t)n*K+b*QBLK+k]/d)+8;if(q<0)q=0;if(q>15)q=15;
            size_t gi=(size_t)n*K+b*QBLK+k; if(gi&1)pk[gi/2]|=(q<<4); else pk[gi/2]|=q;}}
}
// BM 템플릿별 검사: MMA(BM) == 기준 Yg 오차0 + 결정성 30회 bitwise.
template<int BM>
static int checkBM(int M,int N,int K,const int8_t*dXq,const float*dxs,const uint8_t*dW,const float*dsw,const std::vector<float>&Yg){
    int fail=0; float*dYm; CK(cudaMalloc(&dYm,(size_t)M*N*4));
    launch_gemm_mma_w4a8_bm<BM>(dXq,dxs,dW,dsw,dYm,M,N,K); CK(cudaDeviceSynchronize());
    std::vector<float> Ym((size_t)M*N); CK(cudaMemcpy(Ym.data(),dYm,Ym.size()*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(size_t i=0;i<Yg.size();i++){e=fmax(e,fabs((double)Ym[i]-Yg[i]));rms+=Yg[i]*Yg[i];} rms=sqrt(rms/Yg.size());
    bool ok=e/rms<1e-5; printf("  M=%-4d N=%-4d K=%-5d BM=%-3d | MMA vs gemv_w4a8: max err=%.3e (rel=%.2e) %s\n",M,N,K,BM,e,e/rms,ok?"OK":"FAIL"); if(!ok)fail=1;
    // 결정성 가드(cp.async race 회귀 방지 — 과거 wait_group 순서 버그가 flaky 비결정 유발): 30회 == 첫 결과 bitwise
    std::vector<float> Yd((size_t)M*N); bool det=true;
    for(int r=0;r<30 && det;r++){ launch_gemm_mma_w4a8_bm<BM>(dXq,dxs,dW,dsw,dYm,M,N,K); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Yd.data(),dYm,Yd.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<Yd.size();i++) if(Yd[i]!=Ym[i]){det=false;break;} }
    printf("  M=%-4d N=%-4d K=%-5d BM=%-3d | 결정성 30회 %s\n",M,N,K,BM,det?"OK":"FAIL(비결정-race!)"); if(!det)fail=1;
    cudaFree(dYm); return fail;
}
int main(){
    unsigned seed=5; int g_fail=0;
    struct{int M,N,K;} cs[]={{64,128,512},{200,128,512},{256,130,256},{129,256,1024}};
    for(auto c: cs){
        int M=c.M,N=c.N,K=c.K,nblk=K/QBLK;
        std::vector<float> Xf((size_t)M*K),Wf((size_t)N*K); for(auto&v:Xf)v=frand(seed); for(auto&v:Wf)v=frand(seed)*0.3f;
        std::vector<uint8_t> Wpk; std::vector<float> Wsc; quantW(Wf,N,K,Wpk,Wsc);
        // X int8 양자화 (엔진 경로 quantize_x_q8)
        float *dXf; int8_t*dXq; float*dxs; int32_t*dxsum;
        CK(cudaMalloc(&dXf,Xf.size()*4));CK(cudaMalloc(&dXq,(size_t)M*K));CK(cudaMalloc(&dxs,(size_t)M*nblk*4));CK(cudaMalloc(&dxsum,(size_t)M*nblk*4));
        CK(cudaMemcpy(dXf,Xf.data(),Xf.size()*4,cudaMemcpyHostToDevice));
        launch_quantize_x_q8(dXf,dXq,dxs,(int32_t*)dxsum,K,M);
        uint8_t*dW; float*dsw,*dYg;
        CK(cudaMalloc(&dW,Wpk.size()));CK(cudaMalloc(&dsw,Wsc.size()*4));CK(cudaMalloc(&dYg,(size_t)M*N*4));
        CK(cudaMemcpy(dW,Wpk.data(),Wpk.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dsw,Wsc.data(),Wsc.size()*4,cudaMemcpyHostToDevice));
        launch_gemv_w4a8(dXq,dxs,dxsum,dW,dsw,dYg,K,N,M);          // 기준 (IN=K,OUT=N,S=M)
        CK(cudaDeviceSynchronize());
        std::vector<float> Yg((size_t)M*N); CK(cudaMemcpy(Yg.data(),dYg,Yg.size()*4,cudaMemcpyDeviceToHost));
        g_fail|=checkBM<64> (M,N,K,dXq,dxs,dW,dsw,Yg);
        g_fail|=checkBM<128>(M,N,K,dXq,dxs,dW,dsw,Yg);
        g_fail|=checkBM<256>(M,N,K,dXq,dxs,dW,dsw,Yg);
        cudaFree(dXf);cudaFree(dXq);cudaFree(dxs);cudaFree(dxsum);cudaFree(dW);cudaFree(dsw);cudaFree(dYg);
    }
    printf("[test_gemm_mma] %s (MMA W4A8 == gemv_w4a8, BM 64/128/256)\n",g_fail?"FAIL":"PASS");
    return g_fail;
}
