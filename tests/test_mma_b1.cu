// 챕터7 STEP B.1: W4A8 operand MMA + 블록 스케일 통합 == CPU W4A8 dot 오차0.
//   X int8(+xs per-block) · W int4→(wq-8) int8(+sw per-block) → MMA int32 per 블록 ×(xs·sw) → fp32.
//   단일 warp(M=16,N=8,K=가변). B.0 레이아웃 + W unpack + 블록 스케일. 레퍼런스=CPU W4A8(=gemv_w4a8 수식).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/config.h"   // QBLK=32
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
__device__ __forceinline__ int pk4(const int8_t*b){return (b[0]&0xFF)|((b[1]&0xFF)<<8)|((b[2]&0xFF)<<16)|((b[3]&0xFF)<<24);}

// X int8[16,K], xs[16,K/32]; Wq int4 packed[8,K/2], sw[8,K/32] → Y fp32[16,8]
__global__ void mma_w4a8(const int8_t*X,const float*xs,const uint8_t*Wq,const float*sw,float*Y,int K){
    int lane=threadIdx.x&31, g=lane>>2, t=lane&3, nblk=K/QBLK;
    float acc0=0,acc1=0,acc2=0,acc3=0;
    for(int blk=0;blk<nblk;blk++){
        int k0=blk*QBLK; int8_t buf[4]; int a0,a1,a2,a3,b0,b1,c0=0,c1=0,c2=0,c3=0;
        // A = X int8 (B.0 레이아웃, 블록 k0 오프셋)
        for(int j=0;j<4;j++)buf[j]=X[g*K + k0+t*4+j];        a0=pk4(buf);
        for(int j=0;j<4;j++)buf[j]=X[(g+8)*K + k0+t*4+j];    a1=pk4(buf);
        for(int j=0;j<4;j++)buf[j]=X[g*K + k0+16+t*4+j];     a2=pk4(buf);
        for(int j=0;j<4;j++)buf[j]=X[(g+8)*K + k0+16+t*4+j]; a3=pk4(buf);
        // B = W int4→(wq-8) int8. col(N)=g, K=t*4.. (블록 k0). packed: Wq[n*(K/2) + (k0+k)/2]
        auto wdeq=[&](int n,int k)->int8_t{ uint8_t by=Wq[(size_t)n*(K/2)+((k0+k)>>1)]; int q=((k0+k)&1)?(by>>4):(by&0xF); return (int8_t)(q-8); };
        for(int j=0;j<4;j++)buf[j]=wdeq(g, t*4+j);     b0=pk4(buf);
        for(int j=0;j<4;j++)buf[j]=wdeq(g, 16+t*4+j);  b1=pk4(buf);
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        // 블록 스케일: C[m,n]_blk × xs[m,blk]·sw[n,blk]. c0(g,t*2) c1(g,t*2+1) c2(g+8,t*2) c3(g+8,t*2+1)
        acc0 += c0 * xs[g*nblk+blk]     * sw[(t*2)*nblk+blk];
        acc1 += c1 * xs[g*nblk+blk]     * sw[(t*2+1)*nblk+blk];
        acc2 += c2 * xs[(g+8)*nblk+blk] * sw[(t*2)*nblk+blk];
        acc3 += c3 * xs[(g+8)*nblk+blk] * sw[(t*2+1)*nblk+blk];
    }
    Y[g*8 + t*2]=acc0; Y[g*8 + t*2+1]=acc1; Y[(g+8)*8 + t*2]=acc2; Y[(g+8)*8 + t*2+1]=acc3;
}
int main(){
    const int M=16,N=8,K=256; unsigned s=7; auto fr=[&](){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;};
    int nblk=K/QBLK;
    std::vector<float> Xf((size_t)M*K),Wf((size_t)N*K); for(auto&v:Xf)v=fr(); for(auto&v:Wf)v=fr()*0.3f;
    // X int8 per-block (quantize_x_q8 동일): d=amax/127
    std::vector<int8_t> Xq((size_t)M*K); std::vector<float> xs((size_t)M*nblk);
    for(int m=0;m<M;m++)for(int b=0;b<nblk;b++){float am=0;for(int k=0;k<QBLK;k++)am=fmaxf(am,fabsf(Xf[m*K+b*QBLK+k]));
        float d=am/127.f; if(d==0)d=1; xs[m*nblk+b]=d; for(int k=0;k<QBLK;k++){int q=(int)lroundf(Xf[m*K+b*QBLK+k]/d); if(q<-127)q=-127;if(q>127)q=127; Xq[m*K+b*QBLK+k]=(int8_t)q;}}
    // W int4 per-block: scale=maxabs/7, q=round+8
    std::vector<uint8_t> Wq((size_t)N*K/2); std::vector<float> sw((size_t)N*nblk);
    for(int n=0;n<N;n++)for(int b=0;b<nblk;b++){float am=0;for(int k=0;k<QBLK;k++)am=fmaxf(am,fabsf(Wf[n*K+b*QBLK+k]));
        float d=am/7.f; if(d==0)d=1; sw[n*nblk+b]=d; for(int k=0;k<QBLK;k++){int q=(int)lroundf(Wf[n*K+b*QBLK+k]/d)+8;if(q<0)q=0;if(q>15)q=15;
            int gi=n*K+b*QBLK+k; if(gi&1)Wq[gi/2]|=(q<<4); else Wq[gi/2]|=q;}}
    // CPU W4A8 ref: Σ_blk (Σ_k xq·(wq-8)) × xs·sw
    std::vector<float> ref((size_t)M*N);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double a=0;
        for(int b=0;b<nblk;b++){int dot=0; for(int k=0;k<QBLK;k++){int gi=n*K+b*QBLK+k; uint8_t by=Wq[gi/2]; int wq=((gi&1)?(by>>4):(by&0xF))-8; dot+=(int)Xq[m*K+b*QBLK+k]*wq;}
            a+=(double)dot*xs[m*nblk+b]*sw[n*nblk+b];}
        ref[m*N+n]=(float)a;}
    int8_t*dX;float*dxs,*dsw,*dY;uint8_t*dW;
    CK(cudaMalloc(&dX,Xq.size()));CK(cudaMalloc(&dxs,xs.size()*4));CK(cudaMalloc(&dW,Wq.size()));CK(cudaMalloc(&dsw,sw.size()*4));CK(cudaMalloc(&dY,(size_t)M*N*4));
    CK(cudaMemcpy(dX,Xq.data(),Xq.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dxs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dsw,sw.data(),sw.size()*4,cudaMemcpyHostToDevice));
    mma_w4a8<<<1,32>>>(dX,dxs,dW,dsw,dY,K); CK(cudaDeviceSynchronize());
    std::vector<float> Y((size_t)M*N); CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(size_t i=0;i<Y.size();i++){e=fmax(e,fabs((double)Y[i]-ref[i]));rms+=ref[i]*ref[i];} rms=sqrt(rms/Y.size());
    bool ok=e/rms<1e-5;
    printf("[mma_b1] W4A8 MMA(블록스케일) vs CPU W4A8: max err=%.3e (rms=%.2e, rel=%.2e) %s\n",e,rms,e/rms,ok?"OK":"FAIL");
    printf("[test_mma_b1] %s (int8 MMA + 블록 스케일 통합)\n",ok?"PASS":"FAIL");
    return ok?0:1;
}
