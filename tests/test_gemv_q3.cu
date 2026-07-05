// q3 gemv scheme-internal 골든 (STEP A): GPU gemv_q3(packed) == GPU gemv_f32(CPU-dequant Wd) BIT-IDENTICAL
//   (두 커널 누적 순서 동일 — warp당 1출력·lane 32-stride·warp_reduce) + 결정성 30회 + bytes 0.8× 확인
//   + 격리 타이밍 vs gemv_q4(27b MLP dims 17408×5120, S=1 decode — 풀기 비용 예비 측정, in-engine은 별도).
//   스킴: q=clip(round(w/s+3.5),0,7), s=absmax/3.5, deq=(q-3.5)*s (tools/q2_quality.py 동일 — 품질 measured 연속).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../kernels/gemv_f32.cuh"
#include "../kernels/gemv_q4.cuh"
#include "../kernels/gemv_q3.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}

// CPU q3 pack(bit-plane) + dequant ref — 커널 헤더의 규칙과 정확히 일치해야(골든의 기준).
static void quantW3(const std::vector<float>&W,int OUT,int IN,std::vector<uint32_t>&pk,std::vector<float>&sc,std::vector<float>&Wd){
    int ngrp=IN/QBLK; pk.assign((size_t)OUT*ngrp*3,0u); sc.resize((size_t)OUT*ngrp); Wd.resize(W.size());
    for(int o=0;o<OUT;o++)for(int g=0;g<ngrp;g++){
        float mx=0; for(int j=0;j<QBLK;j++) mx=fmaxf(mx,fabsf(W[(size_t)o*IN+g*QBLK+j]));
        float s=mx/3.5f; if(s==0)s=1; sc[(size_t)o*ngrp+g]=s;
        uint32_t lo[2]={0,0},hi=0;
        for(int j=0;j<QBLK;j++){
            int q=(int)lroundf(W[(size_t)o*IN+g*QBLK+j]/s+3.5f); if(q<0)q=0; if(q>7)q=7;
            lo[j>>4] |= (uint32_t)(q&3) << ((j&15)*2);
            hi       |= (uint32_t)((q>>2)&1) << j;
            Wd[(size_t)o*IN+g*QBLK+j] = ((float)q-3.5f)*s;      // CPU dequant(fp32 곱 1회 = GPU와 동일 rounding)
        }
        size_t base=((size_t)o*ngrp+g)*3; pk[base]=lo[0]; pk[base+1]=lo[1]; pk[base+2]=hi;
    }
}
// q4 대조용(격리 타이밍) — 기존 스킴
static void quantW4(const std::vector<float>&W,int OUT,int IN,std::vector<uint8_t>&pk,std::vector<float>&sc){
    int nblk=IN/QBLK; sc.resize((size_t)OUT*nblk); pk.assign((size_t)OUT*IN/2,0);
    for(int o=0;o<OUT;o++)for(int b=0;b<nblk;b++){float mx=0;for(int k=0;k<QBLK;k++)mx=fmaxf(mx,fabsf(W[(size_t)o*IN+b*QBLK+k]));
        float d=mx/7.f;if(d==0)d=1;sc[(size_t)o*nblk+b]=d;
        for(int k=0;k<QBLK;k++){int q=(int)lroundf(W[(size_t)o*IN+b*QBLK+k]/d)+8;if(q<0)q=0;if(q>15)q=15;
            size_t gi=(size_t)o*IN+b*QBLK+k; if(gi&1)pk[gi/2]|=(q<<4); else pk[gi/2]|=q;}}
}
int main(){
    unsigned seed=11; int g_fail=0;
    struct{int OUT,IN,S;} cs[]={{512,1024,1},{300,256,3},{17408,5120,1}};   // 소형 2 + 27b MLP-up dims
    for(auto c: cs){
        int OUT=c.OUT,IN=c.IN,S=c.S; int ngrp=IN/QBLK;
        std::vector<float> Xf((size_t)S*IN),Wf((size_t)OUT*IN);
        for(auto&v:Xf)v=frand(seed); for(auto&v:Wf)v=frand(seed)*0.3f;
        std::vector<uint32_t> W3; std::vector<float> sc3, Wd; quantW3(Wf,OUT,IN,W3,sc3,Wd);
        float *dX,*dWd,*dY3,*dYr; uint32_t*dW3; float*dsc3;
        CK(cudaMalloc(&dX,Xf.size()*4)); CK(cudaMalloc(&dWd,Wd.size()*4));
        CK(cudaMalloc(&dW3,W3.size()*4)); CK(cudaMalloc(&dsc3,sc3.size()*4));
        CK(cudaMalloc(&dY3,(size_t)S*OUT*4)); CK(cudaMalloc(&dYr,(size_t)S*OUT*4));
        CK(cudaMemcpy(dX,Xf.data(),Xf.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWd,Wd.data(),Wd.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW3,W3.data(),W3.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsc3,sc3.data(),sc3.size()*4,cudaMemcpyHostToDevice));
        // ★골든: gemv_q3(packed) vs gemv_f32(CPU-dequant Wd) — 누적 순서 동일 → BIT-ID 기대
        launch_gemv_q3(dX,dW3,dsc3,dY3,IN,OUT,S); CK(cudaDeviceSynchronize());
        { dim3 blk(256); long warps=(long)S*OUT; gemv_f32_kernel<<<(unsigned)((warps*32+255)/256),blk>>>(dX,dWd,dYr,IN,OUT,S); }
        CK(cudaDeviceSynchronize());
        std::vector<float> Y3((size_t)S*OUT),Yr((size_t)S*OUT);
        CK(cudaMemcpy(Y3.data(),dY3,Y3.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Yr.data(),dYr,Yr.size()*4,cudaMemcpyDeviceToHost));
        size_t nbit=0; double emax=0,rms=0;
        for(size_t i=0;i<Y3.size();i++){ if(Y3[i]!=Yr[i])nbit++; emax=fmax(emax,fabs((double)Y3[i]-Yr[i])); rms+=Yr[i]*Yr[i]; }
        rms=sqrt(rms/Yr.size()); double rel=emax/(rms+1e-12);
        bool ok = (nbit==0) || (rel<1e-6);
        printf("  OUT=%-5d IN=%-5d S=%d | q3 vs CPU-ref(f32 gemv): bit-diff=%zu/%zu max|Δ|=%.2e rel=%.2e %s\n",
               OUT,IN,S,nbit,Y3.size(),emax,rel, nbit==0?"BIT-ID✓":(ok?"스킴내":"★FAIL"));
        if(!ok)g_fail=1;
        // 결정성 30회
        std::vector<float> Yd((size_t)S*OUT); bool det=true;
        for(int r=0;r<30&&det;r++){ launch_gemv_q3(dX,dW3,dsc3,dY3,IN,OUT,S); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(Yd.data(),dY3,Yd.size()*4,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<Yd.size();i++) if(Yd[i]!=Y3[i]){det=false;break;} }
        printf("  OUT=%-5d IN=%-5d S=%d | 결정성 30회 %s\n",OUT,IN,S,det?"OK":"FAIL(비결정)"); if(!det)g_fail=1;
        // 격리 타이밍(27b dims만): q3 vs q4 — bytes 0.8× → 시간 몇 %? (풀기 비용 예비. in-engine은 별도)
        if(OUT>=16384){
            std::vector<uint8_t> W4; std::vector<float> sc4; quantW4(Wf,OUT,IN,W4,sc4);
            uint8_t*dW4; float*dsc4,*dY4; CK(cudaMalloc(&dW4,W4.size())); CK(cudaMalloc(&dsc4,sc4.size()*4)); CK(cudaMalloc(&dY4,(size_t)S*OUT*4));
            CK(cudaMemcpy(dW4,W4.data(),W4.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsc4,sc4.data(),sc4.size()*4,cudaMemcpyHostToDevice));
            for(int w=0;w<30;w++){ launch_gemv_q3(dX,dW3,dsc3,dY3,IN,OUT,S); launch_gemv_q4(dX,dW4,dsc4,dY4,IN,OUT,S);} CK(cudaDeviceSynchronize());
            cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b); float m3=0,m4=0; int R=200;
            cudaEventRecord(a); for(int r=0;r<R;r++) launch_gemv_q3(dX,dW3,dsc3,dY3,IN,OUT,S); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&m3,a,b);
            cudaEventRecord(a); for(int r=0;r<R;r++) launch_gemv_q4(dX,dW4,dsc4,dY4,IN,OUT,S); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&m4,a,b);
            double b3=(double)OUT*ngrp*16, b4=(double)OUT*ngrp*20;  // bytes/row-group: q3 12+4, q4 16+4
            printf("  [격리 타이밍 27b MLP-up S=1] q3 %.1fus vs q4 %.1fus = q3/q4 %.3f (bytes비 %.2f — 풀기 비용 예비, in-engine 별도)\n",
                   m3*1000/R,m4*1000/R,m3/m4,b3/b4);
            cudaFree(dW4);cudaFree(dsc4);cudaFree(dY4);
        }
        cudaFree(dX);cudaFree(dWd);cudaFree(dW3);cudaFree(dsc3);cudaFree(dY3);cudaFree(dYr);
    }
    printf("[test_gemv_q3] %s (q3 packed == CPU-dequant ref, bit-plane packing)\n",g_fail?"FAIL":"PASS");
    return g_fail;
}
