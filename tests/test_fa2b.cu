// 챕터7 FA.2.2: bf16 tensor-core FlashAttention == CPU naive(GQA,causal) rel<1e-2.
//   강조1(vLLM 함정): 긴 context(많은 타일 누적)서 fp32 누적 정밀도 유지 — base 32K(2048 타일)까지.
//   강조2: 결정성(여러 번 bitwise). bf16 MMA지만 race 없어야(단일 warp, shared-왕복).
//   bf16이라 fp64-CPU(참값) 대비 rel ~1e-3(bf16 반올림) — 1e-2 기준 통과. GQA NQ24/NKV4 등.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../kernels/attention_flash.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
static void cpu_attn(const std::vector<float>&Q,const std::vector<float>&Kc,const std::vector<float>&Vc,
                     std::vector<float>&O,int Sq,int NQ,int NKV,int HD,int base){
    int GRP=NQ/NKV; double scale=1.0/sqrt((double)HD);
    for(int i=0;i<Sq;i++)for(int h=0;h<NQ;h++){int kv=h/GRP,al=base+i+1;std::vector<double>s(al);
        for(int j=0;j<al;j++){double d=0;for(int t=0;t<HD;t++)d+=(double)Q[((size_t)i*NQ+h)*HD+t]*Kc[((size_t)j*NKV+kv)*HD+t];s[j]=d*scale;}
        double m=-1e300;for(double v:s)m=fmax(m,v);double sm=0;for(int j=0;j<al;j++){s[j]=exp(s[j]-m);sm+=s[j];}
        for(int t=0;t<HD;t++){double a=0;for(int j=0;j<al;j++)a+=s[j]*Vc[((size_t)j*NKV+kv)*HD+t];O[((size_t)i*NQ+h)*HD+t]=(float)(a/sm);}}
}
static double maxrel(const std::vector<float>&A,const std::vector<float>&B){
    double e=0,rms=0;for(size_t i=0;i<A.size();i++){e=fmax(e,fabs((double)A[i]-B[i]));rms+=(double)B[i]*B[i];}
    rms=sqrt(rms/A.size());return e/(rms>0?rms:1);}
static int g_fail=0;
static void test(int Sq,int NQ,int NKV,int base,unsigned&seed,bool det){
    const int HD=256; int Lc=base+Sq;
    std::vector<float> Q((size_t)Sq*NQ*HD),Kc((size_t)Lc*NKV*HD),Vc((size_t)Lc*NKV*HD);
    // bf16 입력으로 통일(FA.2 내부 bf16) → 오차가 알고리즘/누적인지 격리(bf16 입력반올림 상수 제거)
    for(auto&v:Q)v=__bfloat162float(__float2bfloat16(frand(seed)));
    for(auto&v:Kc)v=__bfloat162float(__float2bfloat16(frand(seed)));
    for(auto&v:Vc)v=__bfloat162float(__float2bfloat16(frand(seed)));
    std::vector<float> Oref((size_t)Sq*NQ*HD); cpu_attn(Q,Kc,Vc,Oref,Sq,NQ,NKV,HD,base);
    float*dQ,*dK,*dV,*dO; size_t qb=Q.size()*4,kb=Kc.size()*4,ob=Oref.size()*4;
    CK(cudaMalloc(&dQ,qb));CK(cudaMalloc(&dK,kb));CK(cudaMalloc(&dV,kb));CK(cudaMalloc(&dO,ob));
    CK(cudaMemcpy(dQ,Q.data(),qb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dK,Kc.data(),kb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dV,Vc.data(),kb,cudaMemcpyHostToDevice));
    launch_flash_attn2b_t<HD,16,8>(dQ,dK,dV,dO,Sq,NQ,NKV,base); CK(cudaDeviceSynchronize());
    std::vector<float> Of(Oref.size()); CK(cudaMemcpy(Of.data(),dO,ob,cudaMemcpyDeviceToHost));
    double r=maxrel(Of,Oref); int ntile=(base+Sq+15)/16;
    bool ok=r<1e-2;
    printf("  Sq=%-3d NQ=%-2d NKV=%-2d base=%-6d (타일~%d) | FA.2b vs CPU rel=%.2e %s\n",Sq,NQ,NKV,base,ntile,r,ok?"OK":"FAIL");
    if(!ok)g_fail=1;
    if(det){ std::vector<float> Od(Oref.size()); bool d2=true;
        for(int rep=0;rep<20&&d2;rep++){ launch_flash_attn2b_t<HD,16,8>(dQ,dK,dV,dO,Sq,NQ,NKV,base); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(Od.data(),dO,ob,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<Of.size();i++) if(Od[i]!=Of[i]){d2=false;break;} }
        printf("        └ 결정성 20회 %s\n",d2?"OK":"FAIL(race!)"); if(!d2)g_fail=1; }
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
}
int main(){
    unsigned seed=777;
    printf("=== FA.2b.1: 멀티워프(NW8) FA == CPU naive (GQA,causal,fp32누적) ===\n");
    test(1,  24,4, 0,    seed,true);   // decode Sq=1
    test(1,  24,4, 500,  seed,true);   // decode Sq=1, 다중타일
    test(16, 24,4, 0,    seed,true);   // 1타일
    test(16, 24,4, 100,  seed,true);   // 다중타일
    test(40, 24,4, 500,  seed,true);   // Sq>BR(여러 query타일), GQA group6
    test(16, 16,16,300,  seed,true);   // MHA
    test(7,  16,2, 777,  seed,true);   // 비배수 Sq/bounds, group8
    printf("--- 강조1: 긴 context 누적 (vLLM 함정 — fp32 누적 정밀도 유지?) ---\n");
    test(8,  24,4, 8192, seed,false);  // ~512 타일
    test(4,  24,4, 32768,seed,false);  // ~2048 타일 누적
    printf("\n[test_fa2b] %s\n", g_fail?"FAIL":"PASS");
    return g_fail;
}
