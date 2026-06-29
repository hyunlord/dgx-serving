// 챕터7 FA.2c.0: 2-패스 Flash-Decoding(KV 분할) == CPU naive(decode Sq=1, GQA) 오차0.
//   강조1: 여러 split 결합 정확성 — base 크게(SPLIT_KV=512라 split 다수), split마다 m 다를 때 패스2 rescale 검증.
//   강조2: 결정성(2-패스 race 없음) 20회 bitwise. GQA NQ24/NKV4.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/attention_flash.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
// CPU naive, Sq=1, 위치=base (attend [0..base]). kv=h/(NQ/NKV).
static void cpu_dec(const std::vector<float>&Q,const std::vector<float>&Kc,const std::vector<float>&Vc,
                    std::vector<float>&O,int NQ,int NKV,int HD,int base){
    int GRP=NQ/NKV; double scale=1.0/sqrt((double)HD); int al=base+1;
    for(int h=0;h<NQ;h++){int kv=h/GRP; std::vector<double>s(al);
        for(int j=0;j<al;j++){double d=0;for(int t=0;t<HD;t++)d+=(double)Q[h*HD+t]*Kc[((size_t)j*NKV+kv)*HD+t];s[j]=d*scale;}
        double m=-1e300;for(double v:s)m=fmax(m,v);double sm=0;for(int j=0;j<al;j++){s[j]=exp(s[j]-m);sm+=s[j];}
        for(int t=0;t<HD;t++){double a=0;for(int j=0;j<al;j++)a+=s[j]*Vc[((size_t)j*NKV+kv)*HD+t];O[h*HD+t]=(float)(a/sm);}}
}
static double maxrel(const std::vector<float>&A,const std::vector<float>&B){
    double e=0,rms=0;for(size_t i=0;i<A.size();i++){e=fmax(e,fabs((double)A[i]-B[i]));rms+=(double)B[i]*B[i];}
    rms=sqrt(rms/A.size());return e/(rms>0?rms:1);}
static int g_fail=0;
static void test(int NQ,int NKV,int base){
    const int HD=256;int al=base+1,Lc=al;
    std::vector<float> Q((size_t)NQ*HD),Kc((size_t)Lc*NKV*HD),Vc((size_t)Lc*NKV*HD);
    unsigned sd=base*7+NQ;
    for(auto&v:Q)v=frand(sd);for(auto&v:Kc)v=frand(sd);for(auto&v:Vc)v=frand(sd);
    std::vector<float> Oref((size_t)NQ*HD);cpu_dec(Q,Kc,Vc,Oref,NQ,NKV,HD,base);
    float*dQ,*dK,*dV,*dO;size_t qb=Q.size()*4,kb=Kc.size()*4,ob=Oref.size()*4;
    CK(cudaMalloc(&dQ,qb));CK(cudaMalloc(&dK,kb));CK(cudaMalloc(&dV,kb));CK(cudaMalloc(&dO,ob));
    CK(cudaMemcpy(dQ,Q.data(),qb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dK,Kc.data(),kb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dV,Vc.data(),kb,cudaMemcpyHostToDevice));
    launch_flash_decode<HD>(dQ,dK,dV,dO,NQ,NKV,base);CK(cudaDeviceSynchronize());
    std::vector<float> Of(Oref.size());CK(cudaMemcpy(Of.data(),dO,ob,cudaMemcpyDeviceToHost));
    double r=maxrel(Of,Oref); int NS=(al+FD_SPLIT_KV-1)/FD_SPLIT_KV; bool ok=r<1e-2;
    printf("  base=%-6d NQ=%-2d NKV=%-2d (split=%d) | FlashDecode vs CPU rel=%.2e %s\n",base,NQ,NKV,NS,r,ok?"OK":"FAIL");
    if(!ok)g_fail=1;
    std::vector<float> Od(Oref.size()); bool d2=true;
    for(int rep=0;rep<20&&d2;rep++){ launch_flash_decode<HD>(dQ,dK,dV,dO,NQ,NKV,base);CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Od.data(),dO,ob,cudaMemcpyDeviceToHost)); for(size_t i=0;i<Of.size();i++)if(Od[i]!=Of[i]){d2=false;break;} }
    printf("        └ 결정성 20회 %s\n",d2?"OK":"FAIL(race!)"); if(!d2)g_fail=1;
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
}
int main(){
    printf("=== FA.2c.0: 2-패스 Flash-Decoding == CPU naive (decode Sq=1, GQA) ===\n");
    test(24,4, 100);    // split=1 (trivial)
    test(24,4, 600);    // split=2
    test(24,4, 2000);   // split=4 (다중 — rescale 검증)
    test(24,4, 8192);   // split=16
    test(16,16,1500);   // MHA split=3
    test(16,2, 5000);   // group8 split=10
    printf("\n[test_fa2c] %s\n", g_fail?"FAIL":"PASS");
    return g_fail;
}
