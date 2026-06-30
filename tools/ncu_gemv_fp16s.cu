// STEP fp16-scale de-risk: W4A8 decode gemv scale fp32→fp16 (bytes/token↓ = base 갭 lever).
//   scale은 트래픽 ~20%(per-32 fp32=1b/weight). fp16시 0.5b/weight=llama Q4_0 동일 → 핸디캡 제거(~10% t/s).
//   측정: ① fp16-scale vs fp32-scale 정확성(같은 int4 weight, fp16 반올림만 — int4 quant 1.5e-2 << 보다 작아야)
//         ② t/s(>L2 FFN gate/down). argv[1]=K argv[2]=N.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/gemv_w4a8.cuh"   // baseline gemv_w4a8_kernel + warp_reduce_sum + QBLK
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static unsigned sd=7u; static int ir(int lo,int hi){sd=sd*1664525u+1013904223u;return lo+(int)((sd>>9)%(unsigned)(hi-lo+1));}

// fp16-scale 변형 (gemv_w4a8_kernel와 동일, scale만 __half)
static __global__ void gemv_w4a8_fp16s_kernel(const int8_t *xq, const __half *xs, const int32_t*,
                                              const uint8_t *Wq, const __half *sw,
                                              float *Y, int IN, int OUT, int S){
    long gw=((long)blockIdx.x*blockDim.x+threadIdx.x)>>5; int lane=threadIdx.x&31;
    if(gw>=(long)S*OUT) return;
    int s=(int)(gw/OUT), o=(int)(gw%OUT);
    const int8_t *xqr=xq+(size_t)s*IN; const __half *xsr=xs+(size_t)s*(IN/QBLK);
    const uint8_t *w=Wq+(size_t)o*(IN/2); const __half *swr=sw+(size_t)o*(IN/QBLK);
    auto pk=[](int lo,int hi)->int{
        return (((lo&0xF)-8)&0xFF)|((((lo>>4)&0xF)-8)&0xFF)<<8|(((hi&0xF)-8)&0xFF)<<16|((((hi>>4)&0xF)-8)&0xFF)<<24;};
    float acc=0.f;
    for(int base=lane*8; base<IN; base+=32*8){
        int blk=base/QBLK;
        float sc_b=__half2float(xsr[blk])*__half2float(swr[blk]);
        unsigned W=*reinterpret_cast<const unsigned*>(w+(base>>1));
        int x0=*reinterpret_cast<const int*>(xqr+base), x1=*reinterpret_cast<const int*>(xqr+base+4);
        int dot=__dp4a(x0,pk(W&0xFF,(W>>8)&0xFF),0);
        dot=__dp4a(x1,pk((W>>16)&0xFF,(W>>24)&0xFF),dot);
        acc+=sc_b*(float)dot;
    }
    acc=warp_reduce_sum(acc);
    if(lane==0) Y[(size_t)s*OUT+o]=acc;
}
static void launch_fp16s(const int8_t*xq,const __half*xs,const uint8_t*Wq,const __half*sw,float*Y,int IN,int OUT,int S){
    long warps=(long)S*OUT; long grid=(warps*32+255)/256;
    gemv_w4a8_fp16s_kernel<<<(unsigned)grid,256>>>(xq,xs,nullptr,Wq,sw,Y,IN,OUT,S);
}

int main(int argc,char**argv){
    const int M=1, K=argc>1?atoi(argv[1]):5120, N=argc>2?atoi(argv[2]):17408;
    const int nblk=K/QBLK, K2=K/2;
    double wbytes=(double)N*K2, sbytes32=(double)N*nblk*4;
    printf("[fp16s] M=%d K=%d N=%d | weight=%.1fMB scale(fp32)=%.1fMB(%.0f%%) → fp16시 %.1fMB(트래픽 %.0f%%↓)\n",
        M,K,N, wbytes/1e6, sbytes32/1e6, sbytes32/(wbytes+sbytes32)*100, sbytes32/2e6, sbytes32/2/(wbytes+sbytes32)*100);
    std::vector<int8_t> hX((size_t)M*K); for(auto&x:hX)x=(int8_t)ir(-40,40);
    std::vector<uint8_t> hW((size_t)N*K2); for(auto&x:hW)x=(uint8_t)ir(0,255);
    std::vector<float> hXs((size_t)M*nblk),hSw((size_t)N*nblk);
    for(auto&x:hXs)x=0.02f+ ((sd=sd*1664525u+1013904223u)>>8&0xFFFF)/65536.0f*0.05f;
    for(auto&x:hSw)x=0.01f+ ((sd=sd*1664525u+1013904223u)>>8&0xFFFF)/65536.0f*0.04f;
    std::vector<__half> hXs16((size_t)M*nblk),hSw16((size_t)N*nblk);
    for(size_t i=0;i<hXs.size();i++)hXs16[i]=__float2half(hXs[i]);
    for(size_t i=0;i<hSw.size();i++)hSw16[i]=__float2half(hSw[i]);
    int8_t*dX;uint8_t*dW;float*dXs,*dSw,*dY32,*dY16;__half*dXs16,*dSw16;
    CK(cudaMalloc(&dX,hX.size()));CK(cudaMalloc(&dW,hW.size()));
    CK(cudaMalloc(&dXs,hXs.size()*4));CK(cudaMalloc(&dSw,hSw.size()*4));
    CK(cudaMalloc(&dXs16,hXs.size()*2));CK(cudaMalloc(&dSw16,hSw.size()*2));
    CK(cudaMalloc(&dY32,(size_t)M*N*4));CK(cudaMalloc(&dY16,(size_t)M*N*4));
    CK(cudaMemcpy(dX,hX.data(),hX.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW,hW.data(),hW.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dXs,hXs.data(),hXs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSw,hSw.data(),hSw.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dXs16,hXs16.data(),hXs.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSw16,hSw16.data(),hSw.size()*2,cudaMemcpyHostToDevice));
    // 정확성: fp32 scale vs fp16 scale (같은 int4 weight)
    launch_gemv_w4a8(dX,dXs,nullptr,dW,dSw,dY32,K,N,M);
    launch_fp16s(dX,dXs16,dW,dSw16,dY16,K,N,M); CK(cudaDeviceSynchronize());
    std::vector<float> Y32((size_t)M*N),Y16((size_t)M*N);
    CK(cudaMemcpy(Y32.data(),dY32,Y32.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Y16.data(),dY16,Y16.size()*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(size_t i=0;i<Y32.size();i++){e=fmax(e,fabs((double)Y16[i]-Y32[i]));rms+=Y32[i]*Y32[i];} rms=sqrt(rms/Y32.size());
    printf("[fp16s] fp16-scale vs fp32-scale: max abs=%.3e rel=%.2e (int4 quant 1.5e-2 대비 %s)\n",
        e, e/rms, e/rms<1e-3?"무시가능 OK":"주의");
    // t/s
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);int R=300;float t32=0,t16=0;
    for(int w=0;w<10;w++){launch_gemv_w4a8(dX,dXs,nullptr,dW,dSw,dY32,K,N,M);launch_fp16s(dX,dXs16,dW,dSw16,dY16,K,N,M);} CK(cudaDeviceSynchronize());
    cudaEventRecord(a);for(int r=0;r<R;r++)launch_gemv_w4a8(dX,dXs,nullptr,dW,dSw,dY32,K,N,M);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&t32,a,b);
    cudaEventRecord(a);for(int r=0;r<R;r++)launch_fp16s(dX,dXs16,dW,dSw16,dY16,K,N,M);cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&t16,a,b);
    printf("[fp16s] fp32-scale %.1fus | fp16-scale %.1fus | speedup %.3fx (base 갭 lever)\n", t32/R*1000, t16/R*1000, t32/t16);
    return 0;
}
