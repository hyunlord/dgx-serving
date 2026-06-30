// 챕터9 roofline 실험: gemv_w4a8 MLP↑ 변형 bit-exact + 속도. baseline vs mlp<U=2,4,8>.
//   global-load latency를 outstanding load U개로 숨기나. bit-exact(0-diff) 게이트 후 속도.
//   주: cudaEvent는 warm(L2) — 상대 비교용. 결정적 판정은 in-engine t/s.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <cuda_runtime.h>
#include "../kernels/gemv_w4a8.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static unsigned sd=99u; static int ir(int lo,int hi){sd=sd*1664525u+1013904223u;return lo+(int)((sd>>9)%(unsigned)(hi-lo+1));}
int main(int argc,char**argv){
    const int M=1, K=argc>1?atoi(argv[1]):5120, N=argc>2?atoi(argv[2]):17408;
    const int nblk=K/QBLK, K2=K/2;
    double wbytes=(double)N*K2;
    printf("[mlp] M=%d K=%d N=%d weight=%.1fMB\n",M,K,N,wbytes/1e6);
    std::vector<int8_t> hX((size_t)M*K); for(auto&x:hX)x=(int8_t)ir(-40,40);
    std::vector<uint8_t> hW((size_t)N*K2); for(auto&x:hW)x=(uint8_t)ir(0,255);
    std::vector<float> hXs((size_t)M*nblk),hSw((size_t)N*nblk); for(auto&x:hXs)x=0.03f; for(auto&x:hSw)x=0.02f;
    int8_t*dX;uint8_t*dW;float*dXs,*dSw,*dY,*dYb;
    CK(cudaMalloc(&dX,hX.size()));CK(cudaMalloc(&dW,hW.size()));CK(cudaMalloc(&dXs,hXs.size()*4));
    CK(cudaMalloc(&dSw,hSw.size()*4));CK(cudaMalloc(&dY,(size_t)N*4));CK(cudaMalloc(&dYb,(size_t)N*4));
    CK(cudaMemcpy(dX,hX.data(),hX.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW,hW.data(),hW.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dXs,hXs.data(),hXs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSw,hSw.data(),hSw.size()*4,cudaMemcpyHostToDevice));
    auto bench=[&](const char*nm, const std::function<void()>& run, float* out)->double{
        for(int w=0;w<10;w++) run(); CK(cudaDeviceSynchronize());
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);int R=300;float t=0;
        cudaEventRecord(a); for(int r=0;r<R;r++) run(); cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&t,a,b);
        double us=t/R*1000.0; printf("  %-14s %.1f us | %.1f GB/s (%.0f%% of 273)\n",nm,us,wbytes/(us*1e-6)/1e9,wbytes/(us*1e-6)/1e9/273*100);
        return us;
    };
    // baseline
    auto base=[&](){ launch_gemv_w4a8(dX,dXs,nullptr,dW,dSw,dYb,K,N,M); };
    bench("baseline", base, dYb);
    std::vector<float> hb(N); CK(cudaMemcpy(hb.data(),dYb,(size_t)N*4,cudaMemcpyDeviceToHost));
    auto chk=[&](const char*nm){ std::vector<float> h(N); CK(cudaMemcpy(h.data(),dY,(size_t)N*4,cudaMemcpyDeviceToHost));
        double e=0; for(int i=0;i<N;i++) e=fmax(e,fabs((double)h[i]-hb[i])); printf("    %s bit-exact vs baseline: max|diff|=%.3e %s\n",nm,e,e==0.0?"(0-diff ✓)":"(DIFF!)"); };
    auto m2=[&](){ launch_gemv_w4a8_mlp<2>(dX,dXs,nullptr,dW,dSw,dY,K,N,M); }; bench("mlp U=2", m2, dY); chk("U=2");
    auto m4=[&](){ launch_gemv_w4a8_mlp<4>(dX,dXs,nullptr,dW,dSw,dY,K,N,M); }; bench("mlp U=4", m4, dY); chk("U=4");
    auto m8=[&](){ launch_gemv_w4a8_mlp<8>(dX,dXs,nullptr,dW,dSw,dY,K,N,M); }; bench("mlp U=8", m8, dY); chk("U=8");
    return 0;
}
