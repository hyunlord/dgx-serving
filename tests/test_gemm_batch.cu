// 챕터3 STEP1 검증: batch gemm(B행) == B개 독립 single gemv, 시퀀스별 오차0.
//   Y[B,OUT] = X[B,IN]·Wᵀ. weight 재사용 커널이 행별로 정확한 값을 내는가(CPU fp64 오라클).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../kernels/gemm_batch.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
static int g_fail=0;
static float frand(unsigned&s){ s=s*1664525u+1013904223u; return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f; }

static void run(int IN,int OUT,int B,unsigned&seed){
    std::vector<float> X((size_t)B*IN), W((size_t)OUT*IN), Y((size_t)B*OUT);
    for(auto&v:X)v=frand(seed); for(auto&v:W)v=frand(seed)*0.5f;
    // CPU fp64 골든: 행별 독립 dot
    std::vector<double> gold((size_t)B*OUT);
    for(int b=0;b<B;b++)for(int o=0;o<OUT;o++){ double a=0; for(int i=0;i<IN;i++)a+=(double)X[(size_t)b*IN+i]*W[(size_t)o*IN+i]; gold[(size_t)b*OUT+o]=a; }
    float *dX,*dW,*dY; CK(cudaMalloc(&dX,X.size()*4));CK(cudaMalloc(&dW,W.size()*4));CK(cudaMalloc(&dY,Y.size()*4));
    CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dW,W.data(),W.size()*4,cudaMemcpyHostToDevice));
    launch_gemm_batch_f32(dX,dW,dY,IN,OUT,B); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));
    double e=0; int worst_b=0; for(int b=0;b<B;b++){ double eb=0; for(int o=0;o<OUT;o++) eb=fmax(eb,fabs((double)Y[(size_t)b*OUT+o]-gold[(size_t)b*OUT+o])); if(eb>e){e=eb;worst_b=b;} }
    bool ok=e<1e-2;
    printf("  [IN=%4d OUT=%4d B=%2d] batch gemm == B gemv  max err=%.3e (worst seq %d)  %s\n",IN,OUT,B,e,worst_b,ok?"OK":"FAIL");
    if(!ok)g_fail=1;
    // multi-warp 커널 (챕터4) == 같은 골든 (B=1,2,4,8만 지원)
    if(launch_gemm_mw_f32(dX,dW,dY,IN,OUT,B)){ CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));
        double e2=0; for(int b=0;b<B;b++)for(int o=0;o<OUT;o++) e2=fmax(e2,fabs((double)Y[(size_t)b*OUT+o]-gold[(size_t)b*OUT+o]));
        printf("  [IN=%4d OUT=%4d B=%2d] gemm_mw    == golden   max err=%.3e          %s\n",IN,OUT,B,e2,e2<1e-2?"OK":"FAIL");
        if(e2>=1e-2)g_fail=1;
    }
    cudaFree(dX);cudaFree(dW);cudaFree(dY);
}

int main(){
    unsigned seed=2024;
    printf("=== batch gemm(fp32) vs B개 독립 gemv (시퀀스별 오차0) ===\n");
    for(int B : {1,2,4,8,16}) run(1024, 2048, B, seed);   // q_proj류
    run(3584,1024,8,seed);                                 // FFN down
    run(1024,248320,4,seed);                               // lm_head
    printf("\n[test_gemm_batch] %s\n", g_fail?"FAIL":"PASS");
    return g_fail?1:0;
}
