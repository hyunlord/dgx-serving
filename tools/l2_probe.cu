// L2 진단: GB10 L2 크기·persistence(cudaAccessPolicyWindow) 지원·상주 한도. 값싼 진단(구현 아님).
#include <cstdio>
#include <cuda_runtime.h>
int main(){
    int dev=0; cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
    int l2=0, persistMax=0, winMax=0;
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
    cudaDeviceGetAttribute(&persistMax, cudaDevAttrMaxPersistingL2CacheSize, dev);
    cudaDeviceGetAttribute(&winMax, cudaDevAttrMaxAccessPolicyWindowSize, dev);
    printf("GPU: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("L2 총 크기            : %.1f MB\n", l2/1048576.0);
    printf("persisting L2 최대    : %.1f MB %s\n", persistMax/1048576.0, persistMax>0?"(persistence 지원)":"(=0 → cudaAccessPolicyWindow 미지원)");
    printf("accessPolicy 창 최대  : %.1f MB\n", winMax/1048576.0);
    // persistence set 시도(지원 실확인)
    cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistMax>0?persistMax:1048576);
    printf("SetLimit(persistingL2): %s\n", cudaGetErrorString(e));
    return 0;
}
