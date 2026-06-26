// 빌드 스모크 테스트: -arch=sm_120 컴파일 + GB10 실행 + 공용 커널 1개(add) 동작 확인.
#include <cstdio>
#include <cstdlib>
#include "../kernels/elementwise.cuh"
#include "../runtime/config.h"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

int main() {
    const int n = 1024;
    float *a, *b, *out;
    CK(cudaMallocManaged(&a, n*sizeof(float)));
    CK(cudaMallocManaged(&b, n*sizeof(float)));
    CK(cudaMallocManaged(&out, n*sizeof(float)));
    for (int i=0;i<n;i++){ a[i]=(float)i; b[i]=2.0f*i; }

    int block=256, grid=(n+block-1)/block;
    add_kernel<<<grid,block>>>(out,a,b,n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    double maxerr=0;
    for (int i=0;i<n;i++){ double e=fabs((double)out[i]-(3.0*i)); if(e>maxerr)maxerr=e; }

    // config.h 가 잘 포함되는지(QBLK) 확인
    printf("[test_build] QBLK=%d, add max abs err=%.3e\n", QBLK, maxerr);
    cudaFree(a); cudaFree(b); cudaFree(out);
    if (maxerr < 1e-5) { printf("[test_build] PASS\n"); return 0; }
    printf("[test_build] FAIL\n"); return 1;
}
