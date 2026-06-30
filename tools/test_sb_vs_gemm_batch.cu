#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/gemm_batch.cuh"
#include "../kernels/gemv_w4a8.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}}while(0)
static unsigned sd=7u; static int ir(int lo,int hi){sd=sd*1664525u+1013904223u;return lo+(int)((sd>>9)%(unsigned)(hi-lo+1));}
int main(){
    const int IN=5120, OUT=2048, nb=IN/QBLK;
    std::vector<uint8_t> hW((size_t)OUT*IN/2); for(auto&x:hW)x=(uint8_t)ir(0,255);
    std::vector<float> hSw((size_t)OUT*nb); for(auto&x:hSw)x=0.02f;
    uint8_t*dW; float*dSw,*dYg,*dYs; CK(cudaMalloc(&dW,hW.size())); CK(cudaMalloc(&dSw,hSw.size()*4));
    CK(cudaMalloc(&dYg,(size_t)8*OUT*4)); CK(cudaMalloc(&dYs,(size_t)8*OUT*4));
    CK(cudaMemcpy(dW,hW.data(),hW.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dSw,hSw.data(),hSw.size()*4,cudaMemcpyHostToDevice));
    printf("gemm_batch_w4a8 vs gemv_w4a8_sb bit-exact (IN=%d OUT=%d):\n",IN,OUT);
    for(int B : {2,3,4,5,6,7,8}){
        std::vector<int8_t> hX((size_t)B*IN); for(auto&x:hX)x=(int8_t)ir(-50,50);
        std::vector<float> hXs((size_t)B*nb); for(auto&x:hXs)x=0.03f;
        int8_t*dX; float*dXs; CK(cudaMalloc(&dX,hX.size())); CK(cudaMalloc(&dXs,hXs.size()*4));
        CK(cudaMemcpy(dX,hX.data(),hX.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dXs,hXs.data(),hXs.size()*4,cudaMemcpyHostToDevice));
        launch_gemm_batch_w4a8(dX,dXs,dW,dSw,dYg,IN,OUT,B,0);
        launch_gemv_w4a8_sb(dX,dXs,dW,dSw,dYs,IN,OUT,B,0);
        CK(cudaDeviceSynchronize());
        std::vector<float> g((size_t)B*OUT),s((size_t)B*OUT);
        CK(cudaMemcpy(g.data(),dYg,(size_t)B*OUT*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s.data(),dYs,(size_t)B*OUT*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<(size_t)B*OUT;i++) e=fmax(e,fabs((double)g[i]-s[i]));
        printf("  B=%d: max|gemm_batch - SB| = %.3e %s\n",B,e,e==0.0?"(bit-exact ✓)":"(DIFF)");
        cudaFree(dX); cudaFree(dXs);
    }
    return 0;
}
