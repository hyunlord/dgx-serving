#include <cstdio>
#include <cuda_runtime.h>
__global__ void copyk(const float4* in, float4* out, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]; }
__global__ void readk(const float4* in, float* out, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; float4 v; if(i<n){v=in[i]; float s=v.x+v.y+v.z+v.w; if(s==-1.f) out[0]=s;} }
int main(){
    size_t bytes=4ull<<30;  // 4GB (>L2, DRAM cold)
    size_t n=bytes/sizeof(float4);
    float4 *a,*b; cudaMalloc(&a,bytes); cudaMalloc(&b,bytes); cudaMemset(a,1,bytes);
    float* sink; cudaMalloc(&sink,4);
    int blk=256; size_t grid=(n+blk-1)/blk;
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for(int w=0;w<3;w++) copyk<<<grid,blk>>>(a,b,n);
    cudaDeviceSynchronize(); cudaEventRecord(e0);
    int IT=20; for(int i=0;i<IT;i++) copyk<<<grid,blk>>>(a,b,n);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms,e0,e1); ms/=IT;
    double gbps=(2.0*bytes)/(ms/1e3)/1e9;  // read+write
    printf("copy(R+W) %.2f GB 2x: %.3f ms -> %.1f GB/s (%.0f%% of 273)\n", bytes/1e9, ms, gbps, 100*gbps/273);
    for(int w=0;w<3;w++) readk<<<grid,blk>>>(a,sink,n);
    cudaDeviceSynchronize(); cudaEventRecord(e0);
    for(int i=0;i<IT;i++) readk<<<grid,blk>>>(a,sink,n);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms,e0,e1); ms/=IT;
    double rgbps=(double)bytes/(ms/1e3)/1e9;
    printf("read-only %.2f GB: %.3f ms -> %.1f GB/s (%.0f%% of 273)\n", bytes/1e9, ms, rgbps, 100*rgbps/273);
    return 0;
}
