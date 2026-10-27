#include <cuda_runtime.h>
#include "kernels/attention_flash.cuh"
#ifndef PROBE_HD
#define PROBE_HD 256
#endif
// Compile-only resource probe: instantiates the unchanged 16x16 tile, eight warps.
void instantiate(const float *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v, float *out) {
    launch_flash_attn2bs<PROBE_HD,16,8>(q,k,v,out,16,16,8,0);
}
int main() {
    cudaDeviceProp prop{};
    cudaFuncAttributes attr{};
    int blocks=0;
    cudaError_t error=cudaGetDeviceProperties(&prop,0);
    if(error==cudaSuccess) error=cudaFuncGetAttributes(&attr,flash_attn2bs_p1_kernel<PROBE_HD,16,16,8>);
    if(error==cudaSuccess) error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,flash_attn2bs_p1_kernel<PROBE_HD,16,16,8>,256,0);
    if(error!=cudaSuccess) { fprintf(stderr,"RESOURCE_ERROR %s\n",cudaGetErrorString(error)); return 1; }
    printf("HD=%d regs=%d static_shared=%zu local=%zu active_blocks=%d max_threads_sm=%d occupancy=%.17g shared_limit=%zu shared_optin=%zu\n",PROBE_HD,attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,blocks,prop.maxThreadsPerMultiProcessor,double(blocks*256)/prop.maxThreadsPerMultiProcessor,prop.sharedMemPerBlock,prop.sharedMemPerBlockOptin);
}
