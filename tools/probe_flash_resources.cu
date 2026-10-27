#include <cuda_runtime.h>
#include "kernels/attention_flash.cuh"
#ifndef PROBE_HD
#define PROBE_HD 256
#endif
#ifndef PROBE_FLOAT_KV
#define PROBE_FLOAT_KV 0
#endif
using ProbeKV = std::conditional_t<PROBE_FLOAT_KV, float, __nv_bfloat16>;
// Compile-only resource probe: instantiates the unchanged 16x16 tile, eight warps.
void instantiate(const float *q, const ProbeKV *k, const ProbeKV *v, float *out) {
    launch_flash_attn2bs<PROBE_HD,16,8>(q,k,v,out,16,16,8,0);
}
int main() {
    cudaDeviceProp prop{};
    cudaFuncAttributes attr{};
    int blocks=0;
    cudaError_t error=cudaGetDeviceProperties(&prop,0);
    size_t dynamic_bytes = 0;
#if PROBE_HD == 512
    if(error==cudaSuccess) dynamic_bytes=configure_flash_attn_hd512<16,8,ProbeKV>();
    if(error==cudaSuccess) error=cudaFuncGetAttributes(&attr,flash_attn2bs_p1_custom_kernel<512,16,16,8,ProbeKV>);
    if(error==cudaSuccess) error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,flash_attn2bs_p1_custom_kernel<512,16,16,8,ProbeKV>,256,dynamic_bytes);
#elif PROBE_FLOAT_KV
    if(error==cudaSuccess) error=cudaFuncGetAttributes(&attr,flash_attn2bs_p1_custom_kernel<PROBE_HD,16,16,8,ProbeKV>);
    if(error==cudaSuccess) error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,flash_attn2bs_p1_custom_kernel<PROBE_HD,16,16,8,ProbeKV>,256,0);
#else
    if(error==cudaSuccess) error=cudaFuncGetAttributes(&attr,flash_attn2bs_p1_kernel<PROBE_HD,16,16,8>);
    if(error==cudaSuccess) error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,flash_attn2bs_p1_kernel<PROBE_HD,16,16,8>,256,0);
#endif
    if(error!=cudaSuccess) { fprintf(stderr,"RESOURCE_ERROR %s\n",cudaGetErrorString(error)); return 1; }
    printf("HD=%d KV=%s regs=%d static_shared=%zu dynamic_shared=%zu local=%zu active_blocks=%d max_threads_sm=%d occupancy=%.17g shared_limit=%zu shared_optin=%zu\n",PROBE_HD,PROBE_FLOAT_KV ? "float" : "bf16",attr.numRegs,attr.sharedSizeBytes,dynamic_bytes,attr.localSizeBytes,blocks,prop.maxThreadsPerMultiProcessor,double(blocks*256)/prop.maxThreadsPerMultiProcessor,prop.sharedMemPerBlock,prop.sharedMemPerBlockOptin);
}
