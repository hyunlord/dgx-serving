#ifndef DGX_GEMV_W4A8_R10_D3_CUH
#define DGX_GEMV_W4A8_R10_D3_CUH
// Isolated two-chain candidate; original GEMV remains unchanged.
#include "gemv_w4a8.cuh"

// D3: exactly two independent chains for one output, split by original loop parity.
// Original placement: eight output warps per 256-thread CTA, one output per warp.
// Original chunk: eight adjacent-nibble weights, two int32 activation loads, two DP4A.
// Unlike gemv_w4a8_mlp_kernel<U>, this reassociates each output's float summation.
template<typename ST>
static __global__ void gemv_w4a8_r10_d3_2chain(const int8_t* xq,const float* xs,
                                             const uint8_t* wq,const ST* sw,
                                             float* y,int in,int out) {
    const long row=((long)blockIdx.x*blockDim.x+threadIdx.x)>>5;
    const int lane=threadIdx.x&31;
    if (row>=out) return;
    const uint8_t* w=wq+(size_t)row*(in/2);
    const ST* scales=sw+(size_t)row*(in/QBLK);
    float acc[2]={0.f,0.f};
    constexpr int stride=32*8;
    for (int first=lane*8;first<in;first+=2*stride) {
        #pragma unroll 2
        for (int chain=0;chain<2;chain++) {
            const int base=first+chain*stride;
            if (base<in) {
                const int blk=base/QBLK;
                const float sc=xs[blk]*sval(scales[blk]);
                const unsigned packed=*reinterpret_cast<const unsigned*>(w+(base>>1));
                const int x0=*reinterpret_cast<const int*>(xq+base);
                const int x1=*reinterpret_cast<const int*>(xq+base+4);
                int dot=__dp4a(x0,w4a8_pk(packed&255,(packed>>8)&255),0);
                dot=__dp4a(x1,w4a8_pk((packed>>16)&255,(packed>>24)&255),dot);
                acc[chain]+=sc*(float)dot;
            }
        }
    }
    float result=warp_reduce_sum(acc[0]+acc[1]);
    if (lane==0) y[row]=result;
}

// Sq1 only. Caller must validate QBLK=32, in%32=0 and dtype in {0,1}.
// No environment parsing or model dispatch belongs in this isolated launcher.
static inline void launch_gemv_w4a8_r10_d3_2chain(const int8_t* xq,const float* xs,
                                                 const uint8_t* wq,const void* sw,
                                                 float* y,int in,int out,int dtype,
                                                 cudaStream_t stream=0) {
    constexpr int block=256;
    const unsigned grid=(unsigned)(((long)out*32+block-1)/block);
    if (dtype==1)
        gemv_w4a8_r10_d3_2chain<__half><<<grid,block,0,stream>>>(xq,xs,wq,(const __half*)sw,y,in,out);
    else
        gemv_w4a8_r10_d3_2chain<float><<<grid,block,0,stream>>>(xq,xs,wq,(const float*)sw,y,in,out);
}

#endif
