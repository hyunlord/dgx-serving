#ifndef DGX_GEMV_W4A8_R10_CUH
#define DGX_GEMV_W4A8_R10_CUH
#include "gemv_w4a8.cuh"

// D1 only: pinned llama.cpp GB10 Q4_0 B1 placement is one row / eight warps.
// Keep our eight-element chunk, adjacent nibbles, scales and DP4A arithmetic.
// Cross-warp reduction changes summation order; this is not a bit-identical claim.
template<typename ST>
static __global__ void gemv_w4a8_r10_d1(const int8_t* xq,const float* xs,
                                      const uint8_t* wq,const ST* sw,float* y,int in) {
    const int lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane,row=blockIdx.x;
    const uint8_t* w=wq+(size_t)row*(in/2);
    const ST* scales=sw+(size_t)row*(in/QBLK);
    float acc=0.f;
    for (int base=tid*8;base<in;base+=256*8) {
        const int blk=base/QBLK;
        const float sc=xs[blk]*sval(scales[blk]);
        const unsigned packed=*reinterpret_cast<const unsigned*>(w+(base>>1));
        const int x0=*reinterpret_cast<const int*>(xq+base);
        const int x1=*reinterpret_cast<const int*>(xq+base+4);
        int dot=__dp4a(x0,w4a8_pk(packed&255,(packed>>8)&255),0);
        dot=__dp4a(x1,w4a8_pk((packed>>16)&255,(packed>>24)&255),dot);
        acc+=sc*(float)dot;
    }
    __shared__ float partial[7][32];
    if (warp>0) partial[warp-1][lane]=acc;
    __syncthreads();
    if (warp!=0) return;
    for (int w=0;w<7;w++) acc+=partial[w][lane];
    acc=warp_reduce_sum(acc);
    if (lane==0) y[row]=acc;
}

static inline void launch_gemv_w4a8_r10_d1(const int8_t* xq,const float* xs,
                                          const uint8_t* wq,const void* sw,float* y,
                                          int in,int out,int dtype,cudaStream_t stream=0) {
    if (dtype==1) gemv_w4a8_r10_d1<__half><<<out,dim3(32,8),0,stream>>>(xq,xs,wq,(const __half*)sw,y,in);
    else gemv_w4a8_r10_d1<float><<<out,dim3(32,8),0,stream>>>(xq,xs,wq,(const float*)sw,y,in);
}
#endif
