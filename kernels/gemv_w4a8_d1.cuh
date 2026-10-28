#ifndef DGX_GEMV_W4A8_D1_CUH
#define DGX_GEMV_W4A8_D1_CUH
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "gemv_w4a8.cuh"

static inline bool gemv_d1_enabled() {
    static const bool enabled=[](){ const char* value=getenv("GEMV_D1"); return value && strcmp(value,"1")==0; }();
    return enabled;
}

static inline bool gemv_d1_supported(int IN, int OUT, int S, int scale_dtype) {
    return IN>0 && IN%QBLK==0 && OUT>0 && S>=1 && S<=16 &&
           (scale_dtype==0 || scale_dtype==1);
}

// R10 D1 placement: one output row, eight warps. Each column uses exactly the
// same recurrence; packed weights and weight scales are shared across columns.
// Sum warp partials lane-wise BEFORE the shuffle reduction, including for S=1.
template<typename ST, int SMAX>
static __device__ __forceinline__ void gemv_d1_row_dot(
        const int8_t* xq, const float* xs, const uint8_t* wq, const ST* sw,
        float* y, int IN, int OUT, int S, int row, float* partial) {
    const int lane=threadIdx.x, warp=threadIdx.y, tid=warp*32+lane;
    const uint8_t* w=wq+(size_t)row*(IN/2);
    const ST* scales=sw+(size_t)row*(IN/QBLK);
    float acc[SMAX]={};
    for(int base=tid*8;base<IN;base+=256*8) {
        const int blk=base/QBLK;
        const float weight_scale=sval(scales[blk]);
        const unsigned packed=*reinterpret_cast<const unsigned*>(w+(base>>1));
        const int w0=w4a8_pk(packed&255,(packed>>8)&255);
        const int w1=w4a8_pk((packed>>16)&255,(packed>>24)&255);
        #pragma unroll
        for(int s=0;s<SMAX;s++) if(s<S) {
            const int8_t* x=xq+(size_t)s*IN+base;
            const int x0=*reinterpret_cast<const int*>(x);
            const int x1=*reinterpret_cast<const int*>(x+4);
            int dot=__dp4a(x0,w0,0);
            dot=__dp4a(x1,w1,dot);
            const float scale=__fmul_rn(xs[(size_t)s*(IN/QBLK)+blk],weight_scale);
            acc[s]=__fmaf_rn(scale,(float)dot,acc[s]);
        }
    }
    if(warp>0) {
        #pragma unroll
        for(int s=0;s<SMAX;s++) if(s<S)
            partial[(s*7+warp-1)*32+lane]=acc[s];
    }
    __syncthreads();
    if(warp!=0) return;
    #pragma unroll
    for(int s=0;s<SMAX;s++) if(s<S) {
        #pragma unroll
        for(int w=0;w<7;w++) acc[s]=__fadd_rn(acc[s],partial[(s*7+w)*32+lane]);
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)
            acc[s]=__fadd_rn(acc[s],__shfl_down_sync(0xffffffff,acc[s],offset));
        if(lane==0) y[(size_t)s*OUT+row]=acc[s];
    }
}

template<typename ST, int SMAX>
static __global__ void gemv_w4a8_d1_kernel(
        const int8_t* xq, const float* xs, const uint8_t* wq, const ST* sw,
        float* y, int IN, int OUT, int S) {
    __shared__ float partial[SMAX*7*32];
    gemv_d1_row_dot<ST,SMAX>(xq,xs,wq,sw,y,IN,OUT,S,blockIdx.x,partial);
}

template<typename ST>
static inline void launch_gemv_d1_typed(
        const int8_t* xq, const float* xs, const uint8_t* wq, const ST* sw,
        float* y, int IN, int OUT, int S, cudaStream_t stream) {
    if(S==1) gemv_w4a8_d1_kernel<ST,1><<<OUT,dim3(32,8),0,stream>>>(xq,xs,wq,sw,y,IN,OUT,S);
    else if(S<=8) gemv_w4a8_d1_kernel<ST,8><<<OUT,dim3(32,8),0,stream>>>(xq,xs,wq,sw,y,IN,OUT,S);
    else gemv_w4a8_d1_kernel<ST,16><<<OUT,dim3(32,8),0,stream>>>(xq,xs,wq,sw,y,IN,OUT,S);
}

static inline void launch_gemv_w4a8_d1(
        const int8_t* xq, const float* xs, const uint8_t* wq, const void* sw,
        float* y, int IN, int OUT, int S, int scale_dtype=0, cudaStream_t stream=0) {
    if(!gemv_d1_supported(IN,OUT,S,scale_dtype)) {
        fprintf(stderr,"D1 unsupported shape IN=%d OUT=%d S=%d scale_dtype=%d\n",IN,OUT,S,scale_dtype);
        exit(1);
    }
    if(scale_dtype==1) launch_gemv_d1_typed(xq,xs,wq,(const __half*)sw,y,IN,OUT,S,stream);
    else launch_gemv_d1_typed(xq,xs,wq,(const float*)sw,y,IN,OUT,S,stream);
}
#endif
