#ifndef DGX_MOE_FUSED_CUH
#define DGX_MOE_FUSED_CUH
#include <cstdint>
#include "reduce.cuh"
#include "qscale.cuh"
#include "../runtime/config.h"   // QBLK
// device-fused MoE (모델 독립): naive moe_mlp_block(호스트 루프·D2H·토큰당 gemv 다중 launch)을 grouped gemv로 대체.
//   warp per (slot, output). slot = b*K+k. expert e = didx[slot](★device 상주 — D2H 제거). weight = base + e·stride 간접.
//   ★per-output dot은 gemv_q4와 동일(같은 dequant (q-8)·scale, 같은 lane-stride32 warp_reduce) → naive와 bit-identical.
//   launch: 토큰당 ~640 → 층당 4(gate_up/swiglu/down/reduce). k-order 고정 reduce = 결정적(atomicAdd 회피).

// (1) grouped gate_up: Y[slot, o] = dot(n2[b], W_gu[e][o]),  o∈[0,2·MI), IN=H.  b=slot/K, e=didx[slot].
template<typename ST>
static __global__ void moe_fused_gu_kernel(const float *n2, const uint8_t *gu_wq, const ST *gu_sc,
        const int *didx, float *Y, int H, int MI2, int K, long nslot, long gu_w) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    long tot = nslot * MI2;
    if (gw >= tot) return;
    long slot = gw / MI2; int o = (int)(gw % MI2);
    int b = (int)(slot / K); int e = didx[slot];
    const float   *x  = n2 + (size_t)b * H;
    const uint8_t *w  = gu_wq + (size_t)e * (gu_w/2) + (size_t)o * (H/2);
    const ST      *sca= gu_sc + (size_t)e * (gu_w/QBLK) + (size_t)o * (H/QBLK);
    float acc = 0.f;
    for (int i = lane; i < H; i += 32) {
        uint8_t bb = w[i >> 1];
        int q = (i & 1) ? (bb >> 4) : (bb & 0x0F);
        acc += x[i] * (((float)q - 8.0f) * sval(sca[i / QBLK]));
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) Y[gw] = acc;   // Y[slot*MI2 + o]
}

// (2) grouped swiglu: hid[slot·MI + j] = silu(gu[slot·2MI + j]) · gu[slot·2MI + MI + j].
static __global__ void moe_fused_swiglu_kernel(const float *gu, float *hid, int MI, long nslot) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nslot * MI) return;
    long slot = i / MI; int j = (int)(i % MI);
    const float *g = gu + slot * (2L*MI);
    float gg = g[j], uu = g[MI + j];
    hid[i] = (gg / (1.f + __expf(-gg))) * uu;
}

// (3) grouped down + router-weight scale: Sout[slot, o] = dw[slot] · dot(hid[slot], W_dn[e][o]), o∈[0,H), IN=MI.
template<typename ST>
static __global__ void moe_fused_down_kernel(const float *hid, const uint8_t *dn_wq, const ST *dn_sc,
        const int *didx, const float *dw, float *Sout, int MI, int H, int K, long nslot, long dn_w) {
    long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    long tot = nslot * H;
    if (gw >= tot) return;
    long slot = gw / H; int o = (int)(gw % H);
    int e = didx[slot];
    const float   *x  = hid + slot * MI;
    const uint8_t *w  = dn_wq + (size_t)e * (dn_w/2) + (size_t)o * (MI/2);
    const ST      *sca= dn_sc + (size_t)e * (dn_w/QBLK) + (size_t)o * (MI/QBLK);
    float acc = 0.f;
    for (int i = lane; i < MI; i += 32) {
        uint8_t bb = w[i >> 1];
        int q = (i & 1) ? (bb >> 4) : (bb & 0x0F);
        acc += x[i] * (((float)q - 8.0f) * sval(sca[i / QBLK]));
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) Sout[gw] = dw[slot] * acc;   // wk·eo (naive axpy_acc의 alpha·x와 동일 값)
}

// (4) k-order 고정 reduce: out[b·H + o] = Σ_{k=0}^{K-1} Sout[(b·K+k)·H + o]. 결정적(naive s,k 루프 순서와 동일).
static __global__ void moe_fused_reduce_kernel(const float *Sout, float *out, int H, int K, long B) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H) return;
    long b = i / H; int o = (int)(i % H);
    float acc = 0.f;
    for (int k = 0; k < K; k++) acc += Sout[((long)b*K + k) * H + o];   // k 오름차순 = naive 순서
    out[i] = acc;
}

// 런처 (scale_dtype 1=fp16, else fp32). nslot = B·K.
static inline void launch_moe_fused_gu(const float *n2, const uint8_t *gu_wq, const void *gu_sc,
        const int *didx, float *Y, int H, int MI2, int K, long nslot, long gu_w, int sdtype, cudaStream_t st=0) {
    int block=256; long warps=nslot*MI2; long grid=(warps*32+block-1)/block;
    if(sdtype==1) moe_fused_gu_kernel<__half><<<(unsigned)grid,block,0,st>>>(n2,gu_wq,(const __half*)gu_sc,didx,Y,H,MI2,K,nslot,gu_w);
    else          moe_fused_gu_kernel<float ><<<(unsigned)grid,block,0,st>>>(n2,gu_wq,(const float*)gu_sc,didx,Y,H,MI2,K,nslot,gu_w);
}
static inline void launch_moe_fused_swiglu(const float *gu, float *hid, int MI, long nslot, cudaStream_t st=0) {
    int block=256; long grid=(nslot*MI+block-1)/block;
    moe_fused_swiglu_kernel<<<(unsigned)grid,block,0,st>>>(gu,hid,MI,nslot);
}
static inline void launch_moe_fused_down(const float *hid, const uint8_t *dn_wq, const void *dn_sc,
        const int *didx, const float *dw, float *Sout, int MI, int H, int K, long nslot, long dn_w, int sdtype, cudaStream_t st=0) {
    int block=256; long warps=nslot*H; long grid=(warps*32+block-1)/block;
    if(sdtype==1) moe_fused_down_kernel<__half><<<(unsigned)grid,block,0,st>>>(hid,dn_wq,(const __half*)dn_sc,didx,dw,Sout,MI,H,K,nslot,dn_w);
    else          moe_fused_down_kernel<float ><<<(unsigned)grid,block,0,st>>>(hid,dn_wq,(const float*)dn_sc,didx,dw,Sout,MI,H,K,nslot,dn_w);
}
static inline void launch_moe_fused_reduce(const float *Sout, float *out, int H, int K, long B, cudaStream_t st=0) {
    int block=256; long grid=(B*H+block-1)/block;
    moe_fused_reduce_kernel<<<(unsigned)grid,block,0,st>>>(Sout,out,H,K,B);
}
#endif // DGX_MOE_FUSED_CUH
