#ifndef DGX_ATTENTION_FLASH_CUH
#define DGX_ATTENTION_FLASH_CUH
#include <float.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <type_traits>
#include <limits>

#define FD_SPLIT_KV 512   // KV split 크기(BC16 배수). B(MMA split) + FA.2c(flash_decode) 공용.

// ── attention 커널(정리: B 단일) ───────────────────────────────────────────────────────────
//   launch_flash_attn(엔진 진입, qwen35_decode/batch) → flash_attn2bs(B: MMA + KV split 2-패스).
//   prefill split=1(멀티워프 MMA, 빠름) + decode split>1(KV 분할, GPU 채움). split=f(attendlen)=Sq독립
//   → 단발==청크/verify==decode 자동(단일 커널). 중간단계(FA.1/FA.2/FA.2b non-split/flash_decode) 제거됨.
//   레퍼런스 forward는 attention.cuh launch_attn(독립 교차검증).

__device__ __forceinline__ unsigned fa_pkbf(const __nv_bfloat16 &x, const __nv_bfloat16 &y){
    return (unsigned)(*(const unsigned short*)&x) | ((unsigned)(*(const unsigned short*)&y) << 16);
}

// ── B: MMA(FA.2b.1) + KV split 2-패스 ─────────────────────────────────────────────────────────
//   검증된 두 조각 결합: FA.2b.1(멀티워프 MMA, prefill 빠름) + flash_decode(KV split, decode GPU 채움).
//   split=f(attendlen)(Sq·split 독립 → 완전 일관). prefill split=1(=FA.2b.1) / decode split>1(GPU 채움).
//   패스1: flash_attn2b를 split 구간[s·SPLIT, (s+1)·SPLIT)만 + O 대신 중간버퍼(부분 m/l/acc). 패스2: 결정적 결합.
//   B_SPLIT은 BC 배수(타일 정렬). FD_SPLIT_KV(512=32·BC16) 재사용.
template<int HD, int BR, int BC, int NW>
static __global__ void flash_attn2bs_p1_kernel(const float *Q, const __nv_bfloat16 *Kc, const __nv_bfloat16 *Vc,
                                               float *pm, float *pl, float *pacc, int Sq, int NQ, int NKV, int base, int NS,
                                               const unsigned *anc = nullptr) {  // 챕터9: anc[gq]=tree 조상 비트마스크(null=causal)
    int q0 = blockIdx.x * BR, h = blockIdx.y, sp = blockIdx.z;
    int GRP = NQ / NKV, kv = h / GRP;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    float scale = 1.0f / sqrtf((float)HD);
    const int HW = HD / NW, NT = HW / 8;
    __shared__ __nv_bfloat16 sQ[BR][HD], sK[BC][HD], sV[BC][HD], sP[BR][BC];
    __shared__ float sS[BR][BC], sm[BR], sl[BR], salpha[BR];
    float acc[NT][4];
    #pragma unroll
    for (int n=0;n<NT;n++) for(int c=0;c<4;c++) acc[n][c]=0.f;
    for (int i = tid; i < BR*HD; i += 32*NW) { int r=i/HD,c=i%HD,gq=q0+r;
        sQ[r][c] = (gq<Sq) ? __float2bfloat16(Q[((size_t)gq*NQ+h)*HD+c]) : (__nv_bfloat16)0; }
    if (tid < BR) { sm[tid]=-FLT_MAX; sl[tid]=0.f; }
    __syncthreads();
    int qend = (q0+BR < Sq) ? q0+BR : Sq;
    int maxlen = base + qend;
    int j_lo = sp * FD_SPLIT_KV, j_hi = j_lo + FD_SPLIT_KV; if (j_hi > maxlen) j_hi = maxlen;  // 이 split 구간
    int wcol = warp * HW;
    for (int j0 = j_lo; j0 < j_hi; j0 += BC) {       // split 구간만 순회
        for (int i = tid; i < BC*HD; i += 32*NW) { int r=i/HD,c=i%HD,gk=j0+r; bool v=gk<maxlen;
            sK[r][c]=v?Kc[((size_t)gk*NKV+kv)*HD+c]:(__nv_bfloat16)0;   // KV는 이미 bf16(저장 시 변환) — 적재 변환 제거
            sV[r][c]=v?Vc[((size_t)gk*NKV+kv)*HD+c]:(__nv_bfloat16)0; }
        __syncthreads();
        if (warp==0) {
            #pragma unroll
            for (int ns=0; ns<BC/8; ns++) { float c0=0,c1=0,c2=0,c3=0;
                for (int ks=0; ks<HD; ks+=16) {
                    unsigned a0=*(unsigned*)&sQ[g][ks+2*t], a1=*(unsigned*)&sQ[g+8][ks+2*t];
                    unsigned a2=*(unsigned*)&sQ[g][ks+2*t+8], a3=*(unsigned*)&sQ[g+8][ks+2*t+8];
                    unsigned b0=*(unsigned*)&sK[ns*8+g][ks+2*t], b1=*(unsigned*)&sK[ns*8+g][ks+2*t+8];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                      :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
                }
                sS[g][ns*8+2*t]=c0*scale; sS[g][ns*8+2*t+1]=c1*scale; sS[g+8][ns*8+2*t]=c2*scale; sS[g+8][ns*8+2*t+1]=c3*scale;
            }
        }
        __syncthreads();
        if (tid < BR) { int r=tid, gq=q0+r, qpos=base+gq;
            if (gq < Sq) {
                float tm=-FLT_MAX;
                for (int j=0;j<BC;j++){ int gk=j0+j;
                    // causal(anc=null) 또는 tree mask: prefix(gk<base) 항상, tree노드(gk>=base)는 anc[gq] 비트(조상-or-self)만.
                    bool att = anc ? ((gk<base) || ((gk-base)<=gq && ((anc[gq]>>(gk-base))&1u))) : (gk<=qpos);
                    float s=att?sS[r][j]:-FLT_MAX; sS[r][j]=s; tm=fmaxf(tm,s); }
                float mnew=fmaxf(sm[r],tm), alpha=expf(sm[r]-mnew), tl=0.f;
                for (int j=0;j<BC;j++){ float p=expf(sS[r][j]-mnew); sP[r][j]=__float2bfloat16(p); tl+=p; }
                sl[r]=sl[r]*alpha+tl; sm[r]=mnew; salpha[r]=alpha;
            } else { for (int j=0;j<BC;j++) sP[r][j]=(__nv_bfloat16)0; salpha[r]=1.f; }
        }
        __syncthreads();
        float ag=salpha[g], ag8=salpha[g+8];
        unsigned a0=*(unsigned*)&sP[g][2*t], a1=*(unsigned*)&sP[g+8][2*t];
        unsigned a2=*(unsigned*)&sP[g][2*t+8], a3=*(unsigned*)&sP[g+8][2*t+8];
        #pragma unroll
        for (int nt=0; nt<NT; nt++) { int col0=wcol+nt*8;
            acc[nt][0]*=ag; acc[nt][1]*=ag; acc[nt][2]*=ag8; acc[nt][3]*=ag8;
            unsigned b0=fa_pkbf(sV[2*t][col0+g], sV[2*t+1][col0+g]);
            unsigned b1=fa_pkbf(sV[2*t+8][col0+g], sV[2*t+9][col0+g]);
            float c0=acc[nt][0],c1=acc[nt][1],c2=acc[nt][2],c3=acc[nt][3];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
              :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            acc[nt][0]=c0; acc[nt][1]=c1; acc[nt][2]=c2; acc[nt][3]=c3;
        }
        __syncthreads();
    }
    // 중간버퍼 출력(부분, normalize 안 함): pm/pl/pacc[(gq·NQ+h)·NS + sp]. 빈 split도 write(m=-inf,l=0,acc=0).
    #pragma unroll
    for (int nt=0; nt<NT; nt++) { int col0=wcol+nt*8;
        int gq0=q0+g, gq1=q0+g+8;
        if (gq0<Sq){ size_t oi=((size_t)gq0*NQ+h)*NS+sp; pacc[oi*HD+col0+2*t]=acc[nt][0]; pacc[oi*HD+col0+2*t+1]=acc[nt][1]; }
        if (gq1<Sq){ size_t oi=((size_t)gq1*NQ+h)*NS+sp; pacc[oi*HD+col0+2*t]=acc[nt][2]; pacc[oi*HD+col0+2*t+1]=acc[nt][3]; }
    }
    if (tid < BR) { int gq=q0+tid; if (gq<Sq){ size_t oi=((size_t)gq*NQ+h)*NS+sp; pm[oi]=sm[tid]; pl[oi]=sl[tid]; } }
}
// The legacy bf16/default kernel above is kept intact for Qwen bitwise regression.
template<int HD, int BR, int BC>
struct FlashAttnShared {
    __nv_bfloat16 sQ[BR][HD], sK[BC][HD], sV[BC][HD], sP[BR][BC];
    float sS[BR][BC], sm[BR], sl[BR], salpha[BR];
};
template<int HD, int BR, int BC>
struct FlashAttnSharedAccess {
    static __device__ __forceinline__ FlashAttnShared<HD, BR, BC> &get() {
        __shared__ FlashAttnShared<HD, BR, BC> storage;
        return storage;
    }
};
template<int BR, int BC>
struct FlashAttnSharedAccess<512, BR, BC> {
    static __device__ __forceinline__ FlashAttnShared<512, BR, BC> &get() {
        extern __shared__ __align__(16) unsigned char shared_storage[];
        return *reinterpret_cast<FlashAttnShared<512, BR, BC> *>(shared_storage);
    }
};
__device__ __forceinline__ __nv_bfloat16 fa_kv_bf16(__nv_bfloat16 value) { return value; }
__device__ __forceinline__ __nv_bfloat16 fa_kv_bf16(float value) { return __float2bfloat16(value); }

template<int HD, int BR, int BC, int NW, typename KV>
static __global__ void flash_attn2bs_p1_custom_kernel(const float *Q, const KV *Kc, const KV *Vc,
                                               float *pm, float *pl, float *pacc, int Sq, int NQ, int NKV, int base, int NS,
                                               const unsigned *anc, float requested_scale, int window) {
    int q0 = blockIdx.x * BR, h = blockIdx.y, sp = blockIdx.z;
    int GRP = NQ / NKV, kv = h / GRP;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    float scale = requested_scale == 0.f ? 1.0f / sqrtf((float)HD) : requested_scale;
    const int HW = HD / NW, NT = HW / 8;
    auto &storage = FlashAttnSharedAccess<HD, BR, BC>::get();
    auto &sQ = storage.sQ; auto &sK = storage.sK; auto &sV = storage.sV; auto &sP = storage.sP;
    auto &sS = storage.sS; auto &sm = storage.sm; auto &sl = storage.sl; auto &salpha = storage.salpha;
    float acc[NT][4];
    #pragma unroll
    for (int n=0;n<NT;n++) for(int c=0;c<4;c++) acc[n][c]=0.f;
    for (int i = tid; i < BR*HD; i += 32*NW) { int r=i/HD,c=i%HD,gq=q0+r;
        sQ[r][c] = (gq<Sq) ? __float2bfloat16(Q[((size_t)gq*NQ+h)*HD+c]) : (__nv_bfloat16)0; }
    if (tid < BR) { sm[tid]=-FLT_MAX; sl[tid]=0.f; }
    __syncthreads();
    int qend = (q0+BR < Sq) ? q0+BR : Sq;
    int maxlen = base + qend;
    int j_lo = sp * FD_SPLIT_KV, j_hi = j_lo + FD_SPLIT_KV; if (j_hi > maxlen) j_hi = maxlen;  // 이 split 구간
    int wcol = warp * HW;
    for (int j0 = j_lo; j0 < j_hi; j0 += BC) {       // split 구간만 순회
        for (int i = tid; i < BC*HD; i += 32*NW) { int r=i/HD,c=i%HD,gk=j0+r; bool v=gk<maxlen;
            sK[r][c]=v?fa_kv_bf16(Kc[((size_t)gk*NKV+kv)*HD+c]):(__nv_bfloat16)0;
            sV[r][c]=v?fa_kv_bf16(Vc[((size_t)gk*NKV+kv)*HD+c]):(__nv_bfloat16)0; }
        __syncthreads();
        if (warp==0) {
            #pragma unroll
            for (int ns=0; ns<BC/8; ns++) { float c0=0,c1=0,c2=0,c3=0;
                for (int ks=0; ks<HD; ks+=16) {
                    unsigned a0=*(unsigned*)&sQ[g][ks+2*t], a1=*(unsigned*)&sQ[g+8][ks+2*t];
                    unsigned a2=*(unsigned*)&sQ[g][ks+2*t+8], a3=*(unsigned*)&sQ[g+8][ks+2*t+8];
                    unsigned b0=*(unsigned*)&sK[ns*8+g][ks+2*t], b1=*(unsigned*)&sK[ns*8+g][ks+2*t+8];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                      :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
                }
                sS[g][ns*8+2*t]=c0*scale; sS[g][ns*8+2*t+1]=c1*scale; sS[g+8][ns*8+2*t]=c2*scale; sS[g+8][ns*8+2*t+1]=c3*scale;
            }
        }
        __syncthreads();
        if (tid < BR) { int r=tid, gq=q0+r, qpos=base+gq;
            if (gq < Sq) {
                float tm=-FLT_MAX;
                for (int j=0;j<BC;j++){ int gk=j0+j;
                    // causal(anc=null) 또는 tree mask: prefix(gk<base) 항상, tree노드(gk>=base)는 anc[gq] 비트(조상-or-self)만.
                    bool att = anc ? ((gk<base) || ((gk-base)<=gq && (gk-base)<32 && ((anc[gq]>>(gk-base))&1u))) : (gk<=qpos);
                    att = att && gk < maxlen && (window <= 0 || gk >= qpos-window+1);
                    float s=att?sS[r][j]:-FLT_MAX; sS[r][j]=s; tm=fmaxf(tm,s); }
                if (tm == -FLT_MAX) {
                    for (int j=0;j<BC;j++) sP[r][j]=(__nv_bfloat16)0;
                    salpha[r]=1.f;
                } else {
                    float mnew=fmaxf(sm[r],tm), alpha=expf(sm[r]-mnew), tl=0.f;
                    for (int j=0;j<BC;j++) {
                        float p=sS[r][j] == -FLT_MAX ? 0.f : expf(sS[r][j]-mnew);
                        sP[r][j]=__float2bfloat16(p); tl+=p;
                    }
                    sl[r]=sl[r]*alpha+tl; sm[r]=mnew; salpha[r]=alpha;
                }
            } else { for (int j=0;j<BC;j++) sP[r][j]=(__nv_bfloat16)0; salpha[r]=1.f; }
        }
        __syncthreads();
        float ag=salpha[g], ag8=salpha[g+8];
        unsigned a0=*(unsigned*)&sP[g][2*t], a1=*(unsigned*)&sP[g+8][2*t];
        unsigned a2=*(unsigned*)&sP[g][2*t+8], a3=*(unsigned*)&sP[g+8][2*t+8];
        #pragma unroll
        for (int nt=0; nt<NT; nt++) { int col0=wcol+nt*8;
            acc[nt][0]*=ag; acc[nt][1]*=ag; acc[nt][2]*=ag8; acc[nt][3]*=ag8;
            unsigned b0=fa_pkbf(sV[2*t][col0+g], sV[2*t+1][col0+g]);
            unsigned b1=fa_pkbf(sV[2*t+8][col0+g], sV[2*t+9][col0+g]);
            float c0=acc[nt][0],c1=acc[nt][1],c2=acc[nt][2],c3=acc[nt][3];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
              :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            acc[nt][0]=c0; acc[nt][1]=c1; acc[nt][2]=c2; acc[nt][3]=c3;
        }
        __syncthreads();
    }
    // 중간버퍼 출력(부분, normalize 안 함): pm/pl/pacc[(gq·NQ+h)·NS + sp]. 빈 split도 write(m=-inf,l=0,acc=0).
    #pragma unroll
    for (int nt=0; nt<NT; nt++) { int col0=wcol+nt*8;
        int gq0=q0+g, gq1=q0+g+8;
        if (gq0<Sq){ size_t oi=((size_t)gq0*NQ+h)*NS+sp; pacc[oi*HD+col0+2*t]=acc[nt][0]; pacc[oi*HD+col0+2*t+1]=acc[nt][1]; }
        if (gq1<Sq){ size_t oi=((size_t)gq1*NQ+h)*NS+sp; pacc[oi*HD+col0+2*t]=acc[nt][2]; pacc[oi*HD+col0+2*t+1]=acc[nt][3]; }
    }
    if (tid < BR) { int gq=q0+tid; if (gq<Sq){ size_t oi=((size_t)gq*NQ+h)*NS+sp; pm[oi]=sm[tid]; pl[oi]=sl[tid]; } }
}
// 패스2: grid(Sq·NQ), block=HD. split 결합 → O[(gq·NQ+h)·HD+d].
template<int HD>
static __global__ void flash_attn2bs_p2_kernel(const float *pm, const float *pl, const float *pacc,
                                               float *O, int Sq, int NQ, int NS) {
    int qh = blockIdx.x, d = threadIdx.x;
    float mg=-FLT_MAX; for (int s=0;s<NS;s++) mg=fmaxf(mg, pm[qh*NS+s]);
    float lg=0.f, acc=0.f;
    for (int s=0;s<NS;s++){ float w=expf(pm[qh*NS+s]-mg); lg+=pl[qh*NS+s]*w; acc+=pacc[(size_t)(qh*NS+s)*HD+d]*w; }
    O[(size_t)qh*HD+d] = acc/lg;
}
static float *g_bs_m=nullptr, *g_bs_l=nullptr, *g_bs_acc=nullptr;
static size_t g_bs_cap=0, g_bs_acc_cap=0;
static inline void fa_cuda_check(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "[flash_attn] %s failed: %s\n", operation, cudaGetErrorString(error));
        abort();
    }
}
static inline void reserve_flash_attn_scratch(size_t need, size_t hd) {
    if (need > std::numeric_limits<size_t>::max() / sizeof(float) / hd) {
        fprintf(stderr, "[flash_attn] scratch size overflow\n"); abort();
    }
    if (g_bs_cap < need) {
        if(g_bs_m) fa_cuda_check(cudaFree(g_bs_m), "free m");
        if(g_bs_l) fa_cuda_check(cudaFree(g_bs_l), "free l");
        fa_cuda_check(cudaMalloc(&g_bs_m,need*sizeof(float)), "allocate m");
        fa_cuda_check(cudaMalloc(&g_bs_l,need*sizeof(float)), "allocate l");
        g_bs_cap=need;
    }
    const size_t acc_need=need*hd;
    if (g_bs_acc_cap < acc_need) {
        if(g_bs_acc) fa_cuda_check(cudaFree(g_bs_acc), "free acc");
        fa_cuda_check(cudaMalloc(&g_bs_acc,acc_need*sizeof(float)), "allocate acc");
        g_bs_acc_cap=acc_need;
    }
}
template<int BC, int NW, typename KV>
static inline size_t configure_flash_attn_hd512() {
    constexpr size_t bytes=sizeof(FlashAttnShared<512,16,BC>);
    static const bool configured=[]() {
        int device=0, limit=0;
        fa_cuda_check(cudaGetDevice(&device), "get device");
        fa_cuda_check(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device), "HD512 opt-in query");
        if (limit < (int)sizeof(FlashAttnShared<512,16,BC>)) {
            fprintf(stderr,"[flash_attn] HD512 needs %zu shared bytes, device opt-in limit is %d\n",
                    sizeof(FlashAttnShared<512,16,BC>),limit); abort();
        }
        fa_cuda_check(cudaFuncSetAttribute(flash_attn2bs_p1_custom_kernel<512,16,BC,NW,KV>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      (int)sizeof(FlashAttnShared<512,16,BC>)), "HD512 opt-in");
        return true;
    }();
    (void)configured;
    return bytes;
}
template<int HD, int BC, int NW, typename KV>
static inline void launch_flash_attn2bs(const float *Q, const KV *Kc, const KV *Vc, float *O,
                                        int Sq, int NQ, int NKV, int base, cudaStream_t st=0,
                                        const unsigned *anc=nullptr, float scale=0.f, int window=0) {
    static_assert(std::is_same<KV,float>::value || std::is_same<KV,__nv_bfloat16>::value,"unsupported KV type");
    int maxlen=base+Sq, NS=(maxlen+FD_SPLIT_KV-1)/FD_SPLIT_KV;
    size_t need=(size_t)Sq*NQ*NS;
    reserve_flash_attn_scratch(need,HD);
    dim3 g1((Sq+15)/16,NQ,NS);
    if constexpr (HD != 512 && std::is_same<KV,__nv_bfloat16>::value) {
        if (scale == 0.f && window == 0) {
            flash_attn2bs_p1_kernel<HD,16,BC,NW><<<g1,32*NW,0,st>>>(Q,Kc,Vc,g_bs_m,g_bs_l,g_bs_acc,Sq,NQ,NKV,base,NS,anc);
        } else {
            flash_attn2bs_p1_custom_kernel<HD,16,BC,NW,KV><<<g1,32*NW,0,st>>>(Q,Kc,Vc,g_bs_m,g_bs_l,g_bs_acc,Sq,NQ,NKV,base,NS,anc,scale,window);
        }
    } else {
        size_t shared_bytes=0;
        if constexpr (HD == 512) shared_bytes=configure_flash_attn_hd512<BC,NW,KV>();
        flash_attn2bs_p1_custom_kernel<HD,16,BC,NW,KV><<<g1,32*NW,shared_bytes,st>>>(Q,Kc,Vc,g_bs_m,g_bs_l,g_bs_acc,Sq,NQ,NKV,base,NS,anc,scale,window);
    }
    flash_attn2bs_p2_kernel<HD><<<Sq*NQ,HD,0,st>>>(g_bs_m,g_bs_l,g_bs_acc,O,Sq,NQ,NS);
}

template<typename KV>
static inline void launch_flash_attn(const float *Q, const KV *Kc, const KV *Vc, float *O,
                                     int Sq, int NQ, int NKV, int HD, int base, cudaStream_t st=0,
                                     const unsigned *anc=nullptr, float scale=0.f, int window=0) {
    if      (HD == 256) launch_flash_attn2bs<256,16,8>(Q,Kc,Vc,O,Sq,NQ,NKV,base,st,anc,scale,window);
    else if (HD == 128) launch_flash_attn2bs<128,16,8>(Q,Kc,Vc,O,Sq,NQ,NKV,base,st,anc,scale,window);
    else if (HD == 512) launch_flash_attn2bs<512,16,8>(Q,Kc,Vc,O,Sq,NQ,NKV,base,st,anc,scale,window);
    else {
        fprintf(stderr,"[flash_attn] unsupported head_dim=%d (128/256/512 only)\n",HD); abort();
    }
}

#endif // DGX_ATTENTION_FLASH_CUH
