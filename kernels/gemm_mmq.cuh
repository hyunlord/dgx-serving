#pragma once
#include <cstdint>
#include "qscale.cuh"
#include "../runtime/config.h"   // QBLK
#include "gemv_w4a8.cuh"         // S=1 fallback (gemm_mma와 동일 인터페이스 정신)

// 챕터7 STEP 0i/1: full-MMQ W4A8 GEMM. 128×128 타일 + 큰 K-tile + ldmatrix-A → L1/TEX 84→40%(=llama 39%).
//   Y[M,N] = (X int8 · (Wq int4-8) int8) ×(xs·sw). MMA m16n8k32.s8.s8.s32. A=ldmatrix.x4, B=scalar 4B-unpack.
//   8 warp(2×4): wm∈{0,1} 각 64행(ms 0..3), wn∈{0,3} 각 32열(ns 0..3). KT=K-tile(KS=KT/QBLK substep), cp.async 더블버퍼.
//   STEP 1: KT 템플릿(64/128…) — sync↓ vs shared↑ 트레이드 측정용. KT은 16배수(ldmatrix 16B 정렬).
#define MMQ_BM 128
#define MMQ_BN 128
#define MMQ_KT 64           // 엔진 기본 K-tile(2 QBLK). KTILE↑ = sync↓ (단 shared↑)
#define MMQ_AP 0            // sA K-stride = KT = 16배수 → ldmatrix 16B 정렬
#define MMQ_BPAD 4

// 정렬 4-byte read의 하위 16bit(4 nibble) → (q-8) int8 4개 (bank conflict 회피).
__device__ __forceinline__ int mmq_unpack4v(unsigned h){
    int q0=(int)(h&0xF)-8,q1=(int)((h>>4)&0xF)-8,q2=(int)((h>>8)&0xF)-8,q3=(int)((h>>12)&0xF)-8;
    return (q0&0xFF)|((q1&0xFF)<<8)|((q2&0xFF)<<16)|((q3&0xFF)<<24);
}
__device__ __forceinline__ void mmq_cpa4(void*s,const void*g,bool pred){
    unsigned a=__cvta_generic_to_shared(s); int ss=pred?4:0;
    asm volatile("cp.async.ca.shared.global [%0],[%1],4,%2;\n"::"r"(a),"l"(g),"r"(ss));
}

// __launch_bounds__(256,2): reg≤128 강제 → 2 CTA/SM 목표(llama occ 29%=2CTA, 우리 reg137=1CTA 17% 매칭 시도).
//   de-risk: occupancy↑가 lever인지(KT/sync 아님 — Warp Cycles 5.64 낮음). spill 역효과면 순효과 ncu로.
template<int KT, typename ST>
static __global__ void __launch_bounds__(256,2) gemm_mmq_kernel(const int8_t*X,const float*xs,const uint8_t*Wq,
                                       const ST*sw,float*Y,int M,int N,int K){
    constexpr int MS=MMQ_BM/32, NS=MMQ_BN/32, KS=KT/QBLK;       // 4,4, KT/32
    __shared__ int8_t  sA[2][MMQ_BM][KT+MMQ_AP];     // [buf][m][k] int8
    __shared__ uint8_t sB[2][MMQ_BN][KT/2+MMQ_BPAD]; // [buf][n][k/2] int4 packed
    __shared__ float   sXs[MMQ_BM][KS];              // 행 스케일(QBLK별) — coalesced 1회 적재
    __shared__ float   sSw[MMQ_BN][KS];              // 열 스케일
    int tid=threadIdx.x, warp=tid>>5, lane=tid&31, g=lane>>2, t=lane&3, wm=warp>>2, wn=warp&3;
    int m0=blockIdx.y*MMQ_BM, n0=blockIdx.x*MMQ_BN, nblk=K/QBLK, K2=K/2, niter=(nblk+KS-1)/KS;
    int arow=lane&15, ahalf=lane>>4;
    float acc[MS][NS][4];
    #pragma unroll
    for(int a=0;a<MS;a++)for(int b=0;b<NS;b++)for(int c=0;c<4;c++)acc[a][b][c]=0.f;

    auto load=[&](int buf,int it){
        int k0=it*KT;
        #pragma unroll
        for(int i=tid;i<MMQ_BM*KT/4;i+=256){int m=i/(KT/4),kw=(i%(KT/4))*4,gm=m0+m,gk=k0+kw;
            mmq_cpa4(&sA[buf][m][kw],&X[(size_t)gm*K+gk], gm<M && gk<K);}
        #pragma unroll
        for(int i=tid;i<MMQ_BN*(KT/2)/4;i+=256){int n=i/(KT/2/4),kw=(i%(KT/2/4))*4,gn=n0+n,gk=k0+kw*2;
            mmq_cpa4(&sB[buf][n][kw],&Wq[(size_t)gn*K2+k0/2+kw], gn<N && gk<K);}
        asm volatile("cp.async.commit_group;\n");
    };
    load(0,0);
    for(int it=0;it<niter;it++){
        asm volatile("cp.async.wait_group 0;\n");
        // 스케일 staging: 현 K-tile 행/열 스케일을 shared로 coalesced 1회(uncoalesced per-thread global read 제거). 기존 sync에 흡수.
        #pragma unroll
        for(int i=tid;i<MMQ_BM*KS;i+=256){int m=i/KS,b=i%KS,bk=it*KS+b,gm=m0+m;
            sXs[m][b]=(gm<M&&bk<nblk)?xs[(size_t)gm*nblk+bk]:0.f;}
        #pragma unroll
        for(int i=tid;i<MMQ_BN*KS;i+=256){int n=i/KS,b=i%KS,bk=it*KS+b,gn=n0+n;
            sSw[n][b]=(gn<N&&bk<nblk)?sval(sw[(size_t)gn*nblk+bk]):0.f;}
        __syncthreads();
        if(it+1<niter) load((it+1)&1,it+1);
        int cur=it&1;
        #pragma unroll
        for(int ks=0;ks<KS;ks++){ int blk=it*KS+ks; if(blk>=nblk) continue;  // K 부분 타일 가드
            int af[MS][4];
            #pragma unroll
            for(int ms=0;ms<MS;ms++){
                unsigned ad=__cvta_generic_to_shared(&sA[cur][wm*64+ms*16+arow][ks*32+ahalf*16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];\n"
                  :"=r"(af[ms][0]),"=r"(af[ms][1]),"=r"(af[ms][2]),"=r"(af[ms][3]):"r"(ad));
            }
            int bf[NS][2];
            #pragma unroll
            for(int ns=0;ns<NS;ns++){int nb=wn*32+ns*8, kb=ks*16;
                unsigned w0=*(const unsigned*)&sB[cur][nb+g][kb+(t>>1)*4], w1=*(const unsigned*)&sB[cur][nb+g][kb+8+(t>>1)*4];
                bf[ns][0]=mmq_unpack4v((t&1)?(w0>>16):w0); bf[ns][1]=mmq_unpack4v((t&1)?(w1>>16):w1);
            }
            // 스케일 hoist: QBLK(blk)별 행/열 스케일을 MMA loop 밖에서 1회씩(shared read, staging됨). per-MMA global read는 reg 폭발(255 spill).
            float xr[MS][2], scl[NS][2];
            #pragma unroll
            for(int ms=0;ms<MS;ms++){ int r=wm*64+ms*16+g; xr[ms][0]=sXs[r][ks]; xr[ms][1]=sXs[r+8][ks]; }
            #pragma unroll
            for(int ns=0;ns<NS;ns++){ int cc=wn*32+ns*8+t*2; scl[ns][0]=sSw[cc][ks]; scl[ns][1]=sSw[cc+1][ks]; }
            #pragma unroll
            for(int ms=0;ms<MS;ms++){
                #pragma unroll
                for(int ns=0;ns<NS;ns++){
                    int c0=0,c1=0,c2=0,c3=0;
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                      :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(af[ms][0]),"r"(af[ms][1]),"r"(af[ms][2]),"r"(af[ms][3]),"r"(bf[ns][0]),"r"(bf[ns][1]));
                    acc[ms][ns][0]+=c0*xr[ms][0]*scl[ns][0]; acc[ms][ns][1]+=c1*xr[ms][0]*scl[ns][1];
                    acc[ms][ns][2]+=c2*xr[ms][1]*scl[ns][0]; acc[ms][ns][3]+=c3*xr[ms][1]*scl[ns][1];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int ms=0;ms<MS;ms++){ int mb=wm*64+ms*16;
        #pragma unroll
        for(int ns=0;ns<NS;ns++){ int nb=wn*32+ns*8;
            int rm0=m0+mb+g, rm1=m0+mb+g+8, cn0=n0+nb+t*2, cn1=n0+nb+t*2+1;
            if(rm0<M){ if(cn0<N)Y[(size_t)rm0*N+cn0]=acc[ms][ns][0]; if(cn1<N)Y[(size_t)rm0*N+cn1]=acc[ms][ns][1]; }
            if(rm1<M){ if(cn0<N)Y[(size_t)rm1*N+cn0]=acc[ms][ns][2]; if(cn1<N)Y[(size_t)rm1*N+cn1]=acc[ms][ns][3]; }
        }
    }
}

// 챕터9 stream-K(split-K) BM=16 배치 GEMM: BM=16 few-blocks(grid~40<SM48) 해결 — K를 SPLITK로 쪼개 blockIdx.z로 블록↑.
//   각 split이 K-range 부분합(fp dequant됨)→workspace[split] 기록→reduce가 합산(deterministic, atomic 안 씀). MS=1, 8 warp col.
//   격리 correctness: argmax==sb(fp 순서차 1-2e-5). in-engine 측정이 진실(격리 오도 3전례). cp.async 더블버퍼/MMA/scale 재사용.
// launch_bounds(256,2)=80 regs, 3블록/SM, 이론 occ 50%. occupancy de-risk(측정): minblk 4(64reg)/6(40reg,spill0)로
// 이론 occ 66/100% 올려도 achieved occ 28% 불변(wave quant — BM=16 few-blocks라 채울 블록 없음) → throughput 불변(68.7→67.9).
// = occupancy는 batch matmul lever 아님. 진짜 제약 = BM=16 grid(few M-tiles). minblk 2가 측정 최적.
template<int KT, typename ST>
static __global__ void __launch_bounds__(256,2) gemm_bm16_splitk_kernel(const int8_t*X,const float*xs,const uint8_t*Wq,
                                       const ST*sw,float*out,int M,int N,int K,int SPLITK,long MN){
    constexpr int BM=16, BN=128, NS=2, KS=KT/QBLK;
    __shared__ int8_t  sA[2][BM][KT+MMQ_AP];
    __shared__ uint8_t sB[2][BN][KT/2+MMQ_BPAD];
    __shared__ float   sXs[BM][KS];
    __shared__ float   sSw[BN][KS];
    int tid=threadIdx.x, warp=tid>>5, lane=tid&31, g=lane>>2, t=lane&3, wn=warp;
    int m0=blockIdx.y*BM, n0=blockIdx.x*BN, nblk=K/QBLK, K2=K/2, niter=(nblk+KS-1)/KS;
    int sp=blockIdx.z, itper=(niter+SPLITK-1)/SPLITK, it0=sp*itper, it1=(sp+1)*itper<niter?(sp+1)*itper:niter;
    int arow=lane&15, ahalf=lane>>4;
    float acc[NS][4];
    #pragma unroll
    for(int b=0;b<NS;b++)for(int c=0;c<4;c++)acc[b][c]=0.f;
    float*Yo=out+(size_t)sp*MN;
    if(it0>=it1){  // 빈 split → 0 기록(reduce가 합산)
        for(int ns=0;ns<NS;ns++){int nb=wn*16+ns*8;int rm0=m0+g,rm1=m0+g+8,cn0=n0+nb+t*2,cn1=cn0+1;
            if(rm0<M){if(cn0<N)Yo[(size_t)rm0*N+cn0]=0;if(cn1<N)Yo[(size_t)rm0*N+cn1]=0;}
            if(rm1<M){if(cn0<N)Yo[(size_t)rm1*N+cn0]=0;if(cn1<N)Yo[(size_t)rm1*N+cn1]=0;}}
        return;
    }
    auto load=[&](int buf,int it){
        int k0=it*KT;
        #pragma unroll
        for(int i=tid;i<BM*KT/4;i+=256){int m=i/(KT/4),kw=(i%(KT/4))*4,gm=m0+m,gk=k0+kw;
            mmq_cpa4(&sA[buf][m][kw],&X[(size_t)gm*K+gk], gm<M && gk<K);}
        #pragma unroll
        for(int i=tid;i<BN*(KT/2)/4;i+=256){int n=i/(KT/2/4),kw=(i%(KT/2/4))*4,gn=n0+n,gk=k0+kw*2;
            mmq_cpa4(&sB[buf][n][kw],&Wq[(size_t)gn*K2+k0/2+kw], gn<N && gk<K);}
        asm volatile("cp.async.commit_group;\n");
    };
    load(it0&1,it0);
    for(int it=it0;it<it1;it++){
        asm volatile("cp.async.wait_group 0;\n");
        #pragma unroll
        for(int i=tid;i<BM*KS;i+=256){int m=i/KS,b=i%KS,bk=it*KS+b,gm=m0+m;
            sXs[m][b]=(gm<M&&bk<nblk)?xs[(size_t)gm*nblk+bk]:0.f;}
        for(int i=tid;i<BN*KS;i+=256){int n=i/KS,b=i%KS,bk=it*KS+b,gn=n0+n;
            sSw[n][b]=(gn<N&&bk<nblk)?sval(sw[(size_t)gn*nblk+bk]):0.f;}
        __syncthreads();
        if(it+1<it1) load((it+1)&1,it+1);
        int cur=it&1;
        #pragma unroll
        for(int ks=0;ks<KS;ks++){ int blk=it*KS+ks; if(blk>=nblk) continue;
            int af[4];
            unsigned ad=__cvta_generic_to_shared(&sA[cur][arow][ks*32+ahalf*16]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];\n":"=r"(af[0]),"=r"(af[1]),"=r"(af[2]),"=r"(af[3]):"r"(ad));
            int bf[NS][2];
            #pragma unroll
            for(int ns=0;ns<NS;ns++){int nb=wn*16+ns*8, kb=ks*16;
                unsigned w0=*(const unsigned*)&sB[cur][nb+g][kb+(t>>1)*4], w1=*(const unsigned*)&sB[cur][nb+g][kb+8+(t>>1)*4];
                bf[ns][0]=mmq_unpack4v((t&1)?(w0>>16):w0); bf[ns][1]=mmq_unpack4v((t&1)?(w1>>16):w1);}
            float xr[2], scl[NS][2]; xr[0]=sXs[g][ks]; xr[1]=sXs[g+8][ks];
            #pragma unroll
            for(int ns=0;ns<NS;ns++){ int cc=wn*16+ns*8+t*2; scl[ns][0]=sSw[cc][ks]; scl[ns][1]=sSw[cc+1][ks]; }
            #pragma unroll
            for(int ns=0;ns<NS;ns++){
                int c0=0,c1=0,c2=0,c3=0;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                  :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(af[0]),"r"(af[1]),"r"(af[2]),"r"(af[3]),"r"(bf[ns][0]),"r"(bf[ns][1]));
                acc[ns][0]+=c0*xr[0]*scl[ns][0]; acc[ns][1]+=c1*xr[0]*scl[ns][1];
                acc[ns][2]+=c2*xr[1]*scl[ns][0]; acc[ns][3]+=c3*xr[1]*scl[ns][1];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int ns=0;ns<NS;ns++){ int nb=wn*16+ns*8; int rm0=m0+g,rm1=m0+g+8,cn0=n0+nb+t*2,cn1=cn0+1;
        if(rm0<M){ if(cn0<N)Yo[(size_t)rm0*N+cn0]=acc[ns][0]; if(cn1<N)Yo[(size_t)rm0*N+cn1]=acc[ns][1]; }
        if(rm1<M){ if(cn0<N)Yo[(size_t)rm1*N+cn0]=acc[ns][2]; if(cn1<N)Yo[(size_t)rm1*N+cn1]=acc[ns][3]; }
    }
}
__global__ static void reduce_splitk_kernel(const float*Wksp,float*Y,long MN,int SPLITK){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=MN)return;
    float s=0; for(int sp=0;sp<SPLITK;sp++)s+=Wksp[(size_t)sp*MN+i]; Y[i]=s;
}
// BM=16 split-K 런처(B=9~16). SPLITK auto(SM~48 채움, blocks=ceil(N/128)*SPLITK≥~96) 또는 env SPLITK. workspace 영속 scratch.
static float* g_skwksp=nullptr; static size_t g_skwcap=0;
static inline void launch_gemm_bm16_splitk(const int8_t*X,const float*xs,const uint8_t*Wq,
                                           const void*sw,float*Y,int M,int N,int K,int scale_dtype=0,cudaStream_t st=0){
    int ntile=(N+MMQ_BN-1)/MMQ_BN;
    static int ENV_SK = getenv("SPLITK") ? atoi(getenv("SPLITK")) : 0;
    int SK = ENV_SK>0 ? ENV_SK : (96/ntile>1 ? (96/ntile<8?96/ntile:8) : 1);   // ~96 blocks 목표(2 blk/SM), clamp 1..8
    long MN=(long)M*N; dim3 grid(ntile,(M+15)/16,SK);
    if(SK==1){ if(scale_dtype==1) gemm_bm16_splitk_kernel<MMQ_KT,__half><<<grid,256,0,st>>>(X,xs,Wq,(const __half*)sw,Y,M,N,K,1,MN);
               else               gemm_bm16_splitk_kernel<MMQ_KT,float ><<<grid,256,0,st>>>(X,xs,Wq,(const float*)sw,Y,M,N,K,1,MN); return; }
    size_t need=(size_t)SK*MN; if(g_skwcap<need){ if(g_skwksp)cudaFree(g_skwksp); cudaMalloc(&g_skwksp,need*4); g_skwcap=need; }
    if(scale_dtype==1) gemm_bm16_splitk_kernel<MMQ_KT,__half><<<grid,256,0,st>>>(X,xs,Wq,(const __half*)sw,g_skwksp,M,N,K,SK,MN);
    else               gemm_bm16_splitk_kernel<MMQ_KT,float ><<<grid,256,0,st>>>(X,xs,Wq,(const float*)sw,g_skwksp,M,N,K,SK,MN);
    int th=256; reduce_splitk_kernel<<<(unsigned)((MN+th-1)/th),th,0,st>>>(g_skwksp,Y,MN,SK);
}

// KT 스윕 진입점(de-risk/측정). KT은 16배수(ldmatrix 정렬) + shared 예산(KT=128 더블버퍼 ~53KB OK, 256은 ~106KB 불가).
template<int KT>
static inline void launch_gemm_mmq_w4a8_kt(const int8_t*X,const float*xs,const uint8_t*Wq,
                                           const void*sw,float*Y,int M,int N,int K,int scale_dtype=0,cudaStream_t st=0){
    dim3 grid((N+MMQ_BN-1)/MMQ_BN, (M+MMQ_BM-1)/MMQ_BM);
    if(scale_dtype==1) gemm_mmq_kernel<KT,__half><<<grid,256,0,st>>>(X,xs,Wq,(const __half*)sw,Y,M,N,K);
    else               gemm_mmq_kernel<KT,float ><<<grid,256,0,st>>>(X,xs,Wq,(const float*)sw,Y,M,N,K);
}
// 엔진 경로: KT=64 기본(c7f8ada 검증값). 측정 우월 확인+합의 후에만 KT 교체.
static inline void launch_gemm_mmq_w4a8(const int8_t*X,const float*xs,const uint8_t*Wq,
                                        const void*sw,float*Y,int M,int N,int K,int scale_dtype=0,cudaStream_t st=0){
    launch_gemm_mmq_w4a8_kt<MMQ_KT>(X,xs,Wq,sw,Y,M,N,K,scale_dtype,st);
}
