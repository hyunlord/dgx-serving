#pragma once
#include <cstdint>
#include "qscale.cuh"
#include "../runtime/config.h"   // QBLK
#include "gemm_mmq.cuh"          // mmq_unpack4v, mmq_cpa4, MMQ_AP, MMQ_BPAD 재사용

// (BM/BN POC) gemm_mmq_kernel의 타일-형상 일반화 재구조판.
//   원본 = BM=128,BN=128,워프 2×4 하드코딩. 여기 = template<BM,BN,KT,WM,WN>로 형상 파라미터화.
//   ★재구조화 원칙: 계산(MMA 시퀀스·누적 순서)은 원본과 동일, 인덱스만 형상에서 파생 → <128,128,64,2,4>는 원본과 bit-identical.
//   파생: MS=(BM/WM)/16 (워프-행당 m16 타일 수), NS=(BN/WN)/8 (워프-열당 n8 타일 수), KS=KT/QBLK, NW=WM*WN 워프, NT=NW*32 스레드.
//   제약(컴파일 가능): BM%(WM*16)==0, BN%(WN*8)==0, KT%QBLK==0, smem 48KB, NT≤1024.
//   원본 대비 치환: 64→(MS*16)=BM/WM,  32→(NS*8)=BN/WN,  256→NT,  wm=warp>>2→warp/WN,  wn=warp&3→warp%WN.
template<int BM, int BN, int KT, int WM, int WN, typename ST>
static __global__ void __launch_bounds__(WM*WN*32, 2) gemm_mmq_tmpl_kernel(const int8_t*X,const float*xs,const uint8_t*Wq,
                                       const ST*sw,float*Y,int M,int N,int K){
    constexpr int MS=(BM/WM)/16, NS=(BN/WN)/8, KS=KT/QBLK, NW=WM*WN, NT=NW*32;
    constexpr int RM=MS*16, RN=NS*8;                  // 워프당 담당 행/열(=BM/WM, BN/WN)
    __shared__ int8_t  sA[2][BM][KT+MMQ_AP];
    __shared__ uint8_t sB[2][BN][KT/2+MMQ_BPAD];
    __shared__ float   sXs[BM][KS];
    __shared__ float   sSw[BN][KS];
    int tid=threadIdx.x, warp=tid>>5, lane=tid&31, g=lane>>2, t=lane&3, wm=warp/WN, wn=warp%WN;
    int m0=blockIdx.y*BM, n0=blockIdx.x*BN, nblk=K/QBLK, K2=K/2, niter=(nblk+KS-1)/KS;
    int arow=lane&15, ahalf=lane>>4;
    float acc[MS][NS][4];
    #pragma unroll
    for(int a=0;a<MS;a++)for(int b=0;b<NS;b++)for(int c=0;c<4;c++)acc[a][b][c]=0.f;

    auto load=[&](int buf,int it){
        int k0=it*KT;
        #pragma unroll
        for(int i=tid;i<BM*KT/4;i+=NT){int m=i/(KT/4),kw=(i%(KT/4))*4,gm=m0+m,gk=k0+kw;
            mmq_cpa4(&sA[buf][m][kw],&X[(size_t)gm*K+gk], gm<M && gk<K);}
        #pragma unroll
        for(int i=tid;i<BN*(KT/2)/4;i+=NT){int n=i/(KT/2/4),kw=(i%(KT/2/4))*4,gn=n0+n,gk=k0+kw*2;
            mmq_cpa4(&sB[buf][n][kw],&Wq[(size_t)gn*K2+k0/2+kw], gn<N && gk<K);}
        asm volatile("cp.async.commit_group;\n");
    };
    load(0,0);
    for(int it=0;it<niter;it++){
        asm volatile("cp.async.wait_group 0;\n");
        #pragma unroll
        for(int i=tid;i<BM*KS;i+=NT){int m=i/KS,b=i%KS,bk=it*KS+b,gm=m0+m;
            sXs[m][b]=(gm<M&&bk<nblk)?xs[(size_t)gm*nblk+bk]:0.f;}
        #pragma unroll
        for(int i=tid;i<BN*KS;i+=NT){int n=i/KS,b=i%KS,bk=it*KS+b,gn=n0+n;
            sSw[n][b]=(gn<N&&bk<nblk)?sval(sw[(size_t)gn*nblk+bk]):0.f;}
        __syncthreads();
        if(it+1<niter) load((it+1)&1,it+1);
        int cur=it&1;
        #pragma unroll
        for(int ks=0;ks<KS;ks++){ int blk=it*KS+ks; if(blk>=nblk) continue;
            int af[MS][4];
            #pragma unroll
            for(int ms=0;ms<MS;ms++){
                unsigned ad=__cvta_generic_to_shared(&sA[cur][wm*RM+ms*16+arow][ks*32+ahalf*16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];\n"
                  :"=r"(af[ms][0]),"=r"(af[ms][1]),"=r"(af[ms][2]),"=r"(af[ms][3]):"r"(ad));
            }
            int bf[NS][2];
            #pragma unroll
            for(int ns=0;ns<NS;ns++){int nb=wn*RN+ns*8, kb=ks*16;
                unsigned w0=*(const unsigned*)&sB[cur][nb+g][kb+(t>>1)*4], w1=*(const unsigned*)&sB[cur][nb+g][kb+8+(t>>1)*4];
                bf[ns][0]=mmq_unpack4v((t&1)?(w0>>16):w0); bf[ns][1]=mmq_unpack4v((t&1)?(w1>>16):w1);
            }
            float xr[MS][2], scl[NS][2];
            #pragma unroll
            for(int ms=0;ms<MS;ms++){ int r=wm*RM+ms*16+g; xr[ms][0]=sXs[r][ks]; xr[ms][1]=sXs[r+8][ks]; }
            #pragma unroll
            for(int ns=0;ns<NS;ns++){ int cc=wn*RN+ns*8+t*2; scl[ns][0]=sSw[cc][ks]; scl[ns][1]=sSw[cc+1][ks]; }
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
    for(int ms=0;ms<MS;ms++){ int mb=wm*RM+ms*16;
        #pragma unroll
        for(int ns=0;ns<NS;ns++){ int nb=wn*RN+ns*8;
            int rm0=m0+mb+g, rm1=m0+mb+g+8, cn0=n0+nb+t*2, cn1=n0+nb+t*2+1;
            if(rm0<M){ if(cn0<N)Y[(size_t)rm0*N+cn0]=acc[ms][ns][0]; if(cn1<N)Y[(size_t)rm0*N+cn1]=acc[ms][ns][1]; }
            if(rm1<M){ if(cn0<N)Y[(size_t)rm1*N+cn0]=acc[ms][ns][2]; if(cn1<N)Y[(size_t)rm1*N+cn1]=acc[ms][ns][3]; }
        }
    }
}

// 형상 파라미터화 런처. grid=(N타일, M타일), blockDim=NW*32.
template<int BM, int BN, int KT, int WM, int WN>
static inline void launch_gemm_mmq_tmpl(const int8_t*X,const float*xs,const uint8_t*Wq,
                                        const void*sw,float*Y,int M,int N,int K,int scale_dtype=0,cudaStream_t st=0){
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM); int nt=WM*WN*32;
    if(scale_dtype==1) gemm_mmq_tmpl_kernel<BM,BN,KT,WM,WN,__half><<<grid,nt,0,st>>>(X,xs,Wq,(const __half*)sw,Y,M,N,K);
    else               gemm_mmq_tmpl_kernel<BM,BN,KT,WM,WN,float ><<<grid,nt,0,st>>>(X,xs,Wq,(const float*)sw,Y,M,N,K);
}

// (in-engine A/B) prefill mmq 형상 토글: env MMQ256=1 → 256×64(격리 +14.5%), 기본 0 → 128×128(원본·무회귀).
//   출력 bitwise 동일(같은 K-reduction 순서) → 정확성 불변. 격리≠엔진 검증용 controlled A/B 스위치.
static inline void launch_gemm_mmq_w4a8_auto(const int8_t*X,const float*xs,const uint8_t*Wq,
                                             const void*sw,float*Y,int M,int N,int K,int scale_dtype=0,cudaStream_t st=0){
    static int sh = getenv("MMQ256") ? atoi(getenv("MMQ256")) : 0;
    if(sh) launch_gemm_mmq_tmpl<256,64,64,4,2>(X,xs,Wq,sw,Y,M,N,K,scale_dtype,st);   // tall 타일(W 트래픽 절반)
    else   launch_gemm_mmq_w4a8(X,xs,Wq,sw,Y,M,N,K,scale_dtype,st);                  // 128×128 원본
}
