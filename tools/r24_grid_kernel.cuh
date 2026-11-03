#pragma once
#include "kernels/gemm_mmq.cuh"

// Diagnostic copy from main 4f65227bb534d70475db8de469816f18b2c17100.
// Only the symbol and blockIdx axes differ; launch dimensions are transposed.
template<int KT, typename ST>
static __global__ void __launch_bounds__(256,2) r24_grid_swap_kernel(const int8_t*X,const float*xs,const uint8_t*Wq,
                                       const ST*sw,float*Y,int M,int N,int K){
    constexpr int MS=MMQ_BM/32, NS=MMQ_BN/32, KS=KT/QBLK;       // 4,4, KT/32
    __shared__ int8_t  sA[2][MMQ_BM][KT+MMQ_AP];     // [buf][m][k] int8
    __shared__ uint8_t sB[2][MMQ_BN][KT/2+MMQ_BPAD]; // [buf][n][k/2] int4 packed
    __shared__ float   sXs[MMQ_BM][KS];              // 행 스케일(QBLK별) — coalesced 1회 적재
    __shared__ float   sSw[MMQ_BN][KS];              // 열 스케일
    int tid=threadIdx.x, warp=tid>>5, lane=tid&31, g=lane>>2, t=lane&3, wm=warp>>2, wn=warp&3;
    int m0=blockIdx.x*MMQ_BM, n0=blockIdx.y*MMQ_BN, nblk=K/QBLK, K2=K/2, niter=(nblk+KS-1)/KS;
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
