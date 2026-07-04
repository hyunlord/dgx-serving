#pragma once
#include <cstdint>
#include "qscale.cuh"
#include "../runtime/config.h"
#include "gemm_mmq.cuh"          // mmq_unpack4v, mmq_cpa4, MMQ_AP, MMQ_BPAD

// (파이프 POC) gemm_mmq_tmpl의 multi-stage(깊은 파이프라이닝) 일반화.
//   현 tmpl = depth-1 더블버퍼(STAGES=2). 여기 = template<...,STAGES>로 파이프 깊이 파라미터화.
//   ★핵심: STAGES 버퍼 링, 프롤로그서 STAGES-1 prefetch, 루프 매 스텝 wait_group(STAGES-2)로 가장 오래된 것만 대기
//     → STAGES-1개를 항상 in-flight 유지(더 깊은 latency 은폐). 매 스텝 정확히 1 commit(tail은 empty commit=count 유지).
//   ★정확성: 계산(MMA·누적) 불변, 로드 순서만 깊어짐. STAGES=2는 tmpl과 bit-identical(prologue load0·wait0·buf it%2).
//   ★"안 온 데이터 계산" 차단: wait_group(STAGES-2)가 tile it 도착 보장 후 compute. tail=empty commit로 count 불변.
template<int BM, int BN, int KT, int WM, int WN, int STAGES, typename ST>
static __global__ void __launch_bounds__(WM*WN*32, 2) gemm_mmq_ms_kernel(const int8_t*X,const float*xs,const uint8_t*Wq,
                                       const ST*sw,float*Y,int M,int N,int K){
    constexpr int MS=(BM/WM)/16, NS=(BN/WN)/8, KS=KT/QBLK, NW=WM*WN, NT=NW*32, RM=MS*16, RN=NS*8;
    __shared__ int8_t  sA[STAGES][BM][KT+MMQ_AP];
    __shared__ uint8_t sB[STAGES][BN][KT/2+MMQ_BPAD];
    __shared__ float   sXs[BM][KS];
    __shared__ float   sSw[BN][KS];
    int tid=threadIdx.x, warp=tid>>5, lane=tid&31, g=lane>>2, t=lane&3, wm=warp/WN, wn=warp%WN;
    int m0=blockIdx.y*BM, n0=blockIdx.x*BN, nblk=K/QBLK, K2=K/2, niter=(nblk+KS-1)/KS;
    int arow=lane&15, ahalf=lane>>4;
    float acc[MS][NS][4];
    #pragma unroll
    for(int a=0;a<MS;a++)for(int b=0;b<NS;b++)for(int c=0;c<4;c++)acc[a][b][c]=0.f;

    // cp.async 발행만(commit은 호출자가 명시적으로) — group count 정확 제어
    auto issue=[&](int buf,int it){
        if(it>=niter) return;
        int k0=it*KT;
        #pragma unroll
        for(int i=tid;i<BM*KT/4;i+=NT){int m=i/(KT/4),kw=(i%(KT/4))*4,gm=m0+m,gk=k0+kw;
            mmq_cpa4(&sA[buf][m][kw],&X[(size_t)gm*K+gk], gm<M && gk<K);}
        #pragma unroll
        for(int i=tid;i<BN*(KT/2)/4;i+=NT){int n=i/(KT/2/4),kw=(i%(KT/2/4))*4,gn=n0+n,gk=k0+kw*2;
            mmq_cpa4(&sB[buf][n][kw],&Wq[(size_t)gn*K2+k0/2+kw], gn<N && gk<K);}
    };
    // 프롤로그: STAGES-1 그룹 발행(tile 0..STAGES-2, 범위 밖은 empty commit로 count 유지)
    #pragma unroll
    for(int s=0;s<STAGES-1;s++){ issue(s%STAGES,s); asm volatile("cp.async.commit_group;\n"); }

    for(int it=0;it<niter;it++){
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES-2));   // ≤STAGES-2 남을 때까지 = tile it(최오래) 도착
        // 다음(it+STAGES-1) prefetch — 매 스텝 정확히 1 commit(tail도 empty commit로 count 불변)
        issue((it+STAGES-1)%STAGES, it+STAGES-1);
        asm volatile("cp.async.commit_group;\n");
        int cur=it%STAGES;
        #pragma unroll
        for(int i=tid;i<BM*KS;i+=NT){int m=i/KS,b=i%KS,bk=it*KS+b,gm=m0+m;
            sXs[m][b]=(gm<M&&bk<nblk)?xs[(size_t)gm*nblk+bk]:0.f;}
        #pragma unroll
        for(int i=tid;i<BN*KS;i+=NT){int n=i/KS,b=i%KS,bk=it*KS+b,gn=n0+n;
            sSw[n][b]=(gn<N&&bk<nblk)?sval(sw[(size_t)gn*nblk+bk]):0.f;}
        __syncthreads();
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
template<int BM, int BN, int KT, int WM, int WN, int STAGES>
static inline void launch_gemm_mmq_ms(const int8_t*X,const float*xs,const uint8_t*Wq,
                                      const void*sw,float*Y,int M,int N,int K,int scale_dtype=0,cudaStream_t st=0){
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM); int nt=WM*WN*32;
    if(scale_dtype==1) gemm_mmq_ms_kernel<BM,BN,KT,WM,WN,STAGES,__half><<<grid,nt,0,st>>>(X,xs,Wq,(const __half*)sw,Y,M,N,K);
    else               gemm_mmq_ms_kernel<BM,BN,KT,WM,WN,STAGES,float ><<<grid,nt,0,st>>>(X,xs,Wq,(const float*)sw,Y,M,N,K);
}
