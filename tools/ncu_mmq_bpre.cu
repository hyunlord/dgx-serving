// STEP 1 matmul de-risk: B int4→shared int8 1회 pre-unpack (per-MMA unpack ALU amortize).
//   현 gemm_mmq는 B fragment를 MMA 인너루프마다 unpack(mmq_unpack4v) — B는 wn에만 의존(wm 무관)이라
//   wm=0,1 워프가 같은 B를 2× 중복 unpack(ALU 낭비). probe = 타일당 sB_i8로 1회 pre-unpack → MMA는 int8 직접.
//   측정: ① baseline==probe 오차0(레이아웃 정합 검증) ② duration↓? (ncu로 ALU18%↓·FMA·duration 별도)
//   인자: argv[1] 0=both+timing(기본) / 1=baseline only / 2=probe only (clean ncu용).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/gemm_mmq.cuh"   // baseline gemm_mmq_kernel<KT> + mmq_unpack4v/mmq_cpa4 + QBLK + MMQ_*

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static unsigned g_s=12345u; static int irand(int lo,int hi){g_s=g_s*1664525u+1013904223u;return lo+(int)((g_s>>9)%(unsigned)(hi-lo+1));}
static float frand(){g_s=g_s*1664525u+1013904223u;return ((g_s>>8)&0xFFFFFF)/8388608.0f*0.05f+0.02f;}

// ── probe: B pre-unpack 버전 (baseline gemm_mmq_kernel 복제 + sB_i8 단일버퍼) ──
template<int KT>
static __global__ void __launch_bounds__(256,2) gemm_mmq_bpre_kernel(const int8_t*X,const float*xs,const uint8_t*Wq,
                                       const float*sw,float*Y,int M,int N,int K){
    constexpr int MS=MMQ_BM/32, NS=MMQ_BN/32, KS=KT/QBLK;
    __shared__ int8_t  sA[2][MMQ_BM][KT+MMQ_AP];
    __shared__ uint8_t sB[2][MMQ_BN][KT/2+MMQ_BPAD];      // int4 packed (cp.async 대상)
    __shared__ int8_t  sBi[MMQ_BN][KT+MMQ_BPAD];          // ★ pre-unpacked int8 (단일버퍼)
    __shared__ float   sXs[MMQ_BM][KS];
    __shared__ float   sSw[MMQ_BN][KS];
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
        #pragma unroll
        for(int i=tid;i<MMQ_BM*KS;i+=256){int m=i/KS,b=i%KS,bk=it*KS+b,gm=m0+m;
            sXs[m][b]=(gm<M&&bk<nblk)?xs[(size_t)gm*nblk+bk]:0.f;}
        #pragma unroll
        for(int i=tid;i<MMQ_BN*KS;i+=256){int n=i/KS,b=i%KS,bk=it*KS+b,gn=n0+n;
            sSw[n][b]=(gn<N&&bk<nblk)?sw[(size_t)gn*nblk+bk]:0.f;}
        __syncthreads();
        if(it+1<niter) load((it+1)&1,it+1);
        int cur=it&1;
        // ★ B pre-unpack: sB[cur](int4 packed) → sBi(int8). 타일당 1회(MMQ_BN*KT/4 unpack = wm-중복 제거).
        #pragma unroll
        for(int i=tid;i<MMQ_BN*KT/4;i+=256){int n=i/(KT/4),kq=(i%(KT/4))*4;
            unsigned h=*(const unsigned short*)&sB[cur][n][kq>>1];   // 4 nibble
            *(int*)&sBi[n][kq]=mmq_unpack4v(h);                      // 4 int8 (q-8)
        }
        __syncthreads();   // sBi 준비 (MMA 읽기 전)
        #pragma unroll
        for(int ks=0;ks<KS;ks++){ int blk=it*KS+ks; if(blk>=nblk) continue;
            int af[MS][4];
            #pragma unroll
            for(int ms=0;ms<MS;ms++){
                unsigned ad=__cvta_generic_to_shared(&sA[cur][wm*64+ms*16+arow][ks*32+ahalf*16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];\n"
                  :"=r"(af[ms][0]),"=r"(af[ms][1]),"=r"(af[ms][2]),"=r"(af[ms][3]):"r"(ad));
            }
            int bf[NS][2];
            #pragma unroll
            for(int ns=0;ns<NS;ns++){int nb=wn*32+ns*8, k0=ks*32+(t>>1)*8+(t&1)*4;  // baseline k_start 정합
                bf[ns][0]=*(const int*)&sBi[nb+g][k0];        // ★ int8 직접 (unpack 없음)
                bf[ns][1]=*(const int*)&sBi[nb+g][k0+16];
            }
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

int main(int argc,char**argv){
    int mode=argc>1?atoi(argv[1]):0;
    const int M=argc>2?atoi(argv[2]):512, N=argc>3?atoi(argv[3]):17408, K=argc>4?atoi(argv[4]):5120;
    // 기본 M512·N17408·K5120 = 실 FFN prefill(grid 544=full GPU, llama grid 544 일치). argv로 sweep.
    const int nblk=K/QBLK, K2=K/2;
    printf("[mmq bpre de-risk] M=%d N=%d K=%d KT=%d (probe=B int4→shared int8 pre-unpack)\n",M,N,K,MMQ_KT);
    std::vector<int8_t> hX((size_t)M*K); for(auto&x:hX)x=(int8_t)irand(-40,40);
    std::vector<uint8_t> hW((size_t)N*K2); for(auto&x:hW)x=(uint8_t)irand(0,255);
    std::vector<float> hXs((size_t)M*nblk); for(auto&x:hXs)x=frand();
    std::vector<float> hSw((size_t)N*nblk); for(auto&x:hSw)x=frand();
    int8_t*dX; uint8_t*dW; float*dXs,*dSw,*dYb,*dYp;
    CK(cudaMalloc(&dX,hX.size()));CK(cudaMalloc(&dW,hW.size()));
    CK(cudaMalloc(&dXs,hXs.size()*4));CK(cudaMalloc(&dSw,hSw.size()*4));
    CK(cudaMalloc(&dYb,(size_t)M*N*4));CK(cudaMalloc(&dYp,(size_t)M*N*4));
    CK(cudaMemcpy(dX,hX.data(),hX.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW,hW.data(),hW.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dXs,hXs.data(),hXs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSw,hSw.data(),hSw.size()*4,cudaMemcpyHostToDevice));
    dim3 grid((N+MMQ_BN-1)/MMQ_BN,(M+MMQ_BM-1)/MMQ_BM);

    // 정확성: baseline vs probe 오차0 (레이아웃 정합)
    gemm_mmq_kernel<MMQ_KT><<<grid,256>>>(dX,dXs,dW,dSw,dYb,M,N,K);
    gemm_mmq_bpre_kernel<MMQ_KT><<<grid,256>>>(dX,dXs,dW,dSw,dYp,M,N,K);
    CK(cudaDeviceSynchronize());
    std::vector<float> Yb((size_t)M*N),Yp((size_t)M*N);
    CK(cudaMemcpy(Yb.data(),dYb,Yb.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Yp.data(),dYp,Yp.size()*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(size_t i=0;i<Yb.size();i++){e=fmax(e,fabs((double)Yp[i]-Yb[i]));rms+=Yb[i]*Yb[i];} rms=sqrt(rms/Yb.size());
    printf("[mmq bpre] baseline==probe: max abs err=%.3e (rel=%.2e) %s\n",e,e/rms, e/rms<1e-5?"OK(오차0=레이아웃 정합)":"FAIL");

    // 타이밍 (mode 0=both)
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); int R=50; float tb=0,tp=0;
    if(mode!=2){ for(int w=0;w<5;w++) gemm_mmq_kernel<MMQ_KT><<<grid,256>>>(dX,dXs,dW,dSw,dYb,M,N,K); CK(cudaDeviceSynchronize());
        cudaEventRecord(a); for(int r=0;r<R;r++) gemm_mmq_kernel<MMQ_KT><<<grid,256>>>(dX,dXs,dW,dSw,dYb,M,N,K); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tb,a,b); }
    if(mode!=1){ for(int w=0;w<5;w++) gemm_mmq_bpre_kernel<MMQ_KT><<<grid,256>>>(dX,dXs,dW,dSw,dYp,M,N,K); CK(cudaDeviceSynchronize());
        cudaEventRecord(a); for(int r=0;r<R;r++) gemm_mmq_bpre_kernel<MMQ_KT><<<grid,256>>>(dX,dXs,dW,dSw,dYp,M,N,K); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tp,a,b); }
    if(mode==0) printf("[mmq bpre] baseline %.3f ms | probe(pre-unpack) %.3f ms | speedup %.3fx %s\n",
        tb/R, tp/R, tb/tp, tb/tp>1.02?"(↑유효)":"(평탄/역효과)");
    else if(mode==1) printf("[mmq bpre] baseline only: %.3f ms\n", tb/R);
    else printf("[mmq bpre] probe only: %.3f ms\n", tp/R);
    return 0;
}
