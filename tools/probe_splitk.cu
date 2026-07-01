// stream-K(split-K) BM=16 배치 GEMM: BM=16 few-blocks(grid ~40<SM48) 해결 — K를 SPLITK로 쪼개 blockIdx.z로 블록↑.
//   각 split이 K-range 부분합(fp, dequant된) → workspace[split] 기록 → reduce 커널이 합산(deterministic, atomic 안 씀).
//   correctness: argmax vs sb(격리, warm 무관). 타이밍은 in-engine이 진실(격리 3번 오도) — 여기선 correctness+SPLITK별 격리 참고만.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/gemm_mmq.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// split-K BM=16: grid(N/128, 1, SPLITK). blockIdx.z=sp가 KT-tile 범위 [it0,it1) 담당. out[sp*MN..]에 부분합 기록.
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
    if(it0>=it1){ /* 빈 split: 0 기록 */
        float*Yo=out+(size_t)sp*MN;
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
    float*Yo=out+(size_t)sp*MN;
    #pragma unroll
    for(int ns=0;ns<NS;ns++){ int nb=wn*16+ns*8; int rm0=m0+g,rm1=m0+g+8,cn0=n0+nb+t*2,cn1=cn0+1;
        if(rm0<M){ if(cn0<N)Yo[(size_t)rm0*N+cn0]=acc[ns][0]; if(cn1<N)Yo[(size_t)rm0*N+cn1]=acc[ns][1]; }
        if(rm1<M){ if(cn0<N)Yo[(size_t)rm1*N+cn0]=acc[ns][2]; if(cn1<N)Yo[(size_t)rm1*N+cn1]=acc[ns][3]; }
    }
}
__global__ void reduce_splitk_kernel(const float*Wksp,float*Y,long MN,int SPLITK){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=MN)return;
    float s=0; for(int sp=0;sp<SPLITK;sp++)s+=Wksp[(size_t)sp*MN+i]; Y[i]=s;
}
static float* g_wksp=nullptr; static size_t g_wcap=0;
template<typename ST>
static void launch_splitk(const int8_t*X,const float*xs,const uint8_t*Wq,const ST*sw,float*Y,int M,int N,int K,int SPLITK){
    long MN=(long)M*N; dim3 grid((N+127)/128,(M+15)/16,SPLITK);
    if(SPLITK==1){ gemm_bm16_splitk_kernel<64,ST><<<grid,256>>>(X,xs,Wq,sw,Y,M,N,K,1,MN); return; }
    size_t need=(size_t)SPLITK*MN; if(g_wcap<need){ if(g_wksp)cudaFree(g_wksp); CK(cudaMalloc(&g_wksp,need*4)); g_wcap=need; }
    gemm_bm16_splitk_kernel<64,ST><<<grid,256>>>(X,xs,Wq,sw,g_wksp,M,N,K,SPLITK,MN);
    int th=256; reduce_splitk_kernel<<<(unsigned)((MN+th-1)/th),th>>>(g_wksp,Y,MN,SPLITK);
}

int main(int argc,char**argv){
    int IN=argc>1?atoi(argv[1]):5120, OUT=argc>2?atoi(argv[2]):5120, S=argc>3?atoi(argv[3]):16;
    srand(7);
    std::vector<int8_t> xq((size_t)S*IN); for(auto&v:xq)v=(int8_t)((rand()%255)-127);
    std::vector<float> xs((size_t)S*(IN/QBLK)); for(auto&v:xs)v=((rand()%1000)/1000.f)*0.05f+0.001f;
    std::vector<uint8_t> Wq((size_t)OUT*(IN/2)); for(auto&v:Wq)v=(uint8_t)(rand()%256);
    std::vector<__half> sw((size_t)OUT*(IN/QBLK)); for(auto&v:sw)v=__float2half(((rand()%1000)/1000.f)*0.02f+0.0005f);
    int8_t*dxq;float*dxs;uint8_t*dwq;__half*dsw;float*dY,*dYs;
    CK(cudaMalloc(&dxq,xq.size()));CK(cudaMemcpy(dxq,xq.data(),xq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dxs,xs.size()*4));CK(cudaMemcpy(dxs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dwq,Wq.size()));CK(cudaMemcpy(dwq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsw,sw.size()*2));CK(cudaMemcpy(dsw,sw.data(),sw.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dY,(size_t)S*OUT*4));CK(cudaMalloc(&dYs,(size_t)S*OUT*4));
    launch_gemv_w4a8_sb(dxq,dxs,dwq,dsw,dYs,IN,OUT,S,1); CK(cudaDeviceSynchronize());
    std::vector<float> ys((size_t)S*OUT); CK(cudaMemcpy(ys.data(),dYs,ys.size()*4,cudaMemcpyDeviceToHost));
    printf("[probe_splitk] IN=%d OUT=%d S=%d (SM=48; SPLITK로 few-blocks 해결)\n",IN,OUT,S);
    cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
    for(int SK:{1,2,4,8}){
        launch_splitk<__half>(dxq,dxs,dwq,dsw,dY,S,OUT,IN,SK); CK(cudaDeviceSynchronize());
        std::vector<float> y((size_t)S*OUT); CK(cudaMemcpy(y.data(),dY,y.size()*4,cudaMemcpyDeviceToHost));
        double mx=0;int amok=1; for(int s=0;s<S;s++){int ab=0,as=0;float bv=y[(size_t)s*OUT],sv=ys[(size_t)s*OUT];
            for(int o=0;o<OUT;o++){float b=y[(size_t)s*OUT+o],ss=ys[(size_t)s*OUT+o];double d=fabs(b-ss);if(d>mx)mx=d;if(b>bv){bv=b;ab=o;}if(ss>sv){sv=ss;as=o;}}if(ab!=as)amok=0;}
        for(int i=0;i<20;i++)launch_splitk<__half>(dxq,dxs,dwq,dsw,dY,S,OUT,IN,SK);CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));for(int i=0;i<200;i++)launch_splitk<__half>(dxq,dxs,dwq,dsw,dY,S,OUT,IN,SK);CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));
        float tt;CK(cudaEventElapsedTime(&tt,e0,e1));
        int blocks=((OUT+127)/128)*SK;
        printf("  SPLITK=%d: blocks=%d argmax %s max|Δ|=%.2e  격리 %.1fus (★in-engine이 진실)\n",SK,blocks,amok?"OK":"★MISMATCH",mx,tt/200*1000);
    }
    return 0;
}
