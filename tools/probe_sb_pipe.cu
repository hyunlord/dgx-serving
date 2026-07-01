// STEP2 de-risk: SB 커널 load-latency(long_scoreboard 72.8%)를 소프트웨어 파이프라이닝으로 숨나.
//   현 gemv_w4a8_sb = 동기 load(W 적재→stall→컴퓨트→다음 W). pipe = 다음 base의 W를 미리 prefetch(현 컴퓨트와 겹침).
//   2-stage 레지스터 파이프(Wnext 1개만 추가 = occupancy 영향 최소, SB-MLP의 U-unroll 역효과와 다름).
//   측정: ① bit-exact(같은 수식·누적순서, prefetch만 = 오차0) ② 타이밍 pipe vs 현 ③ (ncu 별도 long_scoreboard).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../kernels/gemv_w4a8.cuh"   // launch_gemv_w4a8_sb + w4a8_pk/sval/warp_reduce_sum
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// 파이프라인 SB: 다음 base의 W를 미리 적재(prefetch) → load latency가 현 base의 DP4A 컴퓨트와 겹침.
template<typename ST, int SMAX>
__global__ void gemv_w4a8_sb_pipe_kernel(const int8_t*xq,const float*xs,const uint8_t*Wq,const ST*sw,float*Y,int IN,int OUT,int S){
    long warp=((long)blockIdx.x*blockDim.x+threadIdx.x)>>5; int lane=threadIdx.x&31;
    if(warp>=(long)OUT)return; int o=(int)warp;
    const uint8_t*w=Wq+(size_t)o*(IN/2); const ST*swr=sw+(size_t)o*(IN/QBLK);
    float acc[SMAX];
    #pragma unroll
    for(int s=0;s<SMAX;s++)acc[s]=0.f;
    int base=lane*8;
    unsigned Wcur=(base<IN)?*reinterpret_cast<const unsigned*>(w+(base>>1)):0u;   // prologue
    for(; base<IN; base+=32*8){
        int nbase=base+32*8;
        unsigned Wnext=(nbase<IN)?*reinterpret_cast<const unsigned*>(w+(nbase>>1)):0u;  // ★ prefetch(현 컴퓨트와 겹침)
        int blk=base/QBLK;
        int pk0=w4a8_pk(Wcur&0xFF,(Wcur>>8)&0xFF), pk1=w4a8_pk((Wcur>>16)&0xFF,(Wcur>>24)&0xFF);
        float swb=sval(swr[blk]);
        for(int s=0;s<S;s++){ const int8_t*xqr=xq+(size_t)s*IN;
            int x0=*reinterpret_cast<const int*>(xqr+base), x1=*reinterpret_cast<const int*>(xqr+base+4);
            int dot=__dp4a(x0,pk0,0); dot=__dp4a(x1,pk1,dot);
            acc[s]+=xs[(size_t)s*(IN/QBLK)+blk]*swb*(float)dot; }
        Wcur=Wnext;
    }
    for(int s=0;s<S;s++){ float r=warp_reduce_sum(acc[s]); if(lane==0)Y[(size_t)s*OUT+o]=r; }
}
template<typename ST,int SMAX>
static void launch_pipe(const int8_t*xq,const float*xs,const uint8_t*Wq,const ST*sw,float*Y,int IN,int OUT,int S){
    int block=256; long grid=((long)OUT*32+block-1)/block;
    gemv_w4a8_sb_pipe_kernel<ST,SMAX><<<(unsigned)grid,block>>>(xq,xs,Wq,sw,Y,IN,OUT,S);
}

int main(int argc,char**argv){
    int IN=argc>1?atoi(argv[1]):5120, OUT=argc>2?atoi(argv[2]):5120, S=argc>3?atoi(argv[3]):16;
    srand(7);
    std::vector<int8_t> xq((size_t)S*IN); for(auto&v:xq)v=(int8_t)((rand()%255)-127);
    std::vector<float> xs((size_t)S*(IN/QBLK)); for(auto&v:xs)v=((rand()%1000)/1000.f)*0.05f+0.001f;
    std::vector<uint8_t> Wq((size_t)OUT*(IN/2)); for(auto&v:Wq)v=(uint8_t)(rand()%256);
    std::vector<__half> sw((size_t)OUT*(IN/QBLK)); for(auto&v:sw)v=__float2half(((rand()%1000)/1000.f)*0.02f+0.0005f);
    int8_t*dxq;float*dxs;uint8_t*dwq;__half*dsw;float*dY0,*dY1;
    CK(cudaMalloc(&dxq,xq.size()));CK(cudaMemcpy(dxq,xq.data(),xq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dxs,xs.size()*4));CK(cudaMemcpy(dxs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dwq,Wq.size()));CK(cudaMemcpy(dwq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsw,sw.size()*2));CK(cudaMemcpy(dsw,sw.data(),sw.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dY0,(size_t)S*OUT*4));CK(cudaMalloc(&dY1,(size_t)S*OUT*4));
    // bit-exact: 현 SB vs pipe
    launch_gemv_w4a8_sb(dxq,dxs,dwq,dsw,dY0,IN,OUT,S,1);
    launch_pipe<__half,8>(dxq,dxs,dwq,dsw,dY1,IN,OUT,S);   // S<=8; S>8은 <16>
    if(S>8){ launch_pipe<__half,16>(dxq,dxs,dwq,dsw,dY1,IN,OUT,S); }
    CK(cudaDeviceSynchronize());
    std::vector<float> y0((size_t)S*OUT),y1((size_t)S*OUT);
    CK(cudaMemcpy(y0.data(),dY0,y0.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(y1.data(),dY1,y1.size()*4,cudaMemcpyDeviceToHost));
    int mm=0;for(size_t i=0;i<y0.size();i++)if(memcmp(&y0[i],&y1[i],4))mm++;
    printf("[probe_sb_pipe] IN=%d OUT=%d S=%d\n  bit-exact 현SB vs pipe: %d/%zu %s\n",IN,OUT,S,mm,y0.size(),mm?"★DIFF":"(bit-identical ✓)");
    // 타이밍
    cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
    auto cur=[&](){ launch_gemv_w4a8_sb(dxq,dxs,dwq,dsw,dY0,IN,OUT,S,1); };
    auto pip=[&](){ if(S>8)launch_pipe<__half,16>(dxq,dxs,dwq,dsw,dY1,IN,OUT,S); else launch_pipe<__half,8>(dxq,dxs,dwq,dsw,dY1,IN,OUT,S); };
    for(int i=0;i<30;i++){cur();} CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));for(int i=0;i<300;i++)cur();CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));
    float tc;CK(cudaEventElapsedTime(&tc,e0,e1));
    for(int i=0;i<30;i++){pip();} CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));for(int i=0;i<300;i++)pip();CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));
    float tp;CK(cudaEventElapsedTime(&tp,e0,e1));
    printf("  타이밍(300 avg): 현SB=%.1fus  pipe=%.1fus  → pipe %.3f× %s\n",tc/300*1000,tp/300*1000,tc/tp,tp<tc?"빠름":"느림/동일");
    printf("  → pipe 빠름 = load-latency 숨김 성공(정식 GO); 동일/느림 = 컴파일러가 이미 reorder or 다른 벽\n");
    return mm?1:0;
}
