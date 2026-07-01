// STEP 3 de-risk: int8 tensor-core MMA(WMMA s8)가 occupancy 50% 벽(우리 gemm_mmq: 72 regs·26.3KB shared·Block Limit
//   Shared AND Reg 동시)을 완화하나. WMMA int8(m16n16k16, IMMA=tensor core)로 대표 GEMM 타일 커널 → regs/shared/occupancy
//   측정 + correctness(scalar int8 GEMM 대비 오차0=MMA 정확성 sm_120 확인). 여러 타일 config로 occ 곡선.
//   판정: MMA 타일이 >50% occ 달성 가능하면 벽 완화(GO), Reg-limit로 ≤50%면 벽 잔존(NO-GO→③).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mma.h>
#include <cuda_runtime.h>
using namespace nvcuda;
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// int8 WMMA GEMM: block이 (WM×16)×(WN×16) 출력 타일. warp 그리드 WM×WN, 각 warp가 16×16 wmma 타일 K-루프 누적.
//   shared: A[(WM*16)×KT] + B[KT×(WN*16)] int8. accumulator = warp당 int32 16×16 fragment(regs).
template<int WM, int WN, int KT>
__global__ void mma_gemm_s8(const int8_t* __restrict__ A, const int8_t* __restrict__ B, int* __restrict__ C, int M, int N, int K){
    const int BM=WM*16, BN=WN*16;
    __shared__ int8_t sA[BM*KT];
    __shared__ int8_t sB[KT*BN];
    int warp = threadIdx.x/32; int wm = warp/WN, wn = warp%WN;   // warp의 타일 좌표
    int blockRow = blockIdx.y*BM, blockCol = blockIdx.x*BN;
    wmma::fragment<wmma::accumulator,16,16,16,int> acc; wmma::fill_fragment(acc,0);
    for(int k0=0;k0<K;k0+=KT){
        // shared 적재(협력): sA[BM×KT] row-major, sB[KT×BN] row-major(=B col-major 뷰)
        for(int i=threadIdx.x;i<BM*KT;i+=blockDim.x){ int r=i/KT,c=i%KT; sA[i]=A[(size_t)(blockRow+r)*K+(k0+c)]; }
        for(int i=threadIdx.x;i<KT*BN;i+=blockDim.x){ int r=i/BN,c=i%BN; sB[i]=B[(size_t)(k0+r)*N+(blockCol+c)]; }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<KT;kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,int8_t,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,int8_t,wmma::row_major> b;
            wmma::load_matrix_sync(a, sA + (wm*16)*KT + kk, KT);
            wmma::load_matrix_sync(b, sB + kk*BN + (wn*16), BN);
            wmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(C + (size_t)(blockRow+wm*16)*N + (blockCol+wn*16), acc, N, wmma::mem_row_major);
}

template<int WM,int WN,int KT>
static void measure(const char* nm){
    int block = WM*WN*32; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa,(const void*)mma_gemm_s8<WM,WN,KT>));
    int nb=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,(const void*)mma_gemm_s8<WM,WN,KT>,block,0));
    int warpsSM=nb*(block/32); double occ=(double)warpsSM/(p.maxThreadsPerMultiProcessor/32);
    size_t smem=(size_t)(WM*16)*KT + (size_t)KT*(WN*16);
    printf("  %-16s tile=%dx%d(warps%d) block=%d regs=%-4d smem=%zuB blocks/SM=%-2d warps/SM=%-3d occ=%.0f%%\n",
           nm,WM*16,WN*16,WM*WN,block,fa.numRegs,smem,nb,warpsSM,occ*100);
}

int main(){
    printf("[probe_mma_occ] int8 WMMA(tensor core) occupancy de-risk — 우리 gemm_mmq baseline: 72 regs·26.3KB shared·occ 50%%\n");
    printf("=== occupancy (CUDA API, register/shared-limited) ===\n");
    measure<2,2,32>("MMA 32x32");    // 4 warps
    measure<2,4,32>("MMA 32x64");    // 8 warps
    measure<4,4,32>("MMA 64x64");    // 16 warps
    measure<4,4,64>("MMA 64x64 KT64");
    measure<2,2,64>("MMA 32x32 KT64");

    // correctness: 64x64x64 int8 GEMM, MMA vs scalar (bit-exact 정확성 sm_120 확인)
    const int M=64,N=64,K=128;
    std::vector<int8_t> hA((size_t)M*K), hB((size_t)K*N); std::vector<int> hC((size_t)M*N,0), ref((size_t)M*N,0);
    srand(1); for(auto&v:hA)v=(int8_t)((rand()%15)-7); for(auto&v:hB)v=(int8_t)((rand()%15)-7);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){int s=0;for(int k=0;k<K;k++)s+=(int)hA[(size_t)m*K+k]*(int)hB[(size_t)k*N+n];ref[(size_t)m*N+n]=s;}
    int8_t*dA,*dB;int*dC; CK(cudaMalloc(&dA,hA.size()));CK(cudaMalloc(&dB,hB.size()));CK(cudaMalloc(&dC,hC.size()*4));
    CK(cudaMemcpy(dA,hA.data(),hA.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,hB.data(),hB.size(),cudaMemcpyHostToDevice));
    dim3 grid(N/64,M/64); mma_gemm_s8<4,4,32><<<grid,4*4*32>>>(dA,dB,dC,M,N,K); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC.data(),dC,hC.size()*4,cudaMemcpyDeviceToHost));
    int mm=0;for(size_t i=0;i<hC.size();i++)if(hC[i]!=ref[i])mm++;
    printf("=== correctness (64x64x128 int8 GEMM, MMA vs scalar): %d mismatch %s ===\n",mm,mm?"★":"(MMA 정확 sm_120 ✓)");
    printf("=== 판정: MMA 타일 중 occ>50%% 있으면 벽 완화(GO), 전부 ≤50%%(Reg/shared-limit)면 벽 잔존 ===\n");
    return 0;
}
