// 챕터7 STEP B.0: 단일 warp-tile int8 MMA(m16n8k32.s8.s8.s32) == CPU int dot 오차0.
//   tensor-core fragment 레이아웃(가장 위험) 격리 검증. A[16,32]·B[8,32](col)→C[16,8] int32.
//   표준 Ampere/Blackwell s8 MMA 레이아웃: groupID=lane>>2, tid=lane&3.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ int pk4(const int8_t*b){
    return (b[0]&0xFF)|((b[1]&0xFF)<<8)|((b[2]&0xFF)<<16)|((b[3]&0xFF)<<24);
}
// A[16][32], B[8][32](col-major: B[n][k]) → C[16][8] int32
__global__ void mma_kernel(const int8_t*A,const int8_t*B,int32_t*C){
    int lane=threadIdx.x&31, g=lane>>2, t=lane&3;
    int8_t buf[4]; int a0,a1,a2,a3,b0,b1,c0=0,c1=0,c2=0,c3=0;
    // A: a0 row=g cols[t*4..](K0-15), a1 row=g+8, a2 row=g K16-31, a3 row=g+8 K16-31
    for(int j=0;j<4;j++)buf[j]=A[g*32 + t*4+j];        a0=pk4(buf);
    for(int j=0;j<4;j++)buf[j]=A[(g+8)*32 + t*4+j];    a1=pk4(buf);
    for(int j=0;j<4;j++)buf[j]=A[g*32 + 16+t*4+j];     a2=pk4(buf);
    for(int j=0;j<4;j++)buf[j]=A[(g+8)*32 + 16+t*4+j]; a3=pk4(buf);
    // B(col): b0 col(N)=g K[t*4..](0-15), b1 K16-31
    for(int j=0;j<4;j++)buf[j]=B[g*32 + t*4+j];        b0=pk4(buf);
    for(int j=0;j<4;j++)buf[j]=B[g*32 + 16+t*4+j];     b1=pk4(buf);
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    // C: c0 row=g col=t*2, c1 row=g col=t*2+1, c2 row=g+8 col=t*2, c3 row=g+8 col=t*2+1
    C[g*8 + t*2]       = c0;
    C[g*8 + t*2+1]     = c1;
    C[(g+8)*8 + t*2]   = c2;
    C[(g+8)*8 + t*2+1] = c3;
}
int main(){
    unsigned s=123; auto rnd=[&](){s=s*1664525u+1013904223u; return (int8_t)((s>>20)%15 - 7);};
    std::vector<int8_t> A(16*32),B(8*32); for(auto&v:A)v=rnd(); for(auto&v:B)v=rnd();
    std::vector<int32_t> ref(16*8,0);
    for(int m=0;m<16;m++)for(int n=0;n<8;n++){int acc=0;for(int k=0;k<32;k++)acc+=(int)A[m*32+k]*(int)B[n*32+k];ref[m*8+n]=acc;}
    int8_t*dA,*dB; int32_t*dC; CK(cudaMalloc(&dA,A.size()));CK(cudaMalloc(&dB,B.size()));CK(cudaMalloc(&dC,16*8*4));
    CK(cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice));
    mma_kernel<<<1,32>>>(dA,dB,dC); CK(cudaDeviceSynchronize());
    std::vector<int32_t> C(16*8); CK(cudaMemcpy(C.data(),dC,16*8*4,cudaMemcpyDeviceToHost));
    int mism=0,first=-1; for(int i=0;i<16*8;i++) if(C[i]!=ref[i]){mism++; if(first<0)first=i;}
    printf("[mma_b0] C[16,8]=A·Bᵀ int8 MMA vs CPU int dot: 불일치=%d/128",mism);
    if(first>=0) printf(" (첫 [m=%d,n=%d] mma=%d cpu=%d)",first/8,first%8,C[first],ref[first]);
    printf("\n[test_mma_b0] %s (fragment 레이아웃 %s)\n",mism==0?"PASS":"FAIL",mism==0?"확정":"틀림");
    return mism?1:0;
}
