// 챕터7 C.3 de-risk: ldmatrix로 적재한 m16n8k32 A 프래그먼트 == 스칼라 적재 == (둘 다 같은 MMA 입력).
//   목적: L1/TEX(shared) 대역폭 벽(82.7%) 정조준 — 스칼라 4-load를 ldmatrix.x4 1명령으로 교체.
//   방법: 같은 sA/sB(shared)에 A를 (1)스칼라(현 커널 방식) (2)ldmatrix 로 적재 → 각각 MMA → int32 결과 32 lane 전부 bitwise 동일?
//   A(16×32 int8)를 b16 16×16으로 보고 ldmatrix.m8n8.x4.b16. 매핑 맞으면 오차0, 틀리면 즉시 드러남(추측 금지, 측정).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// A:[16][32] int8(row-major), B:[8][32] int8(n-major, k안에) — 우리 커널 b 적재 패턴과 동일.
//   scalar A: g=lane>>2,t=lane&3 → a0=A[g][t*4], a1=A[g+8][t*4], a2=A[g][16+t*4], a3=A[g+8][16+t*4].
//   ldmatrix A: A를 int16[16][16]로 보고 lane→(row=lane%16, colhalf=lane/16), addr=&A[row][colhalf*16].
__global__ void k(const int8_t*gA,const int8_t*gB,int*outS,int*outL){
    __shared__ int8_t sA[16][32];
    __shared__ int8_t sB[8][32];
    int lane=threadIdx.x;                     // 0..31, 1 warp
    for(int i=lane;i<16*32;i+=32) ((int8_t*)sA)[i]=gA[i];
    for(int i=lane;i<8*32;i+=32) ((int8_t*)sB)[i]=gB[i];
    __syncthreads();
    int g=lane>>2, t=lane&3;
    // B 프래그먼트(양 경로 동일): 우리 커널과 같은 col-major 레이아웃
    int b0=*(int*)&sB[g][t*4], b1=*(int*)&sB[g][16+t*4];
    // ── (1) scalar A ──
    int aS0=*(int*)&sA[g][t*4], aS1=*(int*)&sA[g+8][t*4], aS2=*(int*)&sA[g][16+t*4], aS3=*(int*)&sA[g+8][16+t*4];
    int cS0=0,cS1=0,cS2=0,cS3=0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+r"(cS0),"+r"(cS1),"+r"(cS2),"+r"(cS3):"r"(aS0),"r"(aS1),"r"(aS2),"r"(aS3),"r"(b0),"r"(b1));
    // ── (2) ldmatrix A ──
    int row=lane&15, colhalf=lane>>4;         // lane 0..15→colhalf0(cols0-15), 16..31→colhalf1(cols16-31)
    unsigned saddr=__cvta_generic_to_shared(&sA[row][colhalf*16]);
    int aL0,aL1,aL2,aL3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];\n"
      :"=r"(aL0),"=r"(aL1),"=r"(aL2),"=r"(aL3):"r"(saddr));
    int cL0=0,cL1=0,cL2=0,cL3=0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+r"(cL0),"+r"(cL1),"+r"(cL2),"+r"(cL3):"r"(aL0),"r"(aL1),"r"(aL2),"r"(aL3),"r"(b0),"r"(b1));
    outS[lane*4+0]=cS0;outS[lane*4+1]=cS1;outS[lane*4+2]=cS2;outS[lane*4+3]=cS3;
    outL[lane*4+0]=cL0;outL[lane*4+1]=cL1;outL[lane*4+2]=cL2;outL[lane*4+3]=cL3;
    // ldmatrix 출력 레지스터 그대로도 덤프(매핑 디버그용)
}
int main(){
    int8_t hA[16*32], hB[8*32];
    for(int i=0;i<16*32;i++) hA[i]=(int8_t)((i*7+3)%17-8);
    for(int i=0;i<8*32;i++)  hB[i]=(int8_t)((i*5+1)%13-6);
    int8_t*dA,*dB; int*dS,*dL; CK(cudaMalloc(&dA,sizeof hA));CK(cudaMalloc(&dB,sizeof hB));
    CK(cudaMalloc(&dS,32*4*4));CK(cudaMalloc(&dL,32*4*4));
    CK(cudaMemcpy(dA,hA,sizeof hA,cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,hB,sizeof hB,cudaMemcpyHostToDevice));
    k<<<1,32>>>(dA,dB,dS,dL); CK(cudaDeviceSynchronize());
    int cS[128],cL[128]; CK(cudaMemcpy(cS,dS,sizeof cS,cudaMemcpyDeviceToHost));CK(cudaMemcpy(cL,dL,sizeof cL,cudaMemcpyDeviceToHost));
    int fail=0,ndiff=0; for(int i=0;i<128;i++) if(cS[i]!=cL[i]){fail=1; if(ndiff<8)printf("  diff lane%d r%d: scalar=%d ldm=%d\n",i/4,i%4,cS[i],cL[i]); ndiff++;}
    printf("[test_ldmatrix] %s (ldmatrix A frag == scalar, %d/128 diff)\n",fail?"FAIL":"PASS",ndiff);
    return fail;
}
