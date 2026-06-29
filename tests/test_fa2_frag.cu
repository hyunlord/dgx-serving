// 챕터7 FA.2.0/2.1: bf16 tensor-core MMA fragment 레이아웃 de-risk (Phase B식).
//   2.0: S = Q·Kᵀ  (m16n8k16.row.col.f32.bf16.bf16.f32) == CPU dot. Br=16,Bc=8,HD=16(1 k-스텝).
//   2.1: O = P·V    같은 MMA로. P[16,8]·V[8,HD]. V 레이아웃 확인.
//   레이아웃(16bit m16n8k16): groupID=lane>>2, t=lane&3. A 4regs(2 bf16), B 2regs, C 4regs fp32.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
static float fr(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}

// 2.0: S[16,8] = Q[16,K]·Kt[8,K] (K=16 한 스텝). A=Q[br][k] row, B=Kk[bc][k] (.col operand=[k,n]=Kk[bc][k]).
__global__ void score_mma(const __nv_bfloat16*Q,const __nv_bfloat16*Kk,float*S,int HD){
    int lane=threadIdx.x&31, g=lane>>2, t=lane&3;
    float c0=0,c1=0,c2=0,c3=0;
    for(int k0=0;k0<HD;k0+=16){
        // A=Q: a0=Q[g][k0+2t..+1], a1=Q[g+8][..], a2=Q[g][k0+2t+8..], a3=Q[g+8][k0+2t+8..]
        unsigned a0=*(unsigned*)&Q[(size_t)g*HD+k0+2*t], a1=*(unsigned*)&Q[(size_t)(g+8)*HD+k0+2*t];
        unsigned a2=*(unsigned*)&Q[(size_t)g*HD+k0+2*t+8], a3=*(unsigned*)&Q[(size_t)(g+8)*HD+k0+2*t+8];
        // B=Kk: b0=Kk[g][k0+2t..], b1=Kk[g][k0+2t+8..]  (B[k][n]=Kk[bc=n][k], n=g)
        unsigned b0=*(unsigned*)&Kk[(size_t)g*HD+k0+2*t], b1=*(unsigned*)&Kk[(size_t)g*HD+k0+2*t+8];
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    // C: c0=S[g][2t] c1=S[g][2t+1] c2=S[g+8][2t] c3=S[g+8][2t+1]
    S[(size_t)g*8+2*t]=c0; S[(size_t)g*8+2*t+1]=c1; S[(size_t)(g+8)*8+2*t]=c2; S[(size_t)(g+8)*8+2*t+1]=c3;
}

// 2.1: O[16,HD] = P[16,Bc]·V[Bc,HD], Bc=16(1 k-스텝). A=P[br][bc] row, B=V[bc][hd](.col operand [k=bc,n=hd]).
//   B 프래그먼트: b0={V[2t][n0+g], V[2t+1][n0+g]} (k=bc=2t.., n=hd=n0+g). 비연속 → 수동 pack.
__device__ __forceinline__ unsigned packbf(const __nv_bfloat16&x,const __nv_bfloat16&y){
    return (unsigned)(*(const unsigned short*)&x) | ((unsigned)(*(const unsigned short*)&y)<<16);
}
__global__ void pv_mma(const __nv_bfloat16*P,const __nv_bfloat16*V,float*O,int HD){
    int lane=threadIdx.x&31, g=lane>>2, t=lane&3; const int Bc=16;
    // A=P[16][16]
    unsigned a0=*(unsigned*)&P[(size_t)g*Bc+2*t], a1=*(unsigned*)&P[(size_t)(g+8)*Bc+2*t];
    unsigned a2=*(unsigned*)&P[(size_t)g*Bc+2*t+8], a3=*(unsigned*)&P[(size_t)(g+8)*Bc+2*t+8];
    for(int n0=0;n0<HD;n0+=8){
        unsigned b0=packbf(V[(size_t)(2*t)*HD+n0+g],   V[(size_t)(2*t+1)*HD+n0+g]);
        unsigned b1=packbf(V[(size_t)(2*t+8)*HD+n0+g], V[(size_t)(2*t+9)*HD+n0+g]);
        float c0=0,c1=0,c2=0,c3=0;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        O[(size_t)g*HD+n0+2*t]=c0; O[(size_t)g*HD+n0+2*t+1]=c1; O[(size_t)(g+8)*HD+n0+2*t]=c2; O[(size_t)(g+8)*HD+n0+2*t+1]=c3;
    }
}
static int test_pv(){
    unsigned sd=23; int HD=64; const int Bc=16;
    std::vector<float> Pf(16*Bc),Vf(Bc*HD); for(auto&v:Pf)v=fr(sd)*0.1f+0.05f; for(auto&v:Vf)v=fr(sd);
    std::vector<__nv_bfloat16> Ph(16*Bc),Vh(Bc*HD);
    for(int i=0;i<16*Bc;i++)Ph[i]=__float2bfloat16(Pf[i]); for(int i=0;i<Bc*HD;i++)Vh[i]=__float2bfloat16(Vf[i]);
    std::vector<float> ref(16*HD);
    for(int i=0;i<16;i++)for(int j=0;j<HD;j++){double a=0;
        for(int k=0;k<Bc;k++)a+=(double)__bfloat162float(Ph[i*Bc+k])*__bfloat162float(Vh[k*HD+j]); ref[i*HD+j]=(float)a;}
    __nv_bfloat16*dP,*dV; float*dO;
    CK(cudaMalloc(&dP,Ph.size()*2));CK(cudaMalloc(&dV,Vh.size()*2));CK(cudaMalloc(&dO,16*HD*4));
    CK(cudaMemcpy(dP,Ph.data(),Ph.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dV,Vh.data(),Vh.size()*2,cudaMemcpyHostToDevice));
    pv_mma<<<1,32>>>(dP,dV,dO,HD); CK(cudaDeviceSynchronize());
    std::vector<float> O(16*HD); CK(cudaMemcpy(O.data(),dO,16*HD*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(int i=0;i<16*HD;i++){e=fmax(e,fabs((double)O[i]-ref[i]));rms+=ref[i]*ref[i];} rms=sqrt(rms/(16*HD));
    bool ok=e/rms<1e-2;
    printf("[FA.2.1] O=P·V bf16 MMA vs CPU: max err=%.3e (rms=%.2e rel=%.2e) %s\n",e,rms,e/rms,ok?"OK":"FAIL");
    return ok?0:1;
}
int main(){
    unsigned sd=11; int HD=64; // 4 k-스텝 (누적 확인)
    std::vector<float> Qf(16*HD),Kf(8*HD);
    for(auto&v:Qf)v=fr(sd); for(auto&v:Kf)v=fr(sd);
    std::vector<__nv_bfloat16> Qh(16*HD),Kh(8*HD);
    for(int i=0;i<16*HD;i++)Qh[i]=__float2bfloat16(Qf[i]);
    for(int i=0;i<8*HD;i++)Kh[i]=__float2bfloat16(Kf[i]);
    // CPU ref: S[i][j]=Σ_k Q[i][k]*K[j][k] (bf16 반올림 입력으로)
    std::vector<float> ref(16*8);
    for(int i=0;i<16;i++)for(int j=0;j<8;j++){double a=0;
        for(int k=0;k<HD;k++)a+=(double)__bfloat162float(Qh[i*HD+k])*__bfloat162float(Kh[j*HD+k]); ref[i*8+j]=(float)a;}
    __nv_bfloat16*dQ,*dK; float*dS;
    CK(cudaMalloc(&dQ,Qh.size()*2));CK(cudaMalloc(&dK,Kh.size()*2));CK(cudaMalloc(&dS,16*8*4));
    CK(cudaMemcpy(dQ,Qh.data(),Qh.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dK,Kh.data(),Kh.size()*2,cudaMemcpyHostToDevice));
    score_mma<<<1,32>>>(dQ,dK,dS,HD); CK(cudaDeviceSynchronize());
    std::vector<float> S(16*8); CK(cudaMemcpy(S.data(),dS,16*8*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(int i=0;i<128;i++){e=fmax(e,fabs((double)S[i]-ref[i]));rms+=ref[i]*ref[i];} rms=sqrt(rms/128);
    bool ok=e/rms<1e-2;
    printf("[FA.2.0] S=Q·Kt bf16 MMA vs CPU: max err=%.3e (rms=%.2e rel=%.2e) %s\n",e,rms,e/rms,ok?"OK":"FAIL");
    int f2=test_pv();
    bool pass = ok && f2==0;
    printf("[test_fa2_frag] %s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
