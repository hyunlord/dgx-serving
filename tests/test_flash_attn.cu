// 챕터7 FA.1a 검증: FlashAttention(online softmax, GQA, causal) == CPU naive 독립 레퍼런스 오차0.
//   두 보강 명시 검증: ① online softmax — 작은 BC(=8)로 타일 다수 강제(running m/l/acc rescale 진짜 검증).
//   ② GQA — NQ=24/NKV=4(group6) 등 각 query head가 올바른 KV head(kv=h/(NQ/NKV)) attend, CPU도 동일.
//   + 현 attn_decode(벽 안 ≤base)와도 대조. 벽 너머(base=20000)는 CPU만(attn_decode는 가드 abort).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/attention_flash.cuh"
#include "../kernels/attention.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}

// CPU naive: O[i,h,d] = Σ_j softmax(q[i,h]·k[j,kv]/√HD)·v[j,kv], j∈[0,base+i], kv=h/(NQ/NKV). double 누적.
static void cpu_attn(const std::vector<float>&Q,const std::vector<float>&Kc,const std::vector<float>&Vc,
                     std::vector<float>&O,int Sq,int NQ,int NKV,int HD,int base){
    int GRP=NQ/NKV; double scale=1.0/sqrt((double)HD);
    for(int i=0;i<Sq;i++)for(int h=0;h<NQ;h++){
        int kv=h/GRP, attendlen=base+i+1;
        std::vector<double> sc(attendlen);
        for(int j=0;j<attendlen;j++){ double dot=0;
            for(int d=0;d<HD;d++) dot+=(double)Q[((size_t)i*NQ+h)*HD+d]*Kc[((size_t)j*NKV+kv)*HD+d];
            sc[j]=dot*scale; }
        double m=-1e300; for(double v:sc)m=fmax(m,v);
        double sum=0; for(int j=0;j<attendlen;j++){sc[j]=exp(sc[j]-m);sum+=sc[j];}
        for(int d=0;d<HD;d++){ double a=0;
            for(int j=0;j<attendlen;j++) a+=sc[j]*Vc[((size_t)j*NKV+kv)*HD+d];
            O[((size_t)i*NQ+h)*HD+d]=(float)(a/sum); }
    }
}

static double maxrel(const std::vector<float>&A,const std::vector<float>&B){
    double e=0,rms=0; for(size_t i=0;i<A.size();i++){e=fmax(e,fabs((double)A[i]-B[i]));rms+=(double)B[i]*B[i];}
    rms=sqrt(rms/A.size()); return e/(rms>0?rms:1);
}

static int g_fail=0;
// BC 템플릿 직접 인스턴스(작은 BC=8로 타일 다수 / 32은 production)
template<int BC>
static void run_fa(const float*Q,const float*Kc,const float*Vc,float*O,int Sq,int NQ,int NKV,int base){
    launch_flash_attn_t<128,BC>(Q,Kc,Vc,O,Sq,NQ,NKV,base);
}
static void test(int Sq,int NQ,int NKV,int base,unsigned&seed,bool cmp_decode){
    const int HD=128; int Lc=base+Sq;
    std::vector<float> Q((size_t)Sq*NQ*HD),Kc((size_t)Lc*NKV*HD),Vc((size_t)Lc*NKV*HD);
    for(auto&v:Q)v=frand(seed); for(auto&v:Kc)v=frand(seed); for(auto&v:Vc)v=frand(seed);
    std::vector<float> Oref((size_t)Sq*NQ*HD); cpu_attn(Q,Kc,Vc,Oref,Sq,NQ,NKV,HD,base);
    int ntile=(base+Sq+7)/8;   // BC=8 기준 최대 타일 수(running 갱신 횟수)
    float *dQ,*dK,*dV,*dO; size_t qb=Q.size()*4,kb=Kc.size()*4,ob=Oref.size()*4;
    CK(cudaMalloc(&dQ,qb));CK(cudaMalloc(&dK,kb));CK(cudaMalloc(&dV,kb));CK(cudaMalloc(&dO,ob));
    CK(cudaMemcpy(dQ,Q.data(),qb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dK,Kc.data(),kb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dV,Vc.data(),kb,cudaMemcpyHostToDevice));
    std::vector<float> O8(Oref.size()),O32(Oref.size());
    run_fa<8>(dQ,dK,dV,dO,Sq,NQ,NKV,base);  CK(cudaDeviceSynchronize()); CK(cudaMemcpy(O8.data(),dO,ob,cudaMemcpyDeviceToHost));
    run_fa<32>(dQ,dK,dV,dO,Sq,NQ,NKV,base); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(O32.data(),dO,ob,cudaMemcpyDeviceToHost));
    double r8=maxrel(O8,Oref), r32=maxrel(O32,Oref);
    bool ok = r8<1e-4 && r32<1e-4;
    printf("  Sq=%-3d NQ=%-2d NKV=%-2d base=%-6d (BC=8 타일~%d) | FA(BC8)vsCPU rel=%.2e FA(BC32)vsCPU rel=%.2e %s\n",
           Sq,NQ,NKV,base,ntile,r8,r32,ok?"OK":"FAIL"); if(!ok)g_fail=1;
    if(cmp_decode){  // 현 attn_decode(벽 안)와 대조 — FA(BC32)와 같은 결과?
        run_fa<32>(dQ,dK,dV,dO,Sq,NQ,NKV,base);  // dO 재사용 전 FA 다시(위 O32와 동일)
        float*dOd; CK(cudaMalloc(&dOd,ob));
        launch_attn_decode(dQ,dK,dV,dOd,Sq,NQ,NKV,HD,base); CK(cudaDeviceSynchronize());
        std::vector<float> Od(Oref.size()); CK(cudaMemcpy(Od.data(),dOd,ob,cudaMemcpyDeviceToHost));
        double rd=maxrel(O32,Od); bool okd=rd<1e-4;
        printf("        └ FA(BC32) vs 현 attn_decode rel=%.2e %s\n",rd,okd?"OK":"FAIL"); if(!okd)g_fail=1;
        cudaFree(dOd);
    }
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
}
int main(){
    unsigned seed=12345;
    printf("=== FA.1a: FlashAttention(online softmax,GQA,causal) == CPU naive 오차0 ===\n");
    // GQA 다양 + 타일 다수(base 큼) + 비배수 attendlen
    test(1,  24,4, 0,    seed,true);   // decode 1토큰, 1타일(trivial)
    test(32, 24,4, 0,    seed,true);   // prefill 청크, attendlen 변동(causal)
    test(32, 24,4, 200,  seed,true);   // 타일 다수(base 200 → BC8서 ~29타일) — online running 검증
    test(64, 24,4, 1000, seed,true);   // 더 많은 타일, GQA group6
    test(17, 16,16,333,  seed,true);   // MHA(group1) 비배수 attendlen
    test(8,  16,2, 777,  seed,true);   // group8, 비배수
    printf("--- 48KB 벽 너머 (현 attn_decode는 abort, FA만 — 벽 제거 증명) ---\n");
    test(4,  24,4, 20000,seed,false);  // base 20000 → attendlen ~20K, FA 상수 shared로 동작
    printf("\n[test_flash_attn] %s\n", g_fail?"FAIL":"PASS");
    return g_fail;
}
