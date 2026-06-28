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
template<int HD,int BC>
static void run_fa(const float*Q,const float*Kc,const float*Vc,float*O,int Sq,int NQ,int NKV,int base){
    launch_flash_attn_t<HD,BC>(Q,Kc,Vc,O,Sq,NQ,NKV,base);
}
// HD 템플릿(실제 모델 head_dim=256, 일반화 위해 128도). BCa=작음(타일 다수), BCb=production.
template<int HD,int BCa,int BCb>
static void test(int Sq,int NQ,int NKV,int base,unsigned&seed,bool cmp_decode){
    int Lc=base+Sq;
    std::vector<float> Q((size_t)Sq*NQ*HD),Kc((size_t)Lc*NKV*HD),Vc((size_t)Lc*NKV*HD);
    for(auto&v:Q)v=frand(seed); for(auto&v:Kc)v=frand(seed); for(auto&v:Vc)v=frand(seed);
    std::vector<float> Oref((size_t)Sq*NQ*HD); cpu_attn(Q,Kc,Vc,Oref,Sq,NQ,NKV,HD,base);
    int ntile=(base+Sq+BCa-1)/BCa;
    float *dQ,*dK,*dV,*dO; size_t qb=Q.size()*4,kb=Kc.size()*4,ob=Oref.size()*4;
    CK(cudaMalloc(&dQ,qb));CK(cudaMalloc(&dK,kb));CK(cudaMalloc(&dV,kb));CK(cudaMalloc(&dO,ob));
    CK(cudaMemcpy(dQ,Q.data(),qb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dK,Kc.data(),kb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dV,Vc.data(),kb,cudaMemcpyHostToDevice));
    std::vector<float> Oa(Oref.size()),Ob(Oref.size());
    run_fa<HD,BCa>(dQ,dK,dV,dO,Sq,NQ,NKV,base); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(Oa.data(),dO,ob,cudaMemcpyDeviceToHost));
    run_fa<HD,BCb>(dQ,dK,dV,dO,Sq,NQ,NKV,base); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(Ob.data(),dO,ob,cudaMemcpyDeviceToHost));
    double ra=maxrel(Oa,Oref), rb=maxrel(Ob,Oref);
    bool ok = ra<1e-4 && rb<1e-4;
    printf("  HD=%d Sq=%-3d NQ=%-2d NKV=%-2d base=%-6d (BC%d 타일~%d) | FA(BC%d)vsCPU=%.2e FA(BC%d)vsCPU=%.2e %s\n",
           HD,Sq,NQ,NKV,base,BCa,ntile,BCa,ra,BCb,rb,ok?"OK":"FAIL"); if(!ok)g_fail=1;
    if(cmp_decode){
        float*dOd; CK(cudaMalloc(&dOd,ob));
        launch_attn_decode(dQ,dK,dV,dOd,Sq,NQ,NKV,HD,base); CK(cudaDeviceSynchronize());
        std::vector<float> Od(Oref.size()); CK(cudaMemcpy(Od.data(),dOd,ob,cudaMemcpyDeviceToHost));
        double rd=maxrel(Ob,Od); bool okd=rd<1e-4;
        printf("        └ FA vs 현 attn_decode rel=%.2e %s\n",rd,okd?"OK":"FAIL"); if(!okd)g_fail=1;
        cudaFree(dOd);
    }
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
}
int main(){
    unsigned seed=12345;
    printf("=== FA.1a: FlashAttention(online softmax,GQA,causal) == CPU naive 오차0 ===\n");
    printf("--- HD=256 (실제 0.8B/27B head_dim) BC8/16 ---\n");
    test<256,8,16>(1,  24,4, 0,    seed,true);   // decode 1토큰
    test<256,8,16>(32, 24,4, 0,    seed,true);   // prefill 청크, causal
    test<256,8,16>(32, 24,4, 200,  seed,true);   // 타일 다수 — online running
    test<256,8,16>(64, 24,4, 1000, seed,true);   // group6, 더 많은 타일
    test<256,8,16>(8,  16,2, 777,  seed,true);   // group8, 비배수
    test<256,8,16>(4,  24,4, 20000,seed,false);  // 벽 너머(attn_decode abort) — FA만, 벽 제거 증명
    printf("--- HD=128 (일반화 커버) BC8/32 ---\n");
    test<128,8,32>(32, 24,4, 200,  seed,true);
    test<128,8,32>(17, 16,16,333,  seed,true);   // MHA group1
    printf("\n[test_flash_attn] %s\n", g_fail?"FAIL":"PASS");
    return g_fail;
}
