// 챕터3 STEP2a 검증: batch decode 커널(attn/gdn/conv) == B개 독립 single-seq 호출, 시퀀스별 오차0.
//   batch가 결과를 안 바꾼다(각 시퀀스 독립). novel 커널 isolated 검증(통합 전 de-risk, STEP1 패턴).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../kernels/attention.cuh"
#include "../kernels/gdn.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
static int g_fail=0;
static float frand(unsigned&s){ s=s*1664525u+1013904223u; return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f; }
static float* up(const std::vector<float>&h){ float*d; CK(cudaMalloc(&d,h.size()*4)); CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static double maxerr(const std::vector<float>&a,const std::vector<float>&b){ double e=0; for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i])); return e; }
static void check(const char*n,double e){ bool ok=e<1e-4; printf("  %-22s max err=%.3e %s\n",n,e,ok?"OK":"FAIL"); if(!ok)g_fail=1; }

int main(){
    unsigned seed=99;
    const int B=8;

    // ===== attention batch vs B single =====
    {
        int NQ=8,NKV=2,HD=256,base=20,maxlen=64; long kvstride=(long)maxlen*NKV*HD;
        std::vector<float> Q((size_t)B*NQ*HD), Kc((size_t)B*kvstride), Vc((size_t)B*kvstride);
        for(auto&v:Q)v=frand(seed); for(auto&v:Kc)v=frand(seed); for(auto&v:Vc)v=frand(seed);
        float *dQ=up(Q),*dKc=up(Kc),*dVc=up(Vc),*dOb; CK(cudaMalloc(&dOb,(size_t)B*NQ*HD*4));
        launch_attn_decode_batch(dQ,dKc,dVc,dOb,B,NQ,NKV,HD,base,kvstride); CK(cudaDeviceSynchronize());
        std::vector<float> Ob((size_t)B*NQ*HD); CK(cudaMemcpy(Ob.data(),dOb,Ob.size()*4,cudaMemcpyDeviceToHost));
        std::vector<float> Os((size_t)B*NQ*HD);
        for(int b=0;b<B;b++){ float*dO1; CK(cudaMalloc(&dO1,(size_t)NQ*HD*4));
            launch_attn_decode(dQ+(size_t)b*NQ*HD, dKc+(size_t)b*kvstride, dVc+(size_t)b*kvstride, dO1, 1,NQ,NKV,HD,base);
            CK(cudaDeviceSynchronize()); CK(cudaMemcpy(Os.data()+(size_t)b*NQ*HD,dO1,(size_t)NQ*HD*4,cudaMemcpyDeviceToHost)); cudaFree(dO1); }
        check("attn_batch==Bx single", maxerr(Ob,Os));
        cudaFree(dQ);cudaFree(dKc);cudaFree(dVc);cudaFree(dOb);
    }

    // ===== conv1d_silu_decode batch vs B single =====
    {
        int C=512,K=4;
        std::vector<float> in((size_t)B*C), prev((size_t)B*(K-1)*C), w((size_t)C*K);
        for(auto&v:in)v=frand(seed); for(auto&v:prev)v=frand(seed); for(auto&v:w)v=frand(seed)*0.3f;
        float *din=up(in),*dprev=up(prev),*dw=up(w),*dob; CK(cudaMalloc(&dob,(size_t)B*C*4));
        launch_gdn_conv1d_silu_decode_batch(din,dprev,dw,dob,B,C,K); CK(cudaDeviceSynchronize());
        std::vector<float> ob((size_t)B*C); CK(cudaMemcpy(ob.data(),dob,ob.size()*4,cudaMemcpyDeviceToHost));
        std::vector<float> os((size_t)B*C);
        for(int b=0;b<B;b++){ float*do1; CK(cudaMalloc(&do1,(size_t)C*4));
            launch_gdn_conv1d_silu_decode(din+(size_t)b*C, dprev+(size_t)b*(K-1)*C, dw, do1, 1,C,K);
            CK(cudaDeviceSynchronize()); CK(cudaMemcpy(os.data()+(size_t)b*C,do1,(size_t)C*4,cudaMemcpyDeviceToHost)); cudaFree(do1); }
        check("conv_batch==Bx single", maxerr(ob,os));
        cudaFree(din);cudaFree(dprev);cudaFree(dw);cudaFree(dob);
    }

    // ===== gdn_step batch vs B single (O + 갱신 state) =====
    {
        int NH=16,D=128; size_t ssz=(size_t)NH*D*D, row=(size_t)NH*D;
        std::vector<float> Q((size_t)B*row),Kk((size_t)B*row),Vv((size_t)B*row),G((size_t)B*NH),Bt((size_t)B*NH),Sin((size_t)B*ssz);
        for(auto&v:Q)v=frand(seed); for(auto&v:Kk)v=frand(seed); for(auto&v:Vv)v=frand(seed);
        for(auto&v:G)v=frand(seed)*0.1f; for(auto&v:Bt)v=frand(seed)*0.3f+0.4f; for(auto&v:Sin)v=frand(seed)*0.05f;
        float *dQ=up(Q),*dK=up(Kk),*dV=up(Vv),*dG=up(G),*dB=up(Bt),*dSin=up(Sin);
        float *dOb,*dSout; CK(cudaMalloc(&dOb,(size_t)B*row*4)); CK(cudaMalloc(&dSout,(size_t)B*ssz*4));
        launch_gdn_step_batch(dQ,dK,dV,dG,dB,dSin,dSout,dOb,B,NH,D,NH); CK(cudaDeviceSynchronize());
        std::vector<float> Ob((size_t)B*row),Sb((size_t)B*ssz);
        CK(cudaMemcpy(Ob.data(),dOb,Ob.size()*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(Sb.data(),dSout,Sb.size()*4,cudaMemcpyDeviceToHost));
        std::vector<float> Os((size_t)B*row),Ss((size_t)B*ssz);
        for(int b=0;b<B;b++){ float*dO1,*dSnap; CK(cudaMalloc(&dO1,row*4)); CK(cudaMalloc(&dSnap,ssz*4));  // Sq=1 → snap[0]=새 state
            launch_gdn_step(dQ+(size_t)b*row,dK+(size_t)b*row,dV+(size_t)b*row,dG+(size_t)b*NH,dB+(size_t)b*NH,dSin+(size_t)b*ssz,dSnap,dO1,1,NH,D,NH);
            CK(cudaDeviceSynchronize()); CK(cudaMemcpy(Os.data()+(size_t)b*row,dO1,row*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(Ss.data()+(size_t)b*ssz,dSnap,ssz*4,cudaMemcpyDeviceToHost)); cudaFree(dO1);cudaFree(dSnap); }
        check("gdn_step_batch O ==Bx", maxerr(Ob,Os));
        check("gdn_step_batch state==Bx", maxerr(Sb,Ss));
        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dG);cudaFree(dB);cudaFree(dSin);cudaFree(dOb);cudaFree(dSout);
    }

    printf("\n[test_batch_kernels] %s\n", g_fail?"FAIL":"PASS");
    return g_fail?1:0;
}
