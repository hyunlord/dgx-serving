// 커널 단위 테스트: 각 공용 커널을 작은 입력에 대해 CPU 레퍼런스(오라클)와 대조.
// torch 불필요 — CPU에서 같은 수학을 독립 구현해 비교. 오차 0(부동소수 잔차) 확인.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cfloat>
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_q4.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../runtime/config.h"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static int g_fail = 0;
static void check(const char* name, double maxerr, double tol) {
    printf("  %-12s max abs err = %.3e  (tol %.0e)  %s\n", name, maxerr, tol,
           maxerr <= tol ? "OK" : "FAIL");
    if (maxerr > tol) g_fail = 1;
}
static float frand(unsigned& s){ s=s*1664525u+1013904223u; return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f; } // [-1,1)

// ---- CPU int4 양자화 (tools/extract_weights.py 와 동일 스킴) ----
static void cpu_quant_int4(const std::vector<float>& w, std::vector<uint8_t>& packed, std::vector<float>& scale){
    int n=w.size(), nblk=n/QBLK; packed.resize(n/2); scale.resize(nblk);
    std::vector<uint8_t> q(n);
    for(int b=0;b<nblk;b++){ float mx=0; for(int j=0;j<QBLK;j++) mx=fmaxf(mx,fabsf(w[b*QBLK+j]));
        float sc=mx/7.0f; if(sc==0) sc=1e-8f; scale[b]=sc;
        for(int j=0;j<QBLK;j++){ int v=(int)lroundf(w[b*QBLK+j]/sc)+8; if(v<0)v=0; if(v>15)v=15; q[b*QBLK+j]=(uint8_t)v; } }
    for(int i=0;i<n/2;i++) packed[i]=(uint8_t)(q[2*i] | (q[2*i+1]<<4));
}
static float deq(uint8_t b,int hi,float sc){ int q=hi?(b>>4):(b&0x0F); return ((float)q-8.0f)*sc; }

int main(){
    unsigned seed=12345;

    // ===== rmsnorm =====
    {
        int rows=3, hh=2048; float eps=1e-6f;
        std::vector<float> X(rows*hh), W(hh), Y(rows*hh), ref(rows*hh);
        for(auto&v:X)v=frand(seed); for(auto&v:W)v=frand(seed)*0.5f+1.0f;
        for(int r=0;r<rows;r++){ double ss=0; for(int i=0;i<hh;i++)ss+=(double)X[r*hh+i]*X[r*hh+i];
            float inv=1.0f/sqrtf((float)(ss/hh)+eps); for(int i=0;i<hh;i++) ref[r*hh+i]=X[r*hh+i]*inv*W[i]; }
        float *dX,*dW,*dY; CK(cudaMalloc(&dX,X.size()*4)); CK(cudaMalloc(&dW,W.size()*4)); CK(cudaMalloc(&dY,Y.size()*4));
        CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dW,W.data(),W.size()*4,cudaMemcpyHostToDevice));
        launch_rmsnorm(dX,dW,dY,rows,hh,eps); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<Y.size();i++) e=fmax(e,fabs((double)Y[i]-ref[i]));
        check("rmsnorm",e,1e-4); cudaFree(dX);cudaFree(dW);cudaFree(dY);
    }

    // ===== gemv_q4 ===== (kernel dequant vs CPU dequant matmul — 동일 weight)
    {
        int IN=512, OUT=128, S=2;
        std::vector<float> Wf(OUT*IN), X(S*IN), Y(S*OUT), ref(S*OUT);
        for(auto&v:Wf)v=frand(seed)*0.3f; for(auto&v:X)v=frand(seed);
        std::vector<uint8_t> packed; std::vector<float> scale; // per row 양자화 (블록은 row 내부)
        cpu_quant_int4(Wf,packed,scale);
        for(int s=0;s<S;s++) for(int o=0;o<OUT;o++){ double acc=0;
            for(int i=0;i<IN;i++){ int gi=o*IN+i; float wq=deq(packed[gi/2], gi&1, scale[gi/QBLK]); acc+=X[s*IN+i]*wq; }
            ref[s*OUT+o]=(float)acc; }
        float *dX,*dY,*dsc; uint8_t* dWq;
        CK(cudaMalloc(&dX,X.size()*4)); CK(cudaMalloc(&dY,Y.size()*4)); CK(cudaMalloc(&dWq,packed.size())); CK(cudaMalloc(&dsc,scale.size()*4));
        CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWq,packed.data(),packed.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsc,scale.data(),scale.size()*4,cudaMemcpyHostToDevice));
        launch_gemv_q4(dX,dWq,dsc,dY,IN,OUT,S); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<Y.size();i++) e=fmax(e,fabs((double)Y[i]-ref[i]));
        check("gemv_q4",e,1e-3); cudaFree(dX);cudaFree(dY);cudaFree(dWq);cudaFree(dsc);
    }

    // ===== rope ===== (HD=128, S=4, NH=2)
    {
        int S=4, NH=2, HD=128, half=HD/2;
        std::vector<float> T(S*NH*HD), cosv(S*HD), sinv(S*HD), ref;
        for(auto&v:T)v=frand(seed);
        for(int s=0;s<S;s++) for(int i=0;i<HD;i++){ float ang=s*0.1f*(1.0f+(i%half)); cosv[s*HD+i]=cosf(ang); sinv[s*HD+i]=sinf(ang); }
        ref=T;
        for(int s=0;s<S;s++)for(int h=0;h<NH;h++)for(int i=0;i<half;i++){ float* b=&ref[((s*NH)+h)*HD];
            float t1=b[i],t2=b[i+half],c=cosv[s*HD+i],sn=sinv[s*HD+i]; b[i]=t1*c-t2*sn; b[i+half]=t2*c+t1*sn; }
        float *dT,*dc,*ds; CK(cudaMalloc(&dT,T.size()*4)); CK(cudaMalloc(&dc,cosv.size()*4)); CK(cudaMalloc(&ds,sinv.size()*4));
        CK(cudaMemcpy(dT,T.data(),T.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dc,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));
        launch_rope(dT,dc,ds,S,NH,HD); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(T.data(),dT,T.size()*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<T.size();i++) e=fmax(e,fabs((double)T[i]-ref[i]));
        check("rope",e,1e-4); cudaFree(dT);cudaFree(dc);cudaFree(ds);
    }

    // ===== attention ===== (GQA causal, S=5, NQ=4, NKV=2, HD=128)
    {
        int S=5,NQ=4,NKV=2,HD=128,GRP=NQ/NKV;
        std::vector<float> Q(S*NQ*HD),K(S*NKV*HD),Vv(S*NKV*HD),O(S*NQ*HD),ref(S*NQ*HD);
        for(auto&v:Q)v=frand(seed)*0.2f; for(auto&v:K)v=frand(seed)*0.2f; for(auto&v:Vv)v=frand(seed);
        for(int qi=0;qi<S;qi++)for(int h=0;h<NQ;h++){ int kv=h/GRP; std::vector<float> sc(qi+1); float m=-FLT_MAX;
            for(int j=0;j<=qi;j++){ double d=0; for(int t=0;t<HD;t++)d+=(double)Q[(qi*NQ+h)*HD+t]*K[(j*NKV+kv)*HD+t]; sc[j]=(float)(d/sqrt((double)HD)); m=fmaxf(m,sc[j]); }
            float sum=0; for(int j=0;j<=qi;j++){ sc[j]=expf(sc[j]-m); sum+=sc[j]; }
            for(int d=0;d<HD;d++){ double acc=0; for(int j=0;j<=qi;j++) acc+=(sc[j]/sum)*Vv[(j*NKV+kv)*HD+d]; ref[(qi*NQ+h)*HD+d]=(float)acc; } }
        float *dQ,*dK,*dV,*dO; CK(cudaMalloc(&dQ,Q.size()*4)); CK(cudaMalloc(&dK,K.size()*4)); CK(cudaMalloc(&dV,Vv.size()*4)); CK(cudaMalloc(&dO,O.size()*4));
        CK(cudaMemcpy(dQ,Q.data(),Q.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dK,K.data(),K.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dV,Vv.data(),Vv.size()*4,cudaMemcpyHostToDevice));
        launch_attn(dQ,dK,dV,dO,S,NQ,NKV,HD); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(O.data(),dO,O.size()*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<O.size();i++) e=fmax(e,fabs((double)O[i]-ref[i]));
        check("attention",e,1e-4); cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    }

    // ===== swiglu + add =====
    {
        int n=4096; std::vector<float> g(n),u(n),Y(n),ref(n);
        for(auto&v:g)v=frand(seed); for(auto&v:u)v=frand(seed);
        for(int i=0;i<n;i++){ float s=g[i]/(1.0f+expf(-g[i])); ref[i]=s*u[i]; }
        float *dg,*du,*dY; CK(cudaMalloc(&dg,n*4)); CK(cudaMalloc(&du,n*4)); CK(cudaMalloc(&dY,n*4));
        CK(cudaMemcpy(dg,g.data(),n*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(du,u.data(),n*4,cudaMemcpyHostToDevice));
        launch_swiglu(dg,du,dY,n); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Y.data(),dY,n*4,cudaMemcpyDeviceToHost));
        double e=0; for(int i=0;i<n;i++) e=fmax(e,fabs((double)Y[i]-ref[i])); check("swiglu",e,1e-5);
        cudaFree(dg);cudaFree(du);cudaFree(dY);
    }

    printf(g_fail? "[test_kernels] FAIL\n":"[test_kernels] PASS\n");
    return g_fail;
}
