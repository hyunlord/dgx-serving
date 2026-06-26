// Qwen3.5 STEP 1 커널 단위 테스트: rmsnorm_zc(1+w), rope_partial(앞 rot_dim), gate_sigmoid.
// CPU 오라클과 격리 대조 (torch 불필요). 1단계 attention/gemv_q4 는 그대로 재사용(여기선 미반복).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../kernels/rmsnorm.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/elementwise.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
static int g_fail=0;
static void check(const char* n,double e,double tol){ printf("  %-16s max abs err=%.3e (tol %.0e) %s\n",n,e,tol,e<=tol?"OK":"FAIL"); if(e>tol)g_fail=1; }
static float frand(unsigned&s){ s=s*1664525u+1013904223u; return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f; }

int main(){
    unsigned seed=777;

    // ===== rmsnorm_zc (1+w), 일반 norm 형태: rows=3, hh=1024 =====
    {
        int rows=3, hh=1024; float eps=1e-6f;
        std::vector<float> X(rows*hh),W(hh),Y(rows*hh),ref(rows*hh);
        for(auto&v:X)v=frand(seed); for(auto&v:W)v=frand(seed)*0.3f;  // weight는 0 근처(zero-centered)
        for(int r=0;r<rows;r++){ double ss=0; for(int i=0;i<hh;i++)ss+=(double)X[r*hh+i]*X[r*hh+i];
            float inv=1.0f/sqrtf((float)(ss/hh)+eps); for(int i=0;i<hh;i++) ref[r*hh+i]=X[r*hh+i]*inv*(1.0f+W[i]); }
        float *dX,*dW,*dY; CK(cudaMalloc(&dX,X.size()*4));CK(cudaMalloc(&dW,W.size()*4));CK(cudaMalloc(&dY,Y.size()*4));
        CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dW,W.data(),W.size()*4,cudaMemcpyHostToDevice));
        launch_rmsnorm_zc(dX,dW,dY,rows,hh,eps); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<Y.size();i++)e=fmax(e,fabs((double)Y[i]-ref[i])); check("rmsnorm_zc",e,1e-4);
        cudaFree(dX);cudaFree(dW);cudaFree(dY);
    }

    // ===== rmsnorm_zc as q/k_norm: rows=S*NH per-head, hh=head_dim=256 =====
    {
        int S=5,NH=8,HD=256; int rows=S*NH; float eps=1e-6f;
        std::vector<float> X(rows*HD),W(HD),Y(rows*HD),ref(rows*HD);
        for(auto&v:X)v=frand(seed); for(auto&v:W)v=frand(seed)*0.3f;
        for(int r=0;r<rows;r++){ double ss=0; for(int i=0;i<HD;i++)ss+=(double)X[r*HD+i]*X[r*HD+i];
            float inv=1.0f/sqrtf((float)(ss/HD)+eps); for(int i=0;i<HD;i++) ref[r*HD+i]=X[r*HD+i]*inv*(1.0f+W[i]); }
        float *dX,*dW,*dY; CK(cudaMalloc(&dX,X.size()*4));CK(cudaMalloc(&dW,W.size()*4));CK(cudaMalloc(&dY,Y.size()*4));
        CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dW,W.data(),W.size()*4,cudaMemcpyHostToDevice));
        launch_rmsnorm_zc(dX,dW,dY,rows,HD,eps); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<Y.size();i++)e=fmax(e,fabs((double)Y[i]-ref[i])); check("qk_norm(head)",e,1e-4);
        cudaFree(dX);cudaFree(dW);cudaFree(dY);
    }

    // ===== rope_partial: HD=256, rot_dim=64 (앞 64만 회전, [64:256] 통과) =====
    {
        int S=4,NH=2,HD=256,rot=64,half=rot/2;
        std::vector<float> T(S*NH*HD),cosv(S*rot),sinv(S*rot),ref;
        for(auto&v:T)v=frand(seed);
        for(int s=0;s<S;s++)for(int i=0;i<rot;i++){ float ang=s*0.2f*(1.0f+(i%half)); cosv[s*rot+i]=cosf(ang); sinv[s*rot+i]=sinf(ang); }
        ref=T;
        for(int s=0;s<S;s++)for(int h=0;h<NH;h++)for(int i=0;i<half;i++){ float* b=&ref[((s*NH)+h)*HD];
            float t1=b[i],t2=b[i+half],c=cosv[s*rot+i],sn=sinv[s*rot+i]; b[i]=t1*c-t2*sn; b[i+half]=t2*c+t1*sn; }
        // dims [rot:HD) must be unchanged — ref leaves them as-is.
        float *dT,*dc,*ds; CK(cudaMalloc(&dT,T.size()*4));CK(cudaMalloc(&dc,cosv.size()*4));CK(cudaMalloc(&ds,sinv.size()*4));
        CK(cudaMemcpy(dT,T.data(),T.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));
        launch_rope_partial(dT,dc,ds,S,NH,HD,rot); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(T.data(),dT,T.size()*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<T.size();i++)e=fmax(e,fabs((double)T[i]-ref[i]));
        // 통과영역이 진짜 안 바뀌었는지도 확인 (위 maxabs에 포함됨)
        check("rope_partial",e,1e-4);
        cudaFree(dT);cudaFree(dc);cudaFree(ds);
    }

    // ===== gate_sigmoid: y *= sigmoid(gate) =====
    {
        int n=4096; std::vector<float> Y(n),G(n),ref(n);
        for(auto&v:Y)v=frand(seed); for(auto&v:G)v=frand(seed);
        for(int i=0;i<n;i++) ref[i]=Y[i]*(1.0f/(1.0f+expf(-G[i])));
        float *dY,*dG; CK(cudaMalloc(&dY,n*4));CK(cudaMalloc(&dG,n*4));
        CK(cudaMemcpy(dY,Y.data(),n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dG,G.data(),n*4,cudaMemcpyHostToDevice));
        launch_gate_sigmoid(dY,dG,n); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Y.data(),dY,n*4,cudaMemcpyDeviceToHost));
        double e=0; for(int i=0;i<n;i++)e=fmax(e,fabs((double)Y[i]-ref[i])); check("gate_sigmoid",e,1e-5);
        cudaFree(dY);cudaFree(dG);
    }

    printf(g_fail?"[test_qwen_kernels] FAIL\n":"[test_qwen_kernels] PASS\n");
    return g_fail;
}
