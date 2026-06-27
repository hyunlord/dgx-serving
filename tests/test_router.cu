// 챕터5 STEP 2: MoE router_topk 커널 격리 검증 (CPU 오라클 = HF 정확 경로, torch 불필요).
//   CPU 오라클: softmax(E 전체) → top-K → renorm (HF Qwen3.5-MoE 방식 그대로).
//   커널: top-K logit → softmax-K (수학 항등). idx 집합 일치 + weight 오차0 확인.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../kernels/router_topk.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}

int main(){
    const int S=20, E=256, K=8; unsigned seed=123;   // qwen3.5_moe: 256 expert, top-8
    std::vector<float> lg((size_t)S*E); for(auto&v:lg)v=frand(seed)*3.f;

    // ===== CPU 오라클: HF 경로 (softmax 256 → top-8 → renorm) =====
    std::vector<int> oidx((size_t)S*K); std::vector<float> ow((size_t)S*K);
    for(int s=0;s<S;s++){
        const float*L=&lg[(size_t)s*E];
        float mx=-1e30f; for(int i=0;i<E;i++)mx=fmaxf(mx,L[i]);
        std::vector<double> p(E); double Z=0; for(int i=0;i<E;i++){p[i]=exp((double)L[i]-mx);Z+=p[i];}
        for(int i=0;i<E;i++)p[i]/=Z;
        std::vector<int> ord(E); for(int i=0;i<E;i++)ord[i]=i;
        std::partial_sort(ord.begin(),ord.begin()+K,ord.end(),[&](int a,int b){return p[a]>p[b];});
        double sw=0; for(int k=0;k<K;k++)sw+=p[ord[k]];
        for(int k=0;k<K;k++){oidx[(size_t)s*K+k]=ord[k];ow[(size_t)s*K+k]=(float)(p[ord[k]]/sw);}
    }

    // ===== 커널 =====
    float *dlg,*dw; int *didx;
    CK(cudaMalloc(&dlg,lg.size()*4)); CK(cudaMalloc(&dw,(size_t)S*K*4)); CK(cudaMalloc(&didx,(size_t)S*K*4));
    CK(cudaMemcpy(dlg,lg.data(),lg.size()*4,cudaMemcpyHostToDevice));
    launch_router_topk(dlg,didx,dw,S,E,K); CK(cudaDeviceSynchronize());
    std::vector<int> kidx((size_t)S*K); std::vector<float> kw((size_t)S*K);
    CK(cudaMemcpy(kidx.data(),didx,(size_t)S*K*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(kw.data(),dw,(size_t)S*K*4,cudaMemcpyDeviceToHost));

    // 비교: idx 집합(순서 무관) + weight (idx 정렬 후 매칭)
    int idx_mis=0; double w_err=0, wsum_err=0;
    for(int s=0;s<S;s++){
        std::vector<std::pair<int,float>> o,k;
        for(int j=0;j<K;j++){o.push_back({oidx[(size_t)s*K+j],ow[(size_t)s*K+j]});
                             k.push_back({kidx[(size_t)s*K+j],kw[(size_t)s*K+j]});}
        std::sort(o.begin(),o.end()); std::sort(k.begin(),k.end());
        double ks=0; for(int j=0;j<K;j++){ ks+=k[j].second;
            if(o[j].first!=k[j].first) idx_mis++;
            else w_err=fmax(w_err,fabs((double)o[j].second-k[j].second)); }
        wsum_err=fmax(wsum_err,fabs(ks-1.0));
    }
    printf("[router] S=%d E=%d K=%d | idx 불일치=%d | weight max err=%.3e | sum-to-1 err=%.3e\n",
           S,E,K,idx_mis,w_err,wsum_err);
    bool pass=(idx_mis==0)&&(w_err<1e-5)&&(wsum_err<1e-6);
    printf("[test_router] %s (커널 top8→softmax8 == HF softmax256→top8→renorm)\n",pass?"PASS":"FAIL");
    cudaFree(dlg);cudaFree(dw);cudaFree(didx);
    return pass?0:1;
}
