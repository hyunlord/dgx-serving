// 챕터6 STEP 1: rope 엔진 내부 생성 검증 (골든 대조 + 32K no-NaN). 골든 rope 의존 제거.
//   골든 rope_cos/sin(HF에서 캡처, 정확)과 생성 테이블을 오차0 대조 → theta/공식 확정.
//   theta 후보 검증(추출 manifest 1e4 vs config 1e7) — 골든과 맞는 쪽이 진짜.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../kernels/rope.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen(p.c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static double maxd(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    std::string GD=argc>1?argv[1]:"artifacts/qwen35-27b/golden";
    const int S=5, rot=64;
    auto gc=rf(GD+"/rope_cos.bin",(size_t)S*rot), gs=rf(GD+"/rope_sin.bin",(size_t)S*rot);

    float *dc,*ds; CK(cudaMalloc(&dc,(size_t)S*rot*4)); CK(cudaMalloc(&ds,(size_t)S*rot*4));
    printf("[rope_gen] 골든 대조 (S=%d rot=%d) — theta 후보별:\n",S,rot);
    double best=1e9; float bestt=0;
    for(float theta : {1.0e4f, 1.0e6f, 1.0e7f}){
        launch_gen_rope(dc,ds,S,rot,theta); CK(cudaDeviceSynchronize());
        std::vector<float> c((size_t)S*rot),s((size_t)S*rot);
        CK(cudaMemcpy(c.data(),dc,c.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s.data(),ds,s.size()*4,cudaMemcpyDeviceToHost));
        double e=fmax(maxd(c,gc),maxd(s,gs));
        printf("    theta=%.0e : cos/sin max err=%.3e %s\n",theta,e,e<1e-5?"← 일치":"");
        if(e<best){best=e;bestt=theta;}
    }
    bool match = best < 1e-5;
    printf("  최적 theta=%.0e (err=%.2e)\n",bestt,best);

    // 32K no-NaN sanity
    const int LONG=32768;
    float *lc,*ls; CK(cudaMalloc(&lc,(size_t)LONG*rot*4)); CK(cudaMalloc(&ls,(size_t)LONG*rot*4));
    launch_gen_rope(lc,ls,LONG,rot,bestt); CK(cudaDeviceSynchronize());
    std::vector<float> hc((size_t)LONG*rot); CK(cudaMemcpy(hc.data(),lc,hc.size()*4,cudaMemcpyDeviceToHost));
    int nan=0; for(float v:hc) if(std::isnan(v)||std::isinf(v)) nan++;
    printf("  32K 생성 no-NaN: %s (NaN/Inf=%d, |범위|<=1: %s)\n",nan==0?"OK":"FAIL",nan,
           (fabs(hc[(size_t)(LONG-1)*rot])<=1.0f)?"OK":"X");

    bool pass = match && nan==0;
    printf("[test_rope_gen] %s (theta=%.0e가 골든과 오차0 → 엔진 rope 생성 확정)\n",pass?"PASS":"FAIL",bestt);
    return pass?0:1;
}
