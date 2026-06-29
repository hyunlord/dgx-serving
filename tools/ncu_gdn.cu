// STEP 1 GDN de-risk: gdn_step(baseline, in-loop block_reduce) vs gdn_qk_norm + gdn_step_prenorm.
//   정확성(오차0) + 타이밍(prenorm이 in-loop reduce 제거로 빨라지나). dims = 27B GDN prefill chunk(NH48,D128,NHk16,Sq256).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/gdn.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
int main(){
    const int NH=48, D=128, NHk=16, Sq=256;
    size_t nqk=(size_t)Sq*NHk*D, nv=(size_t)Sq*NH*D, ng=(size_t)Sq*NH, nsnap=(size_t)NH*D*D;
    std::vector<float> Q(nqk),K(nqk),V(nv),G(ng),Bt(ng); unsigned s=7;
    for(auto&x:Q)x=frand(s); for(auto&x:K)x=frand(s); for(auto&x:V)x=frand(s);
    for(auto&x:G)x=-0.3f-0.2f*fabsf(frand(s));               // g_t=exp(G)<1 (decay)
    for(auto&x:Bt)x=1.0f/(1.0f+expf(-frand(s)));             // beta in (0,1)
    float *dQ,*dK,*dV,*dG,*dB,*dQn,*dKn,*dOr,*dOt,*dSnap;
    CK(cudaMalloc(&dQ,nqk*4));CK(cudaMalloc(&dK,nqk*4));CK(cudaMalloc(&dV,nv*4));CK(cudaMalloc(&dG,ng*4));CK(cudaMalloc(&dB,ng*4));
    CK(cudaMalloc(&dQn,nqk*4));CK(cudaMalloc(&dKn,nqk*4));CK(cudaMalloc(&dOr,nv*4));CK(cudaMalloc(&dOt,nv*4));CK(cudaMalloc(&dSnap,nsnap*4));
    CK(cudaMemcpy(dQ,Q.data(),nqk*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dK,K.data(),nqk*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,V.data(),nv*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dG,G.data(),ng*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,Bt.data(),ng*4,cudaMemcpyHostToDevice));
    // baseline + prenorm
    launch_gdn_step(dQ,dK,dV,dG,dB,nullptr,dSnap,dOr,Sq,NH,D,NHk,0);
    launch_gdn_qk_norm(dQ,dK,dQn,dKn,Sq*NHk,D);
    launch_gdn_step_prenorm(dQn,dKn,dV,dG,dB,nullptr,dSnap,dOt,Sq,NH,D,NHk,0);
    CK(cudaDeviceSynchronize());
    // 정확성 (오차0)
    std::vector<float> Or(nv),Ot(nv);
    CK(cudaMemcpy(Or.data(),dOr,nv*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(Ot.data(),dOt,nv*4,cudaMemcpyDeviceToHost));
    double e=0,rms=0; for(size_t i=0;i<nv;i++){e=fmax(e,fabs((double)Ot[i]-Or[i]));rms+=Or[i]*Or[i];} rms=sqrt(rms/nv);
    printf("[gdn de-risk] prenorm vs baseline: max err=%.3e (rel=%.2e) %s\n",e,e/rms, e/rms<1e-5?"OK(오차0)":"FAIL");
    // 타이밍
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); int R=30; float tb=0,tp=0;
    for(int w=0;w<3;w++){ launch_gdn_step(dQ,dK,dV,dG,dB,nullptr,dSnap,dOr,Sq,NH,D,NHk,0);} CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for(int r=0;r<R;r++) launch_gdn_step(dQ,dK,dV,dG,dB,nullptr,dSnap,dOr,Sq,NH,D,NHk,0); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tb,a,b);
    cudaEventRecord(a); for(int r=0;r<R;r++){ launch_gdn_qk_norm(dQ,dK,dQn,dKn,Sq*NHk,D); launch_gdn_step_prenorm(dQn,dKn,dV,dG,dB,nullptr,dSnap,dOt,Sq,NH,D,NHk,0);} cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tp,a,b);
    printf("[gdn de-risk] baseline gdn_step: %.3f ms/call | prenorm(norm+step): %.3f ms/call | speedup %.2fx\n", tb/R, tp/R, tb/tp);
    // k-split (occupancy↑): norm + ksplit step
    launch_gdn_qk_norm(dQ,dK,dQn,dKn,Sq*NHk,D);
    launch_gdn_step_ksplit<128,2>(dQn,dKn,dV,dG,dB,nullptr,dSnap,dOt,Sq,NH,NHk,0); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(Ot.data(),dOt,nv*4,cudaMemcpyDeviceToHost));
    {double e2=0; for(size_t i=0;i<nv;i++) e2=fmax(e2,fabs((double)Ot[i]-Or[i])); printf("[gdn ksplit<2>] err=%.3e (rel=%.2e) %s\n",e2,e2/rms,e2/rms<1e-5?"OK":"FAIL");}
    launch_gdn_step_ksplit<128,4>(dQn,dKn,dV,dG,dB,nullptr,dSnap,dOt,Sq,NH,NHk,0); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(Ot.data(),dOt,nv*4,cudaMemcpyDeviceToHost));
    {double e4=0; for(size_t i=0;i<nv;i++) e4=fmax(e4,fabs((double)Ot[i]-Or[i])); printf("[gdn ksplit<4>] err=%.3e (rel=%.2e) %s\n",e4,e4/rms,e4/rms<1e-5?"OK":"FAIL");}
    float tk2=0,tk4=0;
    for(int w=0;w<3;w++){launch_gdn_qk_norm(dQ,dK,dQn,dKn,Sq*NHk,D);launch_gdn_step_ksplit<128,2>(dQn,dKn,dV,dG,dB,nullptr,dSnap,dOt,Sq,NH,NHk,0);} CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for(int r=0;r<R;r++){launch_gdn_qk_norm(dQ,dK,dQn,dKn,Sq*NHk,D);launch_gdn_step_ksplit<128,2>(dQn,dKn,dV,dG,dB,nullptr,dSnap,dOt,Sq,NH,NHk,0);} cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tk2,a,b);
    cudaEventRecord(a); for(int r=0;r<R;r++){launch_gdn_qk_norm(dQ,dK,dQn,dKn,Sq*NHk,D);launch_gdn_step_ksplit<128,4>(dQn,dKn,dV,dG,dB,nullptr,dSnap,dOt,Sq,NH,NHk,0);} cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tk4,a,b);
    printf("[gdn de-risk] ksplit<2>(norm+step): %.3f ms | ksplit<4>: %.3f ms | vs baseline %.3f → %.2fx / %.2fx\n", tk2/R, tk4/R, tb/R, tb/tk2, tb/tk4);
    // 결정성 30× bitwise — ksplit 부분합 결합은 고정순서 shared reduction(atomic/cp.async race 없음)이라 bit-identical 기대.
    //   (Phase C 비결정성은 matmul cp.async였음; GDN ksplit엔 cp.async 없음.) 단발 race는 못 잡으니 명시 가드.
    std::vector<float> Oref(nv), Ocmp(nv);
    launch_gdn_qk_norm(dQ,dK,dQn,dKn,Sq*NHk,D);
    launch_gdn_step_ksplit<128,2>(dQn,dKn,dV,dG,dB,nullptr,dSnap,dOt,Sq,NH,NHk,0); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(Oref.data(),dOt,nv*4,cudaMemcpyDeviceToHost));
    int detfail=0;
    for(int r=0;r<30;r++){
        launch_gdn_qk_norm(dQ,dK,dQn,dKn,Sq*NHk,D);
        launch_gdn_step_ksplit<128,2>(dQn,dKn,dV,dG,dB,nullptr,dSnap,dOt,Sq,NH,NHk,0); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(Ocmp.data(),dOt,nv*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<nv;i++) if(Ocmp[i]!=Oref[i]){detfail++;break;}
    }
    printf("[gdn ksplit<2>] 결정성 30× bitwise: %s (%d/30 run 불일치)\n", detfail==0?"OK(bit-identical)":"FAIL", detfail);
    return 0;
}
