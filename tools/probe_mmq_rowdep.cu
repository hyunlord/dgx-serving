// 챕터10 Phase2 golden-debug: gemm_mmq_w4a8 row-독립성 격리 검증.
//   verify_batch concat-mmq가 slot-B(rows 7~13) 발산(A OK). matmul row-독립이면 mmq(M=14)[7..13] == mmq(M=7)[0..6](같은 입력행).
//   같으면 → mmq row-독립(버그는 verify_batch plumbing). 다르면 → mmq row-의존(곱 전제 흔들림).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/gemm_mmq.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
int main(int argc,char**argv){
    int Mbig=argc>1?atoi(argv[1]):14, Msmall=argc>2?atoi(argv[2]):7, N=256, K=512;  // slot B = rows [Msmall..Mbig)
    int off=Mbig-Msmall;                    // slot B 시작행(=Msmall, concat서 두번째 슬롯)
    int nblk=K/QBLK, K2=K/2;
    printf("[mmq rowdep] Mbig=%d Msmall=%d(slotB rows %d..%d) N=%d K=%d QBLK=%d\n",Mbig,Msmall,off,Mbig-1,N,K,QBLK);
    // 호스트 입력(결정적): int8 X[Mbig,K], xs[Mbig,nblk], uint8 Wq[N,K2], sw[N,nblk]. slot B(off..)와 solo(0..Msmall) 같은 값 보장.
    std::vector<int8_t> X(Mbig*K); std::vector<float> xs(Mbig*nblk);
    std::vector<uint8_t> Wq(N*K2); std::vector<float> sw(N*nblk);
    for(size_t i=0;i<X.size();i++) X[i]=(int8_t)((i*7+3)%251-125);
    for(size_t i=0;i<xs.size();i++) xs[i]=0.01f+0.001f*(float)(i%17);
    for(size_t i=0;i<Wq.size();i++) Wq[i]=(uint8_t)((i*13+5)%256);
    for(size_t i=0;i<sw.size();i++) sw[i]=0.02f+0.001f*(float)(i%13);
    // solo(M=Msmall)의 입력 = Mbig의 slot-B 행(off..). X_solo[r]=X[(off+r)], xs_solo[r]=xs[(off+r)].
    std::vector<int8_t> Xs(Msmall*K); std::vector<float> xss(Msmall*nblk);
    memcpy(Xs.data(), X.data()+(size_t)off*K, (size_t)Msmall*K);
    memcpy(xss.data(), xs.data()+(size_t)off*nblk, (size_t)Msmall*nblk*4);
    // device
    int8_t *dX,*dXs; float *dxs,*dxss,*dsw,*dYb,*dYs; uint8_t *dWq;
    CK(cudaMalloc(&dX,X.size())); CK(cudaMalloc(&dXs,Xs.size())); CK(cudaMalloc(&dxs,xs.size()*4)); CK(cudaMalloc(&dxss,xss.size()*4));
    CK(cudaMalloc(&dWq,Wq.size())); CK(cudaMalloc(&dsw,sw.size()*4)); CK(cudaMalloc(&dYb,(size_t)Mbig*N*4)); CK(cudaMalloc(&dYs,(size_t)Msmall*N*4));
    CK(cudaMemcpy(dX,X.data(),X.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dXs,Xs.data(),Xs.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxs,xs.data(),xs.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dxss,xss.data(),xss.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsw,sw.data(),sw.size()*4,cudaMemcpyHostToDevice));
    // mmq(M=Mbig) 전체, mmq(M=Msmall) solo(=slot B 행)
    launch_gemm_mmq_w4a8(dX, dxs, dWq, dsw, dYb, Mbig, N, K, 0);
    launch_gemm_mmq_w4a8(dXs, dxss, dWq, dsw, dYs, Msmall, N, K, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> Yb(Mbig*N), Ys(Msmall*N);
    CK(cudaMemcpy(Yb.data(),dYb,Yb.size()*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(Ys.data(),dYs,Ys.size()*4,cudaMemcpyDeviceToHost));
    // 대조: Yb[off+r] vs Ys[r] (slot-B 행). row-독립이면 bit-identical.
    double maxd=0; int nz=0; for(int r=0;r<Msmall;r++)for(int j=0;j<N;j++){ double d=fabs((double)Yb[(size_t)(off+r)*N+j]-Ys[(size_t)r*N+j]); if(d>maxd)maxd=d; if(d>0)nz++; }
    printf("  Yb[slotB rows %d..%d] vs Ys[solo M=%d]: max|Δ|=%.3e, nonzero_elems=%d/%d\n",off,Mbig-1,Msmall,maxd,nz,Msmall*N);
    printf("  판정: %s\n", maxd==0.0 ? "mmq ROW-독립(bit-identical) → 버그는 verify_batch plumbing(인덱싱)" : "mmq ROW-의존 → concat 곱 전제 흔들림(재설계)");
    return 0;
}
