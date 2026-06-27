// 챕터2b STEP1 검증: W4A8 (int8 activation × int4 weight) 커널 == golden_w4a8 (오차0, 구현 정확성).
//   golden = "X를 per-block-32 int8 양자화 + weight q4 dequant 후 정확 dot"(fp64, 인-테스트 계산).
//   1) quantize_x_q8 + gemv_w4a8 == golden_w4a8 (오차0).
//   2) 품질 델타: W4A8 출력 vs W4-fp32A 출력(같은 q4 weight, fp32 X) — 양자화로 얼마나 바뀌나(기록).
// CPU 오라클 격리(torch 불필요). 랜덤 X + 랜덤 q4 weight(여러 shape).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
static int g_fail=0;
static float frand(unsigned&s){ s=s*1664525u+1013904223u; return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f; }

// q4 pack: weight[OUT,IN] fp -> Wq[OUT,IN/2] (low=even, high=odd), sc[OUT,IN/QBLK]=maxabs/7, q=round(w/sc)+8
static void pack_q4(const std::vector<float>&W,int OUT,int IN,std::vector<uint8_t>&Wq,std::vector<float>&sc){
    int nb=IN/QBLK; Wq.assign((size_t)OUT*(IN/2),0); sc.assign((size_t)OUT*nb,0);
    for(int o=0;o<OUT;o++)for(int b=0;b<nb;b++){
        float mx=0; for(int i=0;i<QBLK;i++) mx=fmaxf(mx,fabsf(W[(size_t)o*IN+b*QBLK+i]));
        float s=mx/7.0f; if(s==0)s=1; sc[(size_t)o*nb+b]=s;
        for(int i=0;i<QBLK;i++){ int n=b*QBLK+i; int q=(int)lrintf(W[(size_t)o*IN+n]/s)+8; if(q<0)q=0;if(q>15)q=15;
            uint8_t&by=Wq[(size_t)o*(IN/2)+(n>>1)]; if(n&1) by=(by&0x0F)|(q<<4); else by=(by&0xF0)|q; }
    }
}

static void run(int IN,int OUT,int S,unsigned&seed){
    std::vector<float> X((size_t)S*IN), W((size_t)OUT*IN);
    for(auto&v:X)v=frand(seed)*2.0f; for(auto&v:W)v=frand(seed)*0.5f;
    std::vector<uint8_t> Wq; std::vector<float> sc; pack_q4(W,OUT,IN,Wq,sc);
    int nb=IN/QBLK;

    // ---- CPU golden_w4a8 (fp64): per-block-32 int8 양자화 X, q4 dequant W, 정확 dot ----
    std::vector<int8_t> xqh((size_t)S*IN); std::vector<float> xsh((size_t)S*nb); std::vector<int> xsmh((size_t)S*nb);
    for(int s=0;s<S;s++)for(int b=0;b<nb;b++){
        float mx=0; for(int i=0;i<QBLK;i++) mx=fmaxf(mx,fabsf(X[(size_t)s*IN+b*QBLK+i]));
        float d=mx/127.0f; int ss=0;
        for(int i=0;i<QBLK;i++){ int n=b*QBLK+i; int q=(d==0)?0:(int)lrintf(X[(size_t)s*IN+n]/d); if(q<-127)q=-127;if(q>127)q=127;
            xqh[(size_t)s*IN+n]=(int8_t)q; ss+=q; }
        xsh[(size_t)s*nb+b]=d; xsmh[(size_t)s*nb+b]=ss;
    }
    std::vector<double> gold((size_t)S*OUT), goldA((size_t)S*OUT);
    for(int s=0;s<S;s++)for(int o=0;o<OUT;o++){
        double acc=0, accA=0;
        for(int b=0;b<nb;b++){ double dot=0; float dx=xsh[(size_t)s*nb+b], sw=sc[(size_t)o*nb+b];
            for(int i=0;i<QBLK;i++){ int n=b*QBLK+i; uint8_t by=Wq[(size_t)o*(IN/2)+(n>>1)];
                int wq=(n&1)?(by>>4):(by&0x0F); double wv=(wq-8)*(double)sw;
                dot += (double)xqh[(size_t)s*IN+n]*dx * wv;          // W4A8: int8-X dequant × q4-W
                accA += (double)X[(size_t)s*IN+n] * wv;              // W4-fp32A: fp32 X × q4-W (품질 기준)
            }
            acc += dot;
        }
        gold[(size_t)s*OUT+o]=acc; goldA[(size_t)s*OUT+o]=accA;
    }

    // ---- CUDA: quantize_x_q8 + gemv_w4a8 ----
    float *dX; int8_t *dxq; float *dxs; int32_t *dxsum; uint8_t *dWq; float *dsc,*dY;
    CK(cudaMalloc(&dX,X.size()*4)); CK(cudaMalloc(&dxq,(size_t)S*IN)); CK(cudaMalloc(&dxs,(size_t)S*nb*4));
    CK(cudaMalloc(&dxsum,(size_t)S*nb*4)); CK(cudaMalloc(&dWq,Wq.size())); CK(cudaMalloc(&dsc,sc.size()*4)); CK(cudaMalloc(&dY,(size_t)S*OUT*4));
    CK(cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsc,sc.data(),sc.size()*4,cudaMemcpyHostToDevice));
    launch_quantize_x_q8(dX,dxq,dxs,dxsum,IN,S);
    launch_gemv_w4a8(dxq,dxs,dxsum,dWq,dsc,dY,IN,OUT,S);
    CK(cudaDeviceSynchronize());
    std::vector<float> Y((size_t)S*OUT); CK(cudaMemcpy(Y.data(),dY,Y.size()*4,cudaMemcpyDeviceToHost));

    // quantize 격리 검증: device xq/xs/xsum == CPU
    std::vector<int8_t> xqd((size_t)S*IN); std::vector<float> xsd((size_t)S*nb); std::vector<int> xsmd((size_t)S*nb);
    CK(cudaMemcpy(xqd.data(),dxq,(size_t)S*IN,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(xsd.data(),dxs,(size_t)S*nb*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(xsmd.data(),dxsum,(size_t)S*nb*4,cudaMemcpyDeviceToHost));
    int qmis=0; for(size_t i=0;i<xqh.size();i++) if(xqh[i]!=xqd[i])qmis++;
    int smis=0; for(size_t i=0;i<xsmh.size();i++) if(xsmh[i]!=xsmd[i])smis++;

    double e=0,rel=0; for(size_t i=0;i<Y.size();i++){ e=fmax(e,fabs((double)Y[i]-gold[i])); }
    // 품질 델타: W4A8(gold) vs W4-fp32A(goldA)
    double qd=0; for(size_t i=0;i<gold.size();i++) qd=fmax(qd,fabs(gold[i]-goldA[i]));
    double amax_out=0; for(size_t i=0;i<goldA.size();i++) amax_out=fmax(amax_out,fabs(goldA[i]));

    bool ok=(e<1e-2)&&(qmis==0)&&(smis==0);
    printf("  [IN=%4d OUT=%4d S=%d] kernel==golden_w4a8 err=%.3e  xq_mis=%d xsum_mis=%d  %s\n",
           IN,OUT,S,e,qmis,smis, ok?"OK":"FAIL");
    printf("                       품질델타 |W4A8 - W4fp32A| max=%.3e (출력 max=%.3e, rel=%.2f%%)\n",
           qd,amax_out,100.0*qd/amax_out);
    if(!ok)g_fail=1;
    cudaFree(dX);cudaFree(dxq);cudaFree(dxs);cudaFree(dxsum);cudaFree(dWq);cudaFree(dsc);cudaFree(dY);
}

int main(){
    unsigned seed=12345;
    printf("=== W4A8 kernel vs golden_w4a8 (구현 오차0) + 품질델타(vs W4-fp32A) ===\n");
    run(1024, 256, 1, seed);    // 작은 gemv (q/o proj류)
    run(1024, 2048, 1, seed);   // FFN gate/up류
    run(3584, 1024, 1, seed);   // FFN down (IN=3584)
    run(1024, 4096, 5, seed);   // S=5 (verify batch)
    printf("\n[test_w4a8] %s\n", g_fail?"FAIL":"PASS");
    return g_fail?1:0;
}
