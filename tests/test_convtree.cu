// 챕터9 STEP 2.2: tree-aware conv1d 정확성 게이트 (STEP 1 gdn_step과 같은 패턴).
//   ① tree(chain parent=t-1) conv == 기존 linear conv1d  → bit-identical (parent walk가 선형 prev와 일치).
//   ② branch sibling conv(node0서 fork) == [node0,node2] ancestor-path chain 레퍼런스 bit-identical.
//   커널 단위(모델 로드 없음, 랜덤). C=128, K=4(conv kernel).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/gdn.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static uint32_t rs=4242; static float frand(){ rs=rs*1664525u+1013904223u; return ((rs>>8)&0xFFFF)/65536.0f-0.5f; }
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static std::vector<float> dn(const float*d,size_t n){std::vector<float>h(n);CK(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost));return h;}
static double maxae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(){
    const int C=128, K=4, Sq=5;
    std::vector<float> in((size_t)Sq*C), prev((size_t)(K-1)*C), w((size_t)C*K);
    for(auto&x:in)x=frand(); for(auto&x:prev)x=frand(); for(auto&x:w)x=frand();
    float *din=up(in),*dprev=up(prev),*dw=up(w);
    float *dlin,*dtree; CK(cudaMalloc(&dlin,(size_t)Sq*C*4)); CK(cudaMalloc(&dtree,(size_t)Sq*C*4));
    int*dparent; CK(cudaMalloc(&dparent,Sq*4));

    // ① chain parent=[-1,0,1,2,3] vs linear
    launch_gdn_conv1d_silu_decode(din,dprev,dw,dlin,Sq,C,K);
    std::vector<int> pc={-1,0,1,2,3}; CK(cudaMemcpy(dparent,pc.data(),Sq*4,cudaMemcpyHostToDevice));
    launch_gdn_conv1d_silu_decode_tree(din,dprev,dparent,dw,dtree,Sq,C,K);
    CK(cudaDeviceSynchronize());
    double e1=maxae(dn(dlin,(size_t)Sq*C),dn(dtree,(size_t)Sq*C));
    printf("=== ① tree(chain parent=t-1) conv vs linear conv1d ===\n  max abs err = %.3e\n",e1);

    // ② branch parent=[-1,0,0,1,1] — node2(0의 자식) conv vs [node0,node2] ancestor chain 레퍼런스
    std::vector<int> pb={-1,0,0,1,1}; CK(cudaMemcpy(dparent,pb.data(),Sq*4,cudaMemcpyHostToDevice));
    float *dbr; CK(cudaMalloc(&dbr,(size_t)Sq*C*4));
    launch_gdn_conv1d_silu_decode_tree(din,dprev,dparent,dw,dbr,Sq,C,K);
    // 레퍼런스: in2 = [in[node0], in[node2]], parent=[-1,0] (2-노드 chain). 같은 prev.
    std::vector<float> in2(2*C);
    for(int c=0;c<C;c++){ in2[c]=in[(size_t)0*C+c]; in2[C+c]=in[(size_t)2*C+c]; }
    float *din2=up(in2),*dref; CK(cudaMalloc(&dref,(size_t)2*C*4));
    int rp[2]={-1,0}; int*drp; CK(cudaMalloc(&drp,2*4)); CK(cudaMemcpy(drp,rp,2*4,cudaMemcpyHostToDevice));
    launch_gdn_conv1d_silu_decode_tree(din2,dprev,drp,dw,dref,2,C,K);
    CK(cudaDeviceSynchronize());
    auto br=dn(dbr,(size_t)Sq*C), ref=dn(dref,(size_t)2*C);
    std::vector<float> br_n2(br.begin()+(size_t)2*C, br.begin()+(size_t)3*C);   // branch node2
    std::vector<float> ref_n1(ref.begin()+(size_t)1*C, ref.begin()+(size_t)2*C); // ref node1(=node2 in chain)
    double e2=maxae(br_n2,ref_n1);
    printf("=== ② branch node2(node0서 fork) conv vs [node0,node2] ancestor chain ===\n  node2 conv err = %.3e\n",e2);

    bool pass=(e1<1e-6)&&(e2<1e-6);
    printf("\n[test_convtree] %s (chain==linear + branch ancestor-path 정확)\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
