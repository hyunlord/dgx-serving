// 챕터9 STEP 1.2: tree-aware gdn_step 정확성 게이트.
//   ① tree(chain parent=t-1) == 기존 linear gdn_step_ksplit  → bit-identical (reload-from-snapshot 메커니즘 검증).
//   ② tree(branch) sibling이 부모서 fork → 2-노드 chain 레퍼런스와 일치 (분기 정확성).
//   커널 단위(모델 로드 없음, 랜덤 입력). D=128(GDN head dim), KSPLIT=2.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/gdn.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static uint32_t rs=999; static float frand(){ rs=rs*1664525u+1013904223u; return ((rs>>8)&0xFFFF)/65536.0f-0.5f; }
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static std::vector<float> dn(const float*d,size_t n){std::vector<float>h(n);CK(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost));return h;}
static double maxae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(){
    const int D=128, KSPLIT=2, NH=4, NHk=4, Sq=5;
    std::vector<float> Q((size_t)Sq*NHk*D), K((size_t)Sq*NHk*D), V((size_t)Sq*NH*D), G((size_t)Sq*NH), Bt((size_t)Sq*NH);
    for(auto&x:Q)x=frand(); for(auto&x:K)x=frand(); for(auto&x:V)x=frand();
    for(auto&x:G)x=frand()*0.1f-0.2f; for(auto&x:Bt)x=frand()*0.5f+0.5f;  // G<0(decay exp<1), beta~0.5
    float *dQ=up(Q),*dK=up(K),*dV=up(V),*dG=up(G),*dBt=up(Bt);
    float *dO_lin,*dS_lin,*dO_tr,*dS_tr;
    CK(cudaMalloc(&dO_lin,(size_t)Sq*NH*D*4)); CK(cudaMalloc(&dS_lin,(size_t)Sq*NH*D*D*4));
    CK(cudaMalloc(&dO_tr,(size_t)Sq*NH*D*4));  CK(cudaMalloc(&dS_tr,(size_t)Sq*NH*D*D*4));

    // ① linear vs tree(chain): parent=[-1,0,1,2,3]
    launch_gdn_step_ksplit<D,KSPLIT>(dQ,dK,dV,dG,dBt,nullptr, dS_lin,dO_lin, Sq,NH,NHk, /*snapshot_all*/1);
    std::vector<int> pc={-1,0,1,2,3};
    int*dparent; CK(cudaMalloc(&dparent,Sq*4)); CK(cudaMemcpy(dparent,pc.data(),Sq*4,cudaMemcpyHostToDevice));
    launch_gdn_step_ksplit_tree<D,KSPLIT>(dQ,dK,dV,dG,dBt,nullptr,dparent, dS_tr,dO_tr, Sq,NH,NHk);
    CK(cudaDeviceSynchronize());
    double eO=maxae(dn(dO_lin,(size_t)Sq*NH*D),dn(dO_tr,(size_t)Sq*NH*D));
    double eS=maxae(dn(dS_lin,(size_t)Sq*NH*D*D),dn(dS_tr,(size_t)Sq*NH*D*D));
    printf("=== ① tree(chain parent=t-1) vs linear gdn_step_ksplit ===\n");
    printf("  O max abs err = %.3e | snapshot max abs err = %.3e\n",eO,eS);

    // ② branch: parent=[-1,0,0,1,1] (node0 root; 1,2=0의 자식; 3,4=1의 자식)
    //    node2(=1의 sibling)의 snapshot이 node0서 fork됐나 → [node0,node2] 2-노드 chain 레퍼런스와 일치 확인.
    std::vector<int> pb={-1,0,0,1,1}; CK(cudaMemcpy(dparent,pb.data(),Sq*4,cudaMemcpyHostToDevice));
    float *dO_br,*dS_br; CK(cudaMalloc(&dO_br,(size_t)Sq*NH*D*4)); CK(cudaMalloc(&dS_br,(size_t)Sq*NH*D*D*4));
    launch_gdn_step_ksplit_tree<D,KSPLIT>(dQ,dK,dV,dG,dBt,nullptr,dparent, dS_br,dO_br, Sq,NH,NHk);
    // 레퍼런스: node0, node2를 2-노드 chain으로 (Q/K/V의 0번,2번 입력). parent=[-1,0].
    std::vector<float> Q2(2*NHk*D),K2(2*NHk*D),V2(2*NH*D),G2(2*NH),Bt2(2*NH);
    auto cp=[&](std::vector<float>&dst,const std::vector<float>&src,int node,int rows,int cols){
        for(int r=0;r<rows;r++)for(int c=0;c<cols;c++) dst[(size_t)r*cols+c]=src[((size_t)node*rows+r)*cols+c]; };
    // node0 → ref idx0, node2 → ref idx1
    for(int c=0;c<NHk*D;c++){Q2[c]=Q[(size_t)0*NHk*D+c]; Q2[NHk*D+c]=Q[(size_t)2*NHk*D+c];
                             K2[c]=K[(size_t)0*NHk*D+c]; K2[NHk*D+c]=K[(size_t)2*NHk*D+c];}
    for(int c=0;c<NH*D;c++){V2[c]=V[(size_t)0*NH*D+c]; V2[NH*D+c]=V[(size_t)2*NH*D+c];}
    for(int c=0;c<NH;c++){G2[c]=G[0*NH+c]; G2[NH+c]=G[2*NH+c]; Bt2[c]=Bt[0*NH+c]; Bt2[NH+c]=Bt[2*NH+c];}
    float *dQ2=up(Q2),*dK2=up(K2),*dV2=up(V2),*dG2=up(G2),*dBt2=up(Bt2),*dO2,*dS2;
    CK(cudaMalloc(&dO2,(size_t)2*NH*D*4)); CK(cudaMalloc(&dS2,(size_t)2*NH*D*D*4));
    launch_gdn_step_ksplit<D,KSPLIT>(dQ2,dK2,dV2,dG2,dBt2,nullptr, dS2,dO2, 2,NH,NHk, 1);
    CK(cudaDeviceSynchronize());
    // branch node2 snapshot(slot2) vs ref node1 snapshot(slot1); branch node2 O vs ref node1 O
    auto sbr=dn(dS_br,(size_t)Sq*NH*D*D), sref=dn(dS2,(size_t)2*NH*D*D);
    auto obr=dn(dO_br,(size_t)Sq*NH*D), oref=dn(dO2,(size_t)2*NH*D);
    std::vector<float> br_n2_S(sbr.begin()+(size_t)2*NH*D*D, sbr.begin()+(size_t)3*NH*D*D);
    std::vector<float> ref_n1_S(sref.begin()+(size_t)1*NH*D*D, sref.begin()+(size_t)2*NH*D*D);
    std::vector<float> br_n2_O(obr.begin()+(size_t)2*NH*D, obr.begin()+(size_t)3*NH*D);
    std::vector<float> ref_n1_O(oref.begin()+(size_t)1*NH*D, oref.begin()+(size_t)2*NH*D);
    double eSb=maxae(br_n2_S,ref_n1_S), eOb=maxae(br_n2_O,ref_n1_O);
    printf("=== ② branch node2(fork from node0) vs [node0,node2] chain 레퍼런스 ===\n");
    printf("  node2 state err = %.3e | node2 O err = %.3e\n",eSb,eOb);

    bool pass=(eO<1e-6)&&(eS<1e-6)&&(eSb<1e-6)&&(eOb<1e-6);
    printf("\n[test_gdntree] %s (chain==linear + branch fork 정확)\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
