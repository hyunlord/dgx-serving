// 챕터9 STEP 3.2: tree attn mask 정확성 게이트.
//   ① chain anc(=(1<<(gq+1))-1) == anc=null(기존 causal) → bit-identical (mask 메커니즘이 causal 안 바꿈).
//   ② branch: node가 조상-path만 attend → tree node O == 그 ancestor-path만 KV에 둔 linear ref O.
//   B 커널(flash_attn2bs) 단위, 모델 로드 없음. HD=128, NQ=NKV=4.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../kernels/attention_flash.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static uint32_t rs=77; static float frand(){ rs=rs*1664525u+1013904223u; return ((rs>>8)&0xFFFF)/65536.0f-0.5f; }
static float* upf(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static __nv_bfloat16* upbf(const std::vector<float>&h){std::vector<__nv_bfloat16>t(h.size());for(size_t i=0;i<h.size();i++)t[i]=__float2bfloat16(h[i]);__nv_bfloat16*d;CK(cudaMalloc(&d,h.size()*2));CK(cudaMemcpy(d,t.data(),h.size()*2,cudaMemcpyHostToDevice));return d;}
static std::vector<float> dn(const float*d,size_t n){std::vector<float>h(n);CK(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost));return h;}
static double maxae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(){
    const int HD=128, NQ=4, NKV=4, base=2, Sq=4;        // prefix 2 + tree 4노드, maxlen=6
    const int maxlen=base+Sq;
    std::vector<float> Q((size_t)Sq*NQ*HD); for(auto&x:Q)x=frand();
    std::vector<float> Kc((size_t)maxlen*NKV*HD), Vc((size_t)maxlen*NKV*HD); for(auto&x:Kc)x=frand(); for(auto&x:Vc)x=frand();
    float*dQ=upf(Q); __nv_bfloat16*dK=upbf(Kc),*dV=upbf(Vc);
    float*dO_null,*dO_anc; CK(cudaMalloc(&dO_null,(size_t)Sq*NQ*HD*4)); CK(cudaMalloc(&dO_anc,(size_t)Sq*NQ*HD*4));
    unsigned*danc; CK(cudaMalloc(&danc,Sq*4));

    // ① chain anc=(1<<(gq+1))-1 vs null(causal)
    launch_flash_attn(dQ,dK,dV,dO_null,Sq,NQ,NKV,HD,base,0,nullptr);
    std::vector<unsigned> ac={1,3,7,15};   // gq=0..3: {0},{0,1},{0,1,2},{0,1,2,3} = causal
    CK(cudaMemcpy(danc,ac.data(),Sq*4,cudaMemcpyHostToDevice));
    launch_flash_attn(dQ,dK,dV,dO_anc,Sq,NQ,NKV,HD,base,0,danc);
    CK(cudaDeviceSynchronize());
    double e1=maxae(dn(dO_null,(size_t)Sq*NQ*HD),dn(dO_anc,(size_t)Sq*NQ*HD));
    printf("=== ① chain anc(causal 비트마스크) vs null(기존 causal) ===\n  O max abs err = %.3e\n",e1);

    // ② branch: tree parent=[-1,0,0,1] (R=0; C0=1,C1=2 = R 자식; GC0=3 = C0 자식).
    //   anc: 0→{0}=1, 1→{0,1}=3, 2→{0,2}=5, 3→{0,1,3}=11. node2(C1)는 slot1(C0) mask, prefix+slot0(R)+slot2(C1)만.
    std::vector<unsigned> tb={1,3,5,11}; CK(cudaMemcpy(danc,tb.data(),Sq*4,cudaMemcpyHostToDevice));
    float*dO_tree; CK(cudaMalloc(&dO_tree,(size_t)Sq*NQ*HD*4));
    launch_flash_attn(dQ,dK,dV,dO_tree,Sq,NQ,NKV,HD,base,0,danc);
    // 레퍼런스: linear [prefix, R(slot0), C1(slot1=tree slot2 복사)] — C1이 prefix+R+C1 attend(causal). Sq=2(R,C1).
    std::vector<float> Kc2((size_t)(base+2)*NKV*HD), Vc2((size_t)(base+2)*NKV*HD), Q2((size_t)2*NQ*HD);
    for(int i=0;i<base*NKV*HD;i++){Kc2[i]=Kc[i];Vc2[i]=Vc[i];}                                   // prefix
    for(int i=0;i<NKV*HD;i++){Kc2[base*NKV*HD+i]=Kc[(size_t)(base+0)*NKV*HD+i]; Vc2[base*NKV*HD+i]=Vc[(size_t)(base+0)*NKV*HD+i];}  // R(slot0)
    for(int i=0;i<NKV*HD;i++){Kc2[(size_t)(base+1)*NKV*HD+i]=Kc[(size_t)(base+2)*NKV*HD+i]; Vc2[(size_t)(base+1)*NKV*HD+i]=Vc[(size_t)(base+2)*NKV*HD+i];}  // C1(tree slot2 → linear slot1)
    for(int i=0;i<NQ*HD;i++){Q2[i]=Q[(size_t)0*NQ*HD+i]; Q2[NQ*HD+i]=Q[(size_t)2*NQ*HD+i];}      // R query, C1 query(tree node2)
    float*dQ2=upf(Q2); __nv_bfloat16*dK2=upbf(Kc2),*dV2=upbf(Vc2); float*dO2; CK(cudaMalloc(&dO2,(size_t)2*NQ*HD*4));
    launch_flash_attn(dQ2,dK2,dV2,dO2,2,NQ,NKV,HD,base,0,nullptr);   // causal: linear node1(C1) attends prefix+R+C1
    CK(cudaDeviceSynchronize());
    auto tree=dn(dO_tree,(size_t)Sq*NQ*HD), ref=dn(dO2,(size_t)2*NQ*HD);
    std::vector<float> tree_n2(tree.begin()+(size_t)2*NQ*HD, tree.begin()+(size_t)3*NQ*HD);  // tree node2(C1)
    std::vector<float> ref_n1(ref.begin()+(size_t)1*NQ*HD, ref.begin()+(size_t)2*NQ*HD);      // ref node1(C1)
    double e2=maxae(tree_n2,ref_n1);
    printf("=== ② branch node2(C1, slot1=C0 mask) vs linear[prefix,R,C1] ref ===\n  node2 O err = %.3e\n",e2);

    bool pass=(e1<1e-6)&&(e2<1e-6);
    printf("\n[test_treeattn] %s (chain==causal + branch ancestor-only attend)\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
