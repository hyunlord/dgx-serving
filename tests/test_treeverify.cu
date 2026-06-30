// 챕터9 STEP 3.2: 1-forward tree verify 통합 정확성 게이트.
//   STEP 1(gdn_step fork) + STEP 2(conv fork) + STEP 3.1(attn mask) + rope-by-depth를 qwen35_decode에 실제 배선,
//   1-forward(weight 1적재)로 tree 전체 처리 == greedy linear forward bit-identical?
//   ① chain tree([-1,0,1], anc=causal, pos=base+slot) == linear forward [R,C0,GC0] → 전체 배선이 chain서 선형과 동일.
//   ② branch tree([-1,0,0]: C0,C1 = R의 두 자식) node2(C1, 조상=R→C1) logits == linear[R,C1] node1 → 분기 노드가
//      조상-path만(sibling C0 무시) fork/attend/rope, full-forward bit-identical.
//   commit=0(verify, st 불변) → 같은 prefill state 재사용. va 불요(logits만 비교; tree snap 버퍼는 d_parent로 Sq-크기).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rfg(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> dn(const float*d,size_t n){std::vector<float>h(n);TCK(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost));return h;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}
static int* up(const std::vector<int>&h){int*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static unsigned* upu(const std::vector<unsigned>&h){unsigned*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/golden")+std::string("/");
    int bq4=(argc>3)?atoi(argv[3]):0;   // 0=fp32 body, 2=W4A8 (직교성)
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size,rot=c.rotary_dim;
    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    int npre=1;                                   // prefix(commit) 길이 (token_ids.bin S 작음 — R,C0,GC0=ids[1..3])
    int R=ids[npre], C0=ids[npre+1], GC0=ids[npre+2], C1=(ids[npre+1]+777)%V;
    if(C1==C0) C1=(C1+1)%V;
    printf("[treeverify] S=%d npre=%d bq4=%d | R=%d C0=%d GC0=%d C1=%d\n",S,npre,bq4,R,C0,GC0,C1);
    std::vector<float> cosv=rfg("rope_cos.bin",(size_t)S*rot), sinv=rfg("rope_sin.bin",(size_t)S*rot);
    if((size_t)(npre+3)>cosv.size()/rot){printf("rope 짧음\n");return 1;}
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cosv.size()*4)); TCK(cudaMalloc(&d_sin,sinv.size()*4));
    TCK(cudaMemcpy(d_cos,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice)); TCK(cudaMemcpy(d_sin,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));
    QwenWeights w=load_weights_qwen(WD.c_str(),c, bq4==0?0:1, bq4==0?1:0);  // (load_q4, load_fp32): bq4=0→fp32만, bq4=2→q4만
    float *dl_lin,*dl_tree,*dl_ref,*dl_br;
    TCK(cudaMalloc(&dl_lin,(size_t)3*V*4)); TCK(cudaMalloc(&dl_tree,(size_t)3*V*4));
    TCK(cudaMalloc(&dl_ref,(size_t)2*V*4)); TCK(cudaMalloc(&dl_br,(size_t)3*V*4));
    int*d_pre=up(std::vector<int>(ids.begin(),ids.begin()+npre));
    const int base=npre;

    DecodeState st=alloc_decode_state(c,S+8);
    qwen35_decode(c,w,st,d_pre,npre,d_cos,d_sin,dl_lin,nullptr,1,nullptr,bq4,nullptr,0);  // prefill commit
    printf("  prefill 완료 n_past=%d (base=%d)\n",st.n_past,base);

    // ① chain: [R,C0,GC0]. linear(d_parent=null) vs tree(chain parent/anc/pos). commit=0(st 불변).
    int*d_chain=up({R,C0,GC0});
    qwen35_decode(c,w,st,d_chain,3,d_cos,d_sin,dl_lin,nullptr,0,nullptr,bq4,nullptr,0);   // linear
    int*p_chain=up({-1,0,1}); unsigned*a_chain=upu({1u,3u,7u}); int*x_chain=up({base,base+1,base+2});
    qwen35_decode(c,w,st,d_chain,3,d_cos,d_sin,dl_tree,nullptr,0,nullptr,bq4,nullptr,0, p_chain,a_chain,x_chain); // tree
    TCK(cudaDeviceSynchronize());
    double e1=mae(dn(dl_lin,(size_t)3*V),dn(dl_tree,(size_t)3*V));
    printf("=== ① chain tree(전체 배선) vs linear forward [R,C0,GC0] ===\n  logits[3,V] max abs err = %.3e\n",e1);

    // ② branch: tree [R,C0,C1] parent=[-1,0,0](C0,C1=R 자식). node2(C1) 조상=R→C1, pos=base+1(깊이1).
    //   레퍼런스: linear [R,C1] (C1이 node1, pos base+1, prefix+R+C1 attend, GDN R→C1 fork) — 같은 prefill state서.
    int*d_RC1=up({R,C1});
    qwen35_decode(c,w,st,d_RC1,2,d_cos,d_sin,dl_ref,nullptr,0,nullptr,bq4,nullptr,0);     // linear ref [R,C1]
    int*d_br=up({R,C0,C1});
    int*p_br=up({-1,0,0}); unsigned*a_br=upu({1u,3u,5u}); int*x_br=up({base,base+1,base+1});
    qwen35_decode(c,w,st,d_br,3,d_cos,d_sin,dl_br,nullptr,0,nullptr,bq4,nullptr,0, p_br,a_br,x_br); // tree branch
    TCK(cudaDeviceSynchronize());
    auto br=dn(dl_br,(size_t)3*V), ref=dn(dl_ref,(size_t)2*V);
    std::vector<float> br_n2(br.begin()+(size_t)2*V,br.begin()+(size_t)3*V);   // tree node2(C1)
    std::vector<float> ref_n1(ref.begin()+(size_t)1*V,ref.begin()+(size_t)2*V); // linear node1(C1)
    double e2=mae(br_n2,ref_n1);
    printf("=== ② branch node2(C1, 조상 R→C1, sibling C0 무시) vs linear[R,C1] node1 ===\n  C1 logits[V] max abs err = %.3e\n",e2);

    // 결정성 30회: tree chain forward 반복 → run0과 bit-identical(고유 1)인가
    auto run0=dn(dl_tree,(size_t)3*V); double dmax=0;
    for(int r=0;r<30;r++){
        qwen35_decode(c,w,st,d_chain,3,d_cos,d_sin,dl_tree,nullptr,0,nullptr,bq4,nullptr,0, p_chain,a_chain,x_chain);
        TCK(cudaDeviceSynchronize());
        dmax=fmax(dmax,mae(run0,dn(dl_tree,(size_t)3*V)));
    }
    printf("=== 결정성: tree forward 30회 vs run0 ===\n  max abs diff = %.3e (%s)\n",dmax,dmax==0.0?"고유1":"비결정");

    double tol = (bq4==0)?1e-3:5e-2;   // fp32 bit-identical 목표(0); W4A8는 양자화 재현성 tol
    bool pass=(e1<tol)&&(e2<tol)&&(dmax==0.0);
    printf("\n해석: ① 전체 tree 배선이 chain서 linear와 동일(배선 정확) ② 분기 노드가 조상-path만 fork/attend/rope(full-forward 정확).\n");
    printf("[test_treeverify] %s (tol=%.0e)\n",pass?"PASS":"FAIL",tol);
    free_decode_state(st); free_weights_qwen(w); free_config_qwen(c);
    return pass?0:1;
}
