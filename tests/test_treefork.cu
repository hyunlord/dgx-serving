// 챕터9 tree-spec GDN-tree fork de-risk: tree 노드별 state fork(부모 snapshot 복원)가 linear과 bit-identical?
//   하드파트 = GDN 48층 recurrence가 tree 분기와 충돌 → 분기 sibling은 부모 state서 fork(선형 아님).
//   토대: test_promote가 "verify(commit=0,Sq=K)+promote(m) committed state == sequential m" <1e-5 검증.
//   이 테스트: tree 한 경로 verify[R,C0,GC0](깊이2) 후 promote(1)로 fork점 R 복원 + sibling C1 처리(commit)가
//     linear[R,C1] committed state(GDN S + conv ctx + full KV + n_past)와 오차0인가.
//   되면 = tree-fork(중간 fork점 복원 + 다른 분기) 정확성 sound → tree-spec 빌드 가능(단 1-forward amortize는 별도 kernel).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rfg(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> dn(const float*d,size_t n){std::vector<float>h(n);TCK(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost));return h;}
static std::vector<float> dnh(const __nv_bfloat16*d,size_t n){std::vector<__nv_bfloat16>t(n);TCK(cudaMemcpy(t.data(),d,n*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));std::vector<float>h(n);for(size_t i=0;i<n;i++)h[i]=__bfloat162float(t[i]);return h;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/golden")+std::string("/");
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size,rot=c.rotary_dim,HD=c.head_dim,NKV=c.num_kv_heads;
    const int nvh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),Kk=c.gdn_conv_kernel;
    const size_t Ssz=(size_t)nvh*D*D, Csz=(size_t)(Kk-1)*convd;
    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    int npre=1;
    // tree: R=ids[npre], C0=ids[npre+1], GC0=ids[npre+2] (한 경로 깊이2), C1=sibling(R의 다른 자식, 구별 토큰)
    int R=ids[npre], C0=ids[npre+1], GC0=ids[npre+2], C1=(ids[npre+1]+777)%V;
    if(C1==C0) C1=(C1+1)%V;
    printf("[treefork] S=%d npre=%d | R=%d C0=%d GC0=%d C1=%d (C1=R의 sibling)\n",S,npre,R,C0,GC0,C1);
    std::vector<float> cosv=rfg("rope_cos.bin",(size_t)S*rot), sinv=rfg("rope_sin.bin",(size_t)S*rot);
    if((size_t)(npre+3)>cosv.size()/rot){printf("rope 짧음(need npre+3)\n");return 1;}
    int *d_R,*d_C0,*d_GC0,*d_C1,*d_pre;
    TCK(cudaMalloc(&d_pre,npre*4)); TCK(cudaMemcpy(d_pre,ids.data(),npre*4,cudaMemcpyHostToDevice));
    auto mk=[&](int t){int*d;TCK(cudaMalloc(&d,4));TCK(cudaMemcpy(d,&t,4,cudaMemcpyHostToDevice));return d;};
    d_R=mk(R); d_C0=mk(C0); d_GC0=mk(GC0); d_C1=mk(C1);
    int d3[3]={R,C0,GC0}; int*d_tree3; TCK(cudaMalloc(&d_tree3,3*4)); TCK(cudaMemcpy(d_tree3,d3,3*4,cudaMemcpyHostToDevice));
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cosv.size()*4)); TCK(cudaMalloc(&d_sin,sinv.size()*4));
    TCK(cudaMemcpy(d_cos,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice)); TCK(cudaMemcpy(d_sin,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));
    QwenWeights w=load_weights_qwen(WD.c_str(),c,0,1);
    float *dl; TCK(cudaMalloc(&dl,(size_t)4*V*4));

    int fl=-1; for(int l=0;l<c.num_layers;l++) if(c.layer_is_full[l]){fl=l;break;}
    auto snap=[&](DecodeState&st)->std::vector<float>{
        std::vector<float> out;
        for(int l=0;l<c.num_layers;l++) if(!c.layer_is_full[l]){
            auto s=dn(st.S[l],Ssz); out.insert(out.end(),s.begin(),s.end());
            auto cc=dn(st.convctx[l],Csz); out.insert(out.end(),cc.begin(),cc.end()); }
        auto kv=dnh(st.Kc[fl],(size_t)st.n_past*NKV*HD); out.insert(out.end(),kv.begin(),kv.end());
        return out;
    };

    // (A) linear: prefill + R(commit) + C1(commit) → committed state after R,C1
    DecodeState sa=alloc_decode_state(c,S+8);
    qwen35_decode(c,w,sa,d_pre,npre,d_cos,d_sin,dl,nullptr,1,nullptr,0);
    qwen35_decode(c,w,sa,d_R,1,d_cos,d_sin,dl,nullptr,1,nullptr,0);
    qwen35_decode(c,w,sa,d_C1,1,d_cos,d_sin,dl,nullptr,1,nullptr,0);
    auto ref=snap(sa); int ref_np=sa.n_past;

    // (B) tree: prefill + verify[R,C0,GC0](commit=0, va, 깊이2 한 경로) + promote(1)로 fork점 R 복원 + C1(commit)
    DecodeState sb=alloc_decode_state(c,S+8);
    qwen35_decode(c,w,sb,d_pre,npre,d_cos,d_sin,dl,nullptr,1,nullptr,0);
    VerifyArtifacts va=alloc_verify_artifacts(c,3);
    qwen35_decode(c,w,sb,d_tree3,3,d_cos,d_sin,dl,nullptr,/*commit*/0,&va,0);  // tree 경로 R→C0→GC0 (GDN state R,C0,GC0 snapshot)
    promote_prefix(c,sb,va,1);   // fork점 = R만 승격(C0,GC0 분기 폐기) → committed = post-R
    qwen35_decode(c,w,sb,d_C1,1,d_cos,d_sin,dl,nullptr,1,nullptr,0);   // sibling C1을 post-R서 처리(commit)
    auto tree=snap(sb); int tree_np=sb.n_past;

    double e=mae(tree,ref);
    printf("\n=== tree-fork(verify[R,C0,GC0]+promote(1)→R+C1) vs linear(R+C1) committed state ===\n");
    printf("  GDN S(48층) + conv ctx + full KV  max abs err = %.3e\n",e);
    printf("  n_past: linear=%d tree=%d %s\n",ref_np,tree_np, ref_np==tree_np?"match":"MISMATCH");
    bool pass=(e<1e-5)&&(ref_np==tree_np);
    printf("\n해석: 오차0 = fork점 복원 후 sibling 처리가 linear와 동일 = GDN tree state-fork 정확성 sound(빌드 가능).\n");
    printf("[test_treefork] %s\n",pass?"PASS":"FAIL");
    free_verify_artifacts(va); free_decode_state(sa); free_decode_state(sb); free_weights_qwen(w); free_config_qwen(c);
    return pass?0:1;
}
