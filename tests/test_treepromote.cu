// 챕터9 STEP 4.2 격리: promote_tree_path(accept된 scattered path 승격 + KV compaction) 정확성.
//   tree [R,C0,C1,GC] parent=[-1,0,0,1] (R root; C0,C1=R 자식; GC=C0 자식). path R→C0→GC = 노드[0,1,3](C1=node2 폐기).
//   (A) linear: prefill + [R,C0,GC] 3-chain commit → committed state.
//   (B) tree: prefill + 1-forward verify[R,C0,C1,GC](commit=0,va) + promote_tree_path([0,1,3],3) → committed state.
//   committed state(GDN S 48 + conv ctx + full-attn KV 전부) bit-identical?  되면 promote_tree_path 정확(scattered→contiguous).
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
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> rfg(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<float> dn(const float*d,size_t n){std::vector<float>h(n);TCK(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost));return h;}
static std::vector<float> dnh(const __nv_bfloat16*d,size_t n){std::vector<__nv_bfloat16>t(n);TCK(cudaMemcpy(t.data(),d,n*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));std::vector<float>h(n);for(size_t i=0;i<n;i++)h[i]=__bfloat162float(t[i]);return h;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}
static int* up(const std::vector<int>&h){int*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static unsigned* upu(const std::vector<unsigned>&h){unsigned*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/golden")+std::string("/");
    int bq4=(argc>3)?atoi(argv[3]):0;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size,rot=c.rotary_dim,HD=c.head_dim,NKV=c.num_kv_heads;
    const int nvh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),Kk=c.gdn_conv_kernel;
    const size_t Ssz=(size_t)nvh*D*D, Csz=(size_t)(Kk-1)*convd;
    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    int npre=1;
    int R=ids[npre], C0=ids[npre+1], GC=ids[npre+2], C1=(ids[npre+1]+777)%V; if(C1==C0)C1=(C1+1)%V;
    printf("[treepromote] S=%d npre=%d bq4=%d | R=%d C0=%d C1=%d GC=%d | path R→C0→GC=[0,1,3]\n",S,npre,bq4,R,C0,C1,GC);
    std::vector<float> cosv=rfg("rope_cos.bin",(size_t)S*rot), sinv=rfg("rope_sin.bin",(size_t)S*rot);
    if((size_t)(npre+3)>cosv.size()/rot){printf("rope 짧음\n");return 1;}
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cosv.size()*4)); TCK(cudaMalloc(&d_sin,sinv.size()*4));
    TCK(cudaMemcpy(d_cos,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice)); TCK(cudaMemcpy(d_sin,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));
    QwenWeights w=load_weights_qwen(WD.c_str(),c, bq4==0?0:1, bq4==0?1:0);
    float *dl; TCK(cudaMalloc(&dl,(size_t)4*V*4));
    int*d_pre=up(std::vector<int>(ids.begin(),ids.begin()+npre));

    // 비교용 snapshot: GDN S(48) + conv ctx(48) + full-attn KV(전부)
    auto snap=[&](DecodeState&st)->std::vector<float>{
        std::vector<float> out;
        for(int l=0;l<c.num_layers;l++) if(!c.layer_is_full[l]){
            auto s=dn(st.S[l],Ssz); out.insert(out.end(),s.begin(),s.end());
            auto cc=dn(st.convctx[l],Csz); out.insert(out.end(),cc.begin(),cc.end()); }
        for(int l=0;l<c.num_layers;l++) if(c.layer_is_full[l]){
            auto kv=dnh(st.Kc[l],(size_t)st.n_past*NKV*HD); out.insert(out.end(),kv.begin(),kv.end());
            auto vv=dnh(st.Vc[l],(size_t)st.n_past*NKV*HD); out.insert(out.end(),vv.begin(),vv.end()); }
        return out;
    };

    // (A) linear: prefill + [R,C0,GC] commit
    DecodeState sa=alloc_decode_state(c,S+8);
    qwen35_decode(c,w,sa,d_pre,npre,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,nullptr,1);
    int*d_lin=up({R,C0,GC});
    qwen35_decode(c,w,sa,d_lin,3,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,nullptr,0);
    auto ref=snap(sa); int ref_np=sa.n_past;

    // (B) tree: prefill + verify[R,C0,C1,GC](commit=0,va) + promote_tree_path([0,1,3])
    DecodeState sb=alloc_decode_state(c,S+8);
    qwen35_decode(c,w,sb,d_pre,npre,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,nullptr,1);
    int base=sb.n_past;
    VerifyArtifacts va=alloc_verify_artifacts(c,4);
    int*d_tree=up({R,C0,C1,GC});
    int*p_tree=up({-1,0,0,1}); unsigned*a_tree=upu({1u,3u,5u,11u}); int*x_tree=up({base,base+1,base+1,base+2});
    qwen35_decode(c,w,sb,d_tree,4,d_cos,d_sin,dl,nullptr,0,&va,bq4,nullptr,0, p_tree,a_tree,x_tree);
    int path[3]={0,1,3};
    promote_tree_path(c,sb,va,path,3);
    auto tree=snap(sb); int tree_np=sb.n_past;

    double e=mae(tree,ref);
    printf("\n=== tree verify[R,C0,C1,GC] + promote_tree_path([0,1,3]) vs linear[R,C0,GC] committed ===\n");
    printf("  GDN S(48) + conv ctx + full KV  max abs err = %.3e\n",e);
    printf("  n_past: linear=%d tree=%d %s\n",ref_np,tree_np, ref_np==tree_np?"match":"MISMATCH");
    double tol=(bq4==0)?1e-5:5e-2;
    bool pass=(e<tol)&&(ref_np==tree_np);
    printf("\n해석: 오차0 = scattered path[0,1,3] 승격(KV compaction+GDN snap[3]+conv along path)이 linear[R,C0,GC]와 동일.\n");
    printf("[test_treepromote] %s (tol=%.0e)\n",pass?"PASS":"FAIL",tol);
    free_verify_artifacts(va); free_decode_state(sa); free_decode_state(sb); free_weights_qwen(w); free_config_qwen(c);
    return pass?0:1;
}
