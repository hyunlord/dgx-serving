// 2c STEP 3 검증: verify(commit=0)+promote_prefix(m) 후 committed 상태가
//   같은 m 토큰을 sequential decode(commit) 한 상태와 오차 0 (모든 GDN state + conv ctx + full KV + n_past).
//   부분 accept (m<K) 로 승격 정확성 시험.
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
// bf16 KV cache 다운로드 → float (지문 비교용; 두 실행 모두 bf16 저장이라 비교는 동일)
static std::vector<float> dnh(const __nv_bfloat16*d,size_t n){std::vector<__nv_bfloat16>t(n);TCK(cudaMemcpy(t.data(),d,n*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));std::vector<float>h(n);for(size_t i=0;i<n;i++)h[i]=__bfloat162float(t[i]);return h;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/golden")+std::string("/");
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size,rot=c.rotary_dim,HD=c.head_dim,NKV=c.num_kv_heads;
    const int nvh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),Kk=c.gdn_conv_kernel;
    const size_t Ssz=(size_t)nvh*D*D, Csz=(size_t)(Kk-1)*convd;
    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    int npre=1, K=S-npre, m=2;   // prefill 1, verify K=4, accept m=2 (부분)
    printf("[promote] S=%d prefill=%d verify_K=%d accept_m=%d\n",S,npre,K,m);
    std::vector<float> cosv=rfg("rope_cos.bin",(size_t)S*rot), sinv=rfg("rope_sin.bin",(size_t)S*rot);
    int *d_ids; TCK(cudaMalloc(&d_ids,S*4)); TCK(cudaMemcpy(d_ids,ids.data(),S*4,cudaMemcpyHostToDevice));
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cosv.size()*4)); TCK(cudaMalloc(&d_sin,sinv.size()*4));
    TCK(cudaMemcpy(d_cos,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice));
    TCK(cudaMemcpy(d_sin,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));
    QwenWeights w=load_weights_qwen(WD.c_str(),c,0,1);
    float *dl; TCK(cudaMalloc(&dl,(size_t)K*V*4));

    // 상태 비교 헬퍼: 모든 GDN S + conv ctx + 첫 full KV(committed 길이) 수집
    int fl=-1; for(int l=0;l<c.num_layers;l++) if(c.layer_is_full[l]){fl=l;break;}
    auto snapshot=[&](DecodeState&st)->std::vector<float>{
        std::vector<float> out;
        for(int l=0;l<c.num_layers;l++){
            if(!c.layer_is_full[l]){ auto s=dn(st.S[l],Ssz); out.insert(out.end(),s.begin(),s.end());
                                     auto cc=dn(st.convctx[l],Csz); out.insert(out.end(),cc.begin(),cc.end()); }
        }
        auto kv=dnh(st.Kc[fl],(size_t)st.n_past*NKV*HD); out.insert(out.end(),kv.begin(),kv.end());
        return out;
    };

    // (A) sequential: prefill npre + decode m (commit)
    DecodeState sa=alloc_decode_state(c,S+8);
    qwen35_decode(c,w,sa,d_ids,npre,d_cos,d_sin,dl,nullptr,1,nullptr,0);
    for(int t=0;t<m;t++) qwen35_decode(c,w,sa,d_ids+npre+t,1,d_cos,d_sin,dl,nullptr,1,nullptr,0);
    auto ref=snapshot(sa); int ref_np=sa.n_past;

    // (B) spec: prefill npre + verify K (commit=0, va) + promote(m)
    DecodeState sb=alloc_decode_state(c,S+8);
    qwen35_decode(c,w,sb,d_ids,npre,d_cos,d_sin,dl,nullptr,1,nullptr,0);
    VerifyArtifacts va=alloc_verify_artifacts(c,K);
    qwen35_decode(c,w,sb,d_ids+npre,K,d_cos,d_sin,dl,nullptr,/*commit*/0,&va,0);
    promote_prefix(c,sb,va,m);
    auto spec=snapshot(sb); int spec_np=sb.n_past;

    double e=mae(spec,ref);
    printf("\n=== promote(m=%d) committed state vs sequential decode(m) ===\n",m);
    printf("  all GDN S + conv ctx + full KV  max abs err = %.3e\n",e);
    printf("  n_past: sequential=%d, spec=%d %s\n",ref_np,spec_np, ref_np==spec_np?"match":"MISMATCH");

    free_verify_artifacts(va); free_decode_state(sa); free_decode_state(sb); free_weights_qwen(w); free_config_qwen(c);
    cudaFree(d_ids);cudaFree(d_cos);cudaFree(d_sin);cudaFree(dl);
    bool pass=(e<1e-5)&&(ref_np==spec_np);
    printf("\n[test_promote] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
