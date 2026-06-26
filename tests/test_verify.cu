// 2c STEP 2 검증: batch verify(commit=0) == sequential decode, + commit=0 이 committed state 불변.
//   verify = qwen35_decode(Sq=K+1, commit=0): committed 캐시 위로 K+1 토큰 일괄 처리.
//   (a) verify per-position logits == 각 토큰을 sequential single decode 한 logits.
//   (b) commit=0 후 committed GDN state S == verify 전 (건드리지 않음).
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
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

// 첫 linear 레이어 인덱스
static int first_linear(const ModelConfig&c){for(int l=0;l<c.num_layers;l++)if(!c.layer_is_full[l])return l;return -1;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/golden")+std::string("/");
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size,rot=c.rotary_dim;
    const int nvh=c.gdn_num_v_heads,D=c.gdn_head_v_dim; const size_t Ssz=(size_t)nvh*D*D;
    int li=first_linear(c);
    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    int npre=2, K=S-npre;   // prefill npre, verify/decode K tokens
    printf("[verify] S=%d prefill=%d verify_chain=%d (linear layer %d)\n",S,npre,K,li);
    std::vector<float> cosv=rf("rope_cos.bin",(size_t)S*rot), sinv=rf("rope_sin.bin",(size_t)S*rot);
    int *d_ids; TCK(cudaMalloc(&d_ids,S*4)); TCK(cudaMemcpy(d_ids,ids.data(),S*4,cudaMemcpyHostToDevice));
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cosv.size()*4)); TCK(cudaMalloc(&d_sin,sinv.size()*4));
    TCK(cudaMemcpy(d_cos,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice));
    TCK(cudaMemcpy(d_sin,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));
    QwenWeights w=load_weights_qwen(WD.c_str(),c,0,1);
    float *dl; TCK(cudaMalloc(&dl,(size_t)(K)*V*4));

    auto prefill=[&](DecodeState&st){ qwen35_decode(c,w,st,d_ids,npre,d_cos,d_sin,dl,nullptr,1,nullptr,0); };

    // ---- (A) sequential: prefill + decode K tokens one at a time (commit) ----
    DecodeState sa=alloc_decode_state(c,S+8); prefill(sa);
    std::vector<float> seq_log((size_t)K*V);
    for(int t=0;t<K;t++){ float*o; TCK(cudaMalloc(&o,(size_t)V*4));
        qwen35_decode(c,w,sa,d_ids+npre+t,1,d_cos,d_sin,o,nullptr,1,nullptr,0);
        TCK(cudaMemcpy(seq_log.data()+(size_t)t*V,o,(size_t)V*4,cudaMemcpyDeviceToHost)); cudaFree(o); }

    // ---- (B) batch verify: fresh prefill, snapshot committed S, verify K (commit=0), snapshot S again ----
    DecodeState sb=alloc_decode_state(c,S+8); prefill(sb);
    std::vector<float> S_before(Ssz); TCK(cudaMemcpy(S_before.data(),sb.S[li],Ssz*4,cudaMemcpyDeviceToHost));
    float *vl; TCK(cudaMalloc(&vl,(size_t)K*V*4));
    qwen35_decode(c,w,sb,d_ids+npre,K,d_cos,d_sin,vl,nullptr,/*commit*/0,nullptr,0);
    std::vector<float> ver_log((size_t)K*V); TCK(cudaMemcpy(ver_log.data(),vl,ver_log.size()*4,cudaMemcpyDeviceToHost));
    std::vector<float> S_after(Ssz); TCK(cudaMemcpy(S_after.data(),sb.S[li],Ssz*4,cudaMemcpyDeviceToHost));

    // ---- 대조 ----
    double el=mae(ver_log,seq_log), es=mae(S_after,S_before);
    printf("\n=== batch verify vs sequential decode ===\n  per-pos logits  max abs err = %.3e\n",el);
    printf("  committed S (commit=0): unchanged err = %.3e\n",es);
    auto am=[&](const std::vector<float>&lg,int p){int b=0;float bv=lg[(size_t)p*V];for(int j=1;j<V;j++){float z=lg[(size_t)p*V+j];if(z>bv){bv=z;b=j;}}return b;};
    bool amok=true; for(int p=0;p<K;p++) if(am(ver_log,p)!=am(seq_log,p)) amok=false;
    printf("  argmax per pos: %s; n_past(verify state)=%d (prefill %d 유지)\n", amok?"ALL match":"MISMATCH", sb.n_past, npre);

    free_decode_state(sa); free_decode_state(sb); free_weights_qwen(w); free_config_qwen(c);
    cudaFree(d_ids);cudaFree(d_cos);cudaFree(d_sin);cudaFree(dl);cudaFree(vl);
    bool pass=(el<1e-2)&&(es<1e-6)&&amok&&(sb.n_past==npre);
    printf("\n[test_verify] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
