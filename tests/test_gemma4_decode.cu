// Gemma4 decode 경로 게이트 (M4 STEP A-1).
//   (1) 프리필 재현: gemma4_decode(base=0, Sq=S) == gemma4_forward(프리필, golden 통과 기준선) — KV/attn_decode 배선.
//   (2) decode 연속성: 프리필 후 1토큰씩 decode → logits가 프리필 마지막 위치를 이어받는지(같은 KV) argmax 일관.
// golden 우선: (1)은 기존 통과 엔진과 bit-level 근접(GPU 재현성), (2)는 프리필 vs decode 경로 일치.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../models/gemma4.h"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/gemma4-31b/weights"; GD=(argc>2?argv[2]:"artifacts/gemma4-31b/golden")+std::string("/");
    ModelConfig c=load_config_gemma4((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size;
    const int hd_f=c.global_head_dim?c.global_head_dim:c.head_dim;
    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    std::vector<float> cf=rf("rope_full_cos.bin",(size_t)S*hd_f), sf=rf("rope_full_sin.bin",(size_t)S*hd_f);
    std::vector<float> cs=rf("rope_sliding_cos.bin",(size_t)S*c.head_dim), ss=rf("rope_sliding_sin.bin",(size_t)S*c.head_dim);
    auto up=[&](std::vector<float>&h){float*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;};
    int *d_ids; TCK(cudaMalloc(&d_ids,S*4)); TCK(cudaMemcpy(d_ids,ids.data(),S*4,cudaMemcpyHostToDevice));
    float *d_cf=up(cf),*d_sf=up(sf),*d_cs=up(cs),*d_ss=up(ss);

    G4Weights w=load_weights_gemma4(WD.c_str(),c,1,0);
    const size_t SH=(size_t)S*H;
    auto am=[&](const std::vector<float>&lg,int pos){int b=0;float bv=lg[(size_t)pos*V];for(int j=1;j<V;j++){float x=lg[(size_t)pos*V+j];if(x>bv){bv=x;b=j;}}return b;};

    // 기준선: gemma4_forward(프리필, golden 통과 경로)
    float *d_logf; TCK(cudaMalloc(&d_logf,(size_t)S*V*4));
    gemma4_forward(c,w,d_ids,S,d_cf,d_sf,d_cs,d_ss,d_logf,nullptr,1);
    std::vector<float> logf((size_t)S*V); TCK(cudaMemcpy(logf.data(),d_logf,logf.size()*4,cudaMemcpyDeviceToHost));

    // (1) 프리필 재현: gemma4_decode(base=0, Sq=S)
    G4DecodeState st=alloc_g4_decode_state(c, S+8);
    float *d_logd; TCK(cudaMalloc(&d_logd,(size_t)S*V*4));
    gemma4_decode(c,w,st,d_ids,S,d_cf,d_sf,d_cs,d_ss,d_logd,/*commit*/1,/*use_q4*/1,/*last_only*/0);
    std::vector<float> logd((size_t)S*V); TCK(cudaMemcpy(logd.data(),d_logd,logd.size()*4,cudaMemcpyDeviceToHost));
    double e1=mae(logf,logd);
    bool am1=true; for(int p=0;p<S;p++) if(am(logf,p)!=am(logd,p)) am1=false;
    printf("[cfg] H=%d L=%d S=%d hd_f=%d n_past(after prefill)=%d\n",H,c.num_layers,S,hd_f,st.n_past);
    printf("(1) 프리필 재현: gemma4_decode(base=0) vs gemma4_forward — logits maxabs=%.3e | argmax %s\n",
           e1, am1?"ALL 일치":"MISMATCH");

    // decode 연속성(base>0, rope 위치 S+) 은 test_gemma4_gen(전체 위치 rope 골든)에서 검증.
    bool pass=(e1<1e-3)&&am1&&(st.n_past==S);
    printf("[test_gemma4_decode] %s (게이트=프리필 재현 logits<1e-3 + argmax 전위치 + n_past)\n",pass?"PASS":"FAIL");
    free_g4_decode_state(st); free_weights_gemma4(w); free_config_gemma4(c);
    return pass?0:1;
}
