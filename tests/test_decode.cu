// 2c STEP 1 검증: incremental decode(KV 캐시) == prefill forward, 오차 0.
//   같은 N토큰을 (a) qwen35_forward prefill 한방, (b) qwen35_decode 한토큰씩(commit) 으로 처리해
//   per-position logits + pre-norm hidden(MTP 입력) 을 대조. fp32 경로.
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
#include "../models/qwen35.h"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/golden")+std::string("/");
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size,rot=c.rotary_dim;
    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    int uq=argc>3?atoi(argv[3]):0;
    printf("[decode] S=%d rot=%d use_q4=%d\n",S,rot,uq);
    std::vector<float> cosv=rf("rope_cos.bin",(size_t)S*rot), sinv=rf("rope_sin.bin",(size_t)S*rot);
    int *d_ids; TCK(cudaMalloc(&d_ids,S*4)); TCK(cudaMemcpy(d_ids,ids.data(),S*4,cudaMemcpyHostToDevice));
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cosv.size()*4)); TCK(cudaMalloc(&d_sin,sinv.size()*4));
    TCK(cudaMemcpy(d_cos,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice));
    TCK(cudaMemcpy(d_sin,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));

    QwenWeights w=load_weights_qwen(WD.c_str(),c,uq,1);

    // (a) prefill ref
    const size_t SH=(size_t)S*H;
    float *pl,*ppn; TCK(cudaMalloc(&pl,(size_t)S*V*4)); TCK(cudaMalloc(&ppn,SH*4));
    qwen35_forward(c,w,d_ids,S,d_cos,d_sin,pl,nullptr,ppn,uq);
    std::vector<float> prefill_log((size_t)S*V), prefill_pn(SH);
    TCK(cudaMemcpy(prefill_log.data(),pl,prefill_log.size()*4,cudaMemcpyDeviceToHost));
    TCK(cudaMemcpy(prefill_pn.data(),ppn,SH*4,cudaMemcpyDeviceToHost));

    // (b) incremental decode, 한 토큰씩
    DecodeState st=alloc_decode_state(c, S+8);
    float *dl,*dpn; TCK(cudaMalloc(&dl,(size_t)V*4)); TCK(cudaMalloc(&dpn,(size_t)H*4));
    std::vector<float> dec_log((size_t)S*V), dec_pn(SH);
    for(int t=0;t<S;t++){
        qwen35_decode(c,w,st,d_ids+t,1,d_cos,d_sin,dl,dpn,1,nullptr,uq);
        TCK(cudaMemcpy(dec_log.data()+(size_t)t*V,dl,(size_t)V*4,cudaMemcpyDeviceToHost));
        TCK(cudaMemcpy(dec_pn.data()+(size_t)t*H,dpn,(size_t)H*4,cudaMemcpyDeviceToHost));
    }
    printf("[decode] n_past after %d steps = %d\n",S,st.n_past);

    double el=mae(dec_log,prefill_log), ep=mae(dec_pn,prefill_pn);
    printf("\n=== incremental decode vs prefill ===\n  logits   max abs err = %.3e\n  pre-norm max abs err = %.3e\n",el,ep);
    // argmax 일치도
    auto am=[&](const std::vector<float>&lg,int p){int b=0;float bv=lg[(size_t)p*V];for(int j=1;j<V;j++){float z=lg[(size_t)p*V+j];if(z>bv){bv=z;b=j;}}return b;};
    bool amok=true; for(int p=0;p<S;p++) if(am(dec_log,p)!=am(prefill_log,p)) amok=false;
    printf("  argmax per pos: %s\n", amok?"ALL match":"MISMATCH");

    free_decode_state(st); free_weights_qwen(w); free_config_qwen(c);
    cudaFree(d_ids);cudaFree(d_cos);cudaFree(d_sin);cudaFree(pl);cudaFree(ppn);cudaFree(dl);cudaFree(dpn);
    bool pass=(el<1e-2)&&(ep<1e-2)&&amok;
    printf("\n[test_decode] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
