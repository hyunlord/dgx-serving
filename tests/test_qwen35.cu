// Qwen3.5-0.8B 본체 통합 검증 (STEP 4): graph 출력 vs HF 골든.
//   (a) 레이어별 hidden 단계 대조 (full·GDN 모두) vs golden_q4
//   (b) pre-norm hidden (MTP 입력) 단독 대조
//   (c) 최종 logits vs golden_q4 (<1e-2) + argmax. golden_fp32 는 정보용(quant budget).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../models/qwen35.h"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen(p.c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"크기 %s\n",p.c_str());exit(1);}fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen(p.c_str(),"rb");if(!f){exit(1);}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    std::string WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights";
    std::string GD=argc>2?argv[2]:"artifacts/qwen35-0.8b/golden";
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size,L=c.num_layers,rot=c.rotary_dim;
    int nf=0; for(int i=0;i<L;i++) nf+=c.layer_is_full[i];
    printf("[cfg] H=%d FFN=%d L=%d (full=%d/GDN=%d) NQ=%d NKV=%d HD=%d rot=%d V=%d gate=%d\n",
           H,c.intermediate_size,L,nf,L-nf,c.num_q_heads,c.num_kv_heads,c.head_dim,rot,V,c.attn_output_gate);
    printf("[cfg] GDN: kheads=%d vheads=%d kdim=%d vdim=%d conv=%d\n",
           c.gdn_num_k_heads,c.gdn_num_v_heads,c.gdn_head_k_dim,c.gdn_head_v_dim,c.gdn_conv_kernel);

    std::vector<int> ids=ri(GD+"/token_ids.bin"); int S=ids.size();
    printf("[in] S=%d tokens=",S); for(int t:ids)printf("%d ",t); printf("\n");
    std::vector<float> cosv=rf(GD+"/rope_cos.bin",(size_t)S*rot), sinv=rf(GD+"/rope_sin.bin",(size_t)S*rot);

    int *d_ids; TCK(cudaMalloc(&d_ids,S*4)); TCK(cudaMemcpy(d_ids,ids.data(),S*4,cudaMemcpyHostToDevice));
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cosv.size()*4)); TCK(cudaMalloc(&d_sin,sinv.size()*4));
    TCK(cudaMemcpy(d_cos,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice));
    TCK(cudaMemcpy(d_sin,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));

    printf("[load] weights %s ...\n",WD.c_str());
    QwenWeights w=load_weights_qwen(WD.c_str(),c);

    const size_t SH=(size_t)S*H;
    float *d_logits,*d_hidden,*d_prenorm;
    TCK(cudaMalloc(&d_logits,(size_t)S*V*4)); TCK(cudaMalloc(&d_hidden,(size_t)(L+1)*SH*4)); TCK(cudaMalloc(&d_prenorm,SH*4));
    printf("[run] forward...\n");
    qwen35_forward(c,w,d_ids,S,d_cos,d_sin,d_logits,d_hidden,d_prenorm);

    std::vector<float> hidden((size_t)(L+1)*SH); TCK(cudaMemcpy(hidden.data(),d_hidden,hidden.size()*4,cudaMemcpyDeviceToHost));
    std::vector<float> logits((size_t)S*V); TCK(cudaMemcpy(logits.data(),d_logits,logits.size()*4,cudaMemcpyDeviceToHost));
    std::vector<float> prenorm(SH); TCK(cudaMemcpy(prenorm.data(),d_prenorm,SH*4,cudaMemcpyDeviceToHost));

    // (a) per-layer hidden vs golden_q4
    printf("\n=== per-layer hidden: engine vs golden_q4 (max abs err) ===\n");
    double worst=0; int wl=-1;
    for(int l=0;l<=L;l++){
        auto g=rf(GD+"/q4_hs_"+std::to_string(l)+".bin",SH);
        std::vector<float> e(hidden.begin()+(size_t)l*SH,hidden.begin()+(size_t)(l+1)*SH);
        double err=mae(e,g); if(err>worst){worst=err;wl=l;}
        if(l<=3||l>=L-1||err>0.02) printf("  hs[%2d] (%s) err=%.4e\n",l, l<L?(c.layer_is_full[l]?"full":"gdn "):"norm",err);
    }
    printf("  → worst per-layer = %.4e at layer %d\n",worst,wl);

    // (b) pre-norm hidden (MTP 입력)
    auto gpre=rf(GD+"/q4_prenorm.bin",SH);
    double pre_err=mae(prenorm,gpre);
    printf("\n=== pre-norm hidden (MTP 입력) vs golden_q4 = %.4e ===\n",pre_err);

    // (c) logits
    auto glq=rf(GD+"/q4_logits.bin",(size_t)S*V), glf=rf(GD+"/fp32_logits.bin",(size_t)S*V);
    double eq=mae(logits,glq), ef=mae(logits,glf);
    printf("\n=== final logits ===\n  vs golden_q4   = %.4e (정확성)\n  vs golden_fp32 = %.4e (quant budget)\n",eq,ef);
    auto am=[&](const std::vector<float>&lg,int pos){int b=0;float bv=lg[(size_t)pos*V];for(int j=1;j<V;j++){float x=lg[(size_t)pos*V+j];if(x>bv){bv=x;b=j;}}return b;};
    printf("  argmax: eng=["); bool mq=true,mf=true;
    for(int p=0;p<S;p++){int ae=am(logits,p),aq=am(glq,p),af=am(glf,p);printf("%d ",ae);if(ae!=aq)mq=false;if(ae!=af)mf=false;}
    printf("]\n  argmax match vs q4=%s vs fp32=%s; last-tok eng=%d fp32=%d\n",mq?"ALL":"partial",mf?"ALL":"partial",am(logits,S-1),am(glf,S-1));

    free_weights_qwen(w); free_config_qwen(c);
    cudaFree(d_ids);cudaFree(d_cos);cudaFree(d_sin);cudaFree(d_logits);cudaFree(d_hidden);cudaFree(d_prenorm);

    bool pass=(eq<1e-2)&&(worst<1e-2)&&(pre_err<1e-2)&&mq;  // golden_q4 정확성 기준 (engine==q4 graph)
    printf("\n[test_qwen35] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
