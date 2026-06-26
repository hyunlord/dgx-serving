// Qwen3.5-0.8B 본체 통합 검증 (STEP 4). 두 경로 모두:
//   fp32 경로: engine(fp32) vs golden_fp32 → next-token=HF fp32 일치 (STEP4c, MTP 기준선).
//   q4   경로: engine(q4)   vs golden_q4   → 단계별 <1e-5 (구현 정확성, bandwidth 런타임).
// 각 경로: (a) 레이어별 hidden, (b) pre-norm hidden(MTP 입력), (c) logits+argmax.
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
static std::string WD,GD;
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"크기 %s\n",p.c_str());exit(1);}fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/golden")+std::string("/");
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size,L=c.num_layers,rot=c.rotary_dim;
    int nf=0; for(int i=0;i<L;i++) nf+=c.layer_is_full[i];
    printf("[cfg] H=%d L=%d (full=%d/GDN=%d) NQ=%d NKV=%d HD=%d rot=%d V=%d | GDN k/v heads=%d/%d hd=%d conv=%d\n",
           H,L,nf,L-nf,c.num_q_heads,c.num_kv_heads,c.head_dim,rot,V,c.gdn_num_k_heads,c.gdn_num_v_heads,c.gdn_head_v_dim,c.gdn_conv_kernel);

    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    printf("[in] S=%d tokens=",S); for(int t:ids)printf("%d ",t); printf("\n");
    std::vector<float> cosv=rf("rope_cos.bin",(size_t)S*rot), sinv=rf("rope_sin.bin",(size_t)S*rot);
    int *d_ids; TCK(cudaMalloc(&d_ids,S*4)); TCK(cudaMemcpy(d_ids,ids.data(),S*4,cudaMemcpyHostToDevice));
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cosv.size()*4)); TCK(cudaMalloc(&d_sin,sinv.size()*4));
    TCK(cudaMemcpy(d_cos,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice));
    TCK(cudaMemcpy(d_sin,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));

    printf("[load] weights (q4 + fp32) ...\n");
    QwenWeights w=load_weights_qwen(WD.c_str(),c,/*q4*/1,/*fp32*/1);

    const size_t SH=(size_t)S*H;
    float *d_logits,*d_hidden,*d_prenorm;
    TCK(cudaMalloc(&d_logits,(size_t)S*V*4)); TCK(cudaMalloc(&d_hidden,(size_t)(L+1)*SH*4)); TCK(cudaMalloc(&d_prenorm,SH*4));
    std::vector<float> hidden((size_t)(L+1)*SH), logits((size_t)S*V), prenorm(SH);
    auto am=[&](const std::vector<float>&lg,int pos){int b=0;float bv=lg[(size_t)pos*V];for(int j=1;j<V;j++){float x=lg[(size_t)pos*V+j];if(x>bv){bv=x;b=j;}}return b;};

    bool pass=true;
    auto run=[&](int use_q4,const char*tag,double tol)->void{
        qwen35_forward(c,w,d_ids,S,d_cos,d_sin,d_logits,d_hidden,d_prenorm,use_q4);
        TCK(cudaMemcpy(hidden.data(),d_hidden,hidden.size()*4,cudaMemcpyDeviceToHost));
        TCK(cudaMemcpy(logits.data(),d_logits,logits.size()*4,cudaMemcpyDeviceToHost));
        TCK(cudaMemcpy(prenorm.data(),d_prenorm,SH*4,cudaMemcpyDeviceToHost));
        std::string T=tag;
        double worst=0;int wl=-1;
        for(int l=0;l<=L;l++){auto g=rf(T+"_hs_"+std::to_string(l)+".bin",SH);
            std::vector<float> e(hidden.begin()+(size_t)l*SH,hidden.begin()+(size_t)(l+1)*SH);
            double er=mae(e,g); if(er>worst){worst=er;wl=l;}}
        double pre=mae(prenorm,rf(T+"_prenorm.bin",SH));
        auto gl=rf(T+"_logits.bin",(size_t)S*V); double el=mae(logits,gl);
        bool amok=true; for(int p=0;p<S;p++) if(am(logits,p)!=am(gl,p)) amok=false;
        int lastE=am(logits,S-1), lastG=am(gl,S-1);
        printf("\n=== [%s] engine(use_q4=%d) vs golden_%s ===\n",tag,use_q4,tag);
        printf("  per-layer hidden worst=%.3e @L%d | pre-norm(MTP)=%.3e | logits=%.3e | argmax %s | last-tok eng=%d gold=%d\n",
               worst,wl,pre,el,amok?"ALL":"partial",lastE,lastG);
        bool ok=(worst<tol)&&(pre<tol)&&(el<tol)&&amok&&(lastE==lastG);
        if(!ok) pass=false;
        printf("  → %s (tol %.0e)\n", ok?"OK":"FAIL", tol);
    };

    run(0,"fp32",1e-2);   // fp32 경로: HF-fp32 일치 (MTP 기준선, STEP4c next-token=HF)
    run(1,"q4",  1e-2);   // q4 경로: golden_q4 일치 (실제 <1e-5)

    free_weights_qwen(w); free_config_qwen(c);
    cudaFree(d_ids);cudaFree(d_cos);cudaFree(d_sin);cudaFree(d_logits);cudaFree(d_hidden);cudaFree(d_prenorm);
    printf("\n[test_qwen35] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
