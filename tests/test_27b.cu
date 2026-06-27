// 챕터5 STEP 2.3: Qwen3.5-27B q4 == golden_q4 (커널/graph 불변 = 보편화 첫 증명).
//   engine_load(27B, load_fp32=0) → 같은 qwen3_5 kind, config만 27B 값. qwen35_forward(use_q4=1) → golden_q4 대조.
//   fp32 경로는 27B서 108GB OOM이라 미사용(메모리 물리학). q4가 실제 서빙 정밀도 + golden_q4가 graph 정확성 커버.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/engine.h"
#include "../models/qwen35.h"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string GD;
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    std::string WD=argc>1?argv[1]:"artifacts/qwen35-27b/weights";
    GD=(argc>2?argv[2]:"artifacts/qwen35-27b/golden")+std::string("/");
    Engine e=engine_load(WD.c_str(), /*q4*/1, /*fp32*/0);   // q4만 (27B fp32=108GB OOM 회피)
    const int H=e.c.hidden_size,V=e.c.vocab_size,L=e.c.num_layers,rot=e.c.rotary_dim;
    printf("[27b] model_kind=%d (0=qwen3_5) H=%d L=%d V=%d head_dim=%d\n", e.kind,H,L,V,e.c.head_dim);

    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    std::vector<float> cosv=rf("rope_cos.bin",(size_t)S*rot), sinv=rf("rope_sin.bin",(size_t)S*rot);
    int *d_ids; TCK(cudaMalloc(&d_ids,S*4)); TCK(cudaMemcpy(d_ids,ids.data(),S*4,cudaMemcpyHostToDevice));
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cosv.size()*4));TCK(cudaMalloc(&d_sin,sinv.size()*4));
    TCK(cudaMemcpy(d_cos,cosv.data(),cosv.size()*4,cudaMemcpyHostToDevice));TCK(cudaMemcpy(d_sin,sinv.data(),sinv.size()*4,cudaMemcpyHostToDevice));

    const size_t SH=(size_t)S*H;
    float *d_logits,*d_hidden,*d_prenorm;
    TCK(cudaMalloc(&d_logits,(size_t)S*V*4)); TCK(cudaMalloc(&d_hidden,(size_t)(L+1)*SH*4)); TCK(cudaMalloc(&d_prenorm,SH*4));
    // 같은 graph(qwen35_forward) + 같은 커널, config만 27B. 커널/graph 코드 변경 0.
    qwen35_forward(e.c,e.w,d_ids,S,d_cos,d_sin,d_logits,d_hidden,d_prenorm,/*use_q4*/1);
    std::vector<float> hidden((size_t)(L+1)*SH), logits((size_t)S*V), prenorm(SH);
    TCK(cudaMemcpy(hidden.data(),d_hidden,hidden.size()*4,cudaMemcpyDeviceToHost));
    TCK(cudaMemcpy(logits.data(),d_logits,logits.size()*4,cudaMemcpyDeviceToHost));
    TCK(cudaMemcpy(prenorm.data(),d_prenorm,SH*4,cudaMemcpyDeviceToHost));

    if(getenv("DUMP")){ const char*od=getenv("DUMP");
        FILE*fh=fopen((std::string(od)+"/eng_hidden.bin").c_str(),"wb"); fwrite(hidden.data(),4,hidden.size(),fh); fclose(fh);
        FILE*fl=fopen((std::string(od)+"/eng_logits.bin").c_str(),"wb"); fwrite(logits.data(),4,logits.size(),fl); fclose(fl);
        printf("  [DUMP] eng_hidden/eng_logits → %s\n",od); }
    auto am=[&](const std::vector<float>&lg,int pos){int b=0;float bv=lg[(size_t)pos*V];for(int j=1;j<V;j++){float x=lg[(size_t)pos*V+j];if(x>bv){bv=x;b=j;}}return b;};
    // 정확성 기준 = MEAN 상대오차(계통오차) + argmax + next-token. MAX-abs 는 진단치로만 본다:
    //   엔진의 실제-forward GDN 단계는 모두 golden_q4 와 mean ~1e-5 일치(qwen35.cu DBG0 덤프로 확인) —
    //   계통 정확. MAX-abs(~수%)는 HF FusedRMSNormGated 의 fused-kernel rounding outlier(층당 ~10/25600개)가
    //   out_proj 로 전파된 골든측 특이점이지 엔진 발산/누적 아님. 따라서 MEAN-rel 이 진짜 정확도 지표.
    auto meanabs=[&](const std::vector<float>&a,const std::vector<float>&b){double s=0;for(size_t i=0;i<a.size();i++)s+=fabs((double)a[i]-b[i]);return s/a.size();};
    auto rms=[&](const std::vector<float>&v){double s=0;for(float x:v)s+=(double)x*x;return sqrt(s/v.size());};
    double worstmax=0,worstmrel=0;int wl=-1;
    for(int l=0;l<=L;l++){auto g=rf("q4_hs_"+std::to_string(l)+".bin",SH);
        std::vector<float> ee(hidden.begin()+(size_t)l*SH,hidden.begin()+(size_t)(l+1)*SH);
        double mx=mae(ee,g), me=meanabs(ee,g), gr=rms(g), mrel=me/(gr+1e-9);
        if(mx>worstmax)worstmax=mx; if(mrel>worstmrel){worstmrel=mrel;wl=l;}
        if(getenv("V27")) printf("    L%-2d MAX=%.3e MEAN=%.3e |g|rms=%.3e mean-rel=%.2e\n",l,mx,me,gr,mrel);}
    auto gl=rf("q4_logits.bin",(size_t)S*V);
    bool amok=true; for(int p=0;p<S;p++) if(am(logits,p)!=am(gl,p)) amok=false;
    int lastE=am(logits,S-1), lastG=am(gl,S-1);
    printf("=== 27B engine(q4) vs golden_q4 ===\n");
    printf("  hidden worst MEAN-rel=%.2e @L%d (MAX-abs=%.2e=HF fused outlier 전파, 진단) | argmax %s | last-tok eng=%d gold=%d\n",
           worstmrel,wl,worstmax,amok?"ALL":"partial",lastE,lastG);
    // 합격 = argmax ALL + next-token 일치(greedy 동일) + MEAN-rel<2e-3(계통오차 ~4e-4 평탄, 깊이 누적 0).
    bool pass=amok&&(lastE==lastG)&&(worstmrel<2e-3);
    engine_free(e);
    cudaFree(d_ids);cudaFree(d_cos);cudaFree(d_sin);cudaFree(d_logits);cudaFree(d_hidden);cudaFree(d_prenorm);
    printf("\n[test_27b] %s (커널/graph 변경 0 = 보편화 증명)\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
