// solo(qwen35_decode) vs batch-B1(qwen35_decode_batch) 단계별 logit 골든 대조.
//   목적: W4A8 req 궤적서 두 그래프가 어디서/얼마나 갈리나. 첫 분기 step의 top-2 gap vs Δlogit →
//   Δ가 gap보다 크면 borderline flip(양성, 정밀도), 구조적 큰 Δ면 batch graph 버그.
//   fp32·W4A8 둘 다. (CLAUDE.md: 끝값 추측 금지, 단계별 골든 대조.)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#include "../models/qwen35_batch.h"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static void top2(const std::vector<float>&lg,int&a1,float&v1,float&v2){a1=0;v1=lg[0];int a2=0;v2=-1e30f;for(size_t j=1;j<lg.size();j++){if(lg[j]>v1){v2=v1;v1=lg[j];a1=j;}else if(lg[j]>v2){v2=lg[j];a2=j;}}(void)a2;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int uq=argc>3?atoi(argv[3]):2; int ps=argc>4?atoi(argv[4]):1, pe=argc>5?atoi(argv[5]):4; int ng=argc>6?atoi(argv[6]):11;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, rot=c.rotary_dim; int maxlen=48;
    std::vector<int> ids=ri("token_ids.bin");
    std::vector<int> prompt(ids.begin()+ps, ids.begin()+pe); int P=prompt.size();
    std::vector<float> cl=rf("rope_cos_long.bin",(size_t)128*rot), sl=rf("rope_sin_long.bin",(size_t)128*rot);
    float*d_cos=up(cl),*d_sin=up(sl);
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,uq==0?1:0);
    int*d_tok;CK(cudaMalloc(&d_tok,maxlen*4));
    printf("[probe solo vs batch-B1] uq=%d prompt[%d..%d)=%d토큰 ng=%d\n",uq,ps,pe,P,ng);

    // solo: chunk prefill + decode, 각 step logit 저장
    std::vector<std::vector<float>> soloLg;
    { DecodeState st=alloc_decode_state(c,maxlen);
      CK(cudaMemcpy(d_tok,prompt.data(),P*4,cudaMemcpyHostToDevice));
      float*dl;CK(cudaMalloc(&dl,(size_t)P*V*4)); qwen35_decode(c,w,st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,uq);
      std::vector<float> lg((size_t)P*V);CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));cudaFree(dl);
      std::vector<float> last(lg.begin()+(size_t)(P-1)*V, lg.end()); soloLg.push_back(last);
      int cur;float v1,v2;top2(last,cur,v1,v2);
      for(int i=0;i<ng;i++){ CK(cudaMemcpy(d_tok,&cur,4,cudaMemcpyHostToDevice));
        float*o;CK(cudaMalloc(&o,(size_t)V*4));qwen35_decode(c,w,st,d_tok,1,d_cos,d_sin,o,nullptr,1,nullptr,uq);
        std::vector<float> hl(V);CK(cudaMemcpy(hl.data(),o,V*4,cudaMemcpyDeviceToHost));cudaFree(o);
        soloLg.push_back(hl); float a,b;int na;top2(hl,na,a,b);cur=na; }
      free_decode_state(st); }

    // batch B=1: token prefill + decode, 각 step logit 저장
    std::vector<std::vector<float>> batLg;
    { BatchDecodeState st=alloc_batch_decode_state(c,1,maxlen); st.B=1;
      float*d_logit;CK(cudaMalloc(&d_logit,(size_t)V*4));
      int pos=0,cur=0; std::vector<int> seq=prompt;
      for(int stp=0; stp<P+ng; stp++){
        int t = (stp<P)? prompt[stp] : cur; int pp=stp;
        CK(cudaMemcpy(d_tok,&t,4,cudaMemcpyHostToDevice));
        int hp[1]={pp}; qwen35_decode_batch(c,w,st,d_tok,d_cos,d_sin,d_logit,uq,hp);
        std::vector<float> hl(V);CK(cudaMemcpy(hl.data(),d_logit,V*4,cudaMemcpyDeviceToHost));
        if(stp>=P-1){ batLg.push_back(hl); int na;float a,b;top2(hl,na,a,b);cur=na; }
      }
      free_batch_decode_state(st); }

    // 단계별 대조: Δlogit vs top-2 gap
    int n=std::min(soloLg.size(),batLg.size()); int firstflip=-1;
    for(int i=0;i<n;i++){
        float mx=0; for(int j=0;j<V;j++){float d=fabsf(soloLg[i][j]-batLg[i][j]);if(d>mx)mx=d;}
        int sa,ba;float sv1,sv2,bv1,bv2; top2(soloLg[i],sa,sv1,sv2); top2(batLg[i],ba,bv1,bv2);
        float gap=sv1-sv2; bool flip=(sa!=ba);
        printf("  step%2d: max|Δlogit|=%.4e  solo top2gap=%.4e  argmax solo=%d bat=%d %s\n",
               i,mx,gap,sa,ba, flip?"◄FLIP":"");
        if(flip&&firstflip<0)firstflip=i;
    }
    if(firstflip>=0){ float sa,sv1,sv2;int a; top2(soloLg[firstflip],a,sv1,sv2);
        float mx=0;for(int j=0;j<V;j++){float d=fabsf(soloLg[firstflip][j]-batLg[firstflip][j]);if(d>mx)mx=d;}
        printf("  ▶ 첫 FLIP step%d: top2gap=%.4e vs max|Δ|=%.4e → %s\n",firstflip,sv1-sv2,mx,
               mx>=(sv1-sv2)?"Δ≥gap: borderline flip(양성, 정밀도)":"Δ<gap: 구조적 차이 의심");
    } else printf("  ▶ FLIP 없음 — 두 그래프 argmax 완전 일치\n");
    return 0;
}
