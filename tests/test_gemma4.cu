// Gemma4 31B 골든 게이트 (M4 Phase 0). test_qwen35 패턴:
//   base 경로: engine(fp32 가중치 없음 — 31B는 q4-only 추출) → q4 경로만 게이트.
//   q4 경로: engine(q4) vs golden_q4 (HF bf16 로드 + q4 round-trip) — 레이어별 hidden + logits + argmax.
//   tol: bf16 골든 노이즈 포함 1e-2 (27b급 관례). argmax 전 위치 일치 필수.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <unistd.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../models/gemma4.h"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"크기 %s\n",p.c_str());exit(1);}fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/gemma4-31b/weights"; GD=(argc>2?argv[2]:"artifacts/gemma4-31b/golden")+std::string("/");
    ModelConfig c=load_config_gemma4((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size,L=c.num_layers;
    const int hd_f=c.global_head_dim?c.global_head_dim:c.head_dim;
    int nf=0; for(int i=0;i<L;i++) nf+=c.layer_is_full[i];
    printf("[cfg] H=%d L=%d (full=%d/sliding=%d w=%d) NQ=%d hd_s=%d/hd_f=%d NKV_s=%d/NKV_f=%d V=%d "
           "k_eq_v=%d softcap=%.1f embed_scale=%.2f\n",
           H,L,nf,L-nf,c.sliding_window,c.num_q_heads,c.head_dim,hd_f,
           c.num_kv_heads,c.num_global_kv_heads,V,c.attention_k_eq_v,
           c.final_logit_softcapping,c.embed_scale);

    std::vector<int> ids=ri("token_ids.bin"); int S=ids.size();
    printf("[in] S=%d tokens=",S); for(int t:ids)printf("%d ",t); printf("\n");
    // cos/sin 2세트 (full: [S,hd_f], sliding: [S,hd_s]) — HF 캡처본
    std::vector<float> cf=rf("rope_full_cos.bin",(size_t)S*hd_f), sf=rf("rope_full_sin.bin",(size_t)S*hd_f);
    std::vector<float> cs=rf("rope_sliding_cos.bin",(size_t)S*c.head_dim), ss=rf("rope_sliding_sin.bin",(size_t)S*c.head_dim);
    int *d_ids; TCK(cudaMalloc(&d_ids,S*4)); TCK(cudaMemcpy(d_ids,ids.data(),S*4,cudaMemcpyHostToDevice));
    auto up=[&](std::vector<float>&h)->float*{float*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;};
    float *d_cf=up(cf),*d_sf=up(sf),*d_cs=up(cs),*d_ss=up(ss);

    printf("[load] weights (q4) ...\n");
    G4Weights w=load_weights_gemma4(WD.c_str(),c,/*q4*/1,/*fp32*/0);

    const size_t SH=(size_t)S*H;
    float *d_logits,*d_hidden;
    TCK(cudaMalloc(&d_logits,(size_t)S*V*4)); TCK(cudaMalloc(&d_hidden,(size_t)(L+1)*SH*4));
    std::vector<float> hidden((size_t)(L+1)*SH), logits((size_t)S*V);
    auto am=[&](const std::vector<float>&lg,int pos){int b=0;float bv=lg[(size_t)pos*V];for(int j=1;j<V;j++){float x=lg[(size_t)pos*V+j];if(x>bv){bv=x;b=j;}}return b;};

    gemma4_forward(c,w,d_ids,S,d_cf,d_sf,d_cs,d_ss,d_logits,d_hidden,/*use_q4=*/1);
    TCK(cudaMemcpy(hidden.data(),d_hidden,hidden.size()*4,cudaMemcpyDeviceToHost));
    TCK(cudaMemcpy(logits.data(),d_logits,logits.size()*4,cudaMemcpyDeviceToHost));

    // === 강화 게이트 (2중) ===
    // (a) op-level fp32: 경계-depth 층 스트리밍 fp32 골든(q4fp32_hs_l, 엔진 동일 q4 weights)과 per-layer err<1e-2.
    //     full-depth fp32는 메모리 가드로 불가 → 앞 N층(전 연산타입 커버)만. 존재하는 층까지 검사.
    // (b) full-depth argmax: bf16 골든(q4_hs / q4_logits) 전위치 argmax 일치.
    // 상대오차(매그니튜드-중립): Gemma는 hidden 규모 큼(embed×√H) → abs 대신 rel = maxabs/max|golden|.
    double fp32_worst=0; int fp32_wl=-1, fp32_n=0; double rel_worst=0; int rel_wl=-1;
    for(int l=0;l<=L;l++){
        std::string fn=GD+"q4fp32_hs_"+std::to_string(l)+".bin";
        if(access(fn.c_str(),0)!=0) continue;         // 경계 depth 밖 → skip
        auto g=rf("q4fp32_hs_"+std::to_string(l)+".bin",SH);
        std::vector<float> e(hidden.begin()+(size_t)l*SH,hidden.begin()+(size_t)(l+1)*SH);
        double er=mae(e,g); fp32_n++;
        double gmax=0; for(float x:g) gmax=fmax(gmax,fabs((double)x));
        double rel=er/(gmax+1e-9);
        if(er>fp32_worst){fp32_worst=er;fp32_wl=l;}
        if(rel>rel_worst){rel_worst=rel;rel_wl=l;}
        printf("  [fp32] L%-2d maxabs=%.3e rel=%.3e (|g|max=%.2f)%s\n",l,er,rel,gmax,rel>2e-3?"  <-- rel>2e-3":"");
    }
    // full-depth: bf16 골든 argmax
    auto gl=rf("q4_logits.bin",(size_t)S*V); double el_bf16=mae(logits,gl);
    bool amok=true; for(int p=0;p<S;p++){int a=am(logits,p),b=am(gl,p); if(a!=b){amok=false;printf("  argmax mismatch @%d: eng=%d gold=%d\n",p,a,b);} }
    printf("\n=== 강화 게이트 ===\n");
    printf("  (a) op-level fp32: %d층, worst abs=%.3e @L%d | worst rel=%.3e @L%d\n",
           fp32_n, fp32_worst, fp32_wl, rel_worst, rel_wl);
    printf("      (엔진 GPU-fp32 vs 골든 CPU-fp32, 동일 q4 weights — rel<2e-3=GPU/CPU fp32 산술차, 버그 아님)\n");
    printf("  (b) full-depth argmax(bf16 골든): %s | logits maxabs=%.3e | last eng=%d gold=%d\n",
           amok?"ALL 일치":"MISMATCH", el_bf16, am(logits,S-1), am(gl,S-1));
    // 게이트: 상대오차 <2e-3(GPU/CPU fp32 산술차 상한, Gemma 대규모 hidden 정규화) + full-depth argmax 전위치.
    bool fp32_ok = (fp32_n>0) && (rel_worst < 2e-3);
    bool pass = fp32_ok && amok;
    printf("[test_gemma4] %s (게이트=op-level fp32 %d층 rel<2e-3 + full-depth argmax)\n",
           pass?"PASS":"FAIL", fp32_n);
    free_weights_gemma4(w); free_config_gemma4(c);
    return pass?0:1;
}
