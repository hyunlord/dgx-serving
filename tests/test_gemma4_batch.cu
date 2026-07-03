// Gemma4 batch decode 게이트 (M4 A-3-2). same-length(모든 슬롯 동일 시퀀스·위치).
//   (a) batch(B=4) 슬롯 서로 bit-identical (같은 입력 → 슬롯 독립성+attn 격리)
//   (b) batch(B=4) 슬롯0 == batch(B=1) bit-identical (gemm_batch B-무관 canonical)
//   (c) argmax == HF greedy(gen 골든) 연속성 B로 확장
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

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/gemma4-31b/weights"; GD=(argc>2?argv[2]:"artifacts/gemma4-31b/golden")+std::string("/");
    ModelConfig c=load_config_gemma4((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size;
    const int hd_f=c.global_head_dim?c.global_head_dim:c.head_dim, hd_s=c.head_dim;
    std::vector<int> full=ri("gen_tokens.bin"); int total=full.size();
    std::vector<int> prompt=ri("token_ids.bin"); int S=prompt.size();
    auto cf=rf("gen_rope_full_cos.bin",(size_t)total*hd_f), sf=rf("gen_rope_full_sin.bin",(size_t)total*hd_f);
    auto cs=rf("gen_rope_sliding_cos.bin",(size_t)total*hd_s), ss=rf("gen_rope_sliding_sin.bin",(size_t)total*hd_s);
    auto up=[&](std::vector<float>&h){float*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;};
    float *d_cf=up(cf),*d_sf=up(sf),*d_cs=up(cs),*d_ss=up(ss);
    G4Weights w=load_weights_gemma4(WD.c_str(),c,1,0);
    auto am=[&](const float*lg){int b=0;float bv=lg[0];for(int j=1;j<V;j++)if(lg[j]>bv){bv=lg[j];b=j;}return b;};

    const int B=4;
    G4BatchState s4=alloc_g4_batch_state(c, B, total+4);
    G4BatchState s1=alloc_g4_batch_state(c, 1, total+4);
    int *d_t4; TCK(cudaMalloc(&d_t4,B*4)); int *d_t1; TCK(cudaMalloc(&d_t1,4));
    float *d_l4; TCK(cudaMalloc(&d_l4,(size_t)B*V*4)); float *d_l1; TCK(cudaMalloc(&d_l1,(size_t)V*4));
    std::vector<float> l4((size_t)B*V), l1(V);

    double slot_diff=0, b_diff=0; int am_match=0, am_checks=0;
    // 전체 시퀀스를 1토큰씩 same-length decode (프롬프트 포함 — 순수 decode 경로 bit-identity 검증)
    for(int t=0;t<total-1;t++){
        int tok=full[t];
        std::vector<int> t4(B,tok); TCK(cudaMemcpy(d_t4,t4.data(),B*4,cudaMemcpyHostToDevice));
        TCK(cudaMemcpy(d_t1,&tok,4,cudaMemcpyHostToDevice));
        gemma4_decode_batch(c,w,s4,d_t4,d_cf,d_sf,d_cs,d_ss,d_l4,1,1);
        gemma4_decode_batch(c,w,s1,d_t1,d_cf,d_sf,d_cs,d_ss,d_l1,1,1);
        TCK(cudaMemcpy(l4.data(),d_l4,l4.size()*4,cudaMemcpyDeviceToHost));
        TCK(cudaMemcpy(l1.data(),d_l1,V*4,cudaMemcpyDeviceToHost));
        // (a) 슬롯 서로
        for(int b=1;b<B;b++)for(int j=0;j<V;j++) slot_diff=fmax(slot_diff,fabs((double)l4[(size_t)b*V+j]-l4[j]));
        // (b) 슬롯0 vs B=1
        for(int j=0;j<V;j++) b_diff=fmax(b_diff,fabs((double)l4[j]-l1[j]));
        // (c) argmax 연속성 (프롬프트 이후 위치만 — teacher forcing)
        if(t>=S-1){ am_checks++; if(am(l4.data())==full[t+1]) am_match++; }
    }
    printf("[cfg] B=%d S=%d total=%d\n",B,S,total);
    printf("(a) batch 슬롯 서로 max|Δ| = %.3e (같은 입력 → 0 기대: 슬롯 독립+attn 격리)\n", slot_diff);
    printf("(b) batch(B=4) 슬롯0 vs batch(B=1) max|Δ| = %.3e (gemm_batch B-무관 → 0 기대)\n", b_diff);
    printf("(c) argmax vs HF greedy: %d/%d\n", am_match, am_checks);
    bool pass = (slot_diff==0.0) && (b_diff==0.0) && (am_match==am_checks) && am_checks>0;
    printf("[test_gemma4_batch] %s (게이트=slot bit-id + B-무관 bit-id + argmax 연속성)\n", pass?"PASS":"FAIL");
    free_g4_batch_state(s4); free_g4_batch_state(s1); free_weights_gemma4(w); free_config_gemma4(c);
    return pass?0:1;
}
