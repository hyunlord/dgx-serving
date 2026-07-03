// Gemma4 decode 연속성 게이트 (M4 STEP A-1, 진짜 연속성).
//   프리필(base=0,S) → base=S+t에 1토큰씩 decode → 각 스텝 argmax == HF greedy 다음 토큰(gen_meta).
//   전체 위치 rope(gen_rope_*, [total,hd])로 decode rope offset. KV 이어받기 정확성 게이트.
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
    // gen_meta: prompt_tokens + gen_steps. gen_tokens=[total], gen_rope_*=[total,hd]
    std::vector<int> full=ri("gen_tokens.bin"); int total=full.size();
    // prompt 길이 S = total - n. n은 gen_rope 크기로 역산(정확히는 meta지만 파서 없음 → S를 프리필 첫 스텝으로 탐지)
    // 간단히: gen_meta.json 대신 첫 프롬프트를 token_ids.bin(=프롬프트)로.
    std::vector<int> prompt=ri("token_ids.bin"); int S=prompt.size();
    int n=total-S;
    printf("[cfg] S=%d n=%d total=%d hd_f=%d hd_s=%d\n",S,n,total,hd_f,hd_s);
    auto cf=rf("gen_rope_full_cos.bin",(size_t)total*hd_f), sf=rf("gen_rope_full_sin.bin",(size_t)total*hd_f);
    auto cs=rf("gen_rope_sliding_cos.bin",(size_t)total*hd_s), ss=rf("gen_rope_sliding_sin.bin",(size_t)total*hd_s);
    auto up=[&](std::vector<float>&h){float*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;};
    float *d_cf=up(cf),*d_sf=up(sf),*d_cs=up(cs),*d_ss=up(ss);

    G4Weights w=load_weights_gemma4(WD.c_str(),c,1,0);
    G4DecodeState st=alloc_g4_decode_state(c, total+4);
    auto am=[&](const std::vector<float>&lg){int b=0;float bv=lg[0];for(int j=1;j<V;j++)if(lg[j]>bv){bv=lg[j];b=j;}return b;};

    // 프리필(base=0, S) last_only — 마지막 위치 logits만
    int *d_p; TCK(cudaMalloc(&d_p,S*4)); TCK(cudaMemcpy(d_p,prompt.data(),S*4,cudaMemcpyHostToDevice));
    float *d_l; TCK(cudaMalloc(&d_l,(size_t)V*4));
    gemma4_decode(c,w,st,d_p,S,d_cf,d_sf,d_cs,d_ss,d_l,/*commit*/1,/*use_q4*/1,/*last_only*/1);
    std::vector<float> lg(V); TCK(cudaMemcpy(lg.data(),d_l,V*4,cudaMemcpyDeviceToHost));
    int pred=am(lg), match=0;
    printf("프리필 후 next=%d (golden 첫 gen=%d) %s\n", pred, full[S], pred==full[S]?"OK":"X");
    if(pred==full[S]) match++;

    // decode 스텝: 각 위치 base=S+t, 입력=golden 토큰 full[S+t](teacher-forcing로 KV 정확성 격리), argmax 대조
    int *d_s; TCK(cudaMalloc(&d_s,4));
    for(int t=0;t<n-1;t++){
        int pos=S+t, inp=full[pos];
        TCK(cudaMemcpy(d_s,&inp,4,cudaMemcpyHostToDevice));
        // 전체 rope 테이블 시작 포인터 전달 — gemma4_decode 내부가 base(=st.n_past=pos) offset 적용(이중 오프셋 금지)
        gemma4_decode(c,w,st,d_s,1,d_cf,d_sf,d_cs,d_ss,d_l,1,1,1);
        TCK(cudaMemcpy(lg.data(),d_l,V*4,cudaMemcpyDeviceToHost));
        int p=am(lg); bool ok=(p==full[pos+1]);
        if(ok)match++;
        printf("  step t=%d pos=%d in=%d pred=%d gold=%d %s\n",t,pos,inp,p,full[pos+1],ok?"OK":"X");
    }
    int checks=n;  // 프리필 1 + (n-1) 스텝
    printf("[test_gemma4_gen] 연속성 argmax %d/%d 일치 | n_past=%d\n", match, checks, st.n_past);
    bool pass=(match==checks)&&(st.n_past==total-1);   // 마지막 gen 토큰은 입력 안 함(n-1 스텝)
    printf("[test_gemma4_gen] %s\n", pass?"PASS":"FAIL");
    free_g4_decode_state(st); free_weights_gemma4(w); free_config_gemma4(c);
    return pass?0:1;
}
