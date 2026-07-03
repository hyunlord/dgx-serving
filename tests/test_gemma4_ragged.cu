// Gemma4 ragged decode_batch 게이트 (STEP 1). 슬롯마다 다른 위치(연속 배칭 본질).
//   시나리오: B=2, 같은 시퀀스를 1스텝 staggered로 진행 → 매 스텝 두 슬롯이 다른 위치(pos_b=[t, t-1]).
//   게이트: ragged batch 슬롯0 == single decode[t] bit-identical, 슬롯1 == single[t-1] bit-identical.
//     → 슬롯별 rope 위치(rope_partial_pos) + 슬롯별 attendlen(attn base_b) + 슬롯별 KV append 정확성 검증.
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
    const int V=c.vocab_size;
    const int hd_f=c.global_head_dim?c.global_head_dim:c.head_dim, hd_s=c.head_dim;
    std::vector<int> full=ri("gen_tokens.bin"); int total=full.size();
    auto cf=rf("gen_rope_full_cos.bin",(size_t)total*hd_f), sf=rf("gen_rope_full_sin.bin",(size_t)total*hd_f);
    auto cs=rf("gen_rope_sliding_cos.bin",(size_t)total*hd_s), ss=rf("gen_rope_sliding_sin.bin",(size_t)total*hd_s);
    auto up=[&](std::vector<float>&h){float*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;};
    float *d_cf=up(cf),*d_sf=up(sf),*d_cs=up(cs),*d_ss=up(ss);
    G4Weights w=load_weights_gemma4(WD.c_str(),c,1,0);
    auto am=[&](const float*lg){int b=0;float bv=lg[0];for(int j=1;j<V;j++)if(lg[j]>bv){bv=lg[j];b=j;}return b;};

    // 기준선: single decode 순차, 각 위치 t의 logits 저장 (같은 gemma4_decode_batch B=1로 — canonical 동일 커널)
    int N=total-1;   // 위치 0..N-1 decode
    std::vector<std::vector<float>> single(N, std::vector<float>(V));
    { G4BatchState s1=alloc_g4_batch_state(c,1,total+4);
      int *d_t; TCK(cudaMalloc(&d_t,4)); float *d_l; TCK(cudaMalloc(&d_l,(size_t)V*4));
      for(int t=0;t<N;t++){ int tok=full[t]; TCK(cudaMemcpy(d_t,&tok,4,cudaMemcpyHostToDevice));
        gemma4_decode_batch(c,w,s1,d_t,d_cf,d_sf,d_cs,d_ss,d_l,1,1);   // same-length(B=1), n_past 전진
        TCK(cudaMemcpy(single[t].data(),d_l,V*4,cudaMemcpyDeviceToHost)); }
      cudaFree(d_t); cudaFree(d_l); free_g4_batch_state(s1); }

    // ragged: B=2 슬롯을 1스텝 staggered. 슬롯0=위치 t, 슬롯1=위치 t-1. 각 슬롯 자기 KV 독립 append.
    G4BatchState s2=alloc_g4_batch_state(c,2,total+4);
    int *d_tok2; TCK(cudaMalloc(&d_tok2,2*4));
    int *d_pos2; TCK(cudaMalloc(&d_pos2,2*4));
    float *d_l2; TCK(cudaMalloc(&d_l2,(size_t)2*V*4));
    std::vector<float> l2((size_t)2*V);
    double d0=0,d1=0; int checks=0, am_ok=0;
    // t=0 워밍: 두 슬롯 위치0 채움(슬롯0의 KV[0] 누락 방지 — 이후 슬롯0=위치 t는 0..t 온전).
    { int hp0[2]={0,0}; int tk0[2]={full[0],full[0]}; TCK(cudaMemcpy(d_pos2,hp0,2*4,cudaMemcpyHostToDevice)); TCK(cudaMemcpy(d_tok2,tk0,2*4,cudaMemcpyHostToDevice));
      gemma4_decode_batch(c,w,s2,d_tok2,d_cf,d_sf,d_cs,d_ss,d_l2,0,1,d_pos2,hp0); }
    for(int t=1;t<N;t++){
        int hp[2]={t, t-1}; int tk[2]={full[t], full[t-1]};   // 슬롯0=위치 t(0..t 온전), 슬롯1=위치 t-1(위치0 재처리 무해=같은 값)
        TCK(cudaMemcpy(d_pos2,hp,2*4,cudaMemcpyHostToDevice));
        TCK(cudaMemcpy(d_tok2,tk,2*4,cudaMemcpyHostToDevice));
        gemma4_decode_batch(c,w,s2,d_tok2,d_cf,d_sf,d_cs,d_ss,d_l2,0,1,d_pos2,hp);   // ragged
        TCK(cudaMemcpy(l2.data(),d_l2,l2.size()*4,cudaMemcpyDeviceToHost));
        double e0=0,e1=0;
        for(int j=0;j<V;j++){ e0=fmax(e0,fabs((double)l2[j]-single[t][j])); e1=fmax(e1,fabs((double)l2[(size_t)V+j]-single[t-1][j])); }
        d0=fmax(d0,e0); d1=fmax(d1,e1);
        checks++;
        if(am(&l2[0])==am(single[t].data()) && am(&l2[V])==am(single[t-1].data())) am_ok++;
    }
    printf("[cfg] total=%d N=%d — ragged B=2 staggered(슬롯0=t, 슬롯1=t-1)\n",total,N);
    printf("(1) 슬롯0(위치 t) vs single[t] max|Δ| = %.3e\n", d0);
    printf("(2) 슬롯1(위치 t-1) vs single[t-1] max|Δ| = %.3e\n", d1);
    printf("(3) argmax 양슬롯 일치: %d/%d\n", am_ok, checks);
    bool pass=(d0==0.0)&&(d1==0.0)&&(am_ok==checks)&&checks>0;
    printf("[test_gemma4_ragged] %s (게이트=ragged 슬롯별 위치 == single bit-identical)\n", pass?"PASS":"FAIL");
    cudaFree(d_tok2);cudaFree(d_pos2);cudaFree(d_l2); free_g4_batch_state(s2);
    free_weights_gemma4(w); free_config_gemma4(c);
    return pass?0:1;
}
