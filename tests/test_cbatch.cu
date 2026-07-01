// 챕터9 continuous batching STEP A+B: ragged batch(per-seq 위치) + staggered 진입 → 각 요청 == solo bit-identical.
//   요청 A(step0 시작), B(step Kenter staggered 진입) → 위치 다름(ragged). st.B 동적(1→2). 각 slot: prompt 1토큰/step
//   ingest 후 decode. h_pos_b=slot별 위치. batch-생성 토큰 == 그 요청 solo greedy(이웃·진입시점 무관, mism=0).
//   최소 scheduler(slot 재사용 없이 staggered 진입만) — ragged 위치 정확성 검증이 목적.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#include "../models/qwen35_batch.h"
#include "../kernels/rope.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int uq=argc>3?atoi(argv[3]):0;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, rot=c.rotary_dim, H=c.hidden_size;
    std::vector<int> ids=ri("token_ids.bin");
    // 요청 A/B: 서로 다른 프롬프트(길이 무관 — staggered 진입이 위치 차이 만듦). ngen 각.
    std::vector<int> pA(ids.begin(), ids.begin()+5);           // 프롬프트 A (5토큰)
    std::vector<int> pB(ids.begin()+1, ids.begin()+9);         // 프롬프트 B (8토큰, 다른 내용)
    int Na=10, Nb=10, Kenter=6;                                // B는 step Kenter에 진입(그때 A는 이미 진행 → 위치 다름)
    int maxlen=64;
    printf("[cbatch] uq=%d | A(prompt %zu, gen %d) step0 시작, B(prompt %zu, gen %d) step %d 진입(staggered)\n",
           uq,pA.size(),Na,pB.size(),Nb,Kenter);
    std::vector<float> cl=rf("rope_cos_long.bin",(size_t)128*rot), sl=rf("rope_sin_long.bin",(size_t)128*rot);
    float*d_cos=up(cl),*d_sin=up(sl);
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,uq==0?1:0);
    int*d_tok; CK(cudaMalloc(&d_tok,8*4)); float*d_logit; CK(cudaMalloc(&d_logit,(size_t)8*V*4));
    auto amax=[&](const std::vector<float>&lg,int b){int bi=0;float bv=lg[(size_t)b*V];for(int j=1;j<V;j++){float z=lg[(size_t)b*V+j];if(z>bv){bv=z;bi=j;}}return bi;};

    // ---- solo greedy 각 요청 (ground truth, single decode) ----
    auto solo=[&](std::vector<int> prompt,int ng){
        DecodeState st=alloc_decode_state(c,maxlen);
        int P=prompt.size(); CK(cudaMemcpy(d_tok,prompt.data(),P*4,cudaMemcpyHostToDevice));
        float*dl; CK(cudaMalloc(&dl,(size_t)P*V*4));
        qwen35_decode(c,w,st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,uq);
        std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
        int cur=amax(lg,P-1); std::vector<int> out;
        for(int i=0;i<ng;i++){ out.push_back(cur); CK(cudaMemcpy(d_tok,&cur,4,cudaMemcpyHostToDevice));
            float*o; CK(cudaMalloc(&o,(size_t)V*4)); qwen35_decode(c,w,st,d_tok,1,d_cos,d_sin,o,nullptr,1,nullptr,uq);
            std::vector<float> hl(V); CK(cudaMemcpy(hl.data(),o,(size_t)V*4,cudaMemcpyDeviceToHost)); cudaFree(o);
            int bi=0;float bv=hl[0];for(int j=1;j<V;j++)if(hl[j]>bv){bv=hl[j];bi=j;} cur=bi; }
        cudaFree(dl); free_decode_state(st); return out;
    };
    std::vector<int> soloA=solo(pA,Na), soloB=solo(pB,Nb);

    // ---- 연속배칭 스케줄러 (2 slot, staggered 진입) ----
    // slot 상태: seq(프롬프트+생성), pos(현재 위치), phase(ingest 프롬프트 / decode), 생성수집.
    struct Slot{ std::vector<int> prompt; int P,ng; int pos=0,gened=0; bool active=false,done=false; std::vector<int> gen; };
    BatchDecodeState st=alloc_batch_decode_state(c,2,maxlen);
    Slot S[2]; S[0]={pA,(int)pA.size(),Na}; S[1]={pB,(int)pB.size(),Nb};
    S[0].active=true;                     // A는 step0부터
    int step=0; bool bstarted=false;
    // 물리 slot 고정(A=idx0, B=idx1, 재사용 없음). st.B는 "시작된 최고 물리 idx+1"을 덮음 — B 진입 후 st.B=2 유지
    //   (A 종료해도 st.B 안 줄임 — 줄이면 B(idx1) 접근 불가·A KV 오염). 종료 slot은 dummy 토큰(무시).
    while(!(S[0].done && S[1].done)){
        if(step==Kenter){ bstarted=true; S[1].active=true; }   // B staggered 진입(fresh slot1)
        int B = bstarted ? 2 : 1; st.B=B;
        std::vector<int> toks(B), pos(B);
        for(int b=0;b<B;b++){ Slot&s=S[b];
            int t = s.done ? s.gen.back()                                   // 종료: dummy(마지막 토큰), pos 고정
                  : (s.pos < s.P ? s.prompt[s.pos] : (s.gen.empty()? s.prompt[s.P-1] : s.gen.back()));
            toks[b]=t; pos[b]=s.pos; }
        CK(cudaMemcpy(d_tok,toks.data(),B*4,cudaMemcpyHostToDevice));
        qwen35_decode_batch(c,w,st,d_tok,d_cos,d_sin,d_logit,uq, pos.data());   // ragged: per-seq 위치
        std::vector<float> lg((size_t)B*V); CK(cudaMemcpy(lg.data(),d_logit,lg.size()*4,cudaMemcpyDeviceToHost));
        for(int b=0;b<B;b++){ Slot&s=S[b]; if(s.done) continue;             // 종료 slot(dummy) 무시·pos 고정
            if(s.pos < s.P-1){ s.pos++; }                                   // 프롬프트 ingest(마지막 전) — logits 무시
            else { int g=amax(lg,b); s.gen.push_back(g); s.gened++; s.pos++;// 마지막 프롬프트 or decode → 생성
                   if(s.gened>=s.ng) s.done=true; }
        }
        step++;
        if(step>maxlen+40) break;  // 안전
    }
    // ---- 검증: batch-생성 == solo (mism=0) ----
    int mmA=0, mmB=0;
    for(int i=0;i<Na;i++) if(i>=(int)S[0].gen.size()||S[0].gen[i]!=soloA[i]) mmA++;
    for(int i=0;i<Nb;i++) if(i>=(int)S[1].gen.size()||S[1].gen[i]!=soloB[i]) mmB++;
    printf("  A: solo   "); for(int i=0;i<Na;i++)printf("%d ",soloA[i]); printf("\n");
    printf("  A: cbatch "); for(int x:S[0].gen)printf("%d ",x); printf("\n");
    printf("  B: solo   "); for(int i=0;i<Nb;i++)printf("%d ",soloB[i]); printf("\n");
    printf("  B: cbatch "); for(int x:S[1].gen)printf("%d ",x); printf("\n");
    printf("  mismatch A=%d/%d B=%d/%d\n",mmA,Na,mmB,Nb);
    bool pass=(mmA==0 && mmB==0);
    printf("\n[test_cbatch] %s (연속배칭 staggered 진입, 각 요청 == solo bit-identical = ragged 위치 정확)\n",pass?"PASS":"FAIL");
    free_batch_decode_state(st); free_weights_qwen(w); free_config_qwen(c);
    return pass?0:1;
}
