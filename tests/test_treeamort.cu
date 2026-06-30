// 챕터9 STEP 3.3: tree 1-forward verify amortize 속도 de-risk (사용자 지적 — STEP 4 풀빌드 전 챕터8 함정 미리).
//   질문: tree verify(Sq=노드수) 1-forward가 weight 1적재 amortize 주나? (노드수만큼 안 늘고, ≈greedy 1토큰)
//   측정(27B W4A8, decode 대역폭-bound 실모델): greedy 1-tok vs tree verify Sq=K(SB) vs linear verify Sq=K.
//     ① amortize factor = K·t(Sq1)/t_tree(K): ≈K면 full amortize(K토큰을 1토큰 시간에), ≈1이면 sequential(amortize 깨짐).
//     ② tree overhead = t_tree(K)/t_linear(K): ≈1이면 tree fork(GDN snapshot/conv walk/attn mask) 오버헤드 무시가능.
//     ③ Sq=16(>8 MMQ): 챕터8 함정(반대역폭) 재현 확인 — tree verify가 SB(≤8) 안에 있어야 함을 정량 대조.
//   ncu dram__bytes는 GB10서 n/a([[mtp-spec-decode-verify-bandwidth]]) → 시간 비율 + 구조 + bit-exact로 amortize 확정.
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
#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rfg(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"open %s 실패\n",(GD+p).c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"open %s 실패\n",(GD+p).c_str());exit(1);}fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static int* up(const std::vector<int>&h){int*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static unsigned* upu(const std::vector<unsigned>&h){unsigned*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-27b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-27b/mtp_golden")+std::string("/");
    int bq4=(argc>3)?atoi(argv[3]):2;        // 기본 W4A8(decode 실경로)
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int rot=c.rotary_dim;
    std::vector<int> prompt=ri("token_ids.bin"); int P=prompt.size(); if(P>16)P=16;   // 짧은 prefix(rope_long 128 한도 + base 작게)
    std::vector<float> cl=rfg("rope_cos_long.bin",(size_t)128*rot), sl=rfg("rope_sin_long.bin",(size_t)128*rot);
    float *d_cos,*d_sin; TCK(cudaMalloc(&d_cos,cl.size()*4)); TCK(cudaMalloc(&d_sin,sl.size()*4));
    TCK(cudaMemcpy(d_cos,cl.data(),cl.size()*4,cudaMemcpyHostToDevice)); TCK(cudaMemcpy(d_sin,sl.data(),sl.size()*4,cudaMemcpyHostToDevice));
    printf("[treeamort] 27B body_q4=%d | prefix P=%d, rope_long=128\n",bq4,P);
    QwenWeights w=load_weights_qwen(WD.c_str(),c, bq4==0?0:1, bq4==0?1:0);
    const int V=c.vocab_size, base=P;
    int*d_pre=up(std::vector<int>(prompt.begin(),prompt.begin()+P));
    float *dl; TCK(cudaMalloc(&dl,(size_t)8*V*4));      // 최대 Sq=8 토큰 logits 여유(Sq=16은 16*V 필요)
    float *dl16; TCK(cudaMalloc(&dl16,(size_t)16*V*4));

    DecodeState st=alloc_decode_state(c, P+32);
    qwen35_decode(c,w,st,d_pre,P,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,nullptr,1);   // prefill commit(마지막 logits만 — dl[8V] overflow 방지) → base=P
    printf("  prefill n_past=%d\n\n",st.n_past);

    // chain tree 자료 (Sq=K): parent=[-1,0,..,K-2], anc=causal 비트마스크, pos=base..base+K-1. 토큰=prompt 반복(타이밍 무관).
    auto mktok=[&](int K){std::vector<int>t(K); for(int i=0;i<K;i++)t[i]=prompt[i%P]; return up(t);};
    auto mkpar=[&](int K){std::vector<int>p(K); p[0]=-1; for(int i=1;i<K;i++)p[i]=i-1; return up(p);};
    auto mkanc=[&](int K){std::vector<unsigned>a(K); for(int i=0;i<K;i++)a[i]=(i>=31)?0xFFFFFFFFu:((1u<<(i+1))-1u); return upu(a);};
    auto mkpos=[&](int K){std::vector<int>x(K); for(int i=0;i<K;i++)x[i]=base+i; return up(x);};

    cudaEvent_t e0,e1; TCK(cudaEventCreate(&e0)); TCK(cudaEventCreate(&e1));
    const int WARM=3, IT=20;
    // tree=1이면 tree 인자, 아니면 linear. lg=logits 버퍼.
    auto timed=[&](int K,bool tree,float*lg)->double{
        int *tok=mktok(K),*par=tree?mkpar(K):nullptr,*pos=tree?mkpos(K):nullptr; unsigned*anc=tree?mkanc(K):nullptr;
        for(int i=0;i<WARM;i++) qwen35_decode(c,w,st,tok,K,d_cos,d_sin,lg,nullptr,0,nullptr,bq4,nullptr,0, par,anc,pos);
        TCK(cudaDeviceSynchronize()); TCK(cudaEventRecord(e0));
        for(int i=0;i<IT;i++)  qwen35_decode(c,w,st,tok,K,d_cos,d_sin,lg,nullptr,0,nullptr,bq4,nullptr,0, par,anc,pos);
        TCK(cudaEventRecord(e1)); TCK(cudaEventSynchronize(e1));
        float ms=0; TCK(cudaEventElapsedTime(&ms,e0,e1)); return ms/IT;
    };

    double t1 = timed(1,false,dl);                                  // greedy 1-tok (gemv_w4a8)
    printf("greedy 1-tok (Sq=1, gemv)          : %.2f ms\n",t1);
    printf("\n  Sq | linear(ms) | tree(ms) | amortize(K·t1/t_tree) | tree/linear | 경로\n");
    printf("  ---+------------+----------+-----------------------+-------------+-----\n");
    for(int K : {2,3,5,7}){
        double tl=timed(K,false,dl), tt=timed(K,true,dl);
        printf("  %2d |   %7.2f  | %7.2f  |        %5.2f×         |    %.3f    | SB(2~8)\n",
               K, tl, tt, (K*t1)/tt, tt/tl);
    }
    // Sq=16: >8 → MMQ 경로(챕터8 함정 대조). tree 16노드(SB 미확장 시 MMQ).
    { int K=16; double tl=timed(K,false,dl16), tt=timed(K,true,dl16);
      printf("  %2d |   %7.2f  | %7.2f  |        %5.2f×         |    %.3f    | MMQ(>8) ← 챕터8 함정 대조\n",
             K, tl, tt, (K*t1)/tt, tt/tl); }

    printf("\n판정: SB(≤8)서 amortize factor가 K에 가까우면(K토큰을 ~1토큰 시간에) → STEP4 풀빌드 정당.\n");
    printf("      tree/linear≈1이면 tree fork 오버헤드(GDN snapshot/conv walk/attn mask) 무시가능.\n");
    printf("      Sq=16 MMQ서 amortize 꺾이면 = 챕터8 함정 재현 → tree>8은 SB 확장 필요(STEP4 기지의 항목).\n");
    free_decode_state(st); free_weights_qwen(w); free_config_qwen(c);
    return 0;
}
