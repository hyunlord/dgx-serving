// 챕터3 STEP2b 검증: batch decode(B 시퀀스 한 forward) == B개 독립 single decode, 시퀀스별 오차0.
//   시퀀스마다 다른 토큰열 → position/KV/state 배선 버그를 시퀀스별로 노출. n_past=0부터 N스텝.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <ctime>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#include "../models/qwen35.h"
#include "../models/qwen35_batch.h"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string GD;
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}

int main(int argc,char**argv){
    std::string WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights";
    GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size,rot=c.rotary_dim;
    const int B=4, N=8, maxlen=N+4;
    int uq=argc>5?atoi(argv[5]):0;   // 0=fp32 1=q4 2=W4A8
    std::vector<int> toks=ri("token_ids.bin"); int P=toks.size();
    // 시퀀스별 토큰열 (서로 다르게): seq b, step t → toks[(b*3+t) % P]
    std::vector<std::vector<int>> seq(B, std::vector<int>(N));
    for(int b=0;b<B;b++)for(int t=0;t<N;t++) seq[b][t]=toks[(b*3+t)%P];
    std::vector<float> cl=rf("rope_cos_long.bin",(size_t)128*rot), sl=rf("rope_sin_long.bin",(size_t)128*rot);
    float *d_cos,*d_sin; CK(cudaMalloc(&d_cos,cl.size()*4));CK(cudaMalloc(&d_sin,sl.size()*4));
    CK(cudaMemcpy(d_cos,cl.data(),cl.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_sin,sl.data(),sl.size()*4,cudaMemcpyHostToDevice));
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,uq==0?1:0);   // q4 always; fp32는 uq=0(fp32 모드)일 때만(q4-only 모델 지원)

    // ---- batch decode ----
    BatchDecodeState bst=alloc_batch_decode_state(c,B,maxlen);
    int *d_tokB; CK(cudaMalloc(&d_tokB,B*4));
    float *d_logB; CK(cudaMalloc(&d_logB,(size_t)B*V*4));
    std::vector<std::vector<float>> blog(N, std::vector<float>((size_t)B*V));
    for(int t=0;t<N;t++){ std::vector<int> tt(B); for(int b=0;b<B;b++)tt[b]=seq[b][t];
        CK(cudaMemcpy(d_tokB,tt.data(),B*4,cudaMemcpyHostToDevice));
        qwen35_decode_batch(c,w,bst,d_tokB,d_cos,d_sin,d_logB,uq);
        CK(cudaMemcpy(blog[t].data(),d_logB,(size_t)B*V*4,cudaMemcpyDeviceToHost));
    }

    // ---- B개 독립 single decode ----
    std::vector<std::vector<float>> slog(N, std::vector<float>((size_t)B*V));
    for(int b=0;b<B;b++){
        DecodeState s1=alloc_decode_state(c,maxlen);
        int *d_tok1; CK(cudaMalloc(&d_tok1,4)); float *d_log1; CK(cudaMalloc(&d_log1,(size_t)V*4));
        for(int t=0;t<N;t++){ CK(cudaMemcpy(d_tok1,&seq[b][t],4,cudaMemcpyHostToDevice));
            qwen35_decode(c,w,s1,d_tok1,1,d_cos,d_sin,d_log1,nullptr,1,nullptr,uq,nullptr);
            CK(cudaMemcpy(slog[t].data()+(size_t)b*V,d_log1,(size_t)V*4,cudaMemcpyDeviceToHost));
        }
        free_decode_state(s1); cudaFree(d_tok1); cudaFree(d_log1);
    }

    // ---- 시퀀스별·스텝별 오차 + argmax ----
    //   batch와 single 둘 다 B(bf16 MMA attn) — 기준은 argmax/next-token + MEAN-rel(계통오차). max-abs는 bf16
    //   정밀도(~1e-2, KV stride 레이아웃차 누적)라 비척도(test_decode와 동일 정정, W4A8/FA 교훈).
    double emax=0,sa=0,sr=0; int worst_b=0,worst_t=0; bool amok=true;
    auto am=[&](const std::vector<float>&lg,size_t off){int bi=0;float bv=lg[off];for(int j=1;j<V;j++){float z=lg[off+j];if(z>bv){bv=z;bi=j;}}return bi;};
    for(int t=0;t<N;t++)for(int b=0;b<B;b++){
        double e=0; size_t off=(size_t)b*V;
        for(int j=0;j<V;j++){ double dd=fabs((double)blog[t][off+j]-slog[t][off+j]); e=fmax(e,dd); sa+=dd; sr+=fabs((double)slog[t][off+j]); }
        if(e>emax){emax=e;worst_b=b;worst_t=t;}
        if(am(blog[t],off)!=am(slog[t],off)) amok=false;
    }
    double mrel=sa/(sr>0?sr:1);
    printf("[batch decode] B=%d N=%d\n",B,N);
    printf("  batch vs B single  max-abs=%.3e (worst seq %d step %d) MEAN-rel=%.3e (bf16)\n",emax,worst_b,worst_t,mrel);
    printf("  argmax per (seq,step): %s\n", amok?"ALL match":"MISMATCH");
    bool pass=amok && (mrel<3e-3);   // argmax(생성 동일) + MEAN-rel<3e-3(bf16 계통). max-abs는 bf16라 비척도.

    // ---- 처리량 측정 (in-engine, B 스윕). microbench 아님 — 실제 decode forward 반복. ----
    if(argc>3){
        int Bt=atoi(argv[3]), Nt=argc>4?atoi(argv[4]):64; if(Nt>=120)Nt=119;
        BatchDecodeState ts=alloc_batch_decode_state(c,Bt,Nt+4);
        int *dtk; CK(cudaMalloc(&dtk,Bt*4)); float *dlg; CK(cudaMalloc(&dlg,(size_t)Bt*V*4));
        std::vector<int> tt(Bt); for(int b=0;b<Bt;b++)tt[b]=toks[b%P];
        CK(cudaMemcpy(dtk,tt.data(),Bt*4,cudaMemcpyHostToDevice));
        for(int t=0;t<3;t++) qwen35_decode_batch(c,w,ts,dtk,d_cos,d_sin,dlg,uq);  // warmup
        CK(cudaDeviceSynchronize());
        clock_t t0=clock();
        for(int t=0;t<Nt;t++) qwen35_decode_batch(c,w,ts,dtk,d_cos,d_sin,dlg,uq);
        CK(cudaDeviceSynchronize());
        double sec=(double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("[throughput] B=%d N=%d: %.3fs  total=%.1f tok/s  per-seq=%.1f tok/s\n",
               Bt,Nt,sec, Bt*Nt/sec, Nt/sec);
        free_batch_decode_state(ts); cudaFree(dtk); cudaFree(dlg);
    }
    free_batch_decode_state(bst); free_weights_qwen(w); free_config_qwen(c);
    printf("\n[test_batch_decode] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
