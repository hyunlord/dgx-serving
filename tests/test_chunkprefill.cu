// chunked prefill mism=0: prefill_batch_slot(청크 병렬) + batch decode == 1토큰/step prefill + batch decode.
//   두 방법 모두 이후 batch decode 동일 → prefill 방식만 차이. 토큰 동일하면 prefill_batch_slot 정확(chunked==1토큰).
//   B=1 slot. fp32 bit-identical / W4A8 argmax(토큰 동일). 커널 재사용 검증(새 prefill 커널 아님).
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
#include "../models/qwen35_batch.h"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int uq=argc>3?atoi(argv[3]):0; int P=argc>4?atoi(argv[4]):40, N=argc>5?atoi(argv[5]):10, chunk=argc>6?atoi(argv[6]):16;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, rot=c.rotary_dim; int maxlen=P+N+4;
    std::vector<int> ids=ri("token_ids.bin"); std::vector<int> prompt(P); for(int i=0;i<P;i++)prompt[i]=ids[i%ids.size()];
    std::vector<float> cl=rf("rope_cos_long.bin",(size_t)128*rot), sl=rf("rope_sin_long.bin",(size_t)128*rot);
    if(maxlen>128){ printf("  (P+N>128: rope 128행 부족 — P 작게)\n"); }
    float*d_cos=up(cl),*d_sin=up(sl);
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,uq==0?1:0);
    int*d_tokP; CK(cudaMalloc(&d_tokP,(size_t)P*4)); CK(cudaMemcpy(d_tokP,prompt.data(),(size_t)P*4,cudaMemcpyHostToDevice));
    int*d_tok1; CK(cudaMalloc(&d_tok1,4)); float*d_lg; CK(cudaMalloc(&d_lg,(size_t)P*V*4));
    printf("[test_chunkprefill] uq=%d(%s) P=%d N=%d chunk=%d\n",uq,uq==0?"fp32":"W4A8",P,N,chunk);
    auto amaxrow=[&](float*dl,int rows){ std::vector<float> h((size_t)rows*V); CK(cudaMemcpy(h.data(),dl,(size_t)rows*V*4,cudaMemcpyDeviceToHost));
        const float*r=&h[(size_t)(rows-1)*V]; int bi=0;float bv=r[0];for(int j=1;j<V;j++)if(r[j]>bv){bv=r[j];bi=j;} return bi; };
    auto decodeN=[&](BatchDecodeState&st,int cur)->std::vector<int>{ std::vector<int> out;
        for(int i=0;i<N;i++){ out.push_back(cur); CK(cudaMemcpy(d_tok1,&cur,4,cudaMemcpyHostToDevice));
            qwen35_decode_batch(c,w,st,d_tok1,d_cos,d_sin,d_lg,uq); // B=1 same-length: st.n_past 사용+증가
            cur=amaxrow(d_lg,1); } return out; };

    // 방법 A: 1토큰/step prefill (현행) + batch decode
    std::vector<int> A;
    { BatchDecodeState st=alloc_batch_decode_state(c,1,maxlen); st.B=1; st.n_past=0;
      for(int i=0;i<P;i++){ CK(cudaMemcpy(d_tok1,&prompt[i],4,cudaMemcpyHostToDevice));
          qwen35_decode_batch(c,w,st,d_tok1,d_cos,d_sin,d_lg,uq); } // st.n_past 0→P
      int cur=amaxrow(d_lg,1); A=decodeN(st,cur); free_batch_decode_state(st); }

    // 방법 B: chunked prefill_batch_slot + batch decode
    std::vector<int> B;
    { BatchDecodeState st=alloc_batch_decode_state(c,1,maxlen); st.B=1; st.n_past=0;
      prefill_batch_slot(c,w,st,0,d_tokP,P,d_cos,d_sin,d_lg,uq,chunk);  // slot0 KV[0..P-1]+GDN 채움
      int lastsq = (P%chunk==0)?chunk:(P%chunk); int cur=amaxrow(d_lg,lastsq); // 마지막 청크 마지막 행=pos P-1
      st.n_past=P;  // decode는 pos P부터
      B=decodeN(st,cur); free_batch_decode_state(st); }

    int mm=0; for(int i=0;i<N;i++) if(A[i]!=B[i]) mm++;
    printf("  방법A(1토큰 prefill) vs B(chunked prefill_batch_slot): %d/%d mismatch %s\n",mm,N, mm?"★":"");
    bool pass=(mm==0);
    printf("[test_chunkprefill] %s (chunked prefill == 1토큰 prefill, batch decode 동일 → prefill_batch_slot 정확)\n",pass?"PASS":"FAIL");
    free_weights_qwen(w); free_config_qwen(c); return pass?0:1;
}
