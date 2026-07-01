// long-rope pos>=128 검증: ext 테이블(8192행)로 P>128 prefill. rope 공식==모델(1 ULP, gen_rope_long) + rope_scaling=None
//   → pos>=128 rope 정확. 여기선 엔진 long-context 자기일관성: chunked prefill P>128 == 1토큰/step prefill P>128
//   (같은 ext rope, 같은 KV). 둘 다 정확한 rope 사용 → 일치하면 엔진 long-context 경로 건전(attention/KV pos>=128 정상).
//   fp32·W4A8. (절대 golden[transformers >128 forward]은 별도 — rope 테이블은 formula-equivalence로 증명됨.)
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
static std::vector<float> rfx(const std::string&full,size_t n){FILE*f=fopen(full.c_str(),"rb");if(!f){fprintf(stderr,"없음: %s\n",full.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int uq=argc>3?atoi(argv[3]):0; int P=argc>4?atoi(argv[4]):256, N=argc>5?atoi(argv[5]):8, chunk=argc>6?atoi(argv[6]):128;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, rot=c.rotary_dim; int maxlen=P+N+8;
    // ext rope 테이블(8192행) 로드 — pos>=128 정확
    std::string cosf=GD+"rope_cos_ext.bin", sinf=GD+"rope_sin_ext.bin";
    FILE*t=fopen(cosf.c_str(),"rb"); if(!t){printf("  rope_cos_ext.bin 없음 — gen_rope_long.py 먼저\n");return 1;}
    fseek(t,0,SEEK_END); long rows=ftell(t)/4/rot; fclose(t);
    if(rows<maxlen){printf("  ext 테이블 %ld행 < maxlen %d — gen_rope_long.py max_len↑\n",rows,maxlen);return 1;}
    std::vector<float> cl=rfx(cosf,(size_t)maxlen*rot), sl=rfx(sinf,(size_t)maxlen*rot);
    float*d_cos=up(cl),*d_sin=up(sl);
    std::vector<int> ids=ri("token_ids.bin"); std::vector<int> prompt(P); for(int i=0;i<P;i++)prompt[i]=ids[i%ids.size()];
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,uq==0?1:0);
    int*d_tokP;CK(cudaMalloc(&d_tokP,(size_t)P*4));CK(cudaMemcpy(d_tokP,prompt.data(),(size_t)P*4,cudaMemcpyHostToDevice));
    int*d_tok1;CK(cudaMalloc(&d_tok1,4)); float*d_lg;CK(cudaMalloc(&d_lg,(size_t)P*V*4));
    printf("[test_longrope] uq=%d(%s) P=%d(>128) N=%d chunk=%d ext_rope=%ld행\n",uq,uq==0?"fp32":"W4A8",P,N,chunk,rows);
    auto amaxrow=[&](float*dl,int r){ std::vector<float> h((size_t)r*V);CK(cudaMemcpy(h.data(),dl,(size_t)r*V*4,cudaMemcpyDeviceToHost));
        const float*rr=&h[(size_t)(r-1)*V];int bi=0;float bv=rr[0];for(int j=1;j<V;j++)if(rr[j]>bv){bv=rr[j];bi=j;}return bi;};
    auto decodeN=[&](BatchDecodeState&st,int cur){std::vector<int> out;for(int i=0;i<N;i++){out.push_back(cur);
        CK(cudaMemcpy(d_tok1,&cur,4,cudaMemcpyHostToDevice)); qwen35_decode_batch(c,w,st,d_tok1,d_cos,d_sin,d_lg,uq); cur=amaxrow(d_lg,1);}return out;};
    // A: 1토큰/step prefill (pos 0..P-1, pos>=128 포함) + decode
    std::vector<int> A;
    { BatchDecodeState st=alloc_batch_decode_state(c,1,maxlen);st.B=1;st.n_past=0;
      for(int i=0;i<P;i++){CK(cudaMemcpy(d_tok1,&prompt[i],4,cudaMemcpyHostToDevice));qwen35_decode_batch(c,w,st,d_tok1,d_cos,d_sin,d_lg,uq);}
      int cur=amaxrow(d_lg,1);A=decodeN(st,cur);free_batch_decode_state(st);}
    // B: chunked prefill_batch_slot (pos>=128 포함) + decode
    std::vector<int> B;
    { BatchDecodeState st=alloc_batch_decode_state(c,1,maxlen);st.B=1;st.n_past=0;
      prefill_batch_slot(c,w,st,0,d_tokP,P,d_cos,d_sin,d_lg,uq,chunk);
      int lastsq=(P%chunk==0)?chunk:(P%chunk);int cur=amaxrow(d_lg,lastsq);st.n_past=P;B=decodeN(st,cur);free_batch_decode_state(st);}
    int mm=0;for(int i=0;i<N;i++)if(A[i]!=B[i])mm++;
    printf("  chunked prefill(P>128) vs 1토큰 prefill(P>128), ext rope: %d/%d mismatch %s\n",mm,N,mm?"★":"");
    bool pass=(mm==0);
    printf("[test_longrope] %s (pos>=128 엔진 long-context 자기일관 + ext rope[공식==모델 1 ULP])\n",pass?"PASS":"FAIL");
    free_weights_qwen(w);free_config_qwen(c);return pass?0:1;
}
