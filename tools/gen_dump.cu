// 챕터9 accept lever de-risk 도구: 임의 프롬프트 greedy 생성 → 토큰 시퀀스 덤프(n-gram accept 통계용).
//   args: weights_dir, prompt_ids.bin(int32), N_gen, [bq4=2], [out.bin]. rope 엔진 생성(self, theta from config).
//   출력: (prompt + generated) int32 시퀀스를 out.bin에. (greedy = spec verify의 ground truth.)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#include "../kernels/rope.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: gen_dump <weights> <prompt_ids.bin> <N_gen> [bq4=2] [out.bin]\n");return 1;}
    std::string WD=argv[1], PF=argv[2]; int N=atoi(argv[3]); int bq4=argc>4?atoi(argv[4]):2;
    std::string OUT=argc>5?argv[5]:(PF+".out.bin");
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, rot=c.rotary_dim, H=c.hidden_size;
    FILE*f=fopen(PF.c_str(),"rb"); if(!f){fprintf(stderr,"prompt 못엶\n");return 1;}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<int> prompt(b/4); if(fread(prompt.data(),4,prompt.size(),f)!=prompt.size())return 1; fclose(f);
    int P=prompt.size(), maxlen=P+N+8;
    printf("[gen_dump] weights=%s P=%d N=%d bq4=%d maxlen=%d\n",WD.c_str(),P,N,bq4,maxlen);
    float *d_cos,*d_sin; CK(cudaMalloc(&d_cos,(size_t)maxlen*rot*4)); CK(cudaMalloc(&d_sin,(size_t)maxlen*rot*4));
    launch_gen_rope(d_cos,d_sin,maxlen,rot,c.rope_theta); CK(cudaDeviceSynchronize());
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,bq4==0?1:0);
    DecodeState st=alloc_decode_state(c,maxlen);
    int*d_tok; CK(cudaMalloc(&d_tok,maxlen*4)); CK(cudaMemcpy(d_tok,prompt.data(),P*4,cudaMemcpyHostToDevice));
    float*dl; CK(cudaMalloc(&dl,(size_t)P*V*4));
    auto amax=[&](const std::vector<float>&lg,int p){int bi=0;float bv=lg[(size_t)p*V];for(int j=1;j<V;j++){float z=lg[(size_t)p*V+j];if(z>bv){bv=z;bi=j;}}return bi;};
    qwen35_decode(c,w,st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,nullptr);
    std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
    int cur=amax(lg,P-1);
    std::vector<int> seq=prompt;
    for(int i=0;i<N;i++){ seq.push_back(cur);
        CK(cudaMemcpy(d_tok,&cur,4,cudaMemcpyHostToDevice));
        float*o; CK(cudaMalloc(&o,(size_t)V*4));
        qwen35_decode(c,w,st,d_tok,1,d_cos,d_sin,o,nullptr,1,nullptr,bq4,nullptr);
        std::vector<float> hl(V); CK(cudaMemcpy(hl.data(),o,(size_t)V*4,cudaMemcpyDeviceToHost)); cudaFree(o);
        int bi=0;float bv=hl[0];for(int j=1;j<V;j++)if(hl[j]>bv){bv=hl[j];bi=j;} cur=bi;
    }
    FILE*of=fopen(OUT.c_str(),"wb"); fwrite(seq.data(),4,seq.size(),of); fclose(of);
    printf("[gen_dump] wrote %zu tokens (P=%d + N=%d) → %s\n",seq.size(),P,N,OUT.c_str());
    free_decode_state(st); free_weights_qwen(w); free_config_qwen(c);
    return 0;
}
