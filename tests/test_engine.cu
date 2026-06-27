// 챕터5 STEP1 검증: 공통 Engine 인터페이스(engine_load/engine_decode_batch) == 직접 qwen35 호출, 오차0.
//   레지스트리/인터페이스가 투명한 래퍼임을 증명(0.8B 회귀 0). additive 레이어라 기존 경로 불변.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/engine.h"
#include "../runtime/manifest_qwen.h"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string GD;
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}

int main(int argc,char**argv){
    std::string WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights";
    GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int uq=argc>3?atoi(argv[3]):1;
    const int B=4,N=6,maxlen=N+4;

    // 레지스트리 확인
    Engine e=engine_load(WD.c_str());
    printf("[engine] model_kind=%d (0=qwen3_5) H=%d L=%d\n", e.kind, e.c.hidden_size, e.c.num_layers);
    const int V=e.c.vocab_size, rot=e.c.rotary_dim;

    std::vector<int> toks=ri("token_ids.bin"); int P=toks.size();
    std::vector<float> cl=rf("rope_cos_long.bin",(size_t)128*rot), sl=rf("rope_sin_long.bin",(size_t)128*rot);
    float *d_cos,*d_sin; CK(cudaMalloc(&d_cos,cl.size()*4));CK(cudaMalloc(&d_sin,sl.size()*4));
    CK(cudaMemcpy(d_cos,cl.data(),cl.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_sin,sl.data(),sl.size()*4,cudaMemcpyHostToDevice));

    // ---- (a) Engine 경로 ----
    BatchDecodeState es=engine_alloc_batch(e,B,maxlen);
    int *dtk; CK(cudaMalloc(&dtk,B*4)); float *dlg; CK(cudaMalloc(&dlg,(size_t)B*V*4));
    std::vector<std::vector<float>> elog(N,std::vector<float>((size_t)B*V));
    for(int t=0;t<N;t++){ std::vector<int> tt(B); for(int b=0;b<B;b++)tt[b]=toks[(b*3+t)%P];
        CK(cudaMemcpy(dtk,tt.data(),B*4,cudaMemcpyHostToDevice));
        engine_decode_batch(e,es,dtk,d_cos,d_sin,dlg,uq);
        CK(cudaMemcpy(elog[t].data(),dlg,(size_t)B*V*4,cudaMemcpyDeviceToHost)); }
    engine_free_batch(es);

    // ---- (b) 직접 호출 경로 (기존 검증된 path) ----
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,1);
    BatchDecodeState ds=alloc_batch_decode_state(c,B,maxlen);
    std::vector<std::vector<float>> dlog(N,std::vector<float>((size_t)B*V));
    for(int t=0;t<N;t++){ std::vector<int> tt(B); for(int b=0;b<B;b++)tt[b]=toks[(b*3+t)%P];
        CK(cudaMemcpy(dtk,tt.data(),B*4,cudaMemcpyHostToDevice));
        qwen35_decode_batch(c,w,ds,dtk,d_cos,d_sin,dlg,uq);
        CK(cudaMemcpy(dlog[t].data(),dlg,(size_t)B*V*4,cudaMemcpyDeviceToHost)); }

    double emax=0; for(int t=0;t<N;t++)for(size_t i=0;i<elog[t].size();i++)emax=fmax(emax,fabs((double)elog[t][i]-dlog[t][i]));
    printf("[engine] engine_decode_batch == 직접 qwen35_decode_batch  max err=%.3e\n",emax);
    bool pass=(emax==0.0);   // 투명 래퍼라 bit-identical 기대
    free_batch_decode_state(ds); free_weights_qwen(w); free_config_qwen(c); engine_free(e);
    printf("\n[test_engine] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
