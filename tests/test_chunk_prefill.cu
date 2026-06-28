// 챕터6 STEP 2: 청크 prefill == 단발 prefill (오차0). decode_state가 청크간 KV/GDN state/conv 연결.
//   N토큰을 (a)단발 qwen35_decode(Sq=N) vs (b)청크(16씩) 처리 후, 이어서 decode → logits 동일 확인.
//   합성 토큰(청크 동등성은 모델내부 속성, 토큰 무관). gen_rope로 임의 길이. 작은 N(안전).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/engine.h"
#include "../runtime/decode_state.h"
#include "../kernels/rope.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

int main(int argc,char**argv){
    std::string WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights";
    const int N=32, CHUNK=16, NDEC=4;   // prefill 32, 청크 16, 이후 4토큰 decode 비교
    Engine e=engine_load(WD.c_str(),1,0);
    const int H=e.c.hidden_size,V=e.c.vocab_size,rot=e.c.rotary_dim;
    int MAXLEN=N+NDEC+8;
    float *d_cos,*d_sin; CK(cudaMalloc(&d_cos,(size_t)MAXLEN*rot*4)); CK(cudaMalloc(&d_sin,(size_t)MAXLEN*rot*4));
    launch_gen_rope(d_cos,d_sin,MAXLEN,rot,e.c.rope_theta); CK(cudaDeviceSynchronize());

    std::vector<int> toks(MAXLEN); for(int i=0;i<MAXLEN;i++) toks[i]=(i*97+13)%V;   // 합성
    int *d_tok; CK(cudaMalloc(&d_tok,MAXLEN*4)); CK(cudaMemcpy(d_tok,toks.data(),MAXLEN*4,cudaMemcpyHostToDevice));
    float *d_logits; CK(cudaMalloc(&d_logits,(size_t)V*4));

    auto run=[&](bool chunked, std::vector<std::vector<float>>&dec_logits){
        DecodeState st=alloc_decode_state(e.c, MAXLEN);
        if(chunked){ for(int t=0;t<N;t+=CHUNK){ int s=(N-t<CHUNK)?(N-t):CHUNK;
            qwen35_decode(e.c,e.w,st,d_tok+t,s,d_cos,d_sin,d_logits,nullptr,1,nullptr,1,nullptr,1); } }
        else qwen35_decode(e.c,e.w,st,d_tok,N,d_cos,d_sin,d_logits,nullptr,1,nullptr,1,nullptr,1);
        // 이어서 NDEC 토큰 decode, 각 logits 저장
        for(int i=0;i<NDEC;i++){
            qwen35_decode(e.c,e.w,st,d_tok+N+i,1,d_cos,d_sin,d_logits,nullptr,1,nullptr,1);
            std::vector<float> lg(V); CK(cudaMemcpy(lg.data(),d_logits,V*4,cudaMemcpyDeviceToHost)); dec_logits.push_back(lg);
        }
        free_decode_state(st);
    };
    std::vector<std::vector<float>> A,B;
    run(false,A); run(true,B);

    double worst=0; for(int i=0;i<NDEC;i++){ double e0=0; for(int j=0;j<V;j++)e0=fmax(e0,fabs((double)A[i][j]-B[i][j]));
        printf("  decode step %d (pos %d): 단발 vs 청크 logits max err=%.3e\n",i,N+i,e0); worst=fmax(worst,e0); }
    bool pass=worst<1e-3;   // q4 GPU 재실행 비결정성 약간 허용(같은 커널, 같은 입력 → ~0 기대)
    printf("[test_chunk_prefill] %s (청크 prefill == 단발, decode_state 청크간 연결 오차0)\n",pass?"PASS":"FAIL");
    engine_free(e); cudaFree(d_cos);cudaFree(d_sin);cudaFree(d_tok);cudaFree(d_logits);
    return pass?0:1;
}
