// prefill-only nsys 프로파일: 청크 prefill 2048만(decode 없음) → 커널 시간 순수 prefill 분해.
#include <cstdio>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/engine.h"
#include "../runtime/decode_state.h"
#include "../models/qwen35.h"
#include "../kernels/rope.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(int argc,char**argv){
    std::string WD=argc>1?argv[1]:"artifacts/qwen35-27b/weights";
    int uq=argc>2?atoi(argv[2]):2;  // W4A8 기본
    int C=argc>3?atoi(argv[3]):2048, PCHUNK=256;
    Engine e=engine_load(WD.c_str(),uq>=1?1:0,uq==0?1:0);
    int rot=e.c.rotary_dim, V=e.c.vocab_size, MAXLEN=C+8;
    float*dcos,*dsin; CK(cudaMalloc(&dcos,(size_t)MAXLEN*rot*4));CK(cudaMalloc(&dsin,(size_t)MAXLEN*rot*4));
    launch_gen_rope(dcos,dsin,MAXLEN,rot,e.c.rope_theta);
    std::vector<int> tok(MAXLEN); for(int i=0;i<MAXLEN;i++)tok[i]=(i*131+7)%V;
    int*dtok; CK(cudaMalloc(&dtok,MAXLEN*4)); CK(cudaMemcpy(dtok,tok.data(),MAXLEN*4,cudaMemcpyHostToDevice));
    float*dlog; CK(cudaMalloc(&dlog,(size_t)V*4));
    DecodeState st=alloc_decode_state(e.c,C+8);
    // warmup 1청크
    qwen35_decode(e.c,e.w,st,dtok,PCHUNK,dcos,dsin,dlog,nullptr,1,nullptr,uq,nullptr,2); CK(cudaDeviceSynchronize());
    reset_decode_state(e.c,st);
    // 측정 대상: 청크 prefill C
    for(int t=0;t<C;t+=PCHUNK){int s=(C-t<PCHUNK)?(C-t):PCHUNK; int last=(t+s>=C)?1:2;
        qwen35_decode(e.c,e.w,st,dtok+t,s,dcos,dsin,dlog,nullptr,1,nullptr,uq,nullptr,last);}
    CK(cudaDeviceSynchronize());
    printf("prefill C=%d uq=%d done\n",C,uq); return 0;
}
