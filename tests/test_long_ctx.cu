// 챕터7 FA.1 성과 검증: 엔진이 이제 >12K context를 정답으로 처리(이전엔 attn_decode 12K 벽 → garbage/abort).
//   golden 없음(레퍼런스 forward S≤128). → 청크 자기-일관성: 같은 16K 시퀀스를 256-청크 vs 512-청크로
//   prefill 후 decode → logits 동일해야(청크 크기 무관). + 모든 logits finite(garbage 아님) 확인.
//   둘 다 청크라 안전(단발 16K는 GDN/메모리 위험). FA가 base>12K에서 동작해야 둘 다 완주.
#include <cstdio>
#include <cmath>
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
    int C=argc>2?atoi(argv[2]):16384, NDEC=4, uq=argc>3?atoi(argv[3]):2;
    Engine e=engine_load(WD.c_str(),1,0);
    int rot=e.c.rotary_dim,V=e.c.vocab_size,MAXLEN=C+NDEC+8;
    float*dcos,*dsin; CK(cudaMalloc(&dcos,(size_t)MAXLEN*rot*4));CK(cudaMalloc(&dsin,(size_t)MAXLEN*rot*4));
    launch_gen_rope(dcos,dsin,MAXLEN,rot,e.c.rope_theta);
    std::vector<int> tok(MAXLEN); for(int i=0;i<MAXLEN;i++)tok[i]=(i*131+7)%V;
    int*dtok; CK(cudaMalloc(&dtok,MAXLEN*4)); CK(cudaMemcpy(dtok,tok.data(),MAXLEN*4,cudaMemcpyHostToDevice));
    float*dlog; CK(cudaMalloc(&dlog,(size_t)V*4));

    auto run=[&](int PCHUNK, std::vector<std::vector<float>>&dec){
        DecodeState st=alloc_decode_state(e.c,MAXLEN);
        for(int t=0;t<C;t+=PCHUNK){ int s=(C-t<PCHUNK)?(C-t):PCHUNK; int last=(t+s>=C)?1:2;
            qwen35_decode(e.c,e.w,st,dtok+t,s,dcos,dsin,dlog,nullptr,1,nullptr,uq,nullptr,last); }
        { std::vector<float> lg(V); CK(cudaMemcpy(lg.data(),dlog,(size_t)V*4,cudaMemcpyDeviceToHost)); dec.push_back(lg); }  // [0]=prefill 최종 logits
        for(int i=0;i<NDEC;i++){ qwen35_decode(e.c,e.w,st,dtok+C+i,1,dcos,dsin,dlog,nullptr,1,nullptr,uq,nullptr,1);
            std::vector<float> lg(V); CK(cudaMemcpy(lg.data(),dlog,(size_t)V*4,cudaMemcpyDeviceToHost)); dec.push_back(lg); }
        free_decode_state(st);
    };
    printf("=== FA.1 긴-context 청크 자기-일관성 C-스윕 (256 vs 512 청크) ===\n");
    int Cs[]={512,2048,4096,8192,16384}; int g_fail=0;
    for(int CC: Cs){ if(CC>C) continue; int sv=C; C=CC;
        std::vector<std::vector<float>> a,b; run(256,a); run(512,b);  // 256 vs 512 청크 일관성
        int nonfinite=0; for(auto&v:a)for(float x:v)if(!std::isfinite(x))nonfinite++;
        double pre=0; for(int j=0;j<V;j++)pre=fmax(pre,fabs((double)a[0][j]-b[0][j]));   // prefill 최종 logits
        double dec=0; for(int i=1;i<(int)a.size();i++) for(int j=0;j<V;j++)dec=fmax(dec,fabs((double)a[i][j]-b[i][j]));
        printf("  C=%-6d | prefill logits=%.3e  decode logits=%.3e  non-finite=%d  %s\n",
               CC,pre,dec,nonfinite, (nonfinite==0&&pre<1e-2&&dec<1e-2)?"OK":"DIVERGE"); if(nonfinite||pre>=1e-2||dec>=1e-2)g_fail=1;
        C=sv;
    }
    printf("[test_long_ctx] %s (divergence가 C에 따라 자라면 누적 fp, 작은 C서도 크면 청크경계 버그)\n", g_fail?"FAIL/조사":"PASS");
    return g_fail;
}
