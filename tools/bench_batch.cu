// 챕터9 batch serving STEP A: batch decode throughput 곡선 (B 곡선 + odd-batch 붕괴 + plateau measure).
//   각 B: 짧은 prefill 후 N decode step 타이밍(cudaEvent) → throughput=B*N/t, per-req t/s. unified 메모리 가드.
//   args: weights, [use_q4=2], [N=32]. gemm_batch_w4a8 경로(현). odd(3,5,6,7)는 default 루프(B× 재적재) 예상.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../models/qwen35_batch.h"
#include "../kernels/rope.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float ev(cudaEvent_t a,cudaEvent_t b){float ms;cudaEventElapsedTime(&ms,a,b);return ms;}
int main(int argc,char**argv){
    std::string WD=argc>1?argv[1]:"artifacts/qwen35-27b-fp16s/weights";
    int use_q4=argc>2?atoi(argv[2]):2, N=argc>3?atoi(argv[3]):32;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, rot=c.rotary_dim;
    int PRE=4, maxlen=PRE+N+8;
    printf("[bench_batch] %s use_q4=%d N=%d (prefill %d) | gemm_batch_w4a8 경로\n",WD.c_str(),use_q4,N,PRE);
    float *d_cos,*d_sin; CK(cudaMalloc(&d_cos,(size_t)maxlen*rot*4)); CK(cudaMalloc(&d_sin,(size_t)maxlen*rot*4));
    launch_gen_rope(d_cos,d_sin,maxlen,rot,c.rope_theta); CK(cudaDeviceSynchronize());
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,use_q4==0?1:0);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    printf("  %-4s %-12s %-14s %-12s %s\n","B","decode ms/step","throughput t/s","per-req t/s","vs B=1");
    double base1=0;
    int bonly = getenv("BONLY") ? atoi(getenv("BONLY")) : 0;   // 튜닝 스윕: 특정 B만(모델 1회 로드, 임의 B 허용). 0=전체
    int reps  = getenv("REPS")  ? atoi(getenv("REPS"))  : 1;   // 결정성: timed 루프 반복(min ms 채택)
    std::vector<int> blist = bonly ? std::vector<int>{bonly} : std::vector<int>{1,2,3,4,5,6,7,8,16};
    for(int B : blist){    // SB는 B≤8 수정(odd 3,5,6,7); 16은 gemm_batch 특수화. (12/24/32 default-collapse는 별도)
        size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB);
        // 대략 예산: GDN S(B*NH*Dv*Dv*4*nGDN) + KV(B*maxlen*NKV*HD*2*nFull) — 가드
        size_t gdnS=(size_t)B*c.gdn_num_v_heads*c.gdn_head_v_dim*c.gdn_head_v_dim*4*48;
        if(gdnS > (size_t)(0.7*freeB)){ printf("  %-4d SKIP(메모리 가드 %.1fGB>0.7×%.1fGB)\n",B,gdnS/1e9,freeB/1e9); continue; }
        BatchDecodeState st=alloc_batch_decode_state(c,B,maxlen);
        std::vector<int> tok(B); for(int j=0;j<B;j++) tok[j]=100+j*7;
        int*dtok; CK(cudaMalloc(&dtok,B*4)); CK(cudaMemcpy(dtok,tok.data(),B*4,cudaMemcpyHostToDevice));
        float*dl; CK(cudaMalloc(&dl,(size_t)B*V*4));
        for(int p=0;p<PRE;p++) qwen35_decode_batch(c,w,st,dtok,d_cos,d_sin,dl,use_q4);  // prefill/warmup
        // 결정성 REPS: rep마다 n_past 리셋 → 동일 position 범위(base..base+N)서 재측정, min ms 채택(OOB 방지).
        int saved_np=st.n_past; float ms=1e30f;
        for(int r=0;r<reps;r++){
            st.n_past=saved_np;
            CK(cudaDeviceSynchronize()); cudaEventRecord(t0);
            for(int i=0;i<N;i++) qwen35_decode_batch(c,w,st,dtok,d_cos,d_sin,dl,use_q4);
            cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); float m=ev(t0,t1); if(m<ms)ms=m;
        }
        double tput=(double)B*N*1000.0/ms, perreq=tput/B; if(B==1)base1=tput;
        printf("  %-4d %-12.2f %-14.1f %-12.1f %.2fx\n",B, ms/N, tput, perreq, base1?tput/base1:1.0); fflush(stdout);
        free_batch_decode_state(st); cudaFree(dtok); cudaFree(dl);
    }
    free_weights_qwen(w); free_config_qwen(c);
    return 0;
}
