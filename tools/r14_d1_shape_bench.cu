// R14 PART B: isolated per-shape timing, D1 vs the existing family (S=1 plain GEMV, S>1 SB).
//   Shapes come from the manifest (every distinct dense linear K×N, incl. lm_head) — no model constants.
//   Synthetic random int4 weights / fp16 scales / int8 activations of the real byte sizes; NBUF weight copies
//   rotate so consecutive launches do not hit L2 (copy count from L2 size). Per (shape,S,family): warm 20,
//   then REPS launches timed with one event pair → mean µs/launch. Optional --ncu: one launch per config
//   (for ncu --launch-skip/count filtering) and no timing.
//   Usage: r14_d1_shape_bench <manifest.json> [reps=200] [--ncu]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <utility>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemv_w4a8_d1.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <manifest.json> [reps] [--ncu]\n",argv[0]); return 2; }
    ModelConfig c=load_config_gemma4(argv[1]);
    const int reps=argc>2&&argv[2][0]!='-'?atoi(argv[2]):200;
    bool ncu=false; for(int i=2;i<argc;i++) if(!strcmp(argv[i],"--ncu")) ncu=true;
    const int H=c.hidden_size, FFN=c.intermediate_size, NQ=c.num_q_heads, V=c.vocab_size;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    std::set<std::pair<int,int>> shapes;   // (K=IN, N=OUT)
    for(int l=0;l<c.num_layers;l++){
        const int full=c.layer_is_full?c.layer_is_full[l]:1;
        const int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        shapes.insert({H,NQ*hd}); shapes.insert({H,nkv*hd});
        if(!(full && c.attention_k_eq_v)) shapes.insert({H,nkv*hd});
        shapes.insert({NQ*hd,H}); shapes.insert({H,FFN}); shapes.insert({FFN,H});
    }
    shapes.insert({H,V});
    int l2=0; CK(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,0));
    const int sdtype=c.scale_dtype;
    const int Svals[4]={1,4,8,16};
    std::mt19937 rng(14);
    printf("R14_D1_CONFIG l2_bytes=%d reps=%d scale_dtype=%d shapes=%zu mode=%s\n",l2,reps,sdtype,shapes.size(),ncu?"ncu":"timing");
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for(auto [K,N]:shapes){
        const size_t wbytes=(size_t)N*K/2, sbytes=(size_t)N*(K/QBLK)*(sdtype==1?2:4);
        const int nbuf=ncu?1:(int)std::max<size_t>(2,(3*(size_t)l2)/(wbytes+sbytes)+2);
        std::vector<uint8_t*> wq(nbuf); std::vector<void*> sc(nbuf);
        std::vector<uint8_t> hw(wbytes); std::vector<__half> hs((size_t)N*(K/QBLK));
        for(auto &x:hw) x=rng()&255;
        for(auto &x:hs) x=__float2half(0.001f+(rng()%1000)*1e-5f);
        for(int i=0;i<nbuf;i++){ CK(cudaMalloc(&wq[i],wbytes)); CK(cudaMalloc(&sc[i],sbytes));
            CK(cudaMemcpy(wq[i],hw.data(),wbytes,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(sc[i],hs.data(),sbytes,cudaMemcpyHostToDevice)); }
        int8_t *xq; float *xs,*y; int32_t *xsum;
        CK(cudaMalloc(&xq,(size_t)16*K)); CK(cudaMalloc(&xs,(size_t)16*(K/QBLK)*4)); CK(cudaMalloc(&xsum,(size_t)16*(K/QBLK)*4));
        CK(cudaMalloc(&y,(size_t)16*N*4));
        std::vector<int8_t> hx((size_t)16*K); for(auto &x:hx) x=(int8_t)((int)(rng()%255)-127);
        std::vector<float> hxs((size_t)16*(K/QBLK)); for(auto &x:hxs) x=0.01f+(rng()%100)*1e-4f;
        CK(cudaMemcpy(xq,hx.data(),hx.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(xs,hxs.data(),hxs.size()*4,cudaMemcpyHostToDevice));
        for(int S:Svals) for(int fam=0;fam<2;fam++){
            auto launch=[&](int i){
                if(fam==1) launch_gemv_w4a8_d1(xq,xs,wq[i],sc[i],y,K,N,S,sdtype,0);
                else if(S==1) launch_gemv_w4a8(xq,xs,xsum,wq[i],sc[i],y,K,N,1,sdtype,0);
                else launch_gemv_w4a8_sb(xq,xs,wq[i],sc[i],y,K,N,S,sdtype,0);
            };
            if(ncu){ launch(0); CK(cudaDeviceSynchronize());
                printf("R14_D1_NCU K=%d N=%d S=%d family=%s\n",K,N,S,fam?"d1":"existing"); continue; }
            for(int i=0;i<20;i++) launch(i%nbuf);
            CK(cudaEventRecord(a));
            for(int i=0;i<reps;i++) launch(i%nbuf);
            CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
            float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
            const double us=1000.0*ms/reps;
            printf("R14_D1_TIME K=%d N=%d S=%d family=%s kernel=%s us=%.4f logical_weight_GBps=%.2f nbuf=%d\n",K,N,S,fam?"d1":"existing",
                   fam?"gemv_w4a8_d1":(S==1?"gemv_w4a8":"gemv_w4a8_sb"),us,(wbytes+sbytes)/(us*1e3),nbuf);
        }
        for(int i=0;i<nbuf;i++){ cudaFree(wq[i]); cudaFree(sc[i]); }
        cudaFree(xq); cudaFree(xs); cudaFree(xsum); cudaFree(y);
    }
    return 0;
}
