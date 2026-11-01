// Gemma 4 dense prefill routing + chunk invariance gate (R21).
//   (1) route: prefill-mode chunks of Sq=1/6/15 take flash; flagless calls with Sq=1..8 (decode/verify shape)
//       take the split-KV kernel (host route counters, one count per call).
//   (2) chunk invariance: one prompt prefilled as 1030 == 1024+6 == 512+512+6 == 16x64+6 (+ 6+1024, 1029+1):
//       every layer's K and V cache bytes for positions 0..P-1 and the last-position logits must be bit-identical.
//       On a mismatch it prints the first differing layer, tensor and position.
//   Usage: ./tests/test_gemma4_prefill_chunk <weights> <tokens.bin (int32, >= P ids)> [P=1030] [use_q4=2]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"
#include "../kernels/kv_store.cuh"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

struct Snapshot { std::vector<std::vector<unsigned char>> k, v; std::vector<float> logits; };

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s <weights> <tokens.bin> [P=1030] [use_q4=2]\n",argv[0]); return 2; }
    const std::string WD=argv[1]; const int P=argc>3?atoi(argv[3]):1030, uq=argc>4?atoi(argv[4]):2;
    ModelConfig c=load_config_gemma4((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, L=c.num_layers, hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    std::vector<int> ids(P);
    { FILE*f=fopen(argv[2],"rb"); if(!f||fread(ids.data(),4,P,f)!=(size_t)P){ fprintf(stderr,"cannot read %d ids from %s\n",P,argv[2]); return 2; } fclose(f); }
    const int ML=P+8;
    float *cf,*sf,*cs,*ss; TCK(cudaMalloc(&cf,(size_t)ML*hd_f*4)); TCK(cudaMalloc(&sf,(size_t)ML*hd_f*4));
    TCK(cudaMalloc(&cs,(size_t)ML*hd_s*4)); TCK(cudaMalloc(&ss,(size_t)ML*hd_s*4));
    if(gemma4_gen_rope_tables((WD+"/manifest.json").c_str(),ML,hd_f,cf,sf,hd_s,cs,ss)){ fprintf(stderr,"rope tables failed\n"); return 2; }
    G4Weights w=load_weights_gemma4(WD.c_str(),c,uq!=0?1:0,uq==0?1:0);
    G4DecodeState st=alloc_g4_decode_state(c,ML);
    int *d_ids; TCK(cudaMalloc(&d_ids,(size_t)P*4)); TCK(cudaMemcpy(d_ids,ids.data(),(size_t)P*4,cudaMemcpyHostToDevice));
    float *d_lg; TCK(cudaMalloc(&d_lg,(size_t)ML*V*4));
    const size_t es=g4_kv_elem();
    printf("[cfg] P=%d uq=%d L=%d kv_elem=%zu\n",P,uq,L,es);
    int fails=0;

    // (1) route counters
    {
        unsigned long f0,s0,f1,s1; reset_g4_decode_state(st);
        for(int sq:{1,6,15}){
            gemma4_attn_route_counts(&f0,&s0);
            gemma4_decode(c,w,st,d_ids,sq,cf,sf,cs,ss,d_lg,0,uq,1,1,nullptr,/*prefill*/1); TCK(cudaDeviceSynchronize());
            gemma4_attn_route_counts(&f1,&s1);
            const bool ok=f1==f0+1 && s1==s0;
            printf("ROUTE prefill=1 Sq=%-2d flash+%lu split+%lu %s\n",sq,f1-f0,s1-s0,ok?"OK":"FAIL"); fails+=!ok;
        }
        for(int sq=1;sq<=8;sq++){
            reset_g4_decode_state(st); st.n_past=32;   // decode/verify shape: queries after an existing prefix
            gemma4_attn_route_counts(&f0,&s0);
            gemma4_decode(c,w,st,d_ids,sq,cf,sf,cs,ss,d_lg,0,uq,0,sq>1?1:0,nullptr); TCK(cudaDeviceSynchronize());
            gemma4_attn_route_counts(&f1,&s1);
            const bool ok=s1==s0+1 && f1==f0;
            printf("ROUTE prefill=0 Sq=%d flash+%lu split+%lu %s\n",sq,f1-f0,s1-s0,ok?"OK":"FAIL"); fails+=!ok;
        }
    }

    // (2) chunk invariance
    auto run=[&](const std::vector<int>& chunks){
        Snapshot s; reset_g4_decode_state(st); int o=0;
        for(int n:chunks){ gemma4_decode(c,w,st,d_ids+o,n,cf,sf,cs,ss,d_lg,1,uq,1,1,nullptr,/*prefill*/1); o+=n; }
        TCK(cudaDeviceSynchronize());
        if(o!=P){ fprintf(stderr,"chunks sum %d != P %d\n",o,P); exit(2); }
        s.k.resize(L); s.v.resize(L);
        for(int l=0;l<L;l++){ const int full=c.layer_is_full?c.layer_is_full[l]:1; const size_t n=(size_t)P*(full?nkv_f*hd_f:nkv_s*hd_s)*es;
            s.k[l].resize(n); s.v[l].resize(n);
            TCK(cudaMemcpy(s.k[l].data(),st.Kc[l],n,cudaMemcpyDeviceToHost)); TCK(cudaMemcpy(s.v[l].data(),st.Vc[l],n,cudaMemcpyDeviceToHost)); }
        s.logits.resize(V); TCK(cudaMemcpy(s.logits.data(),d_lg,(size_t)V*4,cudaMemcpyDeviceToHost));
        return s;
    };
    auto chunked=[](int each,int count,int tail){ std::vector<int> v(count,each); if(tail) v.push_back(tail); return v; };
    struct Case { const char* name; std::vector<int> chunks; };
    std::vector<Case> cases={{"1030",{P}}};
    if(P==1030){
        cases.push_back({"1024+6",{1024,6}}); cases.push_back({"512+512+6",{512,512,6}});
        cases.push_back({"16x64+6",chunked(16,64,6)}); cases.push_back({"6+1024",{6,1024}}); cases.push_back({"1029+1",{1029,1}});
    }
    const Snapshot ref=run(cases[0].chunks);
    for(size_t i=1;i<cases.size();i++){
        const Snapshot s=run(cases[i].chunks); bool ok=true;
        for(int l=0;l<L && ok;l++) for(int t=0;t<2 && ok;t++){
            const auto &a=t?ref.v[l]:ref.k[l], &b=t?s.v[l]:s.k[l];
            if(memcmp(a.data(),b.data(),a.size())){
                size_t j=0; while(a[j]==b[j]) j++;
                const int full=c.layer_is_full?c.layer_is_full[l]:1; const size_t row=(size_t)(full?nkv_f*hd_f:nkv_s*hd_s)*es;
                printf("CHUNK %-10s FAIL first diff layer %d (%s) %s position %zu\n",cases[i].name,l,full?"full":"sliding",t?"V":"K",j/row);
                ok=false; } }
        if(ok && memcmp(ref.logits.data(),s.logits.data(),(size_t)V*4)){ printf("CHUNK %-10s FAIL KV equal, last logits differ\n",cases[i].name); ok=false; }
        if(ok) printf("CHUNK %-10s OK (KV %d layers + last logits bit-identical to %s)\n",cases[i].name,L,cases[0].name);
        fails+=!ok;
    }
    printf("%s (%d failures)\n",fails?"PREFILL_CHUNK_FAIL":"PREFILL_CHUNK_PASS",fails);
    return fails?1:0;
}
