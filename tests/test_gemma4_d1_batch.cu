// Manual R11 batch isolation: each slot equals an independent batch-B1 execution.
// Frozen token IDs are teacher-forced inputs, not an HF greedy reference.
// Usage: GEMV_D1=1 test_gemma4_d1_batch <fp16 weights> <frozen ids.bin> [steps=64]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"

static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p,n*sizeof(T))); }
    ~Device() { if(p) cudaFree(p); }
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
};
static uint32_t bits(float x) { uint32_t b; std::memcpy(&b,&x,4); return b; }
static std::vector<int> read_ids(const char* path) {
    FILE* f=fopen(path,"rb"); if(!f) throw std::runtime_error("token file missing");
    if(fseek(f,0,SEEK_END)) { fclose(f); throw std::runtime_error("token seek failed"); }
    long n=ftell(f); rewind(f);
    if(n<=0 || n%4 || n>16*1024*1024) { fclose(f); throw std::runtime_error("invalid token length"); }
    std::vector<int> result(n/4); bool ok=fread(result.data(),4,result.size(),f)==result.size();
    if(fclose(f) || !ok) throw std::runtime_error("token read failed"); return result;
}
static int parse_steps(const char* text) {
    char* end=nullptr; long n=strtol(text,&end,10);
    if(end==text || *end || n<1 || n>128) throw std::runtime_error("steps must be 1..128");
    return int(n);
}
int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc<3 || argc>4) throw std::runtime_error("usage: GEMV_D1=1 test_gemma4_d1_batch <fp16 weights> <frozen ids.bin> [steps=64]");
    if(!getenv("GEMV_D1") || strcmp(getenv("GEMV_D1"),"1")) throw std::runtime_error("GEMV_D1=1 required for this gate");
    const int steps=argc>3?parse_steps(argv[3]):64,cap=steps+1;
    const std::string wd=argv[1],manifest=wd+"/manifest.json";
    ModelConfig c=load_config_gemma4(manifest.c_str());
    if(c.num_experts || c.scale_dtype!=1 || c.hidden_size!=3840 || c.num_layers!=48)
        throw std::runtime_error("R11 gate requires dense Gemma4-12B fp16-scale weights");
    auto ids=read_ids(argv[2]);
    if(ids.size()<size_t(steps)) throw std::runtime_error("frozen input has fewer tokens than steps");
    for(int t:ids) if(t<0 || t>=c.vocab_size) throw std::runtime_error("token outside vocabulary");
    const int V=c.vocab_size,hs=c.head_dim,hf=c.global_head_dim?c.global_head_dim:hs;
    size_t kv_one=0;
    for(int l=0;l<c.num_layers;l++) {
        bool full=c.layer_is_full?c.layer_is_full[l]:true;
        int hd=full?hf:hs,nkv=full&&c.num_global_kv_heads?c.num_global_kv_heads:c.num_kv_heads;
        kv_one+=(size_t)2*cap*hd*nkv*4;
    }
    const size_t host_ref_bytes=(size_t)12*steps*V*4;
    printf("D1_BATCH_CONFIG steps=%d B=4,8,12 scale_dtype=%d token_file=%s token_count=%zu kv_max_bytes=%zu host_reference_bytes=%zu reserved_gpu_bytes=25769803776\n",
           steps,c.scale_dtype,argv[2],ids.size(),kv_one*12,host_ref_bytes);
    Device<float> cf((size_t)cap*hf),sf((size_t)cap*hf),cs((size_t)cap*hs),ss((size_t)cap*hs);
    if(gemma4_gen_rope_tables(manifest.c_str(),cap,hf,cf.p,sf.p,hs,cs.p,ss.p)) throw std::runtime_error("manifest RoPE generation failed");
    ck(cudaDeviceSynchronize());
    G4Weights w=load_weights_gemma4(wd.c_str(),c,1,0);
    Device<int> dt(12); Device<float> logits((size_t)12*V);
    std::vector<float> reference((size_t)12*steps*V),got((size_t)12*V);
    auto token=[&](int slot,int t) { return ids[(size_t(t)+size_t(slot)*7)%ids.size()]; };
    for(int slot=0;slot<12;slot++) {
        G4BatchState st=alloc_g4_batch_state(c,1,cap);
        for(int t=0;t<steps;t++) {
            int tok=token(slot,t); ck(cudaMemcpy(dt.p,&tok,4,cudaMemcpyHostToDevice));
            gemma4_decode_batch(c,w,st,dt.p,cf.p,sf.p,cs.p,ss.p,logits.p,1,2);
            ck(cudaGetLastError());
            float* dst=reference.data()+((size_t)slot*steps+t)*V;
            ck(cudaMemcpy(dst,logits.p,(size_t)V*4,cudaMemcpyDeviceToHost));
            for(int j=0;j<V;j++) if(!std::isfinite(dst[j])) throw std::runtime_error("nonfinite B1 reference");
        }
        if(st.n_past!=steps) throw std::runtime_error("B1 commit position mismatch");
        free_g4_batch_state(st); printf("D1_BATCH_REFERENCE slot=%d steps=%d complete=1\n",slot,steps);
    }
    for(bool mixed:{false,true}) for(int B:{4,8,12}) {
        G4BatchState st=alloc_g4_batch_state(c,B,cap); std::vector<int> tokens(B);
        for(int t=0;t<steps;t++) {
            for(int b=0;b<B;b++) tokens[b]=token(mixed?b:0,t);
            ck(cudaMemcpy(dt.p,tokens.data(),(size_t)B*4,cudaMemcpyHostToDevice));
            gemma4_decode_batch(c,w,st,dt.p,cf.p,sf.p,cs.p,ss.p,logits.p,1,2);
            ck(cudaGetLastError()); ck(cudaMemcpy(got.data(),logits.p,(size_t)B*V*4,cudaMemcpyDeviceToHost));
            for(int b=0;b<B;b++) {
                const float* ref=reference.data()+((size_t)(mixed?b:0)*steps+t)*V;
                for(int j=0;j<V;j++) if(!std::isfinite(got[(size_t)b*V+j]) || bits(ref[j])!=bits(got[(size_t)b*V+j])) {
                    const float value=got[(size_t)b*V+j];
                    fprintf(stderr,"D1_BATCH_FAIL mode=%s B=%d step=%d slot=%d logit=%d expected=%.9g actual=%.9g expected_bits=%08x actual_bits=%08x\n",
                            mixed?"mixed":"same",B,t,b,j,ref[j],value,bits(ref[j]),bits(value));
                    throw std::runtime_error("batch slot differs from independent B1");
                }
            }
        }
        if(st.n_past!=steps) throw std::runtime_error("batch commit position mismatch");
        free_g4_batch_state(st);
        printf("D1_BATCH_PASS mode=%s B=%d steps=%d compared_floats=%zu reference=batch_B1 mismatches=0\n",mixed?"mixed":"same",B,steps,(size_t)B*steps*V);
    }
    free_weights_gemma4(w); free_config_gemma4(c);
    printf("[test_gemma4_d1_batch] PASS\n"); return 0;
} catch(const std::exception& e) { fprintf(stderr,"[test_gemma4_d1_batch] FAIL: %s\n",e.what()); return 1; }
