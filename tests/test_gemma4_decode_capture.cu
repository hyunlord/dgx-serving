// Before/after allocation-only diagnostic; exit 0 means capture complete, not accuracy PASS.
// Usage: <binary> <QAT weights> <golden directory> <frozen chat ids.bin> <NEW output directory>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"
#include "../models/gemma4_moe.h"

static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> static T* alloc(size_t n) {
    T* p; ck(cudaMalloc(&p,n*sizeof(T))); return p;
}
static std::vector<int> read_ids(const std::string& path) {
    FILE* f=fopen(path.c_str(),"rb");
    if (!f) throw std::runtime_error("missing tokens: "+path);
    if (fseek(f,0,SEEK_END)) throw std::runtime_error("seek failed");
    long bytes=ftell(f);
    if (bytes<=0 || bytes%4 || bytes>16*1024*1024) throw std::runtime_error("invalid token file");
    rewind(f); std::vector<int> ids(bytes/4);
    bool ok=fread(ids.data(),4,ids.size(),f)==ids.size();
    if (fclose(f) || !ok) throw std::runtime_error("token read failed");
    return ids;
}
struct Capture {
    std::string dir;
    FILE* index;
    size_t bytes=0;
    explicit Capture(const std::string& path):dir(path) {
        if (!std::filesystem::create_directory(dir)) throw std::runtime_error("output must be new");
        index=fopen((dir+"/captures.csv").c_str(),"wx");
        if (!index) throw std::runtime_error("index create failed");
        fprintf(index,"file,elements,itemsize\n");
    }
    template<class T> void host(const std::string& name,const T* p,size_t n) {
        FILE* f=fopen((dir+"/"+name+".bin").c_str(),"wx");
        if (!f) throw std::runtime_error("capture exists or cannot be created");
        bool ok=fwrite(p,sizeof(T),n,f)==n;
        if (fclose(f) || !ok) throw std::runtime_error("capture write failed");
        if (fprintf(index,"%s.bin,%zu,%zu\n",name.c_str(),n,sizeof(T))<0)
            throw std::runtime_error("index write failed");
        bytes+=n*sizeof(T);
    }
    void device(const std::string& name,const float* p,size_t n) {
        std::vector<float> h(n); ck(cudaMemcpy(h.data(),p,n*4,cudaMemcpyDeviceToHost));
        for (float v:h) if (!std::isfinite(v)) throw std::runtime_error("nonfinite: "+name);
        host(name,h.data(),n);
    }
    void finish() {
        if (fclose(index)) throw std::runtime_error("index close failed");
        printf("CAPTURE_COMPLETE bytes=%zu; equivalence/accuracy NOT evaluated\n",bytes);
    }
};

int main(int argc,char** argv) try {
    if (argc!=5) throw std::runtime_error("usage: capture <QAT weights> <golden> <chat ids.bin> <NEW output>");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const std::string wd=argv[1], gd=argv[2], manifest=wd+"/manifest.json";
    char* raw=mf_read_file(manifest.c_str());
    if (!raw) throw std::runtime_error("manifest missing");
    for (const char* key:{"rope_theta_full","rope_theta_sliding","rotary_dim_full","rotary_dim_sliding"})
        if (!(mf_num_opt(raw,key,0)>0)) throw std::runtime_error(std::string("required manifest field: ")+key);
    free(raw);
    ModelConfig c=load_config_gemma4(manifest.c_str());
    const int V=c.vocab_size,H=c.hidden_size,hd_s=c.head_dim;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s;
    if (V<=0 || H<=0 || c.rotary_dim_full>hd_f || c.rotary_dim_sliding>hd_s ||
        c.rotary_dim_full%2 || c.rotary_dim_sliding%2) throw std::runtime_error("invalid dimensions");
    auto prompt=read_ids(gd+"/token_ids.bin"), full=read_ids(argv[3]);
    for (const auto* ids:{&prompt,&full}) for (int t:*ids)
        if (t<0 || t>=V) throw std::runtime_error("token outside vocabulary");
    const int P=std::min((int)prompt.size(),128), ML=1024;
    if (full.size()<64) throw std::runtime_error("frozen chat input must contain at least 64 tokens");
    Capture out(argv[4]);
    out.host("source_prompt",prompt.data(),prompt.size()); out.host("source_chat",full.data(),full.size());
    const bool moe=c.num_experts>0;
    printf("uq=2 model=%s P=%d max_len=%d; prefill1024 repeats golden prompt; teacher forcing uses independent frozen chat IDs (not HF golden continuation)\n",
           moe?"moe":"dense",P,ML);
    for (const char* name:{"G4_MMQ_PREFILL","G4_MMQ_MIN_S","G4_FLASH_PREFILL","G4_FLASH_MIN_S","BM16"})
        printf("ENV %s=%s\n",name,getenv(name)?getenv(name):"<unset>");
    float *cf=alloc<float>((size_t)ML*hd_f),*sf=alloc<float>((size_t)ML*hd_f);
    float *cs=alloc<float>((size_t)ML*hd_s),*ss=alloc<float>((size_t)ML*hd_s);
    if (gemma4_gen_rope_tables(manifest.c_str(),ML,hd_f,cf,sf,hd_s,cs,ss))
        throw std::runtime_error("RoPE generation failed");
    ck(cudaDeviceSynchronize());
    out.device("rope_full_cos",cf,(size_t)ML*hd_f); out.device("rope_full_sin",sf,(size_t)ML*hd_f);
    out.device("rope_sliding_cos",cs,(size_t)ML*hd_s); out.device("rope_sliding_sin",ss,(size_t)ML*hd_s);
    G4Weights w=load_weights_gemma4(wd.c_str(),c,1,0);
    int* dt=alloc<int>(1024); float* lg=alloc<float>((size_t)64*V); float* pn=alloc<float>((size_t)1024*H);
    // Sample only the last four valid rows per slot/layer; never dump uninitialized KV capacity.
    auto kv_tail=[&](const std::string& name,float** kc,float** vc,int capacity,int slot,int valid){
        const int count=std::min(valid,4),begin=valid-count;
        for (int l=0;l<c.num_layers;l++) {
            bool is_full=c.layer_is_full?c.layer_is_full[l]:true;
            int hd=is_full?hd_f:hd_s;
            int nkv=is_full && c.num_global_kv_heads?c.num_global_kv_heads:c.num_kv_heads;
            size_t dim=(size_t)hd*nkv,offset=((size_t)slot*capacity+begin)*dim;
            std::string suffix="_slot"+std::to_string(slot)+"_l"+std::to_string(l)+"_rows"+std::to_string(begin)+"_"+std::to_string(valid);
            out.device(name+suffix+"_K",kc[l]+offset,(size_t)count*dim);
            out.device(name+suffix+"_V",vc[l]+offset,(size_t)count*dim);
        }
    };
    auto upload=[&](const int* p,int n){ ck(cudaMemcpy(dt,p,(size_t)n*4,cudaMemcpyHostToDevice)); };
    auto decode=[&](G4DecodeState& st,int n,int commit,int last,int amort){
        if (moe) gemma4_moe_decode(c,w,st,dt,n,cf,sf,cs,ss,lg,commit,2,last);
        else gemma4_decode(c,w,st,dt,n,cf,sf,cs,ss,lg,commit,2,last,amort,pn);
    };
    {
        G4DecodeState st=alloc_g4_decode_state(c,ML);
        upload(prompt.data(),P); decode(st,P,1,1,1);
        out.device("decode_prompt_last",lg,V);
        for (int t=0;t<64;t++) {
            upload(full.data()+t,1); decode(st,1,1,1,0);
            out.device("decode64_"+std::to_string(t),lg,V);
            if (!moe) out.device("decode64_"+std::to_string(t)+"_prenorm",pn,H);
        }
        if (st.n_past!=P+64) throw std::runtime_error("decode commit mismatch");
        kv_tail("decode64_kvtail",st.Kc,st.Vc,st.max_len,0,st.n_past);
        std::vector<int> long_prompt(1024);
        for (int j=0;j<1024;j++) long_prompt[j]=prompt[j%prompt.size()];
        out.host("prefill1024_tokens",long_prompt.data(),long_prompt.size());
        reset_g4_decode_state(st); upload(long_prompt.data(),1024); decode(st,1024,1,1,1);
        out.device("prefill1024_last",lg,V);
        if (!moe) out.device("prefill1024_prenorm",pn,(size_t)1024*H);
        kv_tail("prefill1024_kvtail",st.Kc,st.Vc,st.max_len,0,st.n_past);
        reset_g4_decode_state(st); upload(full.data(),1); decode(st,1,1,1,0);
        out.device("after1024_reset_decode",lg,V);
        int iteration=0;
        for (int n:{1,17,5,64,9,1}) {
            reset_g4_decode_state(st); upload(long_prompt.data(),n); decode(st,n,1,0,0);
            std::string name="varysq_"+std::to_string(iteration++)+"_n"+std::to_string(n);
            out.device(name,lg,(size_t)n*V);
            if (!moe) out.device(name+"_prenorm",pn,(size_t)n*H);
            kv_tail(name+"_kvtail",st.Kc,st.Vc,st.max_len,0,st.n_past);
        }
        free_g4_decode_state(st);
    }
    int iteration=0;
    for (int B:{1,4,1,4}) {
        G4BatchState st=alloc_g4_batch_state(c,B,96);
        std::vector<int> tokens(B);
        for (int t=0;t<64;t++) {
            for (int b=0;b<B;b++) tokens[b]=full[(t+b*7)%full.size()];
            upload(tokens.data(),B);
            if (moe) gemma4_moe_decode_batch(c,w,st,dt,cf,sf,cs,ss,lg,1,2,nullptr,nullptr);
            else gemma4_decode_batch(c,w,st,dt,cf,sf,cs,ss,lg,1,2);
            out.device("batch_"+std::to_string(iteration)+"_B"+std::to_string(B)+"_t"+std::to_string(t),lg,(size_t)B*V);
        }
        if (st.n_past!=64) throw std::runtime_error("batch commit mismatch");
        for (int b=0;b<B;b++) kv_tail("batch_"+std::to_string(iteration)+"_kvtail",st.Kc,st.Vc,st.max_len,b,64);
        free_g4_batch_state(st); iteration++;
    }
    if (!moe) {
        G4BatchState st=alloc_g4_batch_state(c,4,96);
        int slots[4]={0,1,2,3},pos[4]={32,41,50,59};
        for (int b=0;b<4;b++) {
            upload(full.data(),pos[b]);
            gemma4_prefill_batch_slot(c,w,st,b,dt,pos[b],cf,sf,cs,ss,lg,2);
        }
        const int cases[][4]={{5,0,0,0},{5,3,0,0},{5,4,0,0},{5,5,0,0},{5,5,5,0},
            {5,5,6,0},{9,5,3,0},{5,5,5,5},{9,9,9,9},{5,3,0,0},{9,5,3,0},{5,0,0,0}};
        iteration=0;
        for (const auto& nb:cases) {
            int B=0,M=0; std::vector<int> tokens;
            for (int b=0;b<4;b++) if (nb[b]) { B++; M+=nb[b];
                for (int j=0;j<nb[b];j++) tokens.push_back(full[(b*7+j*3)%full.size()]); }
            upload(tokens.data(),M);
            std::string name="concat_"+std::to_string(iteration++)+"_M"+std::to_string(M);
            gemma4_verify_batch(c,w,st,B,slots,nb,pos,dt,cf,sf,cs,ss,lg,2);
            out.device(name+"_canonical",lg,(size_t)M*V);
            gemma4_verify_concat(c,w,st,B,slots,nb,pos,dt,cf,sf,cs,ss,lg,2,pn);
            out.device(name+"_logits",lg,(size_t)M*V); out.device(name+"_prenorm",pn,(size_t)M*H);
            for (int b=0;b<B;b++) kv_tail(name+"_kvtail",st.Kc,st.Vc,st.max_len,b,pos[b]+nb[b]);
        }
        free_g4_batch_state(st);
    } else printf("SKIP verifyconcat: public concat API is dense-only; MoE B4 captured\n");
    for (float* p:{cf,sf,cs,ss,lg,pn}) ck(cudaFree(p));
    ck(cudaFree(dt)); free_weights_gemma4(w); free_config_gemma4(c); out.finish(); return 0;
} catch (const std::exception& e) {
    fprintf(stderr,"CAPTURE_FAILED: %s\n",e.what()); return 1;
}
