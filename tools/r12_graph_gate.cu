// Full-logit capture for R12; successful capture alone does not establish equivalence.
#include "runtime/knobs.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"

static void ck(cudaError_t err) {
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
template<class T> static T* device_alloc(size_t count) {
    T* ptr=nullptr;
    ck(cudaMalloc(&ptr,count*sizeof(T)));
    return ptr;
}
static int integer(const char* value) {
    char* end=nullptr; errno=0;
    long parsed=strtol(value,&end,10);
    if (errno || end==value || *end || parsed<0 || parsed>INT_MAX)
        throw std::runtime_error("invalid nonnegative integer");
    return static_cast<int>(parsed);
}
using File=std::unique_ptr<FILE,decltype(&fclose)>;
static File create_file(const std::string& path) {
    FILE* file=fopen(path.c_str(),"wx");
    if (!file) throw std::runtime_error("output exists or cannot be created: "+path);
    return File(file,&fclose);
}
static std::vector<int> token_pattern(const char* path) {
    if (!path) return {2,105,2364,107,106};
    File file(fopen(path,"rb"),&fclose);
    if (!file || fseek(file.get(),0,SEEK_END)) throw std::runtime_error("cannot read token file");
    const long bytes=ftell(file.get());
    if (bytes<=0 || bytes%sizeof(int32_t) || bytes>16*1024*1024)
        throw std::runtime_error("invalid int32 token file size");
    rewind(file.get());
    std::vector<int> ids(static_cast<size_t>(bytes)/sizeof(int32_t));
    static_assert(sizeof(int)==sizeof(int32_t),"token storage requires int32");
    if (fread(ids.data(),sizeof(int32_t),ids.size(),file.get())!=ids.size())
        throw std::runtime_error("incomplete token read");
    return ids;
}
static void close_file(File& file) {
    if (fclose(file.release())) throw std::runtime_error("output flush/close failed");
}

int main(int argc,char** argv) try {
    if (argc<4 || argc>7)
        throw std::runtime_error("usage: r12_graph_gate WEIGHTS OUTPUT_PREFIX START [STEPS=128] [TOKENS.bin|-] [teacher|greedy]");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const std::string weights=argv[1],prefix=argv[2],manifest=weights+"/manifest.json";
    const int start=integer(argv[3]),steps=argc>4?integer(argv[4]):128;
    if ((start!=0 && start!=300 && start!=900) || steps!=128)
        throw std::runtime_error("R12 gate requires START=0,300,900 and STEPS=128");
    const std::string mode=argc>6?argv[6]:"teacher";
    if (mode!="teacher" && mode!="greedy") throw std::runtime_error("mode must be teacher or greedy");
    auto pattern=token_pattern(argc>5 && std::string(argv[5])!="-"?argv[5]:nullptr);
    char* raw=mf_read_file(manifest.c_str());
    if (!raw) throw std::runtime_error("manifest missing");
    for (const char* key:{"rope_theta_full","rope_theta_sliding","rotary_dim_full","rotary_dim_sliding"}) {
        if (!(mf_num_opt(raw,key,0)>0)) {
            free(raw); throw std::runtime_error(std::string("required manifest field: ")+key);
        }
    }
    free(raw);
    ModelConfig c=load_config_gemma4(manifest.c_str());
    const int V=c.vocab_size,hd_s=c.head_dim,hd_f=c.global_head_dim?c.global_head_dim:hd_s;
    if (c.num_experts || V<=0 || hd_s<=0 || hd_f<=0 || c.rotary_dim_full>hd_f ||
        c.rotary_dim_sliding>hd_s || c.rotary_dim_full%2 || c.rotary_dim_sliding%2)
        throw std::runtime_error("requires valid dense Gemma manifest");
    for (int id:pattern) if (id<0 || id>=V) throw std::runtime_error("token outside vocabulary");
    auto logits_file=create_file(prefix+".logits.f32");
    auto rows_file=create_file(prefix+".rows.csv");
    auto input_file=create_file(prefix+".inputs.i32");
    auto argmax_file=create_file(prefix+".argmax.i32");
    auto pattern_file=create_file(prefix+".pattern.i32");
    if (fwrite(pattern.data(),sizeof(int32_t),pattern.size(),pattern_file.get())!=pattern.size())
        throw std::runtime_error("pattern write failed");
    close_file(pattern_file);
    if (fprintf(rows_file.get(),"step,position,input_token,argmax\n")<0) throw std::runtime_error("CSV write failed");
    const int capacity=start+steps;
    printf("R12_CAPTURE start=%d steps=%d capacity=%d vocab=%d mode=%s uq=2 scale_dtype=%d graph_env=%s\n",
           start,steps,capacity,V,mode.c_str(),c.scale_dtype,knob_raw("G4_CUDA_GRAPH")?knob_raw("G4_CUDA_GRAPH"):"<unset>");
    printf("R12_CAPTURE logits_bytes=%zu host_logits_bytes=%zu prefix_rule=repeat_input_pattern\n",
           static_cast<size_t>(steps)*V*sizeof(float),static_cast<size_t>(V)*sizeof(float));
    float *cf=device_alloc<float>(static_cast<size_t>(capacity)*hd_f),*sf=device_alloc<float>(static_cast<size_t>(capacity)*hd_f);
    float *cs=device_alloc<float>(static_cast<size_t>(capacity)*hd_s),*ss=device_alloc<float>(static_cast<size_t>(capacity)*hd_s);
    if (gemma4_gen_rope_tables(manifest.c_str(),capacity,hd_f,cf,sf,hd_s,cs,ss))
        throw std::runtime_error("RoPE generation failed");
    ck(cudaDeviceSynchronize());
    G4Weights w=load_weights_gemma4(weights.c_str(),c,1,0);
    G4DecodeState state=alloc_g4_decode_state(c,capacity);
    int* tokens=device_alloc<int>(std::max(start,1));
    float* logits=device_alloc<float>(V);
    std::vector<float> host(V);
    if (start) {
        std::vector<int> prompt(start);
        for (int pos=0;pos<start;pos++) prompt[pos]=pattern[static_cast<size_t>(pos)%pattern.size()];
        ck(cudaMemcpy(tokens,prompt.data(),prompt.size()*sizeof(int),cudaMemcpyHostToDevice));
        gemma4_decode(c,w,state,tokens,start,cf,sf,cs,ss,logits,1,2,1,1,nullptr);
    }
    if (state.n_past!=start) throw std::runtime_error("prefix commit mismatch");
    int next=pattern[static_cast<size_t>(start)%pattern.size()];
    for (int step=0;step<steps;step++) {
        const int input=mode=="greedy"?next:pattern[static_cast<size_t>(start+step)%pattern.size()];
        ck(cudaMemcpy(tokens,&input,sizeof(int),cudaMemcpyHostToDevice));
#ifdef R12_BASELINE
        gemma4_decode(c,w,state,tokens,1,cf,sf,cs,ss,logits,1,2,1,0,nullptr);
#else
        gemma4_decode_step(c,w,state,tokens,1,cf,sf,cs,ss,logits,1,2,1,0,nullptr);
#endif
        ck(cudaMemcpy(host.data(),logits,static_cast<size_t>(V)*sizeof(float),cudaMemcpyDeviceToHost));
        for (float value:host) if (!std::isfinite(value)) throw std::runtime_error("nonfinite logit at step "+std::to_string(step));
        next=static_cast<int>(std::max_element(host.begin(),host.end())-host.begin());
        if (fwrite(host.data(),sizeof(float),host.size(),logits_file.get())!=host.size() ||
            fwrite(&input,sizeof(int32_t),1,input_file.get())!=1 ||
            fwrite(&next,sizeof(int32_t),1,argmax_file.get())!=1 ||
            fprintf(rows_file.get(),"%d,%d,%d,%d\n",step,start+step,input,next)<0)
            throw std::runtime_error("capture write failed");
        if (state.n_past!=start+step+1) throw std::runtime_error("decode commit mismatch");
    }
    for (auto* file:{&logits_file,&rows_file,&input_file,&argmax_file}) close_file(*file);
    free_g4_decode_state(state);
    for (float* ptr:{cf,sf,cs,ss,logits}) ck(cudaFree(ptr));
    ck(cudaFree(tokens)); free_weights_gemma4(w); free_config_gemma4(c);
    printf("R12_CAPTURE_COMPLETE start=%d steps=%d mode=%s; compare files externally for bit equivalence\n",start,steps,mode.c_str());
    return 0;
} catch (const std::exception& error) {
    fprintf(stderr,"R12_CAPTURE_FAILED: %s\n",error.what());
    return 1;
}
