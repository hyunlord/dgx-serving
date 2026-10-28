// Capture and timing evidence; equivalence requires comparing separate runs.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/rope_gemma4.h"

static void ck(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> static T* alloc(size_t n) {
    T* p=nullptr; ck(cudaMalloc(&p,n*sizeof(T))); return p;
}
using File=std::unique_ptr<FILE,decltype(&fclose)>;
static File output(const std::string& p) {
    File f(fopen(p.c_str(),"wx"),&fclose);
    if(!f) throw std::runtime_error("cannot exclusively create "+p);
    return f;
}
static void close_file(File& f) {
    if(fclose(f.release())) throw std::runtime_error("output close failed");
}
static int integer(const char* s) {
    char* end=nullptr; errno=0; long x=strtol(s,&end,10);
    if(errno || end==s || *end || x<0 || x>INT_MAX) throw std::runtime_error("invalid integer");
    return static_cast<int>(x);
}
static std::vector<int> read_tokens(const char* p) {
    static_assert(sizeof(int)==sizeof(int32_t),"int32 tokens required");
    File f(fopen(p,"rb"),&fclose);
    if(!f || fseek(f.get(),0,SEEK_END)) throw std::runtime_error("cannot read tokens");
    long bytes=ftell(f.get());
    if(bytes<=0 || bytes%4 || bytes>16*1024*1024) throw std::runtime_error("invalid token file size");
    rewind(f.get()); std::vector<int> v(static_cast<size_t>(bytes)/4);
    if(fread(v.data(),4,v.size(),f.get())!=v.size()) throw std::runtime_error("short token read");
    return v;
}
static double epoch() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
struct Capture {
    bool full;
    int vocab,start,next;
    int exact_top2_ties=0;
    std::vector<float> host;
    std::vector<int> tokens,inputs;
    File logits;
    Capture(bool f,int v,int s,int first,const std::string& prefix,int max_steps)
        :full(f),vocab(v),start(s),next(first),host(f?v:0),logits(nullptr,&fclose) {
        tokens.reserve(max_steps); inputs.reserve(max_steps);
        if(full) logits=output(prefix+".logits.f32");
    }
};
static int capture(const float* d,int token,int step,void* opaque) {
    auto& c=*static_cast<Capture*>(opaque);
    if(step!=static_cast<int>(c.tokens.size()) || token<0 || token>=c.vocab)
        throw std::runtime_error("callback step/token invalid");
    if(c.full) {
        ck(cudaMemcpy(c.host.data(),d,c.host.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(float x:c.host) if(!std::isfinite(x)) throw std::runtime_error("nonfinite logit");
        int argmax=static_cast<int>(std::max_element(c.host.begin(),c.host.end())-c.host.begin());
        float runner_up=-INFINITY;
        for(int i=0;i<c.vocab;i++) if(i!=argmax) runner_up=std::max(runner_up,c.host[i]);
        if(runner_up==c.host[argmax]) ++c.exact_top2_ties;
        if(argmax!=token) throw std::runtime_error("GPU/CPU argmax mismatch at step "+std::to_string(step));
        if(fwrite(c.host.data(),sizeof(float),c.host.size(),c.logits.get())!=c.host.size())
            throw std::runtime_error("logits write failed");
    }
    c.inputs.push_back(c.next); c.tokens.push_back(token); c.next=token;
    return 0;
}
static void save_ids(FILE* f,const std::vector<int>& ids) {
    if(fwrite(ids.data(),sizeof(int32_t),ids.size(),f)!=ids.size()) throw std::runtime_error("token write failed");
}

int main(int argc,char** argv) try {
    if(argc==2 && std::string(argv[1])=="argmax") return gemma4_graph_argmax_test();
    if(argc<6) throw std::runtime_error("usage: r13_graph_gate WEIGHTS PREFIX gate|speed START TOKENS.bin; or WEIGHTS PREFIX eos EOS_ID PROMPT.bin SECOND_PROMPT.bin");
    const std::string weights=argv[1],prefix=argv[2],mode=argv[3],manifest=weights+"/manifest.json";
    const bool eos_mode=mode=="eos",speed=mode=="speed";
    if((!eos_mode && mode!="gate" && !speed) || argc!=(eos_mode?7:6)) throw std::runtime_error("invalid mode/arguments");
    const int argument=integer(argv[4]);
    if(!eos_mode && argument!=0 && argument!=300 && argument!=900) throw std::runtime_error("START must be 0, 300, or 900");
    auto pattern=read_tokens(argv[5]);
    auto second=eos_mode?read_tokens(argv[6]):std::vector<int>();
    if(eos_mode && (pattern.size()>768 || second.size()>896)) throw std::runtime_error("EOS prompts plus generation must fit 1024 positions");
    int capacity=eos_mode?1024:std::max(1024,argument+128);
    char* raw=mf_read_file(manifest.c_str());
    if(!raw) throw std::runtime_error("manifest missing");
    for(const char* key:{"rope_theta_full","rope_theta_sliding","rotary_dim_full","rotary_dim_sliding"}) {
        if(!(mf_num_opt(raw,key,0)>0)) { free(raw); throw std::runtime_error(std::string("missing manifest field ")+key); }
    }
    free(raw);
    ModelConfig c=load_config_gemma4(manifest.c_str());
    const int V=c.vocab_size,hs=c.head_dim,hf=c.global_head_dim?c.global_head_dim:hs;
    if(c.num_experts || V<=0 || hs<=0 || hf<=0 || c.rotary_dim_full>hf || c.rotary_dim_sliding>hs || c.rotary_dim_full%2 || c.rotary_dim_sliding%2)
        throw std::runtime_error("invalid dense Gemma configuration");
    for(const auto* ids:{&pattern,&second}) for(int t:*ids) if(t<0 || t>=V) throw std::runtime_error("invalid token ID");
    if(eos_mode && argument>=V) throw std::runtime_error("invalid EOS ID");
    // Reserve metadata before loading weights so reusing a prefix fails immediately.
    auto metadata=output(prefix+".summary.jsonl");
    auto pattern_file=output(prefix+".pattern.i32"); save_ids(pattern_file.get(),pattern); close_file(pattern_file);
    if(eos_mode) { auto f=output(prefix+".second_pattern.i32"); save_ids(f.get(),second); close_file(f); }
    setvbuf(stdout,nullptr,_IOLBF,0);
    printf("R13 mode=%s capacity=%d uq=2 scale_dtype=%d graph=%s overlap=%s node_update=%s\n",mode.c_str(),capacity,c.scale_dtype,
           getenv("G4_CUDA_GRAPH")?getenv("G4_CUDA_GRAPH"):"0",getenv("G4_GRAPH_OVERLAP")?getenv("G4_GRAPH_OVERLAP"):"default",getenv("G4_GRAPH_NODE_UPDATE")?getenv("G4_GRAPH_NODE_UPDATE"):"default");
    float *cf=alloc<float>(static_cast<size_t>(capacity)*hf),*sf=alloc<float>(static_cast<size_t>(capacity)*hf);
    float *cs=alloc<float>(static_cast<size_t>(capacity)*hs),*ss=alloc<float>(static_cast<size_t>(capacity)*hs);
    if(gemma4_gen_rope_tables(manifest.c_str(),capacity,hf,cf,sf,hs,cs,ss)) throw std::runtime_error("RoPE generation failed");
    ck(cudaDeviceSynchronize());
    G4Weights w=load_weights_gemma4(weights.c_str(),c,1,0);
    G4DecodeState state=alloc_g4_decode_state(c,capacity);
    int* d_tokens=alloc<int>(capacity); float* d_logits=alloc<float>(V);
    const int requests=eos_mode?2:1;
    for(int request=0;request<requests;request++) {
        reset_g4_decode_state(state);
        const auto& source=request?second:pattern;
        const int start=eos_mode?static_cast<int>(source.size())-1:argument;
        const int first=eos_mode?source.back():source[static_cast<size_t>(start)%source.size()];
        const int steps=eos_mode && request==0?256:128;
        const int eos=eos_mode && request==0?argument:-1;
        const std::string out=eos_mode?prefix+(request?".second":".first"):prefix;
        // All output handles exist before the timed interval; speed writes afterward.
        auto inputs=output(out+".inputs.i32"),tokens=output(out+".argmax.i32"),rows=output(out+".rows.csv");
        Capture ctx(!speed,V,start,first,out,steps);
        if(start) {
            std::vector<int> prompt(start);
            for(int i=0;i<start;i++) prompt[i]=source[static_cast<size_t>(i)%source.size()];
            ck(cudaMemcpy(d_tokens,prompt.data(),prompt.size()*sizeof(int),cudaMemcpyHostToDevice));
            gemma4_decode(c,w,state,d_tokens,start,cf,sf,cs,ss,d_logits,1,2,1,1,nullptr);
        }
        ck(cudaDeviceSynchronize());
        if(state.n_past!=start) throw std::runtime_error("prefix commit mismatch");
        G4GenerationStats stats{};
        const double begin=epoch(); const auto clock_begin=std::chrono::steady_clock::now();
        int accepted=gemma4_generate(c,w,state,first,steps,eos,cf,sf,cs,ss,2,capture,&ctx,&stats);
        ck(cudaDeviceSynchronize());
        const auto clock_end=std::chrono::steady_clock::now(); const double end=epoch();
        const double elapsed=std::chrono::duration<double>(clock_end-clock_begin).count();
        if(accepted!=static_cast<int>(ctx.tokens.size()) || state.n_past!=start+accepted)
            throw std::runtime_error("accepted/callback/n_past mismatch");
        if(eos<0 && accepted!=steps) throw std::runtime_error("generation ended early without EOS enabled");
        const bool eos_observed=eos<0 || (accepted>0 && accepted<steps && ctx.tokens.back()==eos);
        save_ids(inputs.get(),ctx.inputs); save_ids(tokens.get(),ctx.tokens);
        if(fprintf(rows.get(),"step,position,input_token,argmax\n")<0) throw std::runtime_error("CSV write failed");
        for(int i=0;i<accepted;i++) if(fprintf(rows.get(),"%d,%d,%d,%d\n",i,start+i,ctx.inputs[i],ctx.tokens[i])<0) throw std::runtime_error("CSV write failed");
        close_file(inputs); close_file(tokens); close_file(rows); if(ctx.full) close_file(ctx.logits);
        printf("DECODE_INTERVAL start=%.17g end=%.17g steps=%d seconds=%.17g tok_s=%.17g diagnostic=%d request=%d\n",begin,end,accepted,elapsed,accepted/elapsed,!speed,request);
        const std::string json="{\"request\":%d,\"start\":%d,\"accepted\":%d,\"n_past\":%d,\"epoch_start\":%.17g,\"epoch_end\":%.17g,\"seconds\":%.17g,\"tok_s\":%.17g,\"planned_captures\":%llu,\"fallback_recaptures\":%llu,\"node_updates\":%llu,\"instantiations\":%llu,\"executed_steps\":%llu,\"accepted_steps\":%llu,\"speculative_steps\":%llu,\"capture_us\":%.17g,\"update_us\":%.17g,\"reinstantiations\":%llu,\"update_failures\":%llu,\"setparams_failures\":%llu,\"top2_checked_steps\":%d,\"exact_top2_ties\":%d,\"graph_updates\":%llu,\"mapping_fallbacks\":%llu}\n";
        if(fprintf(metadata.get(),json.c_str(),request,start,accepted,state.n_past,begin,end,elapsed,accepted/elapsed,
            static_cast<unsigned long long>(stats.planned_captures),static_cast<unsigned long long>(stats.fallback_recaptures),static_cast<unsigned long long>(stats.node_updates),
            static_cast<unsigned long long>(stats.instantiations),static_cast<unsigned long long>(stats.executed_steps),static_cast<unsigned long long>(stats.accepted_steps),
            static_cast<unsigned long long>(stats.speculative_steps),stats.capture_us,stats.update_us,
            static_cast<unsigned long long>(stats.reinstantiations),static_cast<unsigned long long>(stats.update_failures),static_cast<unsigned long long>(stats.setparams_failures),speed?0:accepted,ctx.exact_top2_ties,static_cast<unsigned long long>(stats.graph_updates),static_cast<unsigned long long>(stats.mapping_fallbacks))<0) throw std::runtime_error("summary write failed");
        printf("R13_MAPPING graph_updates=%llu mapping_fallbacks=%llu\n",static_cast<unsigned long long>(stats.graph_updates),static_cast<unsigned long long>(stats.mapping_fallbacks));
        printf("R13_UPDATE_FAILURES reinstantiations=%llu update_failures=%llu setparams_failures=%llu\n",
            static_cast<unsigned long long>(stats.reinstantiations),static_cast<unsigned long long>(stats.update_failures),static_cast<unsigned long long>(stats.setparams_failures));
        printf("R13_TOP2_TIES request=%d start=%d checked_steps=%d exact_equal_steps=%d\n",request,start,speed?0:accepted,ctx.exact_top2_ties);
        printf("R13_STATS planned_captures=%llu fallback_recaptures=%llu node_updates=%llu instantiations=%llu executed=%llu accepted=%llu speculative=%llu capture_us=%.17g update_us=%.17g\n",
            static_cast<unsigned long long>(stats.planned_captures),static_cast<unsigned long long>(stats.fallback_recaptures),static_cast<unsigned long long>(stats.node_updates),
            static_cast<unsigned long long>(stats.instantiations),static_cast<unsigned long long>(stats.executed_steps),static_cast<unsigned long long>(stats.accepted_steps),
            static_cast<unsigned long long>(stats.speculative_steps),stats.capture_us,stats.update_us);
        if(!eos_observed) throw std::runtime_error("EOS gate not exercised before generation limit; captured evidence retained");
        if(speed && elapsed<1.0) throw std::runtime_error("speed interval below one second; measurement invalid");
    }
    close_file(metadata);
    free_g4_decode_state(state);
    for(float* p:{cf,sf,cs,ss,d_logits}) ck(cudaFree(p)); ck(cudaFree(d_tokens));
    free_weights_gemma4(w); free_config_gemma4(c);
    printf("R13_COMPLETE mode=%s; compare output files externally for bit equivalence\n",mode.c_str());
    return 0;
} catch(const std::exception& e) {
    fprintf(stderr,"R13_FAILED: %s\n",e.what()); return 1;
}
