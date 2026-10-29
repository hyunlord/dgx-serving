// Dense Gemma teacher-forced PPL; existing graph/weights remain unchanged.
#include "runtime/knobs.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/rope_gemma4.h"

namespace fs = std::filesystem;
static void require(bool ok, const std::string &why) {
    if (!ok) { fprintf(stderr, "PPL_ERROR %s\n", why.c_str()); exit(1); }
}
#define CK(call) do { cudaError_t e=(call); require(e==cudaSuccess,cudaGetErrorString(e)); } while(0)
static size_t mul(size_t a,size_t b) {
    require(b==0 || a<=std::numeric_limits<size_t>::max()/b,"size overflow"); return a*b;
}
static size_t add(size_t a,size_t b) {
    require(a<=std::numeric_limits<size_t>::max()-b,"size overflow"); return a+b;
}
static std::string quote(const std::string &s) {
    std::ostringstream out; out << '"';
    for(unsigned char ch:s) {
        if(ch=='"' || ch=='\\') out << '\\' << ch;
        else if(ch<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec;
        else out << ch;
    } out << '"'; return out.str();
}
static int positive(const std::string &s) {
    require(!s.empty() && std::all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}),"positive decimal integer required");
    errno=0; char *end=nullptr; long n=strtol(s.c_str(),&end,10);
    require(errno==0 && *end==0 && n>0 && n<=std::numeric_limits<int>::max(),"integer out of range"); return int(n);
}
struct Options { std::string weights,tokens,golden,dump,lengths,mask,save_ref,kld_ref; int window=1024,nwin=32,uq=2; bool position_stats=false,decode=false; };
static Options parse(int argc,char **argv) {
    Options o; std::vector<std::string> seen;
    for(int i=1;i<argc;) {
        std::string key=argv[i++];
        require(std::find(seen.begin(),seen.end(),key)==seen.end(),"duplicate option "+key); seen.push_back(key);
        if(key=="--position-stats") { o.position_stats=true; continue; }
        if(key=="--decode") { o.decode=true; continue; }
        require(i<argc,"every option requires a value"); std::string value=argv[i++];
        if(key=="--weights") o.weights=value;
        else if(key=="--tokens") o.tokens=value;
        else if(key=="--window") o.window=positive(value);
        else if(key=="--nwin") o.nwin=positive(value);
        else if(key=="--uq") o.uq=positive(value);
        else if(key=="--lengths") o.lengths=value;
        else if(key=="--mask") o.mask=value;
        else if(key=="--save-ref") o.save_ref=value;
        else if(key=="--kld-ref") o.kld_ref=value;
        else if(key=="--golden-dir") o.golden=value;
        else if(key=="--dump-dir") o.dump=value;
        else require(false,"unknown option "+key);
    }
    require(!o.weights.empty()&&!o.tokens.empty(),"--weights and --tokens required");
    require(o.uq==1||o.uq==2,"uq must be 1 or2");
    require(o.window>=2&&o.window<=1024,"window must be 2..1024 including prepared BOS");
    require(o.golden.empty()||(o.window==5&&o.nwin==1),"golden validation requires window5 nwin1");
    require(o.dump.empty()||(o.window==1024&&o.nwin==1),"three-row dump requires window1024 nwin1");
    require(o.golden.empty()||o.dump.empty(),"golden and WikiText dump modes are separate");
    require(o.save_ref.empty()||o.kld_ref.empty(),"save-ref and kld-ref are mutually exclusive");
    require((o.golden.empty()&&o.dump.empty())||(o.lengths.empty()&&o.mask.empty()&&o.save_ref.empty()&&o.kld_ref.empty()),"validation modes require original fixed unmasked input"); return o;
}
static void advise_reference_cache(int fd,const std::string &path,bool written) {
    if(written) {
        const int result=fdatasync(fd);
        require(result==0,"reference fdatasync failed "+path+": "+strerror(errno));
    }
    const int error=posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);
    if(error) fprintf(stderr,"warning: posix_fadvise %s: %s\n",path.c_str(),strerror(error));
}
static void release_reference_cache(const std::string &path,bool written) {
    const int fd=open(path.c_str(),written?O_WRONLY:O_RDONLY);
    require(fd>=0,"cannot open reference for cache advice "+path);
    advise_reference_cache(fd,path,written);
    require(close(fd)==0,"reference cache descriptor close failed "+path);
}
template<class T> static std::vector<T> read_exact(const std::string &path,size_t count,bool release_cache=false) {
    require(fs::is_regular_file(path)&&fs::file_size(path)==mul(count,sizeof(T)),"wrong input size "+path);
    std::vector<T> out(count); std::ifstream f(path,std::ios::binary);
    require(bool(f.read(reinterpret_cast<char*>(out.data()),mul(count,sizeof(T)))),"read failed "+path);
    if(release_cache) release_reference_cache(path,false);
    return out;
}
template<class T> static T *device(size_t count) { T *p=nullptr; CK(cudaMalloc(&p,mul(count,sizeof(T)))); return p; }
static void write_new(const std::string &path,const void *data,size_t bytes,bool release_cache=false) {
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600); require(fd>=0,"output must be new "+path);
    const char *p=static_cast<const char*>(data); size_t remaining=bytes;
    while(remaining) { ssize_t n=write(fd,p,remaining); require(n>0,"output write failed"); p+=n; remaining-=n; }
    if(release_cache) advise_reference_cache(fd,path,true);
    require(close(fd)==0,"output close failed");
}
static int argmax(const float *x,int n) {
    int best=0; for(int j=0;j<n;++j) { require(std::isfinite(x[j]),"nonfinite logit"); if(x[j]>x[best]) best=j; } return best;
}
static double nll(const float *x,int n,int target) {
    require(target>=0&&target<n,"target outside vocabulary");
    const double peak=x[argmax(x,n)]; double total=0;
    for(int j=0;j<n;++j) total+=std::exp(double(x[j])-peak);
    double result=(peak-double(x[target]))+std::log(total);
    require(std::isfinite(result)&&result>=0,"nonfinite or negative NLL"); return result;
}
static double epoch_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
static constexpr int position_begin[]={1,16,64,128,256,512};
static constexpr int position_end[]={15,63,127,255,511,1023};
static int position_bucket(int target_position) {
    require(target_position>=1&&target_position<=1023,"scored target position outside statistics range");
    for(int bucket=0;bucket<6;++bucket) if(target_position<=position_end[bucket]) return bucket;
    require(false,"missing position bucket"); return -1;
}
struct PositionStats {
    double sums[6]={}; size_t counts[6]={};
    void accumulate(int target_position,double value) {
        int bucket=position_bucket(target_position); sums[bucket]+=value; ++counts[bucket];
    }
    void print() const {
        for(int bucket=0;bucket<6;++bucket) {
            fprintf(stderr,"PPL_POSITION_STATS target_begin=%d target_end=%d count=%zu sum_nll=%.17g mean_nll=",
                    position_begin[bucket],position_end[bucket],counts[bucket],sums[bucket]);
            if(counts[bucket]) fprintf(stderr,"%.17g\n",sums[bucket]/counts[bucket]);
            else fputs("null\n",stderr);
        }
    }
};
static void self_test() {
    auto options=[](std::vector<std::string> args) {
        std::vector<char*> argv; for(auto &arg:args) argv.push_back(arg.data());
        return parse(int(argv.size()),argv.data());
    };
    require(options({"ppl","--weights","unused","--tokens","unused"}).uq==2,"default uq2");
    require(!options({"ppl","--weights","unused","--tokens","unused"}).decode,"default prefill execution");
    require(options({"ppl","--weights","unused","--tokens","unused","--decode"}).decode,"explicit valueless decode option");
    for(int uq:{1,2}) require(options({"ppl","--weights","unused","--tokens","unused","--uq",std::to_string(uq)}).uq==uq,"explicit uq option");
    const float uniform[]={0,0}; require(nll(uniform,2,1)==std::log(2.0),"uniform NLL");
    const float extreme[]={1000,-1000}; require(nll(extreme,2,1)==2000,"stable extreme NLL");
    const float shifted[]={1100,-900}; require(nll(shifted,2,1)==nll(extreme,2,1),"translation invariance");
    const float tie[]={2,2,1}; require(argmax(tie,3)==0,"first-index argmax tie");
    for(int bucket=0;bucket<6;++bucket) {
        require(position_bucket(position_begin[bucket])==bucket,"position bucket lower boundary");
        require(position_bucket(position_end[bucket])==bucket,"position bucket upper boundary");
    }
    for(int window:{2,5,64,128,256,512,1024}) {
        PositionStats stats;
        for(int win=0;win<2;++win) for(int target=1;target<window;++target) stats.accumulate(target,0.5);
        size_t count=0;
        for(int bucket=0;bucket<6;++bucket) {
            size_t expected=2*std::max(0,std::min(window-1,position_end[bucket])-position_begin[bucket]+1);
            require(stats.counts[bucket]==expected&&stats.sums[bucket]==0.5*expected,"position statistics accounting");
            count+=stats.counts[bucket];
        }
        require(count==size_t(2*(window-1)),"position statistics total");
    }
    puts("PPL_SELF_TEST PASS uniform/extreme/translation/argmax");
}
static void memory_guard(const Options &o,const ModelConfig &c) {
    size_t weights=0,largest=0;
    for(const auto &entry:fs::directory_iterator(o.weights)) if(entry.is_regular_file()&&entry.path().extension()==".bin") {
        size_t n=entry.file_size(); weights=add(weights,n); largest=std::max(largest,n);
    }
    size_t kv=0,qmax=0,kmax=0;
    for(int l=0;l<c.num_layers;++l) {
        bool full=c.layer_is_full[l]; size_t hd=full&&c.global_head_dim?c.global_head_dim:c.head_dim;
        size_t nk=full&&c.num_global_kv_heads?c.num_global_kv_heads:c.num_kv_heads;
        size_t kd=mul(nk,hd); qmax=std::max(qmax,mul(c.num_q_heads,hd)); kmax=std::max(kmax,kd);
        kv=add(kv,mul(mul(o.window,kd),8));
    }
    size_t columns=add(add(mul(6,c.hidden_size),mul(3,qmax)),add(mul(4,kmax),mul(3,c.intermediate_size)));
    size_t activation=mul(mul(o.window,columns),4);
    size_t quant=mul(mul(o.window,std::max({size_t(c.hidden_size),size_t(c.intermediate_size),qmax})),5)/4;
    size_t logits=mul(mul(o.window,c.vocab_size),4),host=mul(mul(std::min(o.window,128),c.vocab_size),4);
    size_t rope=mul(mul(o.window,add(c.head_dim,c.global_head_dim?c.global_head_dim:c.head_dim)),16);
    size_t reference_host=(!o.save_ref.empty()?mul(mul(128,c.vocab_size),sizeof(__half)):(!o.kld_ref.empty()?mul(c.vocab_size,sizeof(__half)):0));
    size_t total=reference_host; for(size_t part:{weights,largest,kv,activation,quant,logits,host,rope,size_t(2)<<30}) total=add(total,part);
    std::ifstream mem("/proc/meminfo"); std::string line; size_t available=0;
    while(std::getline(mem,line)) if(line.rfind("MemAvailable:",0)==0) { std::istringstream in(line.substr(13)); size_t kb=0; in>>kb; available=mul(kb,1024); }
    require(available>0,"MemAvailable unavailable");
    fprintf(stderr,"PPL_MEMORY weights=%zu largest_host_load=%zu kv=%zu activation=%zu quant=%zu device_logits=%zu host_chunk=%zu rope=%zu reserve=%zu estimated=%zu available=%zu\n",weights,largest,kv,activation,quant,logits,host,rope,size_t(2)<<30,total,available);
    require(total<=available/10*7,"memory estimate exceeds 70% MemAvailable");
}

// Normalize the fp16 reference after reading: its rounding need not preserve sum(p)=1.
static double log_normalizer(const float *x,int n) {
    const double peak=x[argmax(x,n)]; double sum=0;
    for(int j=0;j<n;++j) sum+=std::exp(double(x[j])-peak);
    return peak+std::log(sum);
}
static double reference_kl(const __half *ref,const float *current,int n) {
    double peak=-std::numeric_limits<double>::infinity();
    for(int j=0;j<n;++j) { double x=__half2float(ref[j]); require(std::isfinite(x),"nonfinite fp16 reference"); peak=std::max(peak,x); }
    double sum=0; for(int j=0;j<n;++j) sum+=std::exp(double(__half2float(ref[j]))-peak);
    const double ref_z=peak+std::log(sum),current_z=log_normalizer(current,n);
    double kl=0;
    for(int j=0;j<n;++j) { double lp=double(__half2float(ref[j]))-ref_z; kl+=std::exp(lp)*(lp-(double(current[j])-current_z)); }
    require(std::isfinite(kl)&&kl>=-1e-10,"invalid KL divergence"); return std::max(0.0,kl);
}
static void kl_self_test() {
    const float fixtures[][4]={{-2,0,1,3},{1000,-1000,0,999},{2,2,1,-3}};
    const float current[][4]={{-1,0,2,3},{999,-999,1,1000},{2,1,2,-3}};
    for(int i=0;i<3;++i) {
        __half ref[4]; double z=log_normalizer(fixtures[i],4);
        for(int j=0;j<4;++j) ref[j]=__float2half_rn(float(double(fixtures[i][j])-z));
        std::ostringstream out; out<<std::setprecision(17)<<"{\"case\":"<<i<<",\"reference_logp_fp16\":[";
        for(int j=0;j<4;++j) { if(j) out<<','; out<<__half2float(ref[j]); }
        out<<"],\"current_logits\":[";
        for(int j=0;j<4;++j) { if(j) out<<','; out<<current[i][j]; }
        out<<"],\"kl\":"<<reference_kl(ref,current[i],4)<<"}"; puts(out.str().c_str());
    }
}
struct Reference {
    bool save=false,compare=false; int vocab=0; size_t expected=0,written=0;
    std::string directory;
    std::ofstream out_logp,out_argmax; std::ifstream in_logp,in_argmax;
    std::vector<__half> buffer; std::vector<int32_t> tops;
    std::vector<double> divergences; size_t top_matches=0;
    void start(const Options &o,int V,const std::vector<int> &tokens,const std::vector<int32_t> &lengths,const std::vector<uint8_t> &mask,size_t scored) {
        save=!o.save_ref.empty(); compare=!o.kld_ref.empty(); if(!save&&!compare) return;
        vocab=V; expected=scored; directory=save?o.save_ref:o.kld_ref;
        size_t bytes=add(add(mul(mul(scored,V),sizeof(__half)),mul(scored,sizeof(int32_t))),add(mul(tokens.size(),sizeof(int)),add(mul(lengths.size(),sizeof(int32_t)),mask.size())));
        fprintf(stderr,"PPL_REFERENCE_DISK mode=%s scored_rows=%zu vocab=%d payload_bytes=%zu metadata_reserve=4096\n",save?"save":"compare",scored,V,bytes);
        std::string description="PPL_REF_V1\n"+std::to_string(V)+"\n"+std::to_string(scored)+"\n";
        if(save) {
            fs::path parent=fs::path(directory).parent_path(); if(parent.empty()) parent=".";
            require(!fs::exists(directory)&&fs::is_directory(parent),"reference output must be new with existing parent");
            require(fs::space(parent).available>=add(bytes,4096),"insufficient disk space for reference");
            require(fs::create_directory(directory),"cannot create reference directory");
            write_new(directory+"/tokens.bin",tokens.data(),mul(tokens.size(),sizeof(int)),true);
            write_new(directory+"/lengths.bin",lengths.data(),mul(lengths.size(),sizeof(int32_t)),true);
            write_new(directory+"/mask.bin",mask.data(),mask.size(),true);
            write_new(directory+"/format.txt",description.data(),description.size(),true);
            out_logp.open(directory+"/logp.fp16",std::ios::binary); out_argmax.open(directory+"/argmax.i32",std::ios::binary);
            require(bool(out_logp)&&bool(out_argmax),"cannot open reference outputs");
            buffer.reserve(mul(128,V)); tops.reserve(128);
        } else {
            require(fs::is_regular_file(directory+"/complete"),"reference is incomplete");
            auto format=read_exact<char>(directory+"/format.txt",description.size(),true);
            require(std::string(format.begin(),format.end())==description,"reference format/vocabulary/row count mismatch");
            require(read_exact<int>(directory+"/tokens.bin",tokens.size(),true)==tokens,"reference token mismatch");
            require(read_exact<int32_t>(directory+"/lengths.bin",lengths.size(),true)==lengths,"reference window lengths mismatch");
            require(read_exact<uint8_t>(directory+"/mask.bin",mask.size(),true)==mask,"reference scoring mask mismatch");
            require(fs::file_size(directory+"/logp.fp16")==mul(mul(scored,V),sizeof(__half)),"reference logp size mismatch");
            require(fs::file_size(directory+"/argmax.i32")==mul(scored,sizeof(int32_t)),"reference argmax size mismatch");
            in_logp.open(directory+"/logp.fp16",std::ios::binary); in_argmax.open(directory+"/argmax.i32",std::ios::binary);
            require(bool(in_logp)&&bool(in_argmax),"cannot open reference inputs"); buffer.resize(V); divergences.reserve(scored);
        }
    }
    void flush() {
        if(tops.empty()) return;
        out_logp.write(reinterpret_cast<const char*>(buffer.data()),mul(buffer.size(),sizeof(__half)));
        out_argmax.write(reinterpret_cast<const char*>(tops.data()),mul(tops.size(),sizeof(int32_t)));
        out_logp.flush(); out_argmax.flush();
        require(bool(out_logp)&&bool(out_argmax),"reference write failed");
        release_reference_cache(directory+"/logp.fp16",true);
        release_reference_cache(directory+"/argmax.i32",true);
        buffer.clear(); tops.clear();
    }
    void row(const float *logits,int actual) {
        if(!save&&!compare) return;
        if(save) {
            double z=log_normalizer(logits,vocab);
            for(int j=0;j<vocab;++j) { __half value=__float2half_rn(float(double(logits[j])-z)); require(std::isfinite(__half2float(value)),"log-softmax exceeds finite fp16 range"); buffer.push_back(value); }
            tops.push_back(actual); if(tops.size()==128) flush();
        } else {
            int32_t reference_top=-1;
            require(bool(in_logp.read(reinterpret_cast<char*>(buffer.data()),mul(vocab,sizeof(__half))))&&bool(in_argmax.read(reinterpret_cast<char*>(&reference_top),sizeof(reference_top))),"reference read failed");
            require(reference_top>=0&&reference_top<vocab,"invalid reference argmax");
            divergences.push_back(reference_kl(buffer.data(),logits,vocab)); top_matches+=reference_top==actual;
        }
        ++written;
        if(compare && written%128==0) {
            release_reference_cache(directory+"/logp.fp16",false);
            release_reference_cache(directory+"/argmax.i32",false);
        }
    }
    void finish() {
        if(!save&&!compare) return; require(written==expected,"reference row accounting mismatch");
        if(save) { flush(); out_logp.close(); out_argmax.close(); require(!out_logp.fail()&&!out_argmax.fail(),"reference close failed"); const char done[]="complete\n"; write_new(directory+"/complete",done,sizeof(done)-1,true); }
        if(compare && written%128!=0) {
            release_reference_cache(directory+"/logp.fp16",false);
            release_reference_cache(directory+"/argmax.i32",false);
        }
    }
    void json(std::ostringstream &out) const {
        if(!compare) return;
        double sum=0; for(double value:divergences) sum+=value;
        auto sorted=divergences; std::sort(sorted.begin(),sorted.end()); size_t index=(sorted.size()*99+99)/100-1;
        out<<",\"mean_kl\":"<<sum/divergences.size()<<",\"kl_p99\":"<<sorted[index]<<",\"top1_match_rate\":"<<double(top_matches)/written<<",\"kl_ntok\":"<<written;
    }
};
int main(int argc,char **argv) {
    knobs_init("ppl_gemma4");
    if(argc==2&&strcmp(argv[1],"--self-test")==0) { self_test(); return 0; }
    if(argc==2&&strcmp(argv[1],"--kl-self-test")==0) { kl_self_test(); return 0; }
    Options o=parse(argc,argv); const std::string manifest=o.weights+"/manifest.json";
    require(fs::is_regular_file(manifest),"manifest missing");
    ModelConfig c=load_config_gemma4(manifest.c_str());
    require(c.num_experts==0&&c.num_layers>0&&c.hidden_size>0&&c.vocab_size>1&&c.intermediate_size>0,"dense Gemma manifest required");
    require(c.num_q_heads>0&&c.num_kv_heads>0&&c.head_dim>0&&(c.scale_dtype==0||c.scale_dtype==1),"invalid model dimensions or scale dtype");
    require(c.layer_is_full!=nullptr,"layer type configuration missing");
    const int scale_dtype=c.scale_dtype;
    const char *env=knob_raw("G4_MMQ_PREFILL");
    require(env&&(strcmp(env,"0")==0||strcmp(env,"1")==0),"explicit G4_MMQ_PREFILL=0 or1 required");
    const bool mmq=strcmp(env,"1")==0; require(o.golden.empty()||!mmq,"golden validation requires MMQ OFF");
    const char *tile=knob_raw("MMQ256"); require(tile&&strcmp(tile,"0")==0,"explicit MMQ256=0 required");
    std::vector<int32_t> lengths(o.nwin,o.window);
    if(!o.lengths.empty()) lengths=read_exact<int32_t>(o.lengths,o.nwin);
    size_t input_count=0; int maximum=0;
    for(int length:lengths) { require(length>=2&&length<=o.window,"length must be 2..window maximum"); input_count=add(input_count,length); maximum=std::max(maximum,length); }
    o.window=maximum;
    auto tokens=read_exact<int>(o.tokens,input_count);
    std::vector<uint8_t> mask(input_count,1);
    if(!o.mask.empty()) mask=read_exact<uint8_t>(o.mask,input_count);
    size_t expected_targets=0,offset=0;
    for(int length:lengths) {
        if(o.mask.empty()) mask[offset]=0;
        require(mask[offset]==0,"first token of every window must be unscored"); size_t count=0;
        for(int p=0;p<length;++p) { require(mask[offset+p]<=1,"mask values must be 0 or1"); count+=mask[offset+p]; }
        require(count>0,"every window must have at least one scored target"); expected_targets=add(expected_targets,count); offset+=length;
    }
    Reference reference; reference.start(o,c.vocab_size,tokens,lengths,mask,expected_targets);
    require(sizeof(int)==4,"int32 platform required");
    for(int t:tokens) require(t>=0&&t<c.vocab_size,"token outside manifest vocabulary");
    if(!o.golden.empty()) require(read_exact<int>(o.golden+"/token_ids.bin",5)==tokens,"golden input token mismatch");
    if(!o.dump.empty()) require(!fs::exists(o.dump)&&fs::is_directory(fs::path(o.dump).parent_path()),"dump directory must be new with existing parent");
    memory_guard(o,c);
    const int hf=c.global_head_dim?c.global_head_dim:c.head_dim,hs=c.head_dim,V=c.vocab_size;
    G4Weights w=load_weights_gemma4(o.weights.c_str(),c,1,0); G4DecodeState st=alloc_g4_decode_state(c,o.window);
    int *ids=device<int>(o.window); float *logits=device<float>(mul(o.window,V));
    float *cf=device<float>(mul(o.window,hf)),*sf=device<float>(mul(o.window,hf));
    float *cs=device<float>(mul(o.window,hs)),*ss=device<float>(mul(o.window,hs));
    require(gemma4_gen_rope_tables(manifest.c_str(),o.window,hf,cf,sf,hs,cs,ss)==0,"RoPE generation failed");
    if(!o.golden.empty()) {
        const char *names[]={"rope_full_cos.bin","rope_full_sin.bin","rope_sliding_cos.bin","rope_sliding_sin.bin"};
        float *ptrs[]={cf,sf,cs,ss}; int dims[]={hf,hf,hs,hs};
        for(int i=0;i<4;++i) { auto data=read_exact<float>(o.golden+"/"+names[i],mul(5,dims[i])); for(float x:data) require(std::isfinite(x),"nonfinite golden RoPE"); CK(cudaMemcpy(ptrs[i],data.data(),mul(data.size(),4),cudaMemcpyHostToDevice)); }
        fprintf(stderr,"PPL_VALIDATION golden_rope=exact_existing_reference tokens=exact_existing_reference\n");
    }
    if(!o.dump.empty()) require(fs::create_directory(o.dump),"cannot create new dump directory");
    std::vector<float> host(mul(std::min(o.window,128),V)); std::vector<double> windows;
    std::vector<float> golden;
    if(!o.golden.empty()) golden=read_exact<float>(o.golden+"/q4_logits.bin",mul(5,V));
    double total_nll=0; size_t total_tokens=0; int matches=0; PositionStats position_stats;
    size_t window_offset=0;
    for(int win=0;win<o.nwin;++win) {
        const int length=lengths[win]; const int *input=tokens.data()+window_offset; reset_g4_decode_state(st);
        CK(cudaMemcpy(ids,input,mul(length,4),cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize()); const double interval_start=epoch_seconds();
        if(o.decode) {
            for(int p=0;p<length;++p)
                gemma4_decode(c,w,st,ids+p,1,cf,sf,cs,ss,logits+mul(p,V),1,o.uq,0,0,nullptr);
        } else {
            gemma4_decode(c,w,st,ids,length,cf,sf,cs,ss,logits,1,o.uq,0,1,nullptr);
        }
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); const double interval_end=epoch_seconds();
        fprintf(stderr,"PPL_INTERVAL window=%d start=%.17g end=%.17g scope=%s\n",win,interval_start,interval_end,o.decode?"decode_teacher_forced":"gemma4_decode_prefill");
        require(st.n_past==length,"window reset/commit mismatch");
        double window_sum=0; size_t scored=0;
        for(int begin=0;begin<length;begin+=128) {
            int count=std::min(128,length-begin); CK(cudaMemcpy(host.data(),logits+mul(begin,V),mul(mul(count,V),4),cudaMemcpyDeviceToHost));
            for(int j=0;j<count;++j) {
                int position=begin+j; const float *row=host.data()+mul(j,V); int actual=argmax(row,V);
                if(!golden.empty()) { int expected=argmax(golden.data()+mul(position,V),V); matches+=actual==expected; fprintf(stderr,"PPL_GOLDEN row=%d actual=%d reference=%d match=%d\n",position,actual,expected,actual==expected); }
                if(position+1==length||mask[window_offset+position+1]==0) continue;
                reference.row(row,actual);
                const double value=nll(row,V,input[position+1]); window_sum+=value; ++scored;
                if(o.position_stats) position_stats.accumulate(position+1,value);
                if(!o.dump.empty()&&(position==0||position==511||position==1022)) {
                    const std::string prefix=o.dump+"/row_"+std::to_string(position);
                    write_new(prefix+".bin",row,mul(V,4)); std::ostringstream meta; meta<<std::setprecision(17)<<"{\"row\":"<<position<<",\"target\":"<<input[position+1]<<",\"vocab\":"<<V<<",\"nll\":"<<value<<",\"softcap_already_applied\":true}\n";
                    const std::string text=meta.str(); write_new(prefix+".json",text.data(),text.size());
                }
            }
        }
        size_t expected=0; for(int p=1;p<length;++p) expected+=mask[window_offset+p];
        require(scored==expected&&std::isfinite(window_sum),"window accounting failed"); window_offset+=length;
        windows.push_back(window_sum/scored); total_nll+=window_sum; total_tokens+=scored;
        fprintf(stderr,"PPL_WINDOW index=%d targets=%zu sum_nll=%.17g mean_nll=%.17g\n",win,scored,window_sum,windows.back());
    }
    require(total_tokens==expected_targets,"total scoring accounting failed"); reference.finish();
    require(golden.empty()||matches==5,"existing golden argmax validation failed");
    double mean=total_nll/total_tokens,ppl=std::exp(mean); require(std::isfinite(mean)&&std::isfinite(ppl),"aggregate NLL/PPL overflow");
    if(o.position_stats) position_stats.print();
    free_g4_decode_state(st); free_weights_gemma4(w); free_config_gemma4(c);
    for(float *p:{logits,cf,sf,cs,ss}) CK(cudaFree(p)); CK(cudaFree(ids));
    const std::string model=fs::path(o.weights).lexically_normal().parent_path().filename().string();
    std::ostringstream result; result<<std::setprecision(17)<<"{\"model\":"<<quote(model)<<",\"weights\":"<<quote(o.weights)<<",\"path\":"<<quote(o.decode?(o.uq==1?"Q4_DECODE":"W4A8_DECODE"):(o.uq==1?"Q4_F32":(mmq?"MMQ":"SB")))<<",\"uq\":"<<o.uq<<",\"scale_dtype\":"<<scale_dtype<<",\"nwin\":"<<o.nwin<<",\"ntok\":"<<total_tokens<<",\"mean_nll\":"<<mean<<",\"ppl\":"<<ppl<<",\"per_window_nll\":[";
    for(size_t i=0;i<windows.size();++i) { if(i) result<<','; result<<windows[i]; } result<<"]";
    if(o.decode) result<<",\"execution_mode\":\"teacher_forced_decode\"";
    reference.json(result); result<<"}";
    puts(result.str().c_str()); return 0;
}
