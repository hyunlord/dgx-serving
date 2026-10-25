// Dense Gemma teacher-forced PPL; existing graph/weights remain unchanged.
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
struct Options { std::string weights,tokens,golden,dump; int window=1024,nwin=32; };
static Options parse(int argc,char **argv) {
    Options o; std::vector<std::string> seen;
    for(int i=1;i<argc;i+=2) {
        require(i+1<argc,"every option requires a value"); std::string key=argv[i],value=argv[i+1];
        require(std::find(seen.begin(),seen.end(),key)==seen.end(),"duplicate option "+key); seen.push_back(key);
        if(key=="--weights") o.weights=value;
        else if(key=="--tokens") o.tokens=value;
        else if(key=="--window") o.window=positive(value);
        else if(key=="--nwin") o.nwin=positive(value);
        else if(key=="--golden-dir") o.golden=value;
        else if(key=="--dump-dir") o.dump=value;
        else require(false,"unknown option "+key);
    }
    require(!o.weights.empty()&&!o.tokens.empty(),"--weights and --tokens required");
    require(o.window>=2&&o.window<=1024,"window must be 2..1024 including prepared BOS");
    require(o.golden.empty()||(o.window==5&&o.nwin==1),"golden validation requires window5 nwin1");
    require(o.dump.empty()||(o.window==1024&&o.nwin==1),"three-row dump requires window1024 nwin1");
    require(o.golden.empty()||o.dump.empty(),"golden and WikiText dump modes are separate"); return o;
}
template<class T> static std::vector<T> read_exact(const std::string &path,size_t count) {
    require(fs::is_regular_file(path)&&fs::file_size(path)==mul(count,sizeof(T)),"wrong input size "+path);
    std::vector<T> out(count); std::ifstream f(path,std::ios::binary);
    require(bool(f.read(reinterpret_cast<char*>(out.data()),mul(count,sizeof(T)))),"read failed "+path); return out;
}
template<class T> static T *device(size_t count) { T *p=nullptr; CK(cudaMalloc(&p,mul(count,sizeof(T)))); return p; }
static void write_new(const std::string &path,const void *data,size_t bytes) {
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600); require(fd>=0,"output must be new "+path);
    const char *p=static_cast<const char*>(data); size_t remaining=bytes;
    while(remaining) { ssize_t n=write(fd,p,remaining); require(n>0,"output write failed"); p+=n; remaining-=n; }
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
static void self_test() {
    const float uniform[]={0,0}; require(nll(uniform,2,1)==std::log(2.0),"uniform NLL");
    const float extreme[]={1000,-1000}; require(nll(extreme,2,1)==2000,"stable extreme NLL");
    const float shifted[]={1100,-900}; require(nll(shifted,2,1)==nll(extreme,2,1),"translation invariance");
    const float tie[]={2,2,1}; require(argmax(tie,3)==0,"first-index argmax tie");
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
    size_t total=0; for(size_t part:{weights,largest,kv,activation,quant,logits,host,rope,size_t(2)<<30}) total=add(total,part);
    std::ifstream mem("/proc/meminfo"); std::string line; size_t available=0;
    while(std::getline(mem,line)) if(line.rfind("MemAvailable:",0)==0) { std::istringstream in(line.substr(13)); size_t kb=0; in>>kb; available=mul(kb,1024); }
    require(available>0,"MemAvailable unavailable");
    fprintf(stderr,"PPL_MEMORY weights=%zu largest_host_load=%zu kv=%zu activation=%zu quant=%zu device_logits=%zu host_chunk=%zu rope=%zu reserve=%zu estimated=%zu available=%zu\n",weights,largest,kv,activation,quant,logits,host,rope,size_t(2)<<30,total,available);
    require(total<=available/10*7,"memory estimate exceeds 70% MemAvailable");
}
int main(int argc,char **argv) {
    if(argc==2&&strcmp(argv[1],"--self-test")==0) { self_test(); return 0; }
    const Options o=parse(argc,argv); const std::string manifest=o.weights+"/manifest.json";
    require(fs::is_regular_file(manifest),"manifest missing");
    ModelConfig c=load_config_gemma4(manifest.c_str());
    require(c.num_experts==0&&c.num_layers>0&&c.hidden_size>0&&c.vocab_size>1&&c.intermediate_size>0,"dense Gemma manifest required");
    require(c.num_q_heads>0&&c.num_kv_heads>0&&c.head_dim>0&&(c.scale_dtype==0||c.scale_dtype==1),"invalid model dimensions or scale dtype");
    require(c.layer_is_full!=nullptr,"layer type configuration missing");
    const int scale_dtype=c.scale_dtype;
    const char *env=getenv("G4_MMQ_PREFILL");
    require(env&&(strcmp(env,"0")==0||strcmp(env,"1")==0),"explicit G4_MMQ_PREFILL=0 or1 required");
    const bool mmq=strcmp(env,"1")==0; require(o.golden.empty()||!mmq,"golden validation requires MMQ OFF");
    const char *tile=getenv("MMQ256"); require(tile&&strcmp(tile,"0")==0,"explicit MMQ256=0 required");
    auto tokens=read_exact<int>(o.tokens,mul(o.window,o.nwin));
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
    double total_nll=0; size_t total_tokens=0; int matches=0;
    for(int win=0;win<o.nwin;++win) {
        const int *input=tokens.data()+mul(win,o.window); reset_g4_decode_state(st);
        CK(cudaMemcpy(ids,input,mul(o.window,4),cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize()); const double interval_start=epoch_seconds();
        gemma4_decode(c,w,st,ids,o.window,cf,sf,cs,ss,logits,1,2,0,1,nullptr);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); const double interval_end=epoch_seconds();
        fprintf(stderr,"PPL_INTERVAL window=%d start=%.17g end=%.17g scope=gemma4_decode_prefill\n",win,interval_start,interval_end);
        require(st.n_past==o.window,"window reset/commit mismatch");
        double window_sum=0; size_t scored=0;
        for(int begin=0;begin<o.window;begin+=128) {
            int count=std::min(128,o.window-begin); CK(cudaMemcpy(host.data(),logits+mul(begin,V),mul(mul(count,V),4),cudaMemcpyDeviceToHost));
            for(int j=0;j<count;++j) {
                int position=begin+j; const float *row=host.data()+mul(j,V); int actual=argmax(row,V);
                if(!golden.empty()) { int expected=argmax(golden.data()+mul(position,V),V); matches+=actual==expected; fprintf(stderr,"PPL_GOLDEN row=%d actual=%d reference=%d match=%d\n",position,actual,expected,actual==expected); }
                if(position+1==o.window) continue;
                const double value=nll(row,V,input[position+1]); window_sum+=value; ++scored;
                if(!o.dump.empty()&&(position==0||position==511||position==1022)) {
                    const std::string prefix=o.dump+"/row_"+std::to_string(position);
                    write_new(prefix+".bin",row,mul(V,4)); std::ostringstream meta; meta<<std::setprecision(17)<<"{\"row\":"<<position<<",\"target\":"<<input[position+1]<<",\"vocab\":"<<V<<",\"nll\":"<<value<<",\"softcap_already_applied\":true}\n";
                    const std::string text=meta.str(); write_new(prefix+".json",text.data(),text.size());
                }
            }
        }
        require(scored==size_t(o.window-1)&&std::isfinite(window_sum),"window accounting failed");
        windows.push_back(window_sum/scored); total_nll+=window_sum; total_tokens+=scored;
        fprintf(stderr,"PPL_WINDOW index=%d targets=%zu sum_nll=%.17g mean_nll=%.17g\n",win,scored,window_sum,windows.back());
    }
    require(golden.empty()||matches==5,"existing golden argmax validation failed");
    double mean=total_nll/total_tokens,ppl=std::exp(mean); require(std::isfinite(mean)&&std::isfinite(ppl),"aggregate NLL/PPL overflow");
    free_g4_decode_state(st); free_weights_gemma4(w); free_config_gemma4(c);
    for(float *p:{logits,cf,sf,cs,ss}) CK(cudaFree(p)); CK(cudaFree(ids));
    const std::string model=fs::path(o.weights).lexically_normal().parent_path().filename().string();
    std::ostringstream result; result<<std::setprecision(17)<<"{\"model\":"<<quote(model)<<",\"weights\":"<<quote(o.weights)<<",\"path\":"<<quote(mmq?"MMQ":"SB")<<",\"scale_dtype\":"<<scale_dtype<<",\"nwin\":"<<o.nwin<<",\"ntok\":"<<total_tokens<<",\"mean_nll\":"<<mean<<",\"ppl\":"<<ppl<<",\"per_window_nll\":[";
    for(size_t i=0;i<windows.size();++i) { if(i) result<<','; result<<windows[i]; } result<<"]}";
    puts(result.str().c_str()); return 0;
}
