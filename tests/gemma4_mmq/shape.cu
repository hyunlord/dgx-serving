// Manual standalone gate: nvcc -arch=sm_120 -O2 -std=c++17 -I. this-file -o shape
#include "runtime/knobs.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../runtime/manifest_gemma4.h"
#include "../../kernels/gemv_w4a8.cuh"
#include "../../kernels/gemm_mmq_tmpl.cuh"

static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static long long epoch() { return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count(); }
struct Device {
    void* p=nullptr;
    explicit Device(size_t bytes) { ck(cudaMalloc(&p,bytes)); }
    ~Device() { if(p) cudaFree(p); }
    Device(const Device&)=delete; Device& operator=(const Device&)=delete;
    template<class T> T* as() { return static_cast<T*>(p); }
};
struct Random {
    uint64_t state=20261025;
    uint32_t next() { state^=state<<13;state^=state>>7;state^=state<<17;return uint32_t(state>>16); }
    float scale() { return float(1+next()%1999)/1000.0f; }
};
static size_t available() {
    FILE* f=fopen("/proc/meminfo","r"); if(!f) throw std::runtime_error("MemAvailable unavailable");
    char line[256]; unsigned long long kb=0; bool found=false;
    while(fgets(line,sizeof(line),f)) if(sscanf(line,"MemAvailable: %llu kB",&kb)==1){found=true;break;}
    fclose(f); if(!found) throw std::runtime_error("MemAvailable missing");return size_t(kb)*1024;
}
static void canary(Device& d,size_t count) {
    unsigned char bytes[256];ck(cudaMemcpy(bytes,d.as<float>()+count,sizeof(bytes),cudaMemcpyDeviceToHost));
    for(unsigned char b:bytes) if(b!=0xa5) throw std::runtime_error("output trailing canary changed");
}
static void run_case(int M,int N,int K,bool small) {
    if(M<=0||N<=0||K<=0||K%32) throw std::runtime_error("invalid shape");
    const size_t mn=size_t(M)*N,mk=size_t(M)*K,nk=size_t(N)*K;
    const size_t weight=nk/2+(nk/32)*4, inputs=mk+(mk/32)*8;
    const size_t denom_count=small?mn:size_t(N);
    const size_t estimated=2*weight+2*inputs+5*mn*4+denom_count*8+mn*8+1024+(size_t(4)<<30);
    const size_t budget=size_t(10)<<30, free=available();
    printf("CASE M=%d N=%d K=%d small=%d seed=20261025 dtype=0 estimated_bytes=%zu budget_bytes=%zu available_bytes=%zu\n",M,N,K,small,estimated,budget,free);
    if(estimated>budget||budget>free/10*7) throw std::runtime_error("memory guard failed");
    Random rng;
    std::vector<int8_t> x(mk); std::vector<float> xs(mk/32),sw(nk/32);
    std::vector<int32_t> sums(mk/32);std::vector<uint8_t> w(nk/2);
    std::vector<double> denominator(denom_count,0);
    for(size_t b=0;b<mk/32;++b){
        xs[b]=small?rng.scale():1.0f;int total=0;
        for(int j=0;j<32;++j){int v=small?int(rng.next()%255)-127:((rng.next()&1)?1:-1);
            if(small&&b%7==0)v=0;x[b*32+j]=int8_t(v);total+=v;}
        sums[b]=total;
    }
    for(int n=0;n<N;++n)for(int b=0;b<K/32;++b){
        const size_t block=size_t(n)*(K/32)+b;sw[block]=rng.scale();
        for(int j=0;j<32;j+=2){
            const unsigned a=rng.next()%16,c=rng.next()%16;
            w[size_t(n)*(K/2)+b*16+j/2]=uint8_t(a|(c<<4));
            if(!small)denominator[n]+=(std::abs(int(a)-8)+std::abs(int(c)-8))*double(sw[block]);
        }
    }
    if(small)for(int m=0;m<M;++m)for(int n=0;n<N;++n)for(int k=0;k<K;++k){
        const unsigned packed=w[size_t(n)*(K/2)+k/2];const int q=int((packed>>((k%2)*4))&15)-8;
        denominator[size_t(m)*N+n]+=std::abs(double(x[size_t(m)*K+k])*double(xs[size_t(m)*(K/32)+k/32])*double(q)*double(sw[size_t(n)*(K/32)+k/32]));
    }
    const size_t fresh=available();
    printf("PRE_GPU_MEMORY M=%d N=%d K=%d available_bytes=%zu budget_bytes=%zu\n",M,N,K,fresh,budget);
    if(budget>fresh/10*7)throw std::runtime_error("fresh GPU memory guard failed");
    Device dx(mk),ds(xs.size()*4),dsum(sums.size()*4),dw(w.size()),dscale(sw.size()*4);
    Device y0(mn*4+256),y1(mn*4+256);
    ck(cudaMemcpy(dx.p,x.data(),mk,cudaMemcpyHostToDevice));ck(cudaMemcpy(ds.p,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dsum.p,sums.data(),sums.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dw.p,w.data(),w.size(),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dscale.p,sw.data(),sw.size()*4,cudaMemcpyHostToDevice));
    std::vector<float> reference[2];reference[0].resize(mn);reference[1].resize(mn);std::vector<float> scratch(mn);
    const long long begin=epoch();printf("GPU_CASE_START M=%d N=%d K=%d small=%d epoch_ns=%lld\n",M,N,K,small,begin);
    for(int kernel=0;kernel<2;++kernel){Device& y=kernel?y1:y0;
        for(int rep=0;rep<30;++rep){
            ck(cudaMemset(y.p,0xff,mn*4));ck(cudaMemset(y.as<float>()+mn,0xa5,256));
            if(kernel)launch_gemm_mmq_w4a8_auto(dx.as<int8_t>(),ds.as<float>(),dw.as<uint8_t>(),dscale.p,y.as<float>(),M,N,K,0);
            else launch_gemv_w4a8(dx.as<int8_t>(),ds.as<float>(),dsum.as<int32_t>(),dw.as<uint8_t>(),dscale.p,y.as<float>(),K,N,M,0);
            ck(cudaGetLastError());ck(cudaDeviceSynchronize());canary(y,mn);
            ck(cudaMemcpy(scratch.data(),y.p,mn*4,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<mn;++i){
                if(!std::isfinite(scratch[i])){printf("NONFINITE kernel=%d rep=%d m=%zu n=%zu\n",kernel,rep,i/N,i%N);throw std::runtime_error("nonfinite output");}
                if(rep&&memcmp(&scratch[i],&reference[kernel][i],4)){
                    uint32_t got,want;memcpy(&got,&scratch[i],4);memcpy(&want,&reference[kernel][i],4);
                    printf("NONDETERMINISTIC kernel=%d rep=%d m=%zu n=%zu expected_bits=%08x actual_bits=%08x\n",kernel,rep,i/N,i%N,want,got);
                    throw std::runtime_error("bitwise determinism failed");}
            }
            if(!rep)reference[kernel]=scratch;
        }
    }
    printf("GPU_CASE_END M=%d N=%d K=%d small=%d epoch_ns=%lld\n",M,N,K,small,epoch());
    std::vector<double> errors(mn);double maxabs=0,squares=0;size_t fails=0,first=mn;
    for(size_t i=0;i<mn;++i){double diff=std::abs(double(reference[1][i])-double(reference[0][i]));
        const double den=denominator[small?i:i%N];errors[i]=diff/std::max(den,1e-30);
        maxabs=std::max(maxabs,diff);squares+=double(reference[0][i])*double(reference[0][i]);
        if(!(errors[i]<1e-5)){++fails;if(first==mn)first=i;}
    }
    if(fails)printf("FIRST_FAILURE m=%zu n=%zu mmq=%.17g gemv=%.17g denominator=%.17g normalized=%.17g\n",first/N,first%N,double(reference[1][first]),double(reference[0][first]),denominator[small?first:first%N],errors[first]);
    std::sort(errors.begin(),errors.end());
    const double median=mn%2?errors[mn/2]:(errors[mn/2-1]+errors[mn/2])/2;
    const size_t p99=(mn*99+99)/100-1;
    printf("RESULT M=%d N=%d K=%d small=%d count=%zu max=%.17g median=%.17g p99=%.17g maxabs=%.17g gemv_rms=%.17g failures=%zu deterministic_runs=30 status=%s\n",M,N,K,small,mn,errors.back(),median,errors[p99],maxabs,std::sqrt(squares/mn),fails,fails?"FAIL":"PASS");
    if(fails)throw std::runtime_error("normalized error gate failed");
}
int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IOLBF,0);
    try{
        if(argc!=3)throw std::runtime_error("usage: shape manifest12.json manifest31.json");
        if(knob_raw("MMQ256")&&strcmp(knob_raw("MMQ256"),"0"))throw std::runtime_error("set MMQ256=0 for canonical MMQ");
        printf("CONFIG MMQ256=0 launcher=launch_gemm_mmq_w4a8_auto threshold_strict=1e-5 dtype=0\n");
        std::set<std::pair<int,int>> shapes;
        for(int a=1;a<argc;++a){ModelConfig c=load_config_gemma4(argv[a]);
            if(c.num_experts||c.scale_dtype!=0)throw std::runtime_error("requires dense fp32-scale manifests");
            const int H=c.hidden_size,F=c.intermediate_size,hs=c.head_dim,hf=c.global_head_dim?c.global_head_dim:hs;
            const int ks=c.num_kv_heads,kf=c.num_global_kv_heads?c.num_global_kv_heads:ks;
            const int qs=c.num_q_heads*hs,qf=c.num_q_heads*hf;
            const std::pair<int,int> derived[]={{H,qs},{H,qf},{H,ks*hs},{H,kf*hf},{qs,H},{qf,H},{H,F},{F,H},{H,c.vocab_size}};
            for(auto shape:derived){shapes.insert(shape);printf("SHAPE_SOURCE manifest=%s K=%d N=%d\n",argv[a],shape.first,shape.second);}
            free_config_gemma4(c);
        }
        for(int M:{37,513})run_case(M,137,96,true);
        for(auto shape:shapes)for(int M:{37,513})run_case(M,shape.second,shape.first,false);
        printf("SUITE PASS unique_shapes=%zu full_cases=%zu small_cases=2\n",shapes.size(),shapes.size()*2);return 0;
    }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
