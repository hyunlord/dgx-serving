// R25 isolated grid experiment. R23 input generator and manifest shape provenance;
// R24 grid-only clone method. Synthetic weights, unchanged main quantization.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "runtime/manifest_gemma4.h"
#include "kernels/quantize_x.cuh"
#include "r25_grid_kernel.cuh"

namespace {
constexpr size_t budget=size_t(32)<<30, canary=256;
constexpr std::array<int,12> rows={1,4,6,15,16,17,64,128,129,512,1024,2048};
constexpr const char* names[]={"main","r1","r2g4","r2g8","r2g16"};
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
double epoch() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
int positive(const char* s) {
    char* end=nullptr; long n=std::strtol(s,&end,10);
    if(!*s||*end||n<1||n>std::numeric_limits<int>::max()) throw std::runtime_error("invalid positive integer");
    return int(n);
}
int variant(const char* s) { for(int i=0;i<5;i++) if(!std::strcmp(s,names[i])) return i; throw std::runtime_error("invalid variant"); }
struct Buffer {
    void* p=nullptr;
    explicit Buffer(size_t n) { ck(cudaMalloc(&p,n)); }
    ~Buffer() { if(p) cudaFree(p); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    template<class T> T* as() { return static_cast<T*>(p); }
};
struct Random {
    uint64_t state=20261103;
    uint32_t next() { state^=state<<13; state^=state>>7; state^=state<<17; return uint32_t(state>>16); }
    float scale() { return std::ldexp(float(1+next()%2000)/1009.f,int(next()%7)-6); }
};
template<class Kernel> void resource(int v,Kernel k) {
    cudaFuncAttributes a{}; int blocks=0;
    ck(cudaFuncGetAttributes(&a,k)); ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,256,0));
    printf("{\"event\":\"resources\",\"variant\":\"%s\",\"registers\":%d,\"shared_bytes\":%zu,\"local_bytes\":%zu,\"CTA_per_SM\":%d}\n",names[v],a.numRegs,a.sharedSizeBytes,a.localSizeBytes,blocks);
    if(blocks!=2||a.sharedSizeBytes!=27648||a.maxThreadsPerBlock!=256)
        throw std::runtime_error("registered tile/shared/CTA invariant failed");
}
std::set<std::pair<int,int>> shapes(const char* path) {
    ModelConfig c=load_config_gemma4(path);
    if(c.num_experts||c.scale_dtype!=1) throw std::runtime_error("requires dense fp16-scale manifest");
    int H=c.hidden_size,F=c.intermediate_size,hs=c.head_dim,hf=c.global_head_dim?c.global_head_dim:hs;
    int ks=c.num_kv_heads,kf=c.num_global_kv_heads?c.num_global_kv_heads:ks;
    int qs=c.num_q_heads*hs,qf=c.num_q_heads*hf;
    std::set<std::pair<int,int>> result={{H,qs},{H,qf},{H,ks*hs},{H,kf*hf},{qs,H},{qf,H},{H,F},{F,H},{H,c.vocab_size}};
    std::map<std::pair<int,int>,int> calls;
    for(int layer=0;layer<c.num_layers;layer++) {
        bool full=!c.layer_is_full||c.layer_is_full[layer]; int q=full?qf:qs,kv=full?kf*hf:ks*hs;
        ++calls[{H,q}]; ++calls[{H,kv}]; if(!(full&&c.attention_k_eq_v)) ++calls[{H,kv}];
        ++calls[{q,H}]; calls[{H,F}]+=2; ++calls[{F,H}];
    }
    free_config_gemma4(c);
    const std::map<std::pair<int,int>,int> expected={{{3840,512},8},{{3840,2048},80},{{3840,4096},40},{{3840,8192},8},{{3840,15360},96},{{4096,3840},40},{{8192,3840},8},{{15360,3840},48},{{3840,262144},0}};
    for(auto s:result) {
        if(!expected.count(s)||calls[s]!=expected.at(s)) throw std::runtime_error("registered shape/call matrix differs");
        printf("{\"event\":\"shape\",\"K\":%d,\"N\":%d,\"calls\":%d}\n",s.first,s.second,calls[s]);
    }
    if(result.size()!=9) throw std::runtime_error("expected nine shapes");
    return result;
}
struct Data {
    int M,N,K; size_t mk,nk,mn;
    Buffer x,xs,w,ws,y;
    Data(int m,int n,int k):M(m),N(n),K(k),mk(size_t(m)*k),nk(size_t(n)*k),mn(size_t(m)*n),
        x(mk),xs(mk/8),w(nk/2),ws(nk/16),y(mn*4+canary) {
        Random rng; std::vector<float> input(mk); std::vector<uint8_t> hw(nk/2); std::vector<__half> hws(nk/32);
        for(size_t b=0;b<mk/32;b++) {
            float amplitude=rng.scale();
            for(int j=0;j<32;j++) {
                float value=float(int(rng.next()%65535)-32767)*amplitude/127.f;
                if(b%31==0) value=0; else if(b%31==1) value=j%2?-1e20f:1e20f;
                else if(b%31==2) value=j%2?-1e-20f:1e-20f; else if(b%31==3) value=float(j-16)+.5f;
                input[b*32+j]=value;
            }
        }
        for(auto& a:hw) a=uint8_t(rng.next()); for(auto& a:hws) a=__float2half_rn(rng.scale());
        Buffer fp(mk*4),sum(mk/8);
        ck(cudaMemcpy(fp.p,input.data(),mk*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(w.p,hw.data(),nk/2,cudaMemcpyHostToDevice)); ck(cudaMemcpy(ws.p,hws.data(),nk/16,cudaMemcpyHostToDevice));
        launch_quantize_x_q8(fp.as<float>(),x.as<int8_t>(),xs.as<float>(),sum.as<int32_t>(),K,M);
        ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
        printf("{\"event\":\"case\",\"M\":%d,\"N\":%d,\"K\":%d,\"seed\":20261103,\"weights\":\"synthetic_R23\",\"quantization_timed\":false}\n",M,N,K);
    }
    void launch(int v) {
        dim3 g((N-1)/128+1,(M-1)/128+1);
        if(v==1) g=dim3((M-1)/128+1,(N-1)/128+1);
#define R25_ARGS x.as<int8_t>(),xs.as<float>(),w.as<uint8_t>(),ws.as<__half>(),y.as<float>(),M,N,K
        if(v==0) gemm_mmq_kernel<64,__half><<<g,256>>>(R25_ARGS);
        else if(v==1) r25_grid_kernel<1,1,64,__half><<<g,256>>>(R25_ARGS);
        else if(v==2) r25_grid_kernel<2,4,64,__half><<<g,256>>>(R25_ARGS);
        else if(v==3) r25_grid_kernel<2,8,64,__half><<<g,256>>>(R25_ARGS);
        else if(v==4) r25_grid_kernel<2,16,64,__half><<<g,256>>>(R25_ARGS);
        else throw std::runtime_error("invalid variant");
#undef R25_ARGS
        ck(cudaGetLastError());
    }
    std::vector<float> output(int v) {
        ck(cudaMemset(y.p,0xff,mn*4)); ck(cudaMemset(y.as<uint8_t>()+mn*4,0xa5,canary));
        launch(v); ck(cudaDeviceSynchronize()); std::vector<float> out(mn); std::array<uint8_t,canary> tail{};
        ck(cudaMemcpy(out.data(),y.p,mn*4,cudaMemcpyDeviceToHost)); ck(cudaMemcpy(tail.data(),y.as<uint8_t>()+mn*4,canary,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<mn;i++) if(!std::isfinite(out[i])) throw std::runtime_error("nonfinite/unwritten output at "+std::to_string(i));
        for(auto b:tail) if(b!=0xa5) throw std::runtime_error("output canary changed");
        return out;
    }
    void gate() {
        auto ref=output(0);
        for(int v=0;v<5;v++) {
            for(int r=0;r<30;r++) {
                auto actual=output(v);
                if(std::memcmp(ref.data(),actual.data(),mn*4)) for(size_t i=0;i<mn;i++) if(std::memcmp(&ref[i],&actual[i],4)) {
                    uint32_t a,b; std::memcpy(&a,&ref[i],4); std::memcpy(&b,&actual[i],4);
                    printf("{\"event\":\"first_mismatch\",\"variant\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"repeat\":%d,\"index\":%zu,\"main_bits\":%u,\"actual_bits\":%u}\n",names[v],M,N,K,r,i,a,b);
                    throw std::runtime_error("byte identity failed");
                }
            }
            printf("{\"event\":\"case_pass\",\"variant\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"repeats\":30,\"bytes\":%zu}\n",names[v],M,N,K,mn*4);
        }
    }
    double timed(int v,int count,double& start,double& end) {
        cudaEvent_t a,b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b)); ck(cudaDeviceSynchronize());
        start=epoch(); ck(cudaEventRecord(a)); for(int i=0;i<count;i++) launch(v);
        ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b)); end=epoch(); float ms=0;
        ck(cudaEventElapsedTime(&ms,a,b)); ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b)); return ms;
    }
    double arm(int v,int pair) {
        double start=0,end=0,warm=0,cal=0;
        while(warm<10000) { cal=timed(v,32,start,end); warm+=cal; }
        if(cal<=0) throw std::runtime_error("invalid calibration");
        double wanted=std::ceil(1500.*32/cal);
        if(wanted>std::numeric_limits<int>::max()) throw std::runtime_error("launch count overflow");
        int count=std::max(32,int(wanted)); double ms=timed(v,count,start,end);
        printf("{\"event\":\"interval\",\"phase\":\"measure\",\"variant\":\"%s\",\"pair\":%d,\"M\":%d,\"N\":%d,\"K\":%d,\"warm_ms\":%.6f,\"launches\":%d,\"total_ms\":%.6f,\"mean_ms\":%.9f,\"start_epoch\":%.6f,\"end_epoch\":%.6f}\n",names[v],pair,M,N,K,warm,count,ms,ms/count,start,end);
        if(ms<1000) throw std::runtime_error("measurement shorter than one second"); return ms/count;
    }
    void pair(int v,int pair) {
        double base,candidate;
        if(pair%2) { base=arm(0,pair); candidate=arm(v,pair); }
        else { candidate=arm(v,pair); base=arm(0,pair); }
        printf("{\"event\":\"pair_complete\",\"candidate\":\"%s\",\"pair\":%d,\"M\":%d,\"N\":%d,\"K\":%d,\"main_ms\":%.9f,\"candidate_ms\":%.9f,\"ratio\":%.9f}\n",names[v],pair,M,N,K,base,candidate,candidate/base);
    }
};
void limits(int M,int N,int K) {
    if(K%64||M<1||N<1) throw std::runtime_error("invalid dimensions");
    cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p,0));
    int mt=(M-1)/128+1,nt=(N-1)/128+1;
    if(mt>p.maxGridSize[0]||nt>p.maxGridSize[0]||mt>p.maxGridSize[1]||nt>p.maxGridSize[1]) throw std::runtime_error("grid dimensions exceed device limits");
    size_t mk=size_t(M)*K,nk=size_t(N)*K,mn=size_t(M)*N;
    if(12*mk+2*nk+12*mn+(size_t(4)<<30)>budget) throw std::runtime_error("peak host/device memory exceeds budget");
}
void mapping() {
    size_t configs=0;
    for(uint64_t W=1;W<=65;W++) for(uint64_t T=1;T<=33;T++) for(uint64_t G:{1,2,3,4,6,8,12,16,24,32,64,128}) {
        std::vector<bool> seen(W*T);
        for(uint64_t p=0;p<W*T;p++) {
            uint64_t group=p/(G*T),base=group*G,g=std::min(G,W-base),r=p-group*G*T;
            uint64_t w=base+r%g,t=r/g;
            if(w>=W||t>=T||seen[t*W+w]||group*G*T+t*g+w-base!=p) throw std::runtime_error("mapping not bijective");
            seen[t*W+w]=true;
            if((G==1&&(t!=p%T||w!=p/T))||(G>=W&&(t!=p/W||w!=p%W))) throw std::runtime_error("mapping boundary equivalence failed");
        }
        ++configs;
    }
    size_t actual_configs=0, sampled_configs=0, sampled_points=0, beyond_u32=0;
    auto proof=[](uint64_t W,uint64_t T,uint64_t G,uint64_t p) {
        uint64_t group=p/(G*T),base=group*G,g=std::min(G,W-base),r=p-group*G*T;
        uint64_t w=base+r%g,t=r/g;
        if(w>=W||t>=T||group*G*T+t*g+(w-base)!=p)
            throw std::runtime_error("large mapping inverse/bounds failed");
        if((G==1&&(t!=p%T||w!=p/T))||(G>=W&&(t!=p/W||w!=p%W)))
            throw std::runtime_error("large mapping equivalence failed");
        return t*W+w;
    };
    for(uint64_t W:{30,120,2048}) for(uint64_t T:{1,4,8,16}) for(uint64_t G:{1,4,8,16}) {
        std::vector<bool> seen(W*T);
        for(uint64_t p=0;p<W*T;p++) {
            uint64_t index=proof(W,T,G,p);
            if(seen[index]) throw std::runtime_error("actual shape mapping duplicate");
            seen[index]=true;
        }
        ++actual_configs;
    }
    // Pure arithmetic proofs: these grids are not asserted launchable or allocated.
    for(uint64_t W:{65534,65535,65536,131073})
        for(uint64_t T:{65534,65535,65536}) for(uint64_t G:{1,4,8,16}) {
            uint64_t total=W*T, span=G*T, last_group=(W-1)/G;
            std::set<uint64_t> points;
            auto around=[&](uint64_t p) {
                if(p<total) points.insert(p);
                if(p>0&&p-1<total) points.insert(p-1);
                if(p+1<total) points.insert(p+1);
            };
            around(0); around(total-1); around(uint64_t(1)<<32);
            for(uint64_t group:{uint64_t(0),uint64_t(1),last_group/2,last_group}) {
                uint64_t base=group*G,width=std::min(G,W-base),start=group*span;
                around(start); around(start+width); around(start+width*(T-1));
                around(start+width*T);
            }
            std::set<uint64_t> indices;
            for(uint64_t p:points) {
                if(!indices.insert(proof(W,T,G,p)).second)
                    throw std::runtime_error("sampled mapping duplicate");
                if(p>std::numeric_limits<uint32_t>::max()) ++beyond_u32;
                ++sampled_points;
            }
            ++sampled_configs;
        }
    if(!beyond_u32) throw std::runtime_error("large linear index coverage missing");
    printf("{\"event\":\"mapping_pass\",\"configs\":%zu,\"actual_shape_configs\":%zu,\"sampled_large_configs\":%zu,\"sampled_points\":%zu,\"points_beyond_u32\":%zu,\"grid_y_boundary_values\":[65534,65535,65536],\"device_launch_limits_tested\":false}\n",
           configs,actual_configs,sampled_configs,sampled_points,beyond_u32);
}
}
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        if(argc==2&&std::string(argv[1])=="mapping") { mapping(); return 0; }
        if(argc<3) throw std::runtime_error("usage: r25_grid MANIFEST gate | MANIFEST pair CANDIDATE PAIR M N K | MANIFEST interval VARIANT ROUND M N K | MANIFEST profile VARIANT M N K | mapping");
        std::string mode=argv[2];
        if(!((mode=="gate"&&argc==3)||((mode=="pair"||mode=="interval")&&argc==8)||(mode=="profile"&&argc==7))) throw std::runtime_error("invalid arguments");
        auto registered=shapes(argv[1]); size_t free=0,total=0; ck(cudaMemGetInfo(&free,&total));
        if(double(free)*.8<double(budget)) throw std::runtime_error("NOT_RUN_MEMORY_GUARD");
        resource(0,gemm_mmq_kernel<64,__half>); resource(1,r25_grid_kernel<1,1,64,__half>);
        resource(2,r25_grid_kernel<2,4,64,__half>); resource(3,r25_grid_kernel<2,8,64,__half>); resource(4,r25_grid_kernel<2,16,64,__half>);
        if(mode=="gate") {
            for(auto s:registered) for(int M:rows) { limits(M,s.second,s.first); Data d(M,s.second,s.first); d.gate(); }
            printf("{\"event\":\"suite_pass\",\"cases\":108,\"candidate_cases\":432,\"repeats\":30,\"G1\":\"PASS\",\"G2\":\"PASS\"}\n");
        } else {
            const bool timing=mode=="pair"||mode=="interval";
            int v=variant(argv[3]),pair=timing?positive(argv[4]):0,offset=timing?5:4;
            int M=positive(argv[offset]),N=positive(argv[offset+1]),K=positive(argv[offset+2]);
            if(!registered.count({K,N})||std::find(rows.begin(),rows.end(),M)==rows.end()) throw std::runtime_error("unregistered shape/M");
            if(timing&&((mode=="pair"&&v==0)||pair>5||(M!=128&&M!=512&&M!=1024&&M!=2048))) throw std::runtime_error("unregistered timing case");
            limits(M,N,K); Data d(M,N,K);
            if(mode=="pair") d.pair(v,pair);
            else if(mode=="interval") {
                double mean=d.arm(v,pair);
                printf("{\"event\":\"interval_complete\",\"variant\":\"%s\",\"round\":%d,\"M\":%d,\"N\":%d,\"K\":%d,\"mean_ms\":%.9f}\n",names[v],pair,M,N,K,mean);
            }
            else { d.launch(v); ck(cudaDeviceSynchronize()); printf("{\"event\":\"profile_complete\",\"launches\":1}\n"); }
        }
        return 0;
    } catch(const std::exception& e) { fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
