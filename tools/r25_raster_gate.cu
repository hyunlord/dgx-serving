// R25 production-launcher byte gate. Input generator, manifest registration and
// mapping arithmetic derived from tools/r25_grid.cu (R23/R24 provenance).
// Tests the real raster launcher; synthetic inputs are not engine proof.
// Build: nvcc -arch=sm_120 -O2 -std=c++17 -lineinfo -Xptxas=-v -I. tools/r25_raster_gate.cu -o <artifact>/r25_raster_gate
// Run under the 32GiB guard: r25_raster_gate MANIFEST gate; CPU only: mapping
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
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
#include "kernels/gemm_mmq_raster.cuh"

namespace {
constexpr size_t budget=size_t(32)<<30, canary=256;
constexpr std::array<int,12> rows={1,4,6,15,16,17,64,128,129,512,1024,2048};
constexpr const char* names[]={"main","r1","r2g4","r2g8","r2g16"};
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
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
template<class Kernel> void resource(int v,int dtype,Kernel k) {
    cudaFuncAttributes a{}; int blocks=0;
    ck(cudaFuncGetAttributes(&a,k)); ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,256,0));
    printf("{\"event\":\"resources\",\"variant\":\"%s\",\"scale_dtype\":%d,\"registers\":%d,\"shared_bytes\":%zu,\"local_bytes\":%zu,\"CTA_per_SM\":%d}\n",names[v],dtype,a.numRegs,a.sharedSizeBytes,a.localSizeBytes,blocks);
    if(a.numRegs!=128||blocks!=2||a.sharedSizeBytes!=27648||a.maxThreadsPerBlock!=256)
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
    int M,N,K,dtype; size_t mk,nk,mn;
    Buffer x,xs,w,ws,y;
    Data(int m,int n,int k,int sd):M(m),N(n),K(k),dtype(sd),mk(size_t(m)*k),nk(size_t(n)*k),mn(size_t(m)*n),
        x(mk),xs(mk/8),w(nk/2),ws(nk/32*(sd==1?sizeof(__half):sizeof(float))),y(mn*4+canary) {
        Random rng; std::vector<float> input(mk); std::vector<uint8_t> hw(nk/2); std::vector<__half> hws(nk/32); std::vector<float> fws(nk/32);
        for(size_t b=0;b<mk/32;b++) {
            float amplitude=rng.scale();
            for(int j=0;j<32;j++) {
                float value=float(int(rng.next()%65535)-32767)*amplitude/127.f;
                if(b%31==0) value=0; else if(b%31==1) value=j%2?-1e20f:1e20f;
                else if(b%31==2) value=j%2?-1e-20f:1e-20f; else if(b%31==3) value=float(j-16)+.5f;
                input[b*32+j]=value;
            }
        }
        for(auto& a:hw) a=uint8_t(rng.next());
        for(size_t i=0;i<hws.size();i++) { float value=rng.scale(); hws[i]=__float2half_rn(value); fws[i]=value; }
        Buffer fp(mk*4),sum(mk/8);
        ck(cudaMemcpy(fp.p,input.data(),mk*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(w.p,hw.data(),nk/2,cudaMemcpyHostToDevice)); if(dtype==1) ck(cudaMemcpy(ws.p,hws.data(),nk/16,cudaMemcpyHostToDevice));
        else ck(cudaMemcpy(ws.p,fws.data(),nk/8,cudaMemcpyHostToDevice));
        launch_quantize_x_q8(fp.as<float>(),x.as<int8_t>(),xs.as<float>(),sum.as<int32_t>(),K,M);
        ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
        printf("{\"event\":\"case\",\"scale_dtype\":%d,\"M\":%d,\"N\":%d,\"K\":%d,\"seed\":20261103,\"weights\":\"synthetic_R23\",\"quantization_timed\":false}\n",dtype,M,N,K);
    }
    void launch(int v) {
        if(v==0) launch_gemm_mmq_w4a8(x.as<int8_t>(),xs.as<float>(),w.as<uint8_t>(),ws.p,y.as<float>(),M,N,K,dtype);
        else {
            if(v<1||v>4) throw std::runtime_error("invalid variant");
            const int raster=v==1?1:2, group=v==2?4:v==3?8:16;
            launch_gemm_mmq_raster_w4a8(x.as<int8_t>(),xs.as<float>(),w.as<uint8_t>(),ws.p,y.as<float>(),M,N,K,dtype,raster,group);
        }
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
    void gate(bool wide_fallback=false) {
        auto ref=output(0);
        for(int v=0;v<(wide_fallback?2:5);v++) {
            for(int r=0;r<30;r++) {
                auto actual=output(v);
                if(std::memcmp(ref.data(),actual.data(),mn*4)) for(size_t i=0;i<mn;i++) if(std::memcmp(&ref[i],&actual[i],4)) {
                    uint32_t a,b; std::memcpy(&a,&ref[i],4); std::memcpy(&b,&actual[i],4);
                    printf("{\"event\":\"first_mismatch\",\"scale_dtype\":%d,\"variant\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"repeat\":%d,\"index\":%zu,\"main_bits\":%u,\"actual_bits\":%u}\n",dtype,names[v],M,N,K,r,i,a,b);
                    throw std::runtime_error("byte identity failed");
                }
            }
            printf("{\"event\":\"case_pass\",\"scale_dtype\":%d,\"wide_fallback\":%s,\"variant\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"repeats\":30,\"bytes\":%zu}\n",dtype,wide_fallback?"true":"false",names[v],M,N,K,mn*4);
        }
    }
};
void limits(int M,int N,int K,bool wide_fallback=false) {
    if(K%64||M<1||N<1) throw std::runtime_error("invalid dimensions");
    cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p,0));
    int mt=(M-1)/128+1,nt=(N-1)/128+1;
    if(mt>p.maxGridSize[0]||nt>p.maxGridSize[0]||mt>p.maxGridSize[1]||(!wide_fallback&&nt>p.maxGridSize[1])) throw std::runtime_error("grid dimensions exceed device limits");
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
template<class ST> void resources(int dtype) {
    resource(0,dtype,gemm_mmq_kernel<MMQ_KT,ST>);
    resource(1,dtype,gemm_mmq_raster_kernel<1,1,MMQ_KT,ST>);
    resource(2,dtype,gemm_mmq_raster_kernel<2,4,MMQ_KT,ST>);
    resource(3,dtype,gemm_mmq_raster_kernel<2,8,MMQ_KT,ST>);
    resource(4,dtype,gemm_mmq_raster_kernel<2,16,MMQ_KT,ST>);
}
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        if(argc==2&&std::string(argv[1])=="mapping") { mapping(); return 0; }
        if(argc!=3||std::string(argv[2])!="gate") throw std::runtime_error("usage: r25_raster_gate MANIFEST gate | mapping");
        auto registered=shapes(argv[1]); size_t free=0,total=0; ck(cudaMemGetInfo(&free,&total));
        if(double(free)*.8<double(budget)) throw std::runtime_error("NOT_RUN_MEMORY_GUARD");
        resources<__half>(1); resources<float>(0);
        for(int dtype:{1,0}) {
            for(auto shape:registered) for(int M:rows) {
                limits(M,shape.second,shape.first); Data data(M,shape.second,shape.first,dtype); data.gate();
            }
            // 65536 weight tiles exceeds R1 grid.y; the real launcher must use main.
            // This deliberately ragged N also checks the final weight-tile boundary.
            constexpr int wide_n=65535*128+1;
            limits(1,wide_n,64,true); Data wide(1,wide_n,64,dtype); wide.gate(true);
        }
        printf("{\"event\":\"production_suite_pass\",\"shape_M_dtype_cases\":216,\"candidate_cases\":864,\"variant_cases\":1080,\"wide_fallback_cases\":2,\"wide_fallback_variant_cases\":4,\"repeats\":30,\"scale_dtypes\":[1,0],\"G1\":\"PASS\",\"G2\":\"PASS\"}\n");
        return 0;
    } catch(const std::exception& e) { fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
