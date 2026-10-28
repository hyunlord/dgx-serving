// Manual R11/R14 gates: manifest shapes, cross-S bit identity for both families (D1, and the existing
// plain GEMV S=1 vs SB S=2..16 family) and CPU-normalized S1 error of D1 against plain GEMV.
// Usage: test_gemv_d1_unified <dense Gemma manifest.json>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../runtime/manifest_gemma4.h"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemv_w4a8_d1.cuh"

static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p,n*sizeof(T))); }
    ~Device() { if(p) cudaFree(p); }
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
};
static uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
struct Random {
    uint64_t state;
    uint32_t next() {
        state^=state>>12; state^=state<<25; state^=state>>27;
        return uint32_t((state*UINT64_C(2685821657736338717))>>32);
    }
};
struct Shape { std::string label; int in,out; };
static std::vector<Shape> shapes(const ModelConfig& c) {
    std::vector<Shape> v;
    auto add=[&](const char* label,int in,int out) {
        if(in<=0 || out<=0 || in%32) throw std::runtime_error("unaligned or invalid manifest shape");
        for(auto& s:v) if(s.in==in && s.out==out) { s.label+=std::string("+")+label; return; }
        v.push_back({label,in,out});
    };
    const int hf=c.global_head_dim?c.global_head_dim:c.head_dim;
    const int kf=c.num_global_kv_heads?c.num_global_kv_heads:c.num_kv_heads;
    add("q_sliding",c.hidden_size,c.num_q_heads*c.head_dim);
    add("q_full",c.hidden_size,c.num_q_heads*hf);
    add("k_v_sliding",c.hidden_size,c.num_kv_heads*c.head_dim);
    add(c.attention_k_eq_v?"k_full_v_alias":"k_v_full",c.hidden_size,kf*hf);
    add("o_sliding",c.num_q_heads*c.head_dim,c.hidden_size);
    add("o_full",c.num_q_heads*hf,c.hidden_size);
    add("gate_up",c.hidden_size,c.intermediate_size);
    add("down",c.intermediate_size,c.hidden_size);
    add("lm_head",c.hidden_size,c.vocab_size);
    return v;
}
static void run(const Shape& sh,int dtype) {
    const int in=sh.in,out=sh.out,nb=in/QBLK;
    const size_t nw=(size_t)out*in/2,ns=(size_t)out*nb,nx=(size_t)16*in,ny=(size_t)16*out;
    const size_t gpu_bytes=nw+ns*(dtype?2:4)+nx+(size_t)16*nb*4+ny*4*2+(size_t)out*4;
    if(gpu_bytes>size_t(2)*1024*1024*1024) throw std::runtime_error("isolated GPU buffer budget exceeds 2 GiB");
    printf("D1_MEMORY shape=%s in=%d out=%d dtype=%d gpu_bytes=%zu\n",sh.label.c_str(),in,out,dtype,gpu_bytes);
    Random rng{UINT64_C(0x726f773131)+(uint64_t)in*1000003u+(uint64_t)out};
    std::vector<uint8_t> w(nw); for(auto& x:w) x=(uint8_t)rng.next();
    std::vector<float> sw(ns); std::vector<__half> hs(dtype?ns:0);
    for(size_t i=0;i<ns;i++) {
        sw[i]=(i<size_t(nb) || i%97==0)?0.f:((int)(rng.next()%2001)-1000)*0.0000317f;
        if(dtype) { hs[i]=__float2half_rn(sw[i]); sw[i]=__half2float(hs[i]); }
    }
    std::vector<int8_t> x(nx); for(auto& q:x) q=(int8_t)((int)(rng.next()%255)-127);
    std::vector<float> xs((size_t)16*nb);
    for(size_t i=0;i<xs.size();i++) xs[i]=(i%43==0)?0.f:((int)(rng.next()%1001)-500)*0.000127f;
    Device<uint8_t> dw(nw),ds(ns*(dtype?2:4)); Device<int8_t> dx(nx);
    Device<float> dxs(xs.size()),single(ny),multi(ny),old(out);
    ck(cudaMemcpy(dw.p,w.data(),nw,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(ds.p,dtype?(const void*)hs.data():(const void*)sw.data(),ns*(dtype?2:4),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dx.p,x.data(),nx,cudaMemcpyHostToDevice)); ck(cudaMemcpy(dxs.p,xs.data(),xs.size()*4,cudaMemcpyHostToDevice));
    for(int s=0;s<16;s++) launch_gemv_w4a8_d1(dx.p+(size_t)s*in,dxs.p+(size_t)s*nb,dw.p,ds.p,single.p+(size_t)s*out,in,out,1,dtype);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    std::vector<float> ref(ny),got(ny),legacy(out);
    ck(cudaMemcpy(ref.data(),single.p,ny*4,cudaMemcpyDeviceToHost));
    for(int s=2;s<=16;s++) {
        ck(cudaMemset(multi.p,0xff,ny*4));
        launch_gemv_w4a8_d1(dx.p,dxs.p,dw.p,ds.p,multi.p,in,out,s,dtype);
        ck(cudaGetLastError()); ck(cudaMemcpy(got.data(),multi.p,(size_t)s*out*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<(size_t)s*out;i++) if(!std::isfinite(got[i]) || bits(got[i])!=bits(ref[i])) {
            fprintf(stderr,"D1_BIT_FAIL shape=%s in=%d out=%d dtype=%d S=%d column=%zu row=%zu expected=%.9g actual=%.9g expected_bits=%08x actual_bits=%08x\n",
                    sh.label.c_str(),in,out,dtype,s,i/out,i%out,ref[i],got[i],bits(ref[i]),bits(got[i]));
            throw std::runtime_error("cross-S bit identity failed");
        }
        printf("D1_BIT_PASS shape=%s dtype=%d S=%d compared=%zu\n",sh.label.c_str(),dtype,s,(size_t)s*out);
    }
    // Existing family: SB columns S=2..16 must equal plain GEMV S=1 of the same column.
    for(int s=0;s<16;s++) launch_gemv_w4a8(dx.p+(size_t)s*in,dxs.p+(size_t)s*nb,nullptr,dw.p,ds.p,single.p+(size_t)s*out,in,out,1,dtype);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    std::vector<float> plain(ny); ck(cudaMemcpy(plain.data(),single.p,ny*4,cudaMemcpyDeviceToHost));
    for(int s=2;s<=16;s++) {
        ck(cudaMemset(multi.p,0xff,ny*4));
        launch_gemv_w4a8_sb(dx.p,dxs.p,dw.p,ds.p,multi.p,in,out,s,dtype);
        ck(cudaGetLastError()); ck(cudaMemcpy(got.data(),multi.p,(size_t)s*out*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<(size_t)s*out;i++) if(!std::isfinite(got[i]) || bits(got[i])!=bits(plain[i])) {
            fprintf(stderr,"EXISTING_BIT_FAIL shape=%s in=%d out=%d dtype=%d S=%d column=%zu row=%zu expected=%.9g actual=%.9g\n",
                    sh.label.c_str(),in,out,dtype,s,i/out,i%out,plain[i],got[i]);
            throw std::runtime_error("existing-family cross-S bit identity failed");
        }
        printf("EXISTING_BIT_PASS shape=%s dtype=%d S=%d compared=%zu\n",sh.label.c_str(),dtype,s,(size_t)s*out);
    }
    launch_gemv_w4a8(dx.p,dxs.p,nullptr,dw.p,ds.p,old.p,in,out,1,dtype);
    ck(cudaGetLastError()); ck(cudaMemcpy(legacy.data(),old.p,(size_t)out*4,cudaMemcpyDeviceToHost));
    double worst=0,maxabs=0; int worst_row=0;
    for(int r=0;r<out;r++) {
        double denom=0;
        for(int b=0;b<nb;b++) {
            int absdot=0;
            for(int k=b*QBLK;k<(b+1)*QBLK;k++) {
                unsigned packed=w[(size_t)r*in/2+k/2];
                int q=int((packed>>((k&1)*4))&15)-8;
                absdot+=std::abs(q*int(x[k]));
            }
            denom+=double(absdot)*std::abs(double(xs[b])*double(sw[(size_t)r*nb+b]));
        }
        double diff=std::abs(double(ref[r])-double(legacy[r]));
        double e=denom>0?diff/denom:(diff==0?0:INFINITY);
        if(!std::isfinite(ref[r]) || !std::isfinite(legacy[r]) || !std::isfinite(e) || !(e<1e-5)) {
            fprintf(stderr,"D1_ERROR_FAIL shape=%s dtype=%d S=1 column=0 row=%d new=%.9g old=%.9g abs=%.17g denom=%.17g normalized=%.17g\n",sh.label.c_str(),dtype,r,ref[r],legacy[r],diff,denom,e);
            throw std::runtime_error("normalized error gate failed");
        }
        if(e>worst) { worst=e; worst_row=r; } maxabs=std::max(maxabs,diff);
    }
    printf("D1_ERROR_PASS shape=%s dtype=%d S=1 column=0 outputs=%d max_normalized=%.17g worst_row=%d max_abs=%.17g denominator=CPU_double_sumabs\n",sh.label.c_str(),dtype,out,worst,worst_row,maxabs);
}
int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=2) throw std::runtime_error("usage: test_gemv_d1_unified <dense Gemma manifest.json>");
    ModelConfig c=load_config_gemma4(argv[1]);
    if(c.num_experts || c.hidden_size!=3840 || c.num_layers!=48) throw std::runtime_error("R11 gate requires actual dense Gemma4-12B manifest");
    auto ss=shapes(c);
    printf("D1_GATE seed=0x726f773131 shapes=%zu columns=16 S=2..16 normalized_columns=0 dtypes=0,1\n",ss.size());
    for(const auto& sh:ss) for(int dtype:{0,1}) run(sh,dtype);
    free_config_gemma4(c);
    printf("[test_gemv_d1_unified] PASS\n"); return 0;
} catch(const std::exception& e) { fprintf(stderr,"[test_gemv_d1_unified] FAIL: %s\n",e.what()); return 1; }
