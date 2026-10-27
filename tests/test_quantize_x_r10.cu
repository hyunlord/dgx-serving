// TMP-only D4 correctness draft; no timing and no engine wiring.
// Draft (NOT executed): nvcc -arch=sm_120 -O2 -std=c++17 -lineinfo -I. tests/test_quantize_x_r10.cu -o artifacts/r10/test_quantize_x_D4
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/quantize_x_r10.cuh"
#include "../kernels/gemv_w4a8.cuh"

static void ck(cudaError_t e) { if (e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static uint32_t mix(uint32_t n) { n^=n>>16; n*=0x7feb352du; n^=n>>15; n*=0x846ca68bu; return n^(n>>16); }
template<class T> struct Buffer {
    static constexpr size_t guard=16;
    T* allocation=nullptr;
    size_t n;
    explicit Buffer(size_t count):n(count) { ck(cudaMalloc(&allocation,(n+2*guard)*sizeof(T))); reset(); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer() { if (allocation) cudaFree(allocation); }
    T* data() { return allocation+guard; }
    void reset() { ck(cudaMemset(allocation,0xa5,(n+2*guard)*sizeof(T))); }
    void upload(const std::vector<T>& v) {
        if (v.size()!=n) throw std::runtime_error("upload shape mismatch");
        ck(cudaMemcpy(data(),v.data(),n*sizeof(T),cudaMemcpyHostToDevice));
    }
    std::vector<T> read(const char* label,bool untouched=false) {
        std::vector<unsigned char> bytes((n+2*guard)*sizeof(T));
        ck(cudaMemcpy(bytes.data(),allocation,bytes.size(),cudaMemcpyDeviceToHost));
        size_t lo=guard*sizeof(T),hi=lo+n*sizeof(T);
        for (size_t i=0;i<bytes.size();i++) if ((untouched || i<lo || i>=hi) && bytes[i]!=0xa5)
            throw std::runtime_error(std::string("canary damaged: ")+label+" byte="+std::to_string(i));
        std::vector<T> payload(n); std::memcpy(payload.data(),bytes.data()+lo,n*sizeof(T)); return payload;
    }
};
template<class T> static void equal(const std::vector<T>& a,const std::vector<T>& b,const char* label) {
    if (a.size()!=b.size()) throw std::runtime_error("comparison shape mismatch");
    const auto* pa=reinterpret_cast<const unsigned char*>(a.data());
    const auto* pb=reinterpret_cast<const unsigned char*>(b.data());
    for (size_t i=0;i<a.size()*sizeof(T);i++) if (pa[i]!=pb[i])
        throw std::runtime_error(std::string("byte mismatch: ")+label+" byte="+std::to_string(i));
}
static std::vector<float> activation(int in,const std::string& name) {
    std::vector<float> x(in);
    for (int k=0;k<in;k++) {
        int lane=k%32,block=k/32;
        if (name=="zero") x[k]=0.f;
        else if (name=="signed_zero") x[k]=(k&1)?-0.f:0.f;
        else if (name=="ties") {
            const float values[]={0.f,-0.f,0.5f,-0.5f,1.5f,-1.5f,2.5f,-2.5f,126.5f,-126.5f};
            x[k]=values[lane%10];
            if (lane==block%32) x[k]=(block&1)?-127.f:127.f;
        } else if (name=="sparse") x[k]=lane==block%32?((block&1)?-3.25f:19.f):0.f;
        else if (name=="finite") x[k]=((int)(mix(k+91)%200001)-100000)*0.001f;
        else if (name=="large_finite") x[k]=((k&1)?-1.f:1.f)*(std::numeric_limits<float>::max()/4.f)*(1.f-(lane%8)*0.125f);
        else if (name=="tiny_normal") x[k]=((k&1)?-1.f:1.f)*std::numeric_limits<float>::min()*(128.f+lane);
        else throw std::runtime_error("unknown case");
        if (!std::isfinite(x[k])) throw std::runtime_error("nonfinite activation forbidden");
    }
    return x;
}
static void run(int in,const std::string& name) {
    if (QBLK!=32 || in<=0 || in%32) throw std::runtime_error("positive QBLK32-aligned IN required");
    constexpr int out=17; // Exercises a partial final GEMV block.
    int blocks=in/32;
    auto hx=activation(in,name);
    Buffer<float> x(in),old_s(blocks),new_s(blocks),old_y(out),new_y(out);
    Buffer<int8_t> old_q(in),new_q(in);
    Buffer<int32_t> old_sum(blocks),sum_canary(blocks);
    x.upload(hx);
    size_t nq=(size_t)out*in/2,ns=(size_t)out*blocks;
    std::vector<uint8_t> hq(nq); std::vector<float> hs(ns); std::vector<__half> hh(ns);
    for (size_t j=0;j<nq;j++) hq[j]=(uint8_t)mix((uint32_t)j+101);
    // Small, nonzero representable fp16 scales keep the large-finite consumer finite.
    for (size_t j=0;j<ns;j++) { hs[j]=(1+mix((uint32_t)j+719)%4)*0.000001f; hh[j]=__float2half(hs[j]); }
    Buffer<uint8_t> wq(nq); Buffer<float> sf(ns); Buffer<__half> sh(ns);
    wq.upload(hq); sf.upload(hs); sh.upload(hh);
    std::vector<int8_t> first_q;
    std::vector<float> first_s,first_y[2];
    for (int r=0;r<30;r++) {
        old_q.reset(); new_q.reset(); old_s.reset(); new_s.reset(); old_sum.reset(); sum_canary.reset();
        launch_quantize_x_q8(x.data(),old_q.data(),old_s.data(),old_sum.data(),in,1);
        launch_quantize_x_q8_r10_d4(x.data(),new_q.data(),new_s.data(),in,1);
        ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
        auto oq=old_q.read("old xq"),nqv=new_q.read("D4 xq");
        auto os=old_s.read("old xs"),nsv=new_s.read("D4 xs");
        auto sums=old_sum.read("old xsum"); sum_canary.read("D4 unused sum",true);
        equal(oq,nqv,"xq old vs D4"); equal(os,nsv,"xs old vs D4");
        for (int b=0;b<blocks;b++) {
            if (!std::isfinite(os[b])) throw std::runtime_error("nonfinite quantizer scale");
            int expected=0; for (int j=0;j<32;j++) expected+=oq[b*32+j];
            if (sums[b]!=expected) throw std::runtime_error("baseline xsum not populated correctly");
        }
        if (r==0) { first_q=oq; first_s=os; }
        else { equal(first_q,oq,"30-run xq"); equal(first_s,os,"30-run xs"); }
        for (int dtype=0;dtype<2;dtype++) {
            old_y.reset(); new_y.reset();
            const void* scales=dtype?static_cast<void*>(sh.data()):static_cast<void*>(sf.data());
            launch_gemv_w4a8(old_q.data(),old_s.data(),old_sum.data(),wq.data(),scales,old_y.data(),in,out,1,dtype);
            launch_gemv_w4a8(new_q.data(),new_s.data(),sum_canary.data(),wq.data(),scales,new_y.data(),in,out,1,dtype);
            ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
            auto oy=old_y.read("old GEMV"),ny=new_y.read("D4 GEMV");
            for (float v:oy) if (!std::isfinite(v)) throw std::runtime_error("nonfinite consumer output");
            equal(oy,ny,"GEMV old vs D4");
            if (r==0) first_y[dtype]=oy; else equal(first_y[dtype],oy,"30-run GEMV");
            sum_canary.read("unused sum after GEMV",true);
        }
    }
    equal(hx,x.read("activation input"),"activation unchanged");
    equal(hq,wq.read("packed weights"),"weights unchanged");
    equal(hs,sf.read("fp32 scales"),"fp32 scales unchanged");
    equal(hh,sh.read("fp16 scales"),"fp16 scales unchanged");
    printf("CORRECT variant=d4 in=%d S=1 case=%s repeats=30 xq_exact=PASS xs_exact=PASS baseline_sum=PASS sum_canary=PASS guards=PASS consumer_fp32=PASS consumer_fp16=PASS determinism=PASS\n",in,name.c_str());
}
int main() try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    printf("CONFIG variant=d4 mode=correctness_only cases=7 shapes=8 repeats=30 compiler_policy=O2_no_fast_math subnormal_input=excluded tiny_normal_d=normal engine_fp16_eligible=no\n");
    for (int in:{32,64,256,288,3840,4096,8192,15360})
        for (const char* name:{"zero","signed_zero","ties","sparse","finite","large_finite","tiny_normal"}) run(in,name);
    printf("CORRECTNESS_COMPLETE cases=56; no timing or engine acceptance evaluated\n");
    return 0;
} catch (const std::exception& e) {
    fprintf(stderr,"CORRECTNESS_FAILED: %s\n",e.what()); return 1;
}
