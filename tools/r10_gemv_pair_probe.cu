// Isolated D2 pair probe; no model dispatch or quantization changes.
// Frozen mix, warm/calibration/event/pair loops and CLI are duplicated from tools/r10_gemv_probe.cu.
// xsum is unused by both GEMVs: nullptr in both arms; no quantizer is run.
// Build: nvcc -arch=sm_120 -O2 -std=c++17 -lineinfo -I. tools/r10_gemv_pair_probe.cu -o artifacts/r10/r10_gemv_probe_D2
// Usage: probe [gate_up|all] [fp32|fp16|both] [bench|ncu-baseline|ncu-d2]
// NCU brackets exactly old gate+up launches or one D2 launch, after warmup.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "../kernels/gemv_w4a8_r10_d2.cuh"
#include "r10_gemv_pair_oracle.hpp"

struct Shape { const char* name; int in,out; };
static const Shape shapes[]={{"gate_up",3840,15360}};
static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> static T* allocate(size_t n) { T* p; ck(cudaMalloc(&p,n*sizeof(T))); return p; }
template<class T> static void upload(T* p,const std::vector<T>& v) {
    ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
}
static uint32_t mix(uint32_t n) {
    n^=n>>16; n*=0x7feb352du; n^=n>>15; n*=0x846ca68bu; return n^(n>>16);
}
static long long epoch_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
struct Weights { uint8_t* q; void* s; };
struct PairWeights { Weights branch[2]; };

static void run(const Shape& shape,int dtype,const std::string& mode) {
    const int in=shape.in,out=shape.out,blocks=in/QBLK;
    if (QBLK!=32 || in<=0 || in%32 || out<=0) throw std::runtime_error("positive QBLK32 shape required");
    const size_t nq=(size_t)in*out/2,ns=(size_t)blocks*out;
    const size_t weight_bytes=2*(nq+ns*(dtype?2:4));
    const size_t target=96ull*1024*1024,cap=2000000000ull;
    const size_t count=std::max<size_t>(2,(target+weight_bytes-1)/weight_bytes);
    const size_t activation_bytes=(size_t)in+blocks*4ull,output_bytes=2ull*out*4;
    const size_t device_bytes=count*weight_bytes+activation_bytes+output_bytes;
    if (device_bytes>cap) throw std::runtime_error("all explicit device buffers exceed 2GB");
    // Logical unique bytes per PAIR, not a measurement of cache/DRAM traffic.
    const size_t logical_bytes=weight_bytes+activation_bytes+output_bytes;
    printf("CONFIG shape=%s in=%d out=%d dtype=%d QBLK=32 variant=d2 buffers=%zu weight_bytes=%zu rotating_weight_bytes=%zu device_bytes=%zu logical_bytes_per_launch=%zu rows_per_block=8 warps_per_row=1 chunk=8 unit=gate_up_pair baseline_kernels=2 candidate_kernels=1 quantization=excluded geglu=excluded\n",
           shape.name,in,out,dtype,count,weight_bytes,count*weight_bytes,device_bytes,logical_bytes);
    std::vector<int8_t> hx(in); std::vector<float> hxs(blocks);
    std::vector<uint8_t> hq[2]; std::vector<float> hs[2]; std::vector<__half> hh[2];
    for (int i=0;i<in;i++) hx[i]=(int)(mix((uint32_t)i+19)%255)-127;
    for (int b=0;b<blocks;b++) hxs[b]=(1+(mix(b+83)%31))*0.001f;
    for (int branch=0;branch<2;branch++) {
        hq[branch].resize(nq); hs[branch].resize(ns);
        if (dtype) hh[branch].resize(ns);
        for (size_t j=0;j<nq;j++) hq[branch][j]=(uint8_t)mix((uint32_t)j+101+branch*100003u);
        for (size_t j=0;j<ns;j++) {
            float v=(1+(mix((uint32_t)j+719+branch*100019u)%63))*0.0007f;
            if (dtype) { hh[branch][j]=__float2half(v); hs[branch][j]=__half2float(hh[branch][j]); }
            else hs[branch][j]=v;
        }
    }
    int8_t* x=allocate<int8_t>(in); float* xs=allocate<float>(blocks);
    float* y[2]={allocate<float>(out),allocate<float>(out)};
    upload(x,hx); upload(xs,hxs);
    std::vector<PairWeights> weights;
    for (size_t b=0;b<count;b++) {
        PairWeights pair{};
        for (int branch=0;branch<2;branch++) {
            auto& w=pair.branch[branch]; w.q=allocate<uint8_t>(nq); upload(w.q,hq[branch]);
            if (dtype) { auto p=allocate<__half>(ns); upload(p,hh[branch]); w.s=p; }
            else { auto p=allocate<float>(ns); upload(p,hs[branch]); w.s=p; }
        }
        weights.push_back(pair);
    }
    auto launch=[&](bool d2,size_t b) {
        const auto& pair=weights[b%count];
        const auto& g=pair.branch[0]; const auto& u=pair.branch[1];
        if (d2) launch_gemv_w4a8_r10_d2(x,xs,nullptr,g.q,g.s,u.q,u.s,y[0],y[1],in,out,dtype);
        else {
            launch_gemv_w4a8(x,xs,nullptr,g.q,g.s,y[0],in,out,1,dtype);
            launch_gemv_w4a8(x,xs,nullptr,u.q,u.s,y[1],in,out,1,dtype);
        }
    };
    auto warm=[&](bool d2) {
        cudaEvent_t a,b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
        double total_ms=0; size_t launches=0; long long begin=epoch_us();
        while (total_ms<10000.0) {
            ck(cudaEventRecord(a));
            for (size_t r=0;r<16*count;r++) launch(d2,r);
            ck(cudaGetLastError()); ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b));
            float ms=0; ck(cudaEventElapsedTime(&ms,a,b));
            if (!(ms>0) || !std::isfinite(ms)) throw std::runtime_error("invalid warm duration");
            total_ms+=ms; launches+=16*count;
        }
        printf("WARM shape=%s dtype=%d variant=%s reps=%zu event_ms=%.6f epoch_begin_us=%lld epoch_end_us=%lld\n",
               shape.name,dtype,d2?"d2":"baseline",launches,total_ms,begin,epoch_us());
        ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
    };
    ck(cudaDeviceSynchronize());
    if (mode!="bench") {
        warm(mode=="ncu-d2");
        printf("NCU_SINGLE_BEGIN epoch_us=%lld variant=%s\n",epoch_us(),mode.c_str());
        ck(cudaProfilerStart());
        launch(mode=="ncu-d2",0); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
        ck(cudaProfilerStop());
        printf("NCU_SINGLE_END epoch_us=%lld\n",epoch_us());
    } else {
        D2Reference oracle[2]={
            d2_reference(hx.data(),hxs.data(),hq[0].data(),hs[0].data(),in,out),
            d2_reference(hx.data(),hxs.data(),hq[1].data(),hs[1].data(),in,out)};
        std::vector<float> result[2][2];
        for (int variant=0;variant<2;variant++) {
            std::vector<float> actual(out);
            for (int r=0;r<30;r++) {
                launch(variant!=0,(size_t)r); ck(cudaGetLastError());
                for (int branch=0;branch<2;branch++) {
                    ck(cudaMemcpy(actual.data(),y[branch],(size_t)out*4,cudaMemcpyDeviceToHost));
                    for (float v:actual) if (!std::isfinite(v)) throw std::runtime_error("nonfinite output");
                    if (r==0) result[variant][branch]=actual;
                    else if (memcmp(result[variant][branch].data(),actual.data(),(size_t)out*4))
                        throw std::runtime_error("30-run determinism mismatch");
                }
            }
            for (int branch=0;branch<2;branch++)
                printf("DETERMINISTIC shape=%s dtype=%d variant=%s branch=%s exact=30/30\n",
                       shape.name,dtype,variant?"d2":"baseline",branch?"up":"gate");
        }
        auto compare=[&](int branch,const char* label,const auto& candidate,const auto& reference) {
            std::vector<double> errors(out); int first_fail=-1;
            for (int i=0;i<out;i++) {
                errors[i]=d2_normalized_error((double)candidate[i],(double)reference[i],oracle[branch].denominator[i]);
                if (!(errors[i]<1e-5) && first_fail<0) first_fail=i;
            }
            std::sort(errors.begin(),errors.end());
            double median=out%2?errors[out/2]:(errors[out/2-1]+errors[out/2])/2;
            double p99=errors[(size_t)std::ceil(0.99*out)-1];
            printf("CORRECT shape=%s dtype=%d branch=%s comparison=%s denominator=CPUdouble_sum_abs_terms max=%.12g median=%.12g p99=%.12g first_fail_index=%d threshold=1e-5 status=%s\n",
                   shape.name,dtype,branch?"up":"gate",label,errors.back(),median,p99,first_fail,first_fail<0?"PASS":"FAIL");
            return first_fail<0;
        };
        bool numeric_pass=true,exact_pass=true;
        for (int branch=0;branch<2;branch++) {
            bool old_ok=compare(branch,"old_gpu_vs_cpu",result[0][branch],oracle[branch].value);
            bool d2_ok=compare(branch,"d2_vs_cpu",result[1][branch],oracle[branch].value);
            bool pair_ok=compare(branch,"d2_vs_old_gpu",result[1][branch],result[0][branch]);
            numeric_pass=numeric_pass && old_ok && d2_ok && pair_ok;
            size_t mismatch_bytes=0,mismatch_elements=0; int first=-1;
            for (int i=0;i<out;i++) {
                const auto* a=reinterpret_cast<const unsigned char*>(&result[0][branch][i]);
                const auto* b=reinterpret_cast<const unsigned char*>(&result[1][branch][i]);
                size_t differences=0;
                for (size_t j=0;j<sizeof(float);j++) differences+=(a[j]!=b[j]);
                mismatch_bytes+=differences;
                if (differences) { mismatch_elements++; if (first<0) first=i; }
            }
            printf("EXACT shape=%s dtype=%d variant=d2 branch=%s mismatch_elements=%zu mismatch_bytes=%zu first_mismatch_index=%d\n",
                   shape.name,dtype,branch?"up":"gate",mismatch_elements,mismatch_bytes,first);
            exact_pass=exact_pass && mismatch_elements==0;
        }
        if (!numeric_pass) throw std::runtime_error("per-output normalized error gate failed");
        if (!exact_pass) throw std::runtime_error("D2 arithmetic-order byte mismatch; investigate before timing");
        // Drop duplicate host weights; rotating device tuple content stays identical across arms.
        for (int branch=0;branch<2;branch++) {
            std::vector<uint8_t>().swap(hq[branch]);
            std::vector<float>().swap(hs[branch]); std::vector<__half>().swap(hh[branch]);
        }
        cudaEvent_t start,stop; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
        auto measure=[&](bool d2,size_t reps) {
            ck(cudaEventRecord(start));
            for (size_t r=0;r<reps;r++) launch(d2,r);
            ck(cudaGetLastError()); ck(cudaEventRecord(stop)); ck(cudaEventSynchronize(stop));
            float ms=0; ck(cudaEventElapsedTime(&ms,start,stop));
            if (!(ms>0) || !std::isfinite(ms)) throw std::runtime_error("invalid event duration");
            return ms;
        };
        size_t reps[2]={count,count};
        for (int variant=0;variant<2;variant++) {
            while (true) {
                float ms=measure(variant!=0,reps[variant]);
                printf("CALIBRATE shape=%s dtype=%d variant=%s reps=%zu event_ms=%.6f\n",
                       shape.name,dtype,variant?"d2":"baseline",reps[variant],ms);
                if (ms>=1100.f) break;
                double next=std::ceil((double)reps[variant]*std::max(2.0,1200.0/ms));
                if (next>10000000) throw std::runtime_error("calibration repetition budget exceeded");
                reps[variant]=((size_t)next+count-1)/count*count;
            }
        }
        for (int pair=0;pair<5;pair++) for (int arm=0;arm<2;arm++) {
            int variant=arm; float ms=0; long long begin=0,end=0;
            warm(variant!=0);
            begin=epoch_us(); ms=measure(variant!=0,reps[variant]); end=epoch_us();
            if (ms<1000.f) {
                printf("INVALID_SHORT_ARM shape=%s dtype=%d pair=%d variant=%d reps=%zu event_ms=%.6f epoch_begin_us=%lld epoch_end_us=%lld\n",
                       shape.name,dtype,pair,variant,reps[variant],ms,begin,end);
                throw std::runtime_error("timed arm below one second; no selective retry");
            }
            printf("PAIR shape=%s dtype=%d pair=%d order=%d variant=%s reps=%zu event_ms=%.6f us_per_launch=%.9f logical_bytes=%zu unit=gate_up_pair epoch_begin_us=%lld epoch_end_us=%lld\n",
                   shape.name,dtype,pair,arm,variant?"d2":"baseline",reps[variant],ms,
                   (double)ms*1000/reps[variant],logical_bytes,begin,end);
        }
        ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop));
    }
    for (const auto& pair:weights) for (const auto& w:pair.branch) { ck(cudaFree(w.q)); ck(cudaFree(w.s)); }
    ck(cudaFree(x)); ck(cudaFree(xs)); ck(cudaFree(y[0])); ck(cudaFree(y[1]));
}

int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if (argc>4) throw std::runtime_error("usage: probe [gate_up|all] [fp32|fp16|both] [bench|ncu-baseline|ncu-d2]");
    std::string name=argc>1?argv[1]:"all",dtype=argc>2?argv[2]:"both",mode=argc>3?argv[3]:"bench";
    if (dtype!="fp32" && dtype!="fp16" && dtype!="both") throw std::runtime_error("invalid dtype");
    if (mode!="bench" && mode!="ncu-baseline" && mode!="ncu-d2") throw std::runtime_error("invalid mode");
    if (mode!="bench" && (name=="all" || dtype=="both")) throw std::runtime_error("ncu requires one shape and one dtype");
    bool found=false;
    for (const auto& shape:shapes) if (name=="all" || name==shape.name) {
        found=true;
        for (int sd=0;sd<2;sd++) if (dtype=="both" || dtype==(sd?"fp16":"fp32")) run(shape,sd,mode);
    }
    if (!found) throw std::runtime_error("unknown shape");
    printf("PROBE_COMPLETE mode=%s (clock stability and performance acceptance require external analysis)\n",mode.c_str());
    return 0;
} catch (const std::exception& e) {
    fprintf(stderr,"PROBE_FAILED: %s\n",e.what()); return 1;
}
