// Isolated D1 placement probe, not wired into any model dispatch.
// nvcc -arch=sm_120 -O2 -std=c++17 -lineinfo -I. tools/r10_gemv_probe.cu -o /tmp/r10_gemv_probe
// Usage: probe [shape|all] [fp32|fp16|both] [bench|ncu-baseline|ncu-d1]
// ncu --profile-from-start off <binary> q_full fp32 ncu-d1
// ncu modes warm outside profiling, then bracket exactly one GEMV with cudaProfilerStart/Stop.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "../kernels/gemv_w4a8_r10.cuh"

struct Shape { const char* name; int in,out; };
static const Shape shapes[]={{"q_sliding",3840,4096},{"q_full",3840,8192},
    {"o_sliding",4096,3840},{"o_full",8192,3840},{"gate",3840,15360},
    {"up",3840,15360},{"down",15360,3840},{"lm_head",3840,262144}};
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

static void run(const Shape& shape,int dtype,const std::string& mode) {
    const int in=shape.in,out=shape.out,blocks=in/QBLK;
    if (QBLK!=32 || in%32) throw std::runtime_error("QBLK32 required");
    const size_t nq=(size_t)in*out/2,ns=(size_t)blocks*out;
    const size_t weight_bytes=nq+ns*(dtype?2:4);
    const size_t target=96ull*1024*1024,cap=2000000000ull;
    const size_t count=std::max<size_t>(2,(target+weight_bytes-1)/weight_bytes);
    if (count*weight_bytes+(size_t)in+blocks*4ull+out*4ull>cap) throw std::runtime_error("rotating weights exceed 2GB");
    const size_t logical_bytes=weight_bytes+(size_t)in+blocks*4ull+out*4ull;
    printf("CONFIG shape=%s in=%d out=%d dtype=%d QBLK=32 buffers=%zu weight_bytes=%zu rotating_weight_bytes=%zu logical_bytes_per_launch=%zu rows_per_block=1 warps_per_row=8 chunk=8\n",
           shape.name,in,out,dtype,count,weight_bytes,count*weight_bytes,logical_bytes);
    std::vector<int8_t> hx(in); std::vector<float> hxs(blocks);
    std::vector<uint8_t> hq(nq); std::vector<float> hs;
    std::vector<__half> hh;
    for (int i=0;i<in;i++) hx[i]=(int)(mix((uint32_t)i+19)%255)-127;
    for (int b=0;b<blocks;b++) hxs[b]=(1+(mix(b+83)%31))*0.001f;
    for (size_t j=0;j<nq;j++) hq[j]=(uint8_t)mix((uint32_t)j+101);
    if (dtype) hh.resize(ns); else hs.resize(ns);
    for (size_t j=0;j<ns;j++) {
        float v=(1+(mix((uint32_t)j+719)%63))*0.0007f;
        if (dtype) hh[j]=__float2half(v); else hs[j]=v;
    }
    int8_t* x=allocate<int8_t>(in); float* xs=allocate<float>(blocks); float* y=allocate<float>(out);
    upload(x,hx); upload(xs,hxs);
    std::vector<Weights> weights;
    for (size_t b=0;b<count;b++) {
        Weights w{allocate<uint8_t>(nq),nullptr}; upload(w.q,hq);
        if (dtype) { auto p=allocate<__half>(ns); upload(p,hh); w.s=p; }
        else { auto p=allocate<float>(ns); upload(p,hs); w.s=p; }
        weights.push_back(w);
    }
    auto launch=[&](bool d1,size_t b) {
        const auto& w=weights[b%count];
        if (d1) launch_gemv_w4a8_r10_d1(x,xs,w.q,w.s,y,in,out,dtype);
        else launch_gemv_w4a8(x,xs,nullptr,w.q,w.s,y,in,out,1,dtype);
    };
    auto warm=[&](bool d1) {
        cudaEvent_t a,b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
        double total_ms=0; size_t launches=0; long long begin=epoch_us();
        while (total_ms<10000.0) {
            ck(cudaEventRecord(a));
            for (size_t r=0;r<16*count;r++) launch(d1,r);
            ck(cudaGetLastError()); ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b));
            float ms=0; ck(cudaEventElapsedTime(&ms,a,b));
            if (!(ms>0) || !std::isfinite(ms)) throw std::runtime_error("invalid warm duration");
            total_ms+=ms; launches+=16*count;
        }
        printf("WARM shape=%s dtype=%d variant=%s reps=%zu event_ms=%.6f epoch_begin_us=%lld epoch_end_us=%lld\n",
               shape.name,dtype,d1?"d1":"baseline",launches,total_ms,begin,epoch_us());
        ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
    };
    ck(cudaDeviceSynchronize());
    if (mode!="bench") {
        warm(mode=="ncu-d1");
        printf("NCU_SINGLE_BEGIN epoch_us=%lld variant=%s\n",epoch_us(),mode.c_str());
        ck(cudaProfilerStart());
        launch(mode=="ncu-d1",0); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
        ck(cudaProfilerStop());
        printf("NCU_SINGLE_END epoch_us=%lld\n",epoch_us());
    } else {
        // Full CPU-double oracle for the stored fp32/fp16 scales and adjacent-nibble layout.
        std::vector<double> oracle(out),denominator(out);
        for (int row=0;row<out;row++) {
            double sum=0,absolute_sum=0;
            for (int b=0;b<blocks;b++) {
                size_t index=(size_t)row*blocks+b;
                double scale=dtype?(double)__half2float(hh[index]):(double)hs[index];
                for (int j=0;j<32;j++) {
                    int k=b*32+j; uint8_t packed=hq[(size_t)row*(in/2)+k/2];
                    int q=((packed>>(4*(k&1)))&15)-8;
                    double term=(double)q*scale*(double)hx[k]*(double)hxs[b];
                    sum+=term; absolute_sum+=std::abs(term);
                }
            }
            oracle[row]=sum; denominator[row]=absolute_sum;
        }
        std::vector<float> old_gpu,d1_gpu;
        for (bool d1:{false,true}) {
            std::vector<float> first(out),actual(out);
            for (int r=0;r<30;r++) {
                launch(d1,(size_t)r); ck(cudaGetLastError());
                ck(cudaMemcpy(actual.data(),y,(size_t)out*4,cudaMemcpyDeviceToHost));
                if (r==0) first=actual;
                else if (memcmp(first.data(),actual.data(),(size_t)out*4))
                    throw std::runtime_error("30-run determinism mismatch");
                for (int j=0;j<out;j++) {
                    if (!std::isfinite(actual[j])) throw std::runtime_error("nonfinite output");
                }
            }
            (d1?d1_gpu:old_gpu)=std::move(first);
            printf("DETERMINISTIC shape=%s dtype=%d variant=%s exact=30/30\n",
                   shape.name,dtype,d1?"d1":"baseline");
        }
        // Contract: e_i=abs(candidate_i-reference_i)/sum_k(abs(w_ik*s_ik*x_k)).
        // Zero denominator is exact-zero only; no epsilon or output-magnitude normalization.
        auto compare=[&](const char* label,const auto& candidate,const auto& reference) {
            std::vector<double> errors(out); int first_fail=-1;
            for (int i=0;i<out;i++) {
                double diff=std::abs((double)candidate[i]-(double)reference[i]);
                errors[i]=denominator[i]>0?diff/denominator[i]:
                    (diff==0?0:std::numeric_limits<double>::infinity());
                if (!(errors[i]<1e-5) && first_fail<0) first_fail=i;
            }
            std::sort(errors.begin(),errors.end());
            double median=out%2?errors[out/2]:(errors[out/2-1]+errors[out/2])/2;
            double p99=errors[(size_t)std::ceil(0.99*out)-1];
            printf("CORRECT shape=%s dtype=%d comparison=%s denominator=CPUdouble_sum_abs_terms max=%.12g median=%.12g p99=%.12g first_fail_index=%d threshold=1e-5 status=%s\n",
                   shape.name,dtype,label,errors.back(),median,p99,first_fail,first_fail<0?"PASS":"FAIL");
            return first_fail<0;
        };
        bool old_sanity=compare("old_gpu_vs_cpu",old_gpu,oracle);
        bool d1_sanity=compare("d1_vs_cpu",d1_gpu,oracle);
        bool d1_gate=compare("d1_vs_old_gpu",d1_gpu,old_gpu);
        if (!old_sanity || !d1_sanity || !d1_gate) throw std::runtime_error("per-output normalized error gate failed");
        // Drop host matrix storage before timing; device weights rotate through >=96 MiB.
        std::vector<uint8_t>().swap(hq); std::vector<float>().swap(hs); std::vector<__half>().swap(hh);
        cudaEvent_t start,stop; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
        auto measure=[&](bool d1,size_t reps) {
            ck(cudaEventRecord(start));
            for (size_t r=0;r<reps;r++) launch(d1,r);
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
                       shape.name,dtype,variant?"d1":"baseline",reps[variant],ms);
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
            printf("PAIR shape=%s dtype=%d pair=%d order=%d variant=%s reps=%zu event_ms=%.6f us_per_launch=%.9f logical_bytes=%zu epoch_begin_us=%lld epoch_end_us=%lld\n",
                   shape.name,dtype,pair,arm,variant?"d1":"baseline",reps[variant],ms,
                   (double)ms*1000/reps[variant],logical_bytes,begin,end);
        }
        ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop));
    }
    for (const auto& w:weights) { ck(cudaFree(w.q)); ck(cudaFree(w.s)); }
    ck(cudaFree(x)); ck(cudaFree(xs)); ck(cudaFree(y));
}

int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if (argc>4) throw std::runtime_error("usage: probe [shape|all] [fp32|fp16|both] [bench|ncu-baseline|ncu-d1]");
    std::string name=argc>1?argv[1]:"all",dtype=argc>2?argv[2]:"both",mode=argc>3?argv[3]:"bench";
    if (dtype!="fp32" && dtype!="fp16" && dtype!="both") throw std::runtime_error("invalid dtype");
    if (mode!="bench" && mode!="ncu-baseline" && mode!="ncu-d1") throw std::runtime_error("invalid mode");
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
