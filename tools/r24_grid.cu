#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "r24_grid_kernel.cuh"

namespace {
constexpr size_t budget = size_t(32) << 30;
constexpr size_t guard_bytes = 256;
void ck(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
double epoch() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { ck(cudaMalloc(&p, n)); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() { return static_cast<T*>(p); }
};
struct Shape { int m, n, k; };
const std::vector<Shape> cases = {{1024,15360,3840},{1024,3840,15360},{1024,3840,4096},
                                 {2048,15360,3840},{2048,3840,15360},{2048,3840,4096}};
int attr(cudaDeviceAttr a) { int v=0; ck(cudaDeviceGetAttribute(&v,a,0)); return v; }
void device() {
    cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p,0));
    std::printf("DEVICE name=\"%s\" cc=%d.%d SM=%d L2_bytes=%d shared_per_SM=%d registers_per_SM=%d max_clock_kHz=%d max_threads_per_SM=%d\n",
        p.name,p.major,p.minor,attr(cudaDevAttrMultiProcessorCount),attr(cudaDevAttrL2CacheSize),
        attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor),attr(cudaDevAttrMaxRegistersPerMultiprocessor),
        attr(cudaDevAttrClockRate),attr(cudaDevAttrMaxThreadsPerMultiProcessor));
}
struct Data {
    Shape s;
    size_t mk,nk,mn;
    Buffer x,xs,w,ws,y;
    explicit Data(Shape v):s(v),mk(size_t(v.m)*v.k),nk(size_t(v.n)*v.k),mn(size_t(v.m)*v.n),
        x(mk),xs(mk/32*sizeof(float)),w(nk/2),ws(nk/32*sizeof(__half)),y(mn*sizeof(float)+guard_bytes) {
        std::vector<int8_t> hx(mk);
        std::vector<uint8_t> hw(nk/2);
        std::vector<float> hs(mk/32);
        std::vector<__half> hws(nk/32);
        uint32_t state=20261103;
        auto next=[&]() { state^=state<<13; state^=state>>17; state^=state<<5; return state; };
        for(auto& a:hx) a=static_cast<int8_t>(int(next()%255)-127);
        for(auto& a:hw) a=static_cast<uint8_t>(next());
        for(auto& a:hs) a=float(1+next()%127)/4096.f;
        for(auto& a:hws) a=__float2half(float(1+next()%127)/1024.f);
        ck(cudaMemcpy(x.p,hx.data(),mk,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(xs.p,hs.data(),hs.size()*sizeof(float),cudaMemcpyHostToDevice));
        ck(cudaMemcpy(w.p,hw.data(),hw.size(),cudaMemcpyHostToDevice));
        ck(cudaMemcpy(ws.p,hws.data(),hws.size()*sizeof(__half),cudaMemcpyHostToDevice));
        size_t read=mk+mk/8+nk/2+nk/16;
        std::printf("CASE M=%d N=%d K=%d weight_bytes=%zu weight_scale_bytes=%zu activation_bytes=%zu activation_scale_bytes=%zu output_write_bytes=%zu theoretical_read_bytes=%zu theoretical_read_write_bytes=%zu allocation_bytes=%zu\n",
            s.m,s.n,s.k,nk/2,nk/16,mk,mk/8,mn*4,read,read+mn*4,read+mn*4+guard_bytes);
    }
    void launch(bool swap) {
        dim3 grid(swap?(s.m+127)/128:(s.n+127)/128,swap?(s.n+127)/128:(s.m+127)/128);
        if(swap) r24_grid_swap_kernel<64,__half><<<grid,256>>>(x.as<int8_t>(),xs.as<float>(),w.as<uint8_t>(),ws.as<__half>(),y.as<float>(),s.m,s.n,s.k);
        else gemm_mmq_kernel<64,__half><<<grid,256>>>(x.as<int8_t>(),xs.as<float>(),w.as<uint8_t>(),ws.as<__half>(),y.as<float>(),s.m,s.n,s.k);
        ck(cudaGetLastError());
    }
    void resources(bool swap) {
        cudaFuncAttributes f{}; int blocks=0;
        if(swap) {
            ck(cudaFuncGetAttributes(&f,r24_grid_swap_kernel<64,__half>));
            ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,r24_grid_swap_kernel<64,__half>,256,0));
        } else {
            ck(cudaFuncGetAttributes(&f,gemm_mmq_kernel<64,__half>));
            ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,gemm_mmq_kernel<64,__half>,256,0));
        }
        int gx=swap?(s.m+127)/128:(s.n+127)/128,gy=swap?(s.n+127)/128:(s.m+127)/128;
        int capacity=blocks*attr(cudaDevAttrMultiProcessorCount),total=gx*gy;
        int last=capacity?((total-1)%capacity+1):0;
        std::printf("RESOURCES variant=%s M=%d N=%d K=%d grid_x=%d grid_y=%d threads=256 registers=%d shared_bytes=%zu local_bytes=%zu CTA_per_SM=%d waves=%.6f last_wave_efficiency=%.6f\n",
            swap?"swap":"main",s.m,s.n,s.k,gx,gy,f.numRegs,f.sharedSizeBytes,f.localSizeBytes,blocks,
            capacity?double(total)/capacity:0,capacity?double(last)/capacity:0);
    }
    std::vector<float> output(bool swap) {
        ck(cudaMemset(y.p,0xff,mn*4));
        ck(cudaMemset(y.as<uint8_t>()+mn*4,0xa5,guard_bytes));
        launch(swap); ck(cudaDeviceSynchronize());
        std::vector<float> out(mn); std::vector<uint8_t> tail(guard_bytes);
        ck(cudaMemcpy(out.data(),y.p,mn*4,cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(tail.data(),y.as<uint8_t>()+mn*4,guard_bytes,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<mn;i++) if(!std::isfinite(out[i])) throw std::runtime_error("nonfinite/unwritten output at index "+std::to_string(i));
        if(!std::all_of(tail.begin(),tail.end(),[](uint8_t b){return b==0xa5;})) throw std::runtime_error("output tail overwritten");
        return out;
    }
    void check() {
        auto a=output(false),b=output(true);
        for(size_t i=0;i<mn;i++) if(std::memcmp(&a[i],&b[i],sizeof(float))) {
            uint32_t av,bv; std::memcpy(&av,&a[i],4); std::memcpy(&bv,&b[i],4);
            std::printf("MISMATCH M=%d N=%d K=%d index=%zu row=%zu col=%zu main=%08x swap=%08x\n",s.m,s.n,s.k,i,i/s.n,i%s.n,av,bv);
            throw std::runtime_error("byte identity failed");
        }
        std::printf("CHECK M=%d N=%d K=%d elements=%zu bytes=%zu identical=1 finite=1 canary=1\n",s.m,s.n,s.k,mn,mn*4);
    }
    double timed(bool swap,int count,const char* phase,int pair) {
        cudaEvent_t a,b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
        ck(cudaDeviceSynchronize()); double start=epoch(); ck(cudaEventRecord(a));
        for(int i=0;i<count;i++) launch(swap);
        ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b)); double end=epoch();
        float ms=0; ck(cudaEventElapsedTime(&ms,a,b)); ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
        std::printf("INTERVAL phase=%s variant=%s pair=%d M=%d N=%d K=%d launches=%d total_ms=%.6f mean_ms=%.9f start_epoch=%.6f end_epoch=%.6f\n",phase,swap?"swap":"main",pair,s.m,s.n,s.k,count,ms,ms/count,start,end);
        return ms;
    }
    void sweep() {
        double warm=0;
        while(warm<10000) warm+=timed(false,32,"warm",-1);
        double cal=timed(false,32,"calibrate",-1);
        if(cal<=0) throw std::runtime_error("invalid calibration");
        int count=std::max(32,int(std::ceil(1200.0*32/cal)));
        for(int repeat=0;repeat<5;repeat++) {
            double ms=timed(false,count,"measure",repeat);
            if(ms<1000) throw std::runtime_error("measurement shorter than one second");
        }
    }
    void bench() {
        check();
        for(int pair=0;pair<5;pair++) {
            double means[2]={};
            for(int position=0;position<2;position++) {
                bool swap=(pair+position)%2;
                double warm=0; while(warm<10000) warm+=timed(swap,32,"warm",pair);
                double cal=timed(swap,32,"calibrate",pair);
                if(cal<=0) throw std::runtime_error("invalid calibration");
                int count=std::max(32,int(std::ceil(1200.0*32/cal)));
                double ms=timed(swap,count,"measure",pair);
                if(ms<1000) throw std::runtime_error("measurement shorter than one second");
                means[swap]=ms/count;
            }
            std::printf("PAIR pair=%d M=%d N=%d K=%d order=%s main_ms=%.9f swap_ms=%.9f swap_over_main=%.9f\n",pair,s.m,s.n,s.k,pair%2?"BA":"AB",means[0],means[1],means[1]/means[0]);
        }
    }
};
Shape parse(char** a) {
    int v[3]; for(int i=0;i<3;i++) { char* end=nullptr; long n=std::strtol(a[i],&end,10); if(!*a[i]||*end||n<=0||n>32768) throw std::runtime_error("invalid dimension"); v[i]=int(n); }
    if(v[2]%64) throw std::runtime_error("K must be divisible by 64");
    return {v[0],v[1],v[2]};
}
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        if(argc<2) throw std::runtime_error("usage: r24_grid device | check [M N K] | bench [M N K] | sweep [M N K] | profile main|swap M N K");
        std::string mode=argv[1];
        if(mode=="device"&&argc==2) { device(); return 0; }
        bool profile=mode=="profile",swap=false;
        std::vector<Shape> shapes=cases;
        if(mode=="sweep") {
            if(argc==5) shapes={parse(argv+2)};
            else if(argc==2) {
                shapes.clear();
                for(int m:{128,256,512,1024,2048}) {
                    shapes.push_back({m,15360,3840});
                    shapes.push_back({m,3840,15360});
                    shapes.push_back({m,3840,4096});
                }
            } else throw std::runtime_error("expected optional M N K");
        } else if(profile) {
            if(argc!=6||(std::string(argv[2])!="main"&&std::string(argv[2])!="swap")) throw std::runtime_error("profile main|swap M N K required");
            swap=std::string(argv[2])=="swap"; shapes={parse(argv+3)};
        } else if(mode=="check"||mode=="bench") {
            if(argc==5) shapes={parse(argv+2)};
            else if(argc!=2) throw std::runtime_error("expected optional M N K");
        } else throw std::runtime_error("unknown mode");
        size_t free=0,total=0; ck(cudaMemGetInfo(&free,&total));
        std::printf("MEMORY budget_bytes=%zu free_bytes=%zu total_bytes=%zu\n",budget,free,total);
        if(double(free)*0.8<double(budget)) throw std::runtime_error("NOT_RUN_MEMORY_GUARD");
        for(Shape s:shapes) {
            size_t mk=size_t(s.m)*s.k,nk=size_t(s.n)*s.k,mn=size_t(s.m)*s.n;
            if(mk+mk/8+nk/2+nk/16+mn*4+guard_bytes>budget) throw std::runtime_error("allocation exceeds budget");
            Data d(s); d.resources(false); if(mode!="sweep") d.resources(true);
            if(profile) { double start=epoch(); d.launch(swap); ck(cudaDeviceSynchronize()); std::printf("PROFILE_COMPLETE variant=%s M=%d N=%d K=%d launches=1 start_epoch=%.6f end_epoch=%.6f correctness=external_check_required\n",swap?"swap":"main",s.m,s.n,s.k,start,epoch()); }
            else if(mode=="sweep") d.sweep();
            else if(mode=="check") d.check();
            else d.bench();
        }
        std::printf("COMPLETE mode=%s cases=%zu\n",mode.c_str(),shapes.size()); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
