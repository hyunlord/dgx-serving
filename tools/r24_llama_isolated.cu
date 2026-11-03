// R24 measurement-only: call the existing df9beac Q4_0 MMQ shared-library kernel.
// No llama sources are copied or rebuilt. Quantization is outside timed intervals.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/quantize.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static double epoch() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
static void run(ggml_backend_t backend,int m,int n,int k) {
    const size_t estimate=size_t(n)*k/2+size_t(n)*k/16+size_t(m)*k*6+size_t(m)*n*8+64*1024*1024;
    if(estimate>1024ULL*1024*1024) throw std::runtime_error("shape exceeds registered one-GiB budget");
    auto &ctx=*static_cast<ggml_backend_cuda_context *>(backend->context);
    cudaStream_t stream=ctx.stream();
    ggml_init_params params={ggml_tensor_overhead()*4,nullptr,true};
    ggml_context *gc=ggml_init(params);
    if(!gc) throw std::runtime_error("ggml_init failed");
    ggml_tensor *w=ggml_new_tensor_2d(gc,GGML_TYPE_Q4_0,k,n);
    ggml_tensor *x=ggml_new_tensor_2d(gc,GGML_TYPE_F32,k,m);
    ggml_tensor *ref=ggml_mul_mat(gc,w,x);
    ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(gc,backend);
    if(!buf) throw std::runtime_error("backend allocation failed");
    std::vector<block_q4_0> hw(size_t(n)*k/32);
    for(size_t i=0;i<hw.size();++i) {
        hw[i].d=ggml_fp32_to_fp16(0.03125f+float(i%11)/256.f);
        for(int j=0;j<16;++j) hw[i].qs[j]=uint8_t((i*37+j*19+11)&255);
    }
    std::vector<float> hx(size_t(m)*k);
    for(size_t i=0;i<hx.size();++i) hx[i]=float(int((i*17+3)%251)-125)/127.f;
    ggml_backend_tensor_set(w,hw.data(),0,ggml_nbytes(w));
    ggml_backend_tensor_set(x,hx.data(),0,ggml_nbytes(x));
    ggml_cuda_mul_mat_q(ctx,w,x,nullptr,ref);
    ck(cudaStreamSynchronize(stream));
    std::vector<float> expected(size_t(m)*n),actual(expected.size());
    ggml_backend_tensor_get(ref,expected.data(),0,ggml_nbytes(ref));
    const int kp=GGML_PAD(k,MATRIX_ROW_PADDING);
    const size_t qb=size_t(m)*kp*sizeof(block_q8_1_mmq)/QK8_1_MMQ+128*sizeof(block_q8_1_mmq);
    char *q=nullptr;float *out=nullptr;
    ck(cudaMalloc(&q,qb));ck(cudaMalloc(&out,size_t(m)*n*sizeof(float)));
    ck(cudaMemsetAsync(q,0,qb,stream));
    quantize_mmq_q8_1_cuda(static_cast<const float *>(x->data),nullptr,q,GGML_TYPE_Q4_0,k,k,size_t(k)*m,size_t(k)*m,kp,m,1,1,stream);
    mmq_args a={};
    a.x=static_cast<const char *>(w->data);a.type_x=GGML_TYPE_Q4_0;a.y=reinterpret_cast<const int *>(q);a.dst=out;
    a.ncols_x=k;a.nrows_x=n;a.ncols_dst=m;a.stride_row_x=k/32;a.ncols_y=m;a.nrows_dst=n;
    a.nchannels_x=a.nchannels_y=a.nsamples_x=a.nsamples_y=1;
    a.stride_channel_x=a.stride_sample_x=size_t(n)*k/32;
    a.stride_channel_y=a.stride_sample_y=size_t(m)*kp*sizeof(block_q8_1)/(QK8_1*sizeof(int));
    a.stride_channel_dst=a.stride_sample_dst=size_t(m)*n;
    a.ncols_max=m;a.J_best=128;
    auto launch=[&] { mul_mat_q_case<GGML_TYPE_Q4_0,GGML_PREC_Q8>(ctx,a,stream); };
    launch();ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));
    ck(cudaMemcpy(actual.data(),out,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<actual.size();++i) if(!std::isfinite(actual[i])||std::memcmp(&actual[i],&expected[i],sizeof(float))) throw std::runtime_error("direct call differs from llama wrapper at element "+std::to_string(i));
    std::printf("CHECK variant=llama M=%d N=%d K=%d elements=%zu wrapper_byte_identical=1 synthetic=1 quantization_timed=0\n",m,n,k,actual.size());
    cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
    auto timed=[&](int count,const char *phase,int rep) {
        ck(cudaStreamSynchronize(stream));double t0=epoch();ck(cudaEventRecord(start,stream));
        for(int i=0;i<count;++i) launch();
        ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));double t1=epoch();float ms=0;ck(cudaEventElapsedTime(&ms,start,stop));
        std::printf("INTERVAL phase=%s variant=llama pair=%d M=%d N=%d K=%d launches=%d total_ms=%.6f mean_ms=%.9f start_epoch=%.6f end_epoch=%.6f\n",phase,rep,m,n,k,count,ms,ms/count,t0,t1);return ms;
    };
    double warm=0;while(warm<10000) warm+=timed(32,"warm",-1);
    double cal=timed(32,"calibrate",-1);if(cal<=0) throw std::runtime_error("invalid calibration");
    int count=std::max(32,int(std::ceil(1200.0*32/cal)));
    for(int rep=0;rep<5;++rep) if(timed(count,"measure",rep)<1000) throw std::runtime_error("measurement shorter than one second");
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));ck(cudaFree(q));ck(cudaFree(out));ggml_backend_buffer_free(buf);ggml_free(gc);
}
int main(int argc,char **argv) {
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    try {
        if(argc!=1&&argc!=4) throw std::runtime_error("usage: r24_llama_isolated [M N K]");
        int dims[3]={};
        if(argc==4) for(int i=0;i<3;++i) { char *end=nullptr;long v=std::strtol(argv[i+1],&end,10);if(!*argv[i+1]||*end||v<=0||v>32768) throw std::runtime_error("invalid dimension");dims[i]=int(v); }
        if(argc==4&&(dims[0]<128||dims[0]%128||dims[1]%128||dims[2]%256)) throw std::runtime_error("requires M,N multiples128, K multiple256");
        ggml_backend_t backend=ggml_backend_cuda_init(0);if(!backend) throw std::runtime_error("CUDA backend initialization failed");
        size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total));constexpr size_t budget=32ULL*1024*1024*1024;
        std::printf("MEMORY budget_bytes=%zu free_bytes=%zu total_bytes=%zu\n",budget,free,total);
        if(double(free)*0.8<double(budget)) throw std::runtime_error("NOT_RUN_MEMORY_GUARD");
        if(argc==4) run(backend,dims[0],dims[1],dims[2]);
        else for(int m:{128,256,512,1024,2048}) { run(backend,m,15360,3840);run(backend,m,3840,15360);run(backend,m,3840,4096); }
        ggml_backend_free(backend);std::puts("COMPLETE variant=llama");return 0;
    } catch(const std::exception &e) { std::fprintf(stderr,"FAIL %s\n",e.what());return 1; }
}
