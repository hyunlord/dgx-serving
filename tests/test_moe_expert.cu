// 챕터5 STEP 2-②: MoE 단일 expert FFN 격리 + gate_up concat 분리 규약 검증.
//   HF: gate,up = linear(x, gate_up_proj[e]).chunk(2,-1) → [gate(512); up(512)] 블록분리, silu(gate)*up.
//   엔진 expert 경로(gemv_q4 H→1024 → [:512]|[512:] → swiglu → gemv_q4 512→H)를 CPU 오라클(동일 q4
//   weight, double 누적)과 대조. 분리 규약 틀리면(interleave) 엉뚱한 값끼리 곱해 큰 오차 → 규약 검출.
//   디스크 배치 q4(L0 experts)에서 expert 0 슬라이스를 offset 인덱싱 — 엔진 graph가 쓸 인덱싱 방식.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include "../kernels/gemv_q4.cuh"
#include "../kernels/swiglu.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float frand(unsigned&s){s=s*1664525u+1013904223u;return ((s>>8)&0xFFFFFF)/8388608.0f-1.0f;}
static std::string WD="artifacts/qwen35-35b-a3b/weights/";
static std::vector<uint8_t> rdb(const std::string&p){FILE*f=fopen((WD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint8_t>v(n);if(fread(v.data(),1,n,f)!=(size_t)n)exit(1);fclose(f);return v;}
static std::vector<float> rdf(const std::string&p){auto b=rdb(p);std::vector<float>v(b.size()/4);memcpy(v.data(),b.data(),b.size());return v;}
static inline float deq(uint8_t pk,int hi,float sc){return (((hi?(pk>>4):(pk&0x0F))-8))*sc;}

int main(){
    const int H=2048, INTER=512, E=256, eidx=0; unsigned seed=42;  // 35B-A3B MoE
    // 배치 q4에서 expert eidx 슬라이스 (offset 인덱싱 = 엔진 graph 방식)
    auto gu_pk=rdb("L0_mlp.experts.gate_up_proj_q4.bin"); auto gu_sc=rdf("L0_mlp.experts.gate_up_proj_scale.bin");
    auto dn_pk=rdb("L0_mlp.experts.down_proj_q4.bin");    auto dn_sc=rdf("L0_mlp.experts.down_proj_scale.bin");
    const int GU_OUT=2*INTER;                              // 1024 (gate|up concat)
    size_t gu_w=(size_t)GU_OUT*H, dn_w=(size_t)H*INTER;    // expert당 weight 수
    uint8_t *gu_e=gu_pk.data()+eidx*gu_w/2;  float *gu_es=gu_sc.data()+eidx*gu_w/32;  // expert offset
    uint8_t *dn_e=dn_pk.data()+eidx*dn_w/2;  float *dn_es=dn_sc.data()+eidx*dn_w/32;

    std::vector<float> x(H); for(auto&v:x)v=frand(seed);

    // ===== CPU 오라클 (동일 q4 weight, [gate;up] 블록분리, double) =====
    std::vector<double> gu(GU_OUT);
    for(int o=0;o<GU_OUT;o++){double a=0; for(int i=0;i<H;i++){size_t gi=(size_t)o*H+i; a+=(double)x[i]*deq(gu_e[gi/2],gi&1,gu_es[gi/32]);} gu[o]=a;}
    std::vector<double> hmid(INTER);
    for(int i=0;i<INTER;i++){double g=gu[i], u=gu[INTER+i]; hmid[i]=(g/(1.0+exp(-g)))*u;}  // silu(gate)*up
    std::vector<double> ref(H);
    for(int o=0;o<H;o++){double a=0; for(int i=0;i<INTER;i++){size_t gi=(size_t)o*INTER+i; a+=hmid[i]*deq(dn_e[gi/2],gi&1,dn_es[gi/32]);} ref[o]=a;}

    // ===== 엔진 커널 경로 =====
    float *dx,*dgu,*dh,*dout; uint8_t *dguq,*ddnq; float *dgus,*ddns;
    CK(cudaMalloc(&dx,H*4)); CK(cudaMalloc(&dgu,GU_OUT*4)); CK(cudaMalloc(&dh,INTER*4)); CK(cudaMalloc(&dout,H*4));
    CK(cudaMalloc(&dguq,gu_w/2)); CK(cudaMalloc(&dgus,gu_w/32*4)); CK(cudaMalloc(&ddnq,dn_w/2)); CK(cudaMalloc(&ddns,dn_w/32*4));
    CK(cudaMemcpy(dx,x.data(),H*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dguq,gu_e,gu_w/2,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dgus,gu_es,gu_w/32*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddnq,dn_e,dn_w/2,cudaMemcpyHostToDevice)); CK(cudaMemcpy(ddns,dn_es,dn_w/32*4,cudaMemcpyHostToDevice));
    launch_gemv_q4(dx,dguq,dgus,dgu,H,GU_OUT,1);              // x → gate_up [1024]
    launch_swiglu(dgu, dgu+INTER, dh, (size_t)INTER);         // gate=[:512] up=[512:] → silu(gate)*up
    launch_gemv_q4(dh,ddnq,ddns,dout,INTER,H,1);             // → [H]
    CK(cudaDeviceSynchronize());
    std::vector<float> out(H); CK(cudaMemcpy(out.data(),dout,H*4,cudaMemcpyDeviceToHost));

    double e=0,rms=0; for(int o=0;o<H;o++){e=fmax(e,fabs(out[o]-ref[o])); rms+=ref[o]*ref[o];} rms=sqrt(rms/H);
    printf("[moe_expert] H=%d inter=%d expert=%d | engine vs CPU오라클: max=%.3e (out rms=%.3e, rel=%.2e)\n",H,INTER,eidx,e,rms,e/rms);
    bool pass=e/rms<1e-4;   // gemv_q4 GPU-fp32 vs double 누적 floor
    printf("[test_moe_expert] %s (gate_up [gate;up] 블록분리 + SwiGLU 재사용)\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
