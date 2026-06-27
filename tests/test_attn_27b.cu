// 챕터5 STEP 3+: full-attention 경로 GQA 그룹핑 보편화 격리 검증 (27B, NQ=24/NKV=4, group 6).
//   GDN처럼 0.8B(8/2 group4)서 우연히 맞던 게 27B(24/4 group6)서 깨지나? attn 커널은 kv=h/(NQ/NKV) config.
//   골든의 q/k/v_proj 출력을 받아 엔진 커널만(q_norm·k_norm·partial rope·GQA attn·output_gate) 격리 실행 →
//   o_proj 입력(=attn*sigmoid(gate)) 골든과 대조. proj 가중치 불필요(q4-only 추출이라 fp32 미보유).
//   골든 bf16 로드 → tol 2e-2 (grouping 버그=O(0.1~1)이라 충분히 검출).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../kernels/rmsnorm.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/elementwise.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string GD="artifacts/qwen35-27b/golden-attn/", WD="artifacts/qwen35-27b/weights/";
static std::vector<float> rd(const std::string&base,const std::string&p,size_t n){std::string fp=base+p+".bin";FILE*f=fopen(fp.c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",fp.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"크기 %s\n",fp.c_str());exit(1);}fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static double maxabs(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(){
    const int S=5, NQ=24, NKV=4, HD=256, rot=64; const float eps=1e-6f;  // 27B full-attn config
    printf("[attn_27b] S=%d NQ=%d NKV=%d HD=%d rot=%d (GQA group=%d, 격리 full-attn)\n",S,NQ,NKV,HD,rot,NQ/NKV);

    auto qproj=rd(GD,"attn_qproj",(size_t)S*NQ*2*HD);   // [S, NQ, 2*HD] (Q|gate per head)
    auto kproj=rd(GD,"attn_kproj",(size_t)S*NKV*HD);
    auto vproj=rd(GD,"attn_vproj",(size_t)S*NKV*HD);
    auto o_in =rd(GD,"attn_o_in",(size_t)S*NQ*HD);      // 비교 타깃: attn*sigmoid(gate)
    auto cosv =rd(GD,"rope_cos",(size_t)S*rot), sinv=rd(GD,"rope_sin",(size_t)S*rot);
    auto qnw  =rd(WD,"L3_self_attn.q_norm",(size_t)HD), knw=rd(WD,"L3_self_attn.k_norm",(size_t)HD);

    // per-head split: HF view[S,NQ,2HD] chunk → Q=[:HD], gate=[HD:2HD]
    std::vector<float> qbuf((size_t)S*NQ*HD), gate((size_t)S*NQ*HD);
    for(size_t r=0;r<(size_t)S*NQ;r++) for(int d=0;d<HD;d++){
        qbuf[r*HD+d]=qproj[r*2*HD+d]; gate[r*HD+d]=qproj[r*2*HD+HD+d];
    }

    float *dqb=up(qbuf),*dgate=up(gate),*dk=up(kproj),*dv=up(vproj);
    float *dcos=up(cosv),*dsin=up(sinv),*dqnw=up(qnw),*dknw=up(knw);
    float *dqn,*dkn,*dout;
    CK(cudaMalloc(&dqn,(size_t)S*NQ*HD*4)); CK(cudaMalloc(&dkn,(size_t)S*NKV*HD*4)); CK(cudaMalloc(&dout,(size_t)S*NQ*HD*4));

    launch_rmsnorm_zc(dqb, dqnw, dqn, S*NQ, HD, eps);   // q_norm (per head_dim)
    launch_rmsnorm_zc(dk,  dknw, dkn, S*NKV, HD, eps);  // k_norm
    launch_rope_partial(dqn, dcos, dsin, S, NQ, HD, rot);
    launch_rope_partial(dkn, dcos, dsin, S, NKV, HD, rot);
    launch_attn(dqn, dkn, dv, dout, S, NQ, NKV, HD);    // scale=1/sqrt(HD), kv=h/(NQ/NKV)
    launch_gate_sigmoid(dout, dgate, (int)((size_t)S*NQ*HD));
    CK(cudaDeviceSynchronize());

    std::vector<float> out((size_t)S*NQ*HD); CK(cudaMemcpy(out.data(),dout,out.size()*4,cudaMemcpyDeviceToHost));
    double e=maxabs(out,o_in), me=0; for(size_t i=0;i<out.size();i++) me+=fabs((double)out[i]-o_in[i]); me/=out.size();
    double rms=0; for(float x:o_in)rms+=(double)x*x; rms=sqrt(rms/o_in.size());
    // numpy fp32 레퍼런스(같은 골든 입력)도 golden과 정확히 max 9.56e-3/mean 1.6e-5 → 엔진==numpy.
    // 즉 max는 HF bf16 단일 outlier, mean(~1e-5)이 진짜 정확도. grouping 버그면 mean도 O(0.1).
    printf("  engine full-attn vs golden o_proj입력: max=%.3e mean=%.3e (golden rms=%.3e)\n",e,me,rms);
    bool pass=(e<=2e-2)&&(me<=1e-3);  // mean이 핵심: grouping 정확하면 ~1e-5, 버그면 O(0.1)
    printf("  [max tol 2e-2(bf16 outlier), mean tol 1e-3(GQA group=%d 정확성 핵심)] %s\n",NQ/NKV,pass?"OK":"FAIL");
    printf("[test_attn_27b] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
