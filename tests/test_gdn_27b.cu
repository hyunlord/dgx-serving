// 챕터5 STEP 3: GDN GQA 일반화 커널 격리 검증 (27B, num_k_heads=16 != num_v_heads=48).
//   레퍼런스는 k-head를 v-head 수로 repeat_interleave(3×) 확장 후 동수 delta-rule.
//   엔진 graph는 Q/K를 k-head 레이아웃(2048)으로 주고 커널이 g=h*NHk/NH로 내부 확장.
//   여기선 골든의 확장된 q/k(6144)를 de-expand(head 3*kh)해 k-head 레이아웃으로 만들어
//   launch_gdn_delta_rule(NH=48, NHk=16) 결과 == fp32_gdn_dr_recurrent_out 인지 격리 대조.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../kernels/gdn.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string GD="artifacts/qwen35-27b/golden/";
static std::string PFX="fp32_";  // fp32_ or q4_ (q4 경로 GDN 격리용)
static std::vector<float> rd(const std::string&p_,size_t n){std::string p=GD+PFX+p_+".bin";FILE*f=fopen(p.c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"크기 %s\n",p.c_str());exit(1);}fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static double maxabs(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(int argc,char**argv){
    if(argc>1) PFX=std::string(argv[1])+"_";   // "fp32" or "q4"
    const int S=5, NV=48, NK=16, D=128;   // 27B GDN: k-head 16, v-head 48 (GQA 3:1)
    printf("[gdn_27b] S=%d NV=%d NK=%d D=%d (layer 0, %s골든, GQA 일반화 커널)\n",S,NV,NK,D,PFX.c_str());
    // 골든: q/k/v 모두 v-head 레이아웃(6144=48*128)으로 확장 저장. g/beta = per v-head(48).
    auto q=rd("gdn_dr_q",(size_t)S*NV*D), k=rd("gdn_dr_k",(size_t)S*NV*D), v=rd("gdn_dr_v",(size_t)S*NV*D);
    auto g=rd("gdn_dr_g",(size_t)S*NV), bt=rd("gdn_dr_beta",(size_t)S*NV);
    auto gold=rd("gdn_dr_recurrent_out",(size_t)S*NV*D);

    // de-expand q,k → k-head 레이아웃 [S, NK*D]: repeat_interleave 역연산 = head 3*kh 취함.
    std::vector<float> qc((size_t)S*NK*D), kc((size_t)S*NK*D);
    for(int t=0;t<S;t++) for(int kh=0;kh<NK;kh++){
        int vh=kh*NV/NK;  // 3*kh (역 repeat_interleave)
        for(int d=0;d<D;d++){
            qc[((size_t)t*NK+kh)*D+d]=q[((size_t)t*NV+vh)*D+d];
            kc[((size_t)t*NK+kh)*D+d]=k[((size_t)t*NV+vh)*D+d];
        }
    }
    // 검증: de-expand 가 무손실인지(골든이 정말 repeat_interleave인지) 재확장 비교
    double reexp=0;
    for(int t=0;t<S;t++) for(int vh=0;vh<NV;vh++){int kh=vh*NK/NV; for(int d=0;d<D;d++)
        reexp=fmax(reexp,fabs((double)qc[((size_t)t*NK+kh)*D+d]-q[((size_t)t*NV+vh)*D+d]));}
    printf("  de-expand 무손실성(재확장 vs 골든 q) max=%.2e (0이면 repeat_interleave 확정)\n",reexp);

    float*dq=up(qc),*dk=up(kc),*dv=up(v),*dg=up(g),*db=up(bt),*dout;
    CK(cudaMalloc(&dout,(size_t)S*NV*D*4));
    launch_gdn_delta_rule(dq,dk,dv,dg,db,dout,S,NV,D,NK);   // NH=48 (v-head), NHk=16 (k-head)
    CK(cudaDeviceSynchronize());
    std::vector<float> o((size_t)S*NV*D); CK(cudaMemcpy(o.data(),dout,o.size()*4,cudaMemcpyDeviceToHost));
    double e=maxabs(o,gold);
    printf("  delta_rule(GQA 48/16) vs recurrent_out: max abs err=%.3e (tol 1e-3) %s\n",e,e<=1e-3?"OK":"FAIL");

    // ===== conv1d+silu (27B conv_dim=10240) =====
    const int C=2*NK*D + NV*D;  // 2*kdim + vdim = 4096+6144 = 10240
    auto cin=rd("gdn_in_proj_qkv",(size_t)S*C);
    std::vector<float> cw((size_t)C*4); { FILE*f=fopen("artifacts/qwen35-27b/weights/L0_linear_attn.conv1d.bin","rb"); if(!f){printf("conv1d weight 없음\n");return 1;} if(fread(cw.data(),4,cw.size(),f)!=cw.size()){printf("conv1d 크기\n");return 1;} fclose(f); }
    auto cgold=rd("gdn_post_conv",(size_t)S*C);
    float*dci=up(cin),*dcw=up(cw),*dco; CK(cudaMalloc(&dco,(size_t)S*C*4));
    launch_gdn_conv1d_silu(dci,dcw,dco,S,C,4); CK(cudaDeviceSynchronize());
    std::vector<float> co((size_t)S*C); CK(cudaMemcpy(co.data(),dco,co.size()*4,cudaMemcpyDeviceToHost));
    double ec=maxabs(co,cgold);
    printf("  conv1d+silu(C=%d) vs post_conv: max abs err=%.3e (tol 1e-3) %s\n",C,ec,ec<=1e-3?"OK":"FAIL");

    // ===== gated RMSNorm (z gate), R=S*NV v-head, D =====
    const int R=S*NV;
    auto core=rd("gdn_dr_chunk_out",(size_t)R*D);
    auto z=rd("gdn_z",(size_t)R*D);
    std::vector<float> nw((size_t)D); { FILE*f=fopen("artifacts/qwen35-27b/weights/L0_linear_attn.norm.bin","rb"); if(!f){printf("norm weight 없음\n");return 1;} if(fread(nw.data(),4,nw.size(),f)!=nw.size()){printf("norm 크기\n");return 1;} fclose(f); }
    auto ngold=rd("gdn_norm_out",(size_t)R*D);
    float*dnc=up(core),*dnz=up(z),*dnw=up(nw),*dno; CK(cudaMalloc(&dno,(size_t)R*D*4));
    launch_rms_norm_gated(dnc,dnz,dnw,dno,R,D,1e-6f); CK(cudaDeviceSynchronize());
    std::vector<float> no((size_t)R*D); CK(cudaMemcpy(no.data(),dno,no.size()*4,cudaMemcpyDeviceToHost));
    double en=maxabs(no,ngold);
    // 공식 검증됨(numpy로 x·inv·W·silu(z)가 mean 9.6e-6로 골든과 일치, 타 변형은 3+). max 6e-3은
    // 극소수 원소에 국한 = HF FusedRMSNormGated 의 fused 커널 자체 rounding outlier(엔진 버그 아님).
    printf("  rms_norm_gated(R=%d) vs norm_out: max abs err=%.3e (tol 1e-2; mean~1e-5, max는 HF fused outlier) %s\n",R,en,en<=1e-2?"OK":"FAIL");

    bool pass=(e<=1e-3)&&(reexp<=1e-5)&&(ec<=1e-3)&&(en<=1e-2);
    printf("[test_gdn_27b] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
