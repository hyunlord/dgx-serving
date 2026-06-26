// GDN 커널 격리 검증 (STEP 2): conv1d+silu / delta_rule(naive recurrent) / gated_norm 을
// HF fp32 골든 하위단계와 단독 대조. 입력=골든이 기록한 실제 텐서 → 커널 알고리즘만 격리.
// delta_rule 은 사용자 요구대로 단일 head·t=0,1,2 먼저 점검 후 전체.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../kernels/gdn.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static int g_fail=0;
static std::string GD="artifacts/qwen35-0.8b/golden/", WD="artifacts/qwen35-0.8b/weights/";
static std::vector<float> rd(const std::string&p_,size_t n){std::string p=p_+".bin";FILE*f=fopen(p.c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"크기 %s\n",p.c_str());exit(1);}fclose(f);return v;}
static int rdS(){FILE*f=fopen((GD+"token_ids.bin").c_str(),"rb");fseek(f,0,SEEK_END);long b=ftell(f);fclose(f);return b/4;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static double maxabs(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}
static void chk(const char*n,double e,double tol){printf("  %-22s max abs err=%.3e (tol %.0e) %s\n",n,e,tol,e<=tol?"OK":"FAIL");if(e>tol)g_fail=1;}

int main(){
    int S=rdS(); const int C=6144,NH=16,D=128; const float eps=1e-6f;
    printf("[gdn] S=%d C=%d NH=%d D=%d (layer 0, fp32 golden)\n",S,C,NH,D);

    // ===== (1) conv1d + silu =====
    {
        auto in=rd(GD+"fp32_gdn_in_proj_qkv",(size_t)S*C);   // [S,6144]
        auto w =rd(WD+"L0_linear_attn.conv1d",(size_t)C*4);  // [6144,1,4]->[6144,4]
        auto gold=rd(GD+"fp32_gdn_post_conv",(size_t)S*C);
        float*din=up(in),*dw=up(w),*dout; CK(cudaMalloc(&dout,(size_t)S*C*4));
        launch_gdn_conv1d_silu(din,dw,dout,S,C,4); CK(cudaDeviceSynchronize());
        std::vector<float>o((size_t)S*C); CK(cudaMemcpy(o.data(),dout,o.size()*4,cudaMemcpyDeviceToHost));
        chk("conv1d+silu",maxabs(o,gold),1e-3);
        cudaFree(din);cudaFree(dw);cudaFree(dout);
    }

    // ===== (2) delta_rule (naive recurrent) =====
    {
        auto q=rd(GD+"fp32_gdn_dr_q",(size_t)S*NH*D), k=rd(GD+"fp32_gdn_dr_k",(size_t)S*NH*D), v=rd(GD+"fp32_gdn_dr_v",(size_t)S*NH*D);
        auto g=rd(GD+"fp32_gdn_dr_g",(size_t)S*NH), bt=rd(GD+"fp32_gdn_dr_beta",(size_t)S*NH);
        auto gold=rd(GD+"fp32_gdn_dr_recurrent_out",(size_t)S*NH*D);
        float*dq=up(q),*dk=up(k),*dv=up(v),*dg=up(g),*db=up(bt),*dout; CK(cudaMalloc(&dout,(size_t)S*NH*D*4));
        launch_gdn_delta_rule(dq,dk,dv,dg,db,dout,S,NH,D); CK(cudaDeviceSynchronize());
        std::vector<float>o((size_t)S*NH*D); CK(cudaMemcpy(o.data(),dout,o.size()*4,cudaMemcpyDeviceToHost));
        // --- 단일 head(0) t=0,1,2 먼저 점검 (state 첫 업데이트 방향 확인) ---
        printf("  [delta_rule head0 t=0..2 점검]\n");
        for(int t=0;t<3&&t<S;t++){ size_t b=((size_t)t*NH+0)*D; double e=0; for(int d=0;d<D;d++)e=fmax(e,fabs((double)o[b+d]-gold[b+d]));
            printf("    t=%d: eng[0..2]=[% .5f % .5f % .5f] gold=[% .5f % .5f % .5f] maxerr=%.2e\n",
                   t,o[b],o[b+1],o[b+2],gold[b],gold[b+1],gold[b+2],e); }
        chk("delta_rule(full)",maxabs(o,gold),1e-3);
        cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dg);cudaFree(db);cudaFree(dout);
    }

    // ===== (3) gated RMSNorm (z gate) =====
    {
        int R=S*NH; // [80,128]
        auto core=rd(GD+"fp32_gdn_dr_chunk_out",(size_t)R*D); // HF norm 입력 = chunk_out
        auto z   =rd(GD+"fp32_gdn_z",(size_t)R*D);            // [S,2048]=[80,128]
        auto w   =rd(WD+"L0_linear_attn.norm",(size_t)D);
        auto gold=rd(GD+"fp32_gdn_norm_out",(size_t)R*D);
        float*dc=up(core),*dz=up(z),*dw=up(w),*dout; CK(cudaMalloc(&dout,(size_t)R*D*4));
        launch_rms_norm_gated(dc,dz,dw,dout,R,D,eps); CK(cudaDeviceSynchronize());
        std::vector<float>o((size_t)R*D); CK(cudaMemcpy(o.data(),dout,o.size()*4,cudaMemcpyDeviceToHost));
        chk("rms_norm_gated",maxabs(o,gold),1e-4);
        cudaFree(dc);cudaFree(dz);cudaFree(dw);cudaFree(dout);
    }

    printf(g_fail?"[test_gdn] FAIL\n":"[test_gdn] PASS\n");
    return g_fail;
}
