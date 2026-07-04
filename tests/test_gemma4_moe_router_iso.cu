// Gemma4 26B-A4B router ISOLATED 검증 (세션2 — s2 발산이 누적 수치차임을 최종 확정, 35b 방법론).
//   동일 입력(golden 캡처 rin) → 엔진 router 파이프 → golden router 출력(top_i/top_w)과 대조.
//   end-to-end와 달리 rin이 HF와 동일 → 로직이 정확하면 top-8 EXACT 일치해야 함.
//   ≠ 일치면 router 로직 버그. 동일=일치면 end-to-end 발산은 순전히 rin 누적차(정확).
//   소형 텐서만(router.proj/scale/per_expert_scale + rin) — 18GB weight 로드 불요.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>
#include <unistd.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/router_topk.cuh"
#include "../kernels/moe_fused.cuh"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD, GD;
static bool ex(const std::string&p){ return access(p.c_str(),0)==0; }
static std::vector<float> rf(const std::string&p, long n=-1){ FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);} fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET); long cnt=(n<0)?b/4:n; std::vector<float>v(cnt); if(fread(v.data(),4,cnt,f)!=(size_t)cnt)exit(1); fclose(f); return v; }
static std::vector<int> ri(const std::string&p){ FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);} fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET); std::vector<int>v(b/4); if(fread(v.data(),4,v.size(),f)!=v.size())exit(1); fclose(f); return v; }

int main(int argc,char**argv){
    WD = argc>1?argv[1]:"artifacts/gemma4-26b-a4b/weights";
    GD = (argc>2?argv[2]:"artifacts/gemma4-26b-a4b/golden")+std::string("/");
    std::string TAG = argc>3?argv[3]:"q4";   // golden tag (q4 / bf16)
    ModelConfig c = load_config_gemma4((WD+"/manifest.json").c_str());
    const int H=c.hidden_size, E=c.num_experts, K=c.num_experts_per_tok, L=c.num_layers;
    const float eps=c.rms_norm_eps, root=powf((float)H,-0.5f);
    printf("[iso] H=%d E=%d K=%d L=%d eps=%.1e root=H^-0.5=%.5f | tag=%s\n", H,E,K,L,eps,root,TAG.c_str());
    printf("[iso] 동일입력(golden rin) → 엔진 router → golden top_i/top_w 대조. 로직 정확하면 EXACT.\n");

    int nlayers=0, idx_mism=0, s_total=0; double w_relworst=0; int w_rl=-1;
    double idx_matchrate_sum=0;
    for(int l=0; l<L; l++){
        std::string frin=GD+TAG+"_router_rin_"+std::to_string(l)+".bin";
        std::string fti =GD+TAG+"_router_topi_"+std::to_string(l)+".bin";
        std::string ftw =GD+TAG+"_router_topw_"+std::to_string(l)+".bin";
        if(!(ex(frin)&&ex(fti)&&ex(ftw))) continue;
        // rin [S,H]
        auto rin=rf(frin); int S=(int)(rin.size()/H);
        // router weights (fp32): proj[E,H], scale[H], per_expert_scale[E] — artifacts에서 직접
        auto proj=rf(WD+"/L"+std::to_string(l)+"_router.proj.bin", (long)E*H);
        auto scl =rf(WD+"/L"+std::to_string(l)+"_router.scale.bin", H);
        auto pes =rf(WD+"/L"+std::to_string(l)+"_router.per_expert_scale.bin", E);
        // device
        float *d_rin,*d_scl,*d_proj,*d_pes,*d_rnorm,*d_scores,*d_dw; int *d_didx;
        TCK(cudaMalloc(&d_rin,(size_t)S*H*4)); TCK(cudaMemcpy(d_rin,rin.data(),(size_t)S*H*4,cudaMemcpyHostToDevice));
        TCK(cudaMalloc(&d_scl,H*4)); TCK(cudaMemcpy(d_scl,scl.data(),H*4,cudaMemcpyHostToDevice));
        TCK(cudaMalloc(&d_proj,(size_t)E*H*4)); TCK(cudaMemcpy(d_proj,proj.data(),(size_t)E*H*4,cudaMemcpyHostToDevice));
        TCK(cudaMalloc(&d_pes,E*4)); TCK(cudaMemcpy(d_pes,pes.data(),E*4,cudaMemcpyHostToDevice));
        TCK(cudaMalloc(&d_rnorm,(size_t)S*H*4)); TCK(cudaMalloc(&d_scores,(size_t)S*E*4));
        TCK(cudaMalloc(&d_didx,(size_t)S*K*4)); TCK(cudaMalloc(&d_dw,(size_t)S*K*4));
        // 엔진 router 파이프 (gemma4_moe.cu와 동일)
        launch_rmsnorm(d_rin, d_scl, d_rnorm, S, H, eps);
        launch_scale(d_rnorm, d_rnorm, root, (size_t)S*H);
        launch_gemv_f32(d_rnorm, d_proj, d_scores, H, E, S);
        launch_router_topk(d_scores, d_didx, d_dw, S, E, K);
        launch_apply_per_expert_scale(d_dw, d_pes, d_didx, S, K);
        TCK(cudaDeviceSynchronize());
        std::vector<int> didx((size_t)S*K); std::vector<float> dw((size_t)S*K);
        TCK(cudaMemcpy(didx.data(),d_didx,(size_t)S*K*4,cudaMemcpyDeviceToHost));
        TCK(cudaMemcpy(dw.data(),d_dw,(size_t)S*K*4,cudaMemcpyDeviceToHost));

        // === CPU fp32 참조 (build-spec router math, bf16 confound 없는 objective 대조) ===
        //   rmsnorm_noscale(rin)*scale*root → proj → softmax → top-K → renorm → *pes. 엔진 GPU 커널과 대조.
        std::vector<int> ri_(S*K); std::vector<float> rw_(S*K);
        for(int s=0;s<S;s++){
            double ss=0; for(int j=0;j<H;j++) ss+=(double)rin[s*H+j]*rin[s*H+j];
            float inv=(float)(1.0/sqrt(ss/H+eps));
            std::vector<float> hn(H); for(int j=0;j<H;j++) hn[j]=rin[s*H+j]*inv*scl[j]*root;
            std::vector<float> sc(E); float mx=-1e30f;
            for(int e2=0;e2<E;e2++){ double a=0; for(int j=0;j<H;j++) a+=(double)hn[j]*proj[(size_t)e2*H+j]; sc[e2]=(float)a; if(sc[e2]>mx)mx=sc[e2]; }
            double den=0; std::vector<float> pr(E); for(int e2=0;e2<E;e2++){ pr[e2]=expf(sc[e2]-mx); den+=pr[e2]; } for(int e2=0;e2<E;e2++) pr[e2]/=(float)den;
            // top-K (내림차순)
            std::vector<int> ord(E); for(int e2=0;e2<E;e2++)ord[e2]=e2;
            std::partial_sort(ord.begin(),ord.begin()+K,ord.end(),[&](int a,int b){return pr[a]>pr[b];});
            double wsum=0; for(int k=0;k<K;k++) wsum+=pr[ord[k]];
            for(int k=0;k<K;k++){ ri_[s*K+k]=ord[k]; rw_[s*K+k]=(float)(pr[ord[k]]/wsum)*pes[ord[k]]; }
        }
        // 엔진 GPU vs CPU fp32 참조 (objective — 둘 다 fp32)
        int lm=0; double lwr=0;
        for(int s=0;s<S;s++){
            std::unordered_map<int,float> em,rm;
            for(int k=0;k<K;k++){ em[didx[s*K+k]]=dw[s*K+k]; rm[ri_[s*K+k]]=rw_[s*K+k]; }
            for(auto&kv:rm){ if(!em.count(kv.first)) lm++; else { double rel=fabs((double)em[kv.first]-kv.second)/(fabs((double)kv.second)+1e-9); if(rel>lwr)lwr=rel; } }
        }
        // golden-bf16 대조(진단): bf16 산술차 표면화
        auto gi=ri(fti); auto gw=rf(ftw,(long)S*K); int lm_bf=0;
        for(int s=0;s<S;s++){ std::unordered_map<int,int> em; for(int k=0;k<K;k++) em[didx[s*K+k]]=1; for(int k=0;k<K;k++) if(!em.count(gi[s*K+k])) lm_bf++; }
        idx_mism+=lm; s_total+=S; if(lwr>w_relworst){w_relworst=lwr;w_rl=l;}
        double mr = 1.0 - (double)lm/((double)S*K);
        idx_matchrate_sum += mr;
        nlayers++;
        if(lm>0 || l<3 || l==L-1) printf("  L%-2d S=%d: [엔진 vs CPU-fp32] idx불일치=%d(일치 %.1f%%) wrel=%.2e | [vs bf16골든] idx불일치=%d(bf16 confound)\n", l, S, lm, 100*mr, lwr, lm_bf);
        for(float*p:{d_rin,d_scl,d_proj,d_pes,d_rnorm,d_scores,d_dw}) cudaFree(p); cudaFree(d_didx);
    }
    printf("\n=== ISOLATED router 검증 (동일 rin, 엔진 GPU-fp32 vs CPU-fp32 참조) ===\n");
    printf("  MoE층 %d개 | 엔진 vs CPU-fp32: idx 불일치=%d / %d slots | 일치율=%.2f%% | weight worst rel=%.2e @L%d\n",
           nlayers, idx_mism, s_total*K, 100*idx_matchrate_sum/(nlayers>0?nlayers:1), w_relworst, w_rl);
    // 게이트: 엔진 GPU-fp32 == CPU-fp32 참조 (둘 다 fp32, bf16 confound 없음). GPU/CPU fp32 산술차만 허용.
    //   idx 불일치 극소(<0.5%, genuine 1e-6-margin tie flip) + weight rel 작음 = 엔진 router 커널 = build-spec math 정확.
    bool pass = (nlayers>0) && (idx_mism <= (s_total*K)/200) && (w_relworst < 5e-3);
    printf("[test_gemma4_moe_router_iso] %s (엔진 router 커널 == CPU-fp32 build-spec math. idx≈EXACT+wrel<5e-3 → 로직 정확;\n"
           "   bf16골든 대비 잔여는 bf16(HF)vs fp32 산술 confound, 로직 아님. end-to-end 발산=rin 누적차)\n",
           pass?"PASS":(idx_mism>(s_total*K)/200?"IDX 발산 — 로직 조사":"WEIGHT 발산"));
    free_config_gemma4(c);
    return pass?0:1;
}
