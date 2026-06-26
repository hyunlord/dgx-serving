// Qwen3.5 MTP head — CUDA 구현이 torch 레퍼런스(golden) 재현하는지 검증 (2b STEP 2-3).
// fp32 경로 (2b 기준선). 2a 커널 재사용 + eh_proj 앞단 신규. post-norm h / P1 pos.
// 부품별 대조: x(eh_proj) / cur2(attn+residual) / logits, 그리고 accept(=draft vs body argmax) 재현.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/embedding.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string GD="artifacts/qwen35-0.8b/mtp_golden/", WD="artifacts/qwen35-0.8b/weights/";
static std::vector<float> rf(const std::string&d,const std::string&n,size_t c){FILE*f=fopen((d+n+".bin").c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",n.c_str());exit(1);}std::vector<float>v(c);if(fread(v.data(),4,c,f)!=c){fprintf(stderr,"크기 %s\n",n.c_str());exit(1);}fclose(f);return v;}
static std::vector<int> ri(const std::string&n){FILE*f=fopen((GD+n+".bin").c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static double mae(const std::vector<float>&a,const std::vector<float>&b){double e=0;for(size_t i=0;i<a.size();i++)e=fmax(e,fabs((double)a[i]-b[i]));return e;}

int main(){
    const int H=1024,NQ=8,NKV=2,HD=256,FFN=3584,V=248320,rot=64; const float eps=1e-6f;
    const int qdim2=NQ*HD*2, kvdim=NKV*HD, adim=NQ*HD;
    std::vector<int> ids=ri("token_ids"); int S=ids.size(); int M=S-1;
    printf("[mtp] S=%d M=%d H=%d NQ=%d NKV=%d HD=%d rot=%d\n",S,M,H,NQ,NKV,HD,rot);

    // ---- weights (fp32) ----
    float *E=up(rf(WD,"w_embed",(size_t)V*H));               // [V,H] embed=lm_head(tie)
    float *fc=up(rf(GD,"mtp_fc_f32",(size_t)H*2*H));
    float *enorm=up(rf(GD,"mtp_enorm",H)), *hnorm=up(rf(GD,"mtp_hnorm",H)), *mnorm=up(rf(GD,"mtp_norm",H));
    float *iln=up(rf(GD,"mtp_input_layernorm",H)), *pln=up(rf(GD,"mtp_post_attention_layernorm",H));
    float *qn_w=up(rf(GD,"mtp_q_norm",HD)), *kn_w=up(rf(GD,"mtp_k_norm",HD));
    float *Wq=up(rf(GD,"mtp_q_proj_f32",(size_t)qdim2*H)), *Wk=up(rf(GD,"mtp_k_proj_f32",(size_t)kvdim*H));
    float *Wv=up(rf(GD,"mtp_v_proj_f32",(size_t)kvdim*H)), *Wo=up(rf(GD,"mtp_o_proj_f32",(size_t)H*adim));
    float *Wg=up(rf(GD,"mtp_gate_proj_f32",(size_t)FFN*H)), *Wu=up(rf(GD,"mtp_up_proj_f32",(size_t)FFN*H)), *Wd=up(rf(GD,"mtp_down_proj_f32",(size_t)H*FFN));

    // ---- inputs ----
    std::vector<int> eids(ids.begin()+1, ids.end());          // token[1..S-1], M개
    int *d_eids; CK(cudaMalloc(&d_eids,M*4)); CK(cudaMemcpy(d_eids,eids.data(),M*4,cudaMemcpyHostToDevice));
    auto h_post=rf(GD,"body_h_post",(size_t)S*H);             // [S,H], h=rows 0..M-1
    float *d_h=up(h_post);                                     // 0..M-1 사용
    float *d_cos=up(rf(GD,"rope_cos",(size_t)M*rot)), *d_sin=up(rf(GD,"rope_sin",(size_t)M*rot));

    // ---- buffers ----
    auto B=[&](size_t n){float*d;CK(cudaMalloc(&d,n*4));return d;};
    float *e=B((size_t)M*H),*en=B((size_t)M*H),*hn=B((size_t)M*H),*cat=B((size_t)M*2*H),*x=B((size_t)M*H);
    float *cur=B((size_t)M*H),*qg=B((size_t)M*qdim2),*qb=B((size_t)M*adim),*gate=B((size_t)M*adim),*qn=B((size_t)M*adim);
    float *kb=B((size_t)M*kvdim),*kn=B((size_t)M*kvdim),*vb=B((size_t)M*kvdim),*attn=B((size_t)M*adim),*ob=B((size_t)M*H);
    float *cur2=B((size_t)M*H),*ff=B((size_t)M*H),*gm=B((size_t)M*FFN),*um=B((size_t)M*FFN),*act=B((size_t)M*FFN),*dn=B((size_t)M*H);
    float *cur3=B((size_t)M*H),*fn=B((size_t)M*H),*logits=B((size_t)M*V);

    // ---- MTP forward (eh_proj → full-attn → ffn → head) ----
    launch_embedding(d_eids, E, e, M, H);                     // e = embed(token[P+1])
    launch_rmsnorm_zc(e, enorm, en, M, H, eps);
    launch_rmsnorm_zc(d_h, hnorm, hn, M, H, eps);             // h = body_h_post[0..M-1]
    CK(cudaMemcpy2D(cat,      2*H*4, en, H*4, H*4, M, cudaMemcpyDeviceToDevice));   // concat[:, :H]=e_n
    CK(cudaMemcpy2D(cat+H,    2*H*4, hn, H*4, H*4, M, cudaMemcpyDeviceToDevice));   // concat[:, H:]=h_n
    launch_gemv_f32(cat, fc, x, 2*H, H, M);                   // eh_proj
    launch_rmsnorm_zc(x, iln, cur, M, H, eps);
    launch_gemv_f32(cur, Wq, qg, H, qdim2, M);
    CK(cudaMemcpy2D(qb,   HD*4, qg,    2*HD*4, HD*4, (size_t)M*NQ, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy2D(gate, HD*4, qg+HD, 2*HD*4, HD*4, (size_t)M*NQ, cudaMemcpyDeviceToDevice));
    launch_rmsnorm_zc(qb, qn_w, qn, M*NQ, HD, eps);
    launch_gemv_f32(cur, Wk, kb, H, kvdim, M); launch_rmsnorm_zc(kb, kn_w, kn, M*NKV, HD, eps);
    launch_gemv_f32(cur, Wv, vb, H, kvdim, M);
    launch_rope_partial(qn, d_cos, d_sin, M, NQ, HD, rot);
    launch_rope_partial(kn, d_cos, d_sin, M, NKV, HD, rot);
    launch_attn(qn, kn, vb, attn, M, NQ, NKV, HD);            // causal prefix attend
    launch_gate_sigmoid(attn, gate, (int)((size_t)M*adim));
    launch_gemv_f32(attn, Wo, ob, adim, H, M);
    launch_add(cur2, x, ob, (int)((size_t)M*H));
    launch_rmsnorm_zc(cur2, pln, ff, M, H, eps);
    launch_gemv_f32(ff, Wg, gm, H, FFN, M); launch_gemv_f32(ff, Wu, um, H, FFN, M);
    launch_swiglu(gm, um, act, (size_t)M*FFN);
    launch_gemv_f32(act, Wd, dn, FFN, H, M);
    launch_add(cur3, cur2, dn, (int)((size_t)M*H));
    launch_rmsnorm_zc(cur3, mnorm, fn, M, H, eps);
    launch_gemv_f32(fn, E, logits, H, V, M);                  // lm_head tie
    CK(cudaDeviceSynchronize());

    // ---- 부품별 대조 ----
    std::vector<float> hx((size_t)M*H), hc2((size_t)M*H), hlg((size_t)M*V);
    CK(cudaMemcpy(hx.data(),x,hx.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc2.data(),cur2,hc2.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hlg.data(),logits,hlg.size()*4,cudaMemcpyDeviceToHost));
    double ex=mae(hx, rf(GD,"mtp_x",(size_t)M*H));
    double ec2=mae(hc2, rf(GD,"mtp_cur2",(size_t)M*H));
    double el=mae(hlg, rf(GD,"mtp_logits",(size_t)M*V));
    printf("\n=== 부품별 vs torch golden (max abs err) ===\n");
    printf("  eh_proj x   = %.3e\n  attn cur2   = %.3e\n  logits      = %.3e\n", ex, ec2, el);

    // ---- accept (draft vs body argmax[P+1]) ----
    std::vector<int> body=ri("body_argmax");
    int ok=0; for(int p=0;p<M;p++){ int b=0; float bv=hlg[(size_t)p*V]; for(int j=1;j<V;j++){float z=hlg[(size_t)p*V+j];if(z>bv){bv=z;b=j;}} if(b==body[p+1])ok++; }
    double acc=100.0*ok/M;
    printf("\n=== accept (CUDA MTP) = %.1f%% (%d/%d) — torch golden 78.6%% ===\n", acc, ok, M);

    bool pass = (ex<1e-2)&&(ec2<1e-2)&&(el<1e-2)&&(acc>70.0);
    printf("\n[test_mtp] %s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
