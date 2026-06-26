#include "../runtime/decode_state.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_q4.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/gdn.cuh"

#define DK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float *dm(size_t n){ float*p; DK(cudaMalloc(&p,n*4)); return p; }
static void mm(const QMat &m, const float *X, float *Y, int IN, int OUT, int S, int use_q4) {
    if (use_q4) launch_gemv_q4(X, m.wq, m.sc, Y, IN, OUT, S);
    else        launch_gemv_f32(X, m.wf, Y, IN, OUT, S);
}

// ---- state alloc/free ----
DecodeState alloc_decode_state(const ModelConfig &c, int max_len) {
    DecodeState s{}; s.max_len = max_len; s.n_past = 0; s.L = c.num_layers;
    int L = c.num_layers; const int HD = c.head_dim, NKV = c.num_kv_heads;
    const int nh = c.gdn_num_v_heads, D = c.gdn_head_v_dim, convd = cfg_gdn_conv_dim(&c), K = c.gdn_conv_kernel;
    s.Kc = (float**)calloc(L,sizeof(float*)); s.Vc = (float**)calloc(L,sizeof(float*));
    s.S = (float**)calloc(L,sizeof(float*)); s.convctx = (float**)calloc(L,sizeof(float*));
    for (int l=0;l<L;l++){
        if (c.layer_is_full[l]) {
            s.Kc[l] = dm((size_t)max_len*NKV*HD); s.Vc[l] = dm((size_t)max_len*NKV*HD);
        } else {
            s.S[l] = dm((size_t)nh*D*D); s.convctx[l] = dm((size_t)(K-1)*convd);
        }
    }
    reset_decode_state(c, s);
    return s;
}
void reset_decode_state(const ModelConfig &c, DecodeState &s) {
    s.n_past = 0;
    int L=c.num_layers; const int nh=c.gdn_num_v_heads, D=c.gdn_head_v_dim, convd=cfg_gdn_conv_dim(&c), K=c.gdn_conv_kernel;
    for(int l=0;l<L;l++) if(!c.layer_is_full[l]){ DK(cudaMemset(s.S[l],0,(size_t)nh*D*D*4)); DK(cudaMemset(s.convctx[l],0,(size_t)(K-1)*convd*4)); }
}
void free_decode_state(DecodeState &s) {
    auto F=[](float*p){ if(p) cudaFree(p); };
    for(int l=0;l<s.L;l++){ F(s.Kc[l]); F(s.Vc[l]); F(s.S[l]); F(s.convctx[l]); }
    free(s.Kc); free(s.Vc); free(s.S); free(s.convctx);
    s.Kc=s.Vc=s.S=s.convctx=nullptr;
}

void qwen35_decode(const ModelConfig &c, const QwenWeights &w, DecodeState &st,
                   const int *d_tokens, int Sq, const float *d_cos, const float *d_sin,
                   float *d_logits, float *d_prenorm, int commit, float *S_snap, int use_q4) {
    const int H=c.hidden_size, FFN=c.intermediate_size, HD=c.head_dim, V=c.vocab_size;
    const int NQ=c.num_q_heads, NKV=c.num_kv_heads, rot=c.rotary_dim;
    const float eps=c.rms_norm_eps;
    const int qdim2=NQ*HD*(c.attn_output_gate?2:1), qdim=NQ*HD, kvdim=NKV*HD;
    const int convd=cfg_gdn_conv_dim(&c), vdim=cfg_gdn_value_dim(&c), kdim=cfg_gdn_key_dim(&c);
    const int nvh=c.gdn_num_v_heads, hv=c.gdn_head_v_dim, Kk=c.gdn_conv_kernel;
    const int base=st.n_past;
    const size_t SH=(size_t)Sq*H;

    float *x=dm(SH),*norm=dm(SH),*after=dm(SH),*n2=dm(SH);
    float *qg=dm((size_t)Sq*qdim2),*qb=dm((size_t)Sq*qdim),*gate=dm((size_t)Sq*qdim),*qn=dm((size_t)Sq*qdim);
    float *kb=dm((size_t)Sq*kvdim),*kn=dm((size_t)Sq*kvdim),*vb=dm((size_t)Sq*kvdim),*attn=dm((size_t)Sq*qdim),*ob=dm(SH);
    float *mixed=dm((size_t)Sq*convd),*mconv=dm((size_t)Sq*convd);
    float *qg2=dm((size_t)Sq*kdim),*kg2=dm((size_t)Sq*kdim),*vg2=dm((size_t)Sq*vdim);
    float *zb=dm((size_t)Sq*vdim),*ab=dm((size_t)Sq*nvh),*bb=dm((size_t)Sq*nvh),*gg=dm((size_t)Sq*nvh),*beta=dm((size_t)Sq*nvh);
    float *core=dm((size_t)Sq*vdim),*normed=dm((size_t)Sq*vdim),*gout=dm(SH);
    float *gm=dm((size_t)Sq*FFN),*um=dm((size_t)Sq*FFN),*act=dm((size_t)Sq*FFN),*dbuf=dm(SH),*fn=dm(SH);
    float *snap=dm((size_t)Sq*nvh*hv*hv);   // per-layer GDN snapshot scratch
    float *convfull=dm((size_t)(Kk-1+Sq)*convd);

    launch_embedding(d_tokens, w.embed, x, Sq, H);
    int lin=0;  // linear-layer 카운터 (S_snap 인덱싱)
    for (int l=0;l<c.num_layers;l++){
        const QLayer &L=w.layers[l];
        launch_rmsnorm_zc(x, L.input_ln, norm, Sq, H, eps);
        if (L.is_full) {
            mm(L.q,norm,qg,H,qdim2,Sq,use_q4);
            DK(cudaMemcpy2D(qb,HD*4,qg,2*HD*4,HD*4,(size_t)Sq*NQ,cudaMemcpyDeviceToDevice));
            DK(cudaMemcpy2D(gate,HD*4,qg+HD,2*HD*4,HD*4,(size_t)Sq*NQ,cudaMemcpyDeviceToDevice));
            launch_rmsnorm_zc(qb,L.q_norm,qn,Sq*NQ,HD,eps);
            mm(L.k,norm,kb,H,kvdim,Sq,use_q4); launch_rmsnorm_zc(kb,L.k_norm,kn,Sq*NKV,HD,eps);
            mm(L.v,norm,vb,H,kvdim,Sq,use_q4);
            // rope (위치 base..base+Sq-1) on q and on k (k는 저장 전 roped)
            launch_rope_partial(qn,d_cos+(size_t)base*rot,d_sin+(size_t)base*rot,Sq,NQ,HD,rot);
            launch_rope_partial(kn,d_cos+(size_t)base*rot,d_sin+(size_t)base*rot,Sq,NKV,HD,rot);
            // append roped K, V to cache at slot base..
            DK(cudaMemcpy(st.Kc[l]+(size_t)base*kvdim, kn, (size_t)Sq*kvdim*4, cudaMemcpyDeviceToDevice));
            DK(cudaMemcpy(st.Vc[l]+(size_t)base*kvdim, vb, (size_t)Sq*kvdim*4, cudaMemcpyDeviceToDevice));
            launch_attn_decode(qn, st.Kc[l], st.Vc[l], attn, Sq, NQ, NKV, HD, base);
            launch_gate_sigmoid(attn, gate, (int)((size_t)Sq*qdim));
            mm(L.o,attn,ob,qdim,H,Sq,use_q4);
            launch_add(after,x,ob,(int)SH);
        } else {
            mm(L.in_qkv,norm,mixed,H,convd,Sq,use_q4);
            // conv: prev = convctx[l] (직전 K-1 토큰)
            launch_gdn_conv1d_silu_decode(mixed, st.convctx[l], L.conv1d, mconv, Sq, convd, Kk);
            DK(cudaMemcpy2D(qg2,kdim*4,mconv,convd*4,kdim*4,Sq,cudaMemcpyDeviceToDevice));
            DK(cudaMemcpy2D(kg2,kdim*4,mconv+kdim,convd*4,kdim*4,Sq,cudaMemcpyDeviceToDevice));
            DK(cudaMemcpy2D(vg2,vdim*4,mconv+2*kdim,convd*4,vdim*4,Sq,cudaMemcpyDeviceToDevice));
            mm(L.in_z,norm,zb,H,vdim,Sq,use_q4);
            launch_gemv_f32(norm,L.in_a,ab,H,nvh,Sq); launch_gemv_f32(norm,L.in_b,bb,H,nvh,Sq);
            launch_gdn_decay_beta(ab,bb,L.A_log,L.dt_bias,gg,beta,Sq,nvh);
            // gdn_step from committed S, Sq steps, snapshot
            launch_gdn_step(qg2,kg2,vg2,gg,beta, st.S[l], snap, core, Sq, nvh, hv);
            if (S_snap) {  // verify: per-step snapshots → caller buffer [Sq, n_linear, NH, D, D]
                // (STEP2에서 사용) 여기선 단순 복사 생략 가능 — STEP2 구현시 채움
            }
            if (commit) {
                // 최종 state = snap[Sq-1]
                DK(cudaMemcpy(st.S[l], snap+(size_t)(Sq-1)*nvh*hv*hv, (size_t)nvh*hv*hv*4, cudaMemcpyDeviceToDevice));
                // conv ctx 갱신 = [convctx ; mixed] 의 마지막 K-1 토큰
                DK(cudaMemcpy(convfull, st.convctx[l], (size_t)(Kk-1)*convd*4, cudaMemcpyDeviceToDevice));
                DK(cudaMemcpy(convfull+(size_t)(Kk-1)*convd, mixed, (size_t)Sq*convd*4, cudaMemcpyDeviceToDevice));
                DK(cudaMemcpy(st.convctx[l], convfull+(size_t)Sq*convd, (size_t)(Kk-1)*convd*4, cudaMemcpyDeviceToDevice));
            }
            launch_rms_norm_gated(core,zb,L.gdn_norm,normed,Sq*nvh,hv,eps);
            mm(L.out_p,normed,gout,vdim,H,Sq,use_q4);
            launch_add(after,x,gout,(int)SH);
            lin++;
        }
        launch_rmsnorm_zc(after,L.post_ln,n2,Sq,H,eps);
        mm(L.gate,n2,gm,H,FFN,Sq,use_q4); mm(L.up,n2,um,H,FFN,Sq,use_q4);
        launch_swiglu(gm,um,act,(size_t)Sq*FFN);
        mm(L.down,act,dbuf,FFN,H,Sq,use_q4);
        launch_add(x,after,dbuf,(int)SH);
    }
    if (d_prenorm) DK(cudaMemcpy(d_prenorm,x,SH*4,cudaMemcpyDeviceToDevice));
    launch_rmsnorm_zc(x,w.final_norm,fn,Sq,H,eps);
    mm(w.lm_head,fn,d_logits,H,V,Sq,use_q4);
    if (commit) st.n_past += Sq;
    DK(cudaDeviceSynchronize());

    float* all[]={x,norm,after,n2,qg,qb,gate,qn,kb,kn,vb,attn,ob,mixed,mconv,qg2,kg2,vg2,zb,ab,bb,gg,beta,core,normed,gout,gm,um,act,dbuf,fn,snap,convfull};
    for(float*p:all) cudaFree(p);
    (void)lin;
}
