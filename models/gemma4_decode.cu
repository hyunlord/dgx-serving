#include "../runtime/decode_state_gemma4.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_tiled.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"

#define DK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float *g4d_dm(size_t n){ float*p; DK(cudaMalloc(&p,n*4)); return p; }

// ---- KV 캐시 alloc/free (A-1: 캡=max_len 선형; A-2에서 sliding 캡) ----
G4DecodeState alloc_g4_decode_state(const ModelConfig &c, int max_len) {
    G4DecodeState s{}; s.max_len=max_len; s.n_past=0; s.L=c.num_layers;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    s.kv_cap=(int*)calloc(c.num_layers,sizeof(int));
    s.Kc=(float**)calloc(c.num_layers,sizeof(float*)); s.Vc=(float**)calloc(c.num_layers,sizeof(float*));
    for(int l=0;l<c.num_layers;l++){
        int full=c.layer_is_full?c.layer_is_full[l]:1;
        int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        // A-2: sliding 층은 min(max_len, sliding_window)로 캡. A-1(sliding_window==0 취급 or full)은 max_len.
        int cap = (!full && c.sliding_window>0) ? (c.sliding_window<max_len?c.sliding_window:max_len) : max_len;
        s.kv_cap[l]=cap;
        s.Kc[l]=g4d_dm((size_t)cap*nkv*hd); s.Vc[l]=g4d_dm((size_t)cap*nkv*hd);
    }
    return s;
}
void free_g4_decode_state(G4DecodeState &s){
    for(int l=0;l<s.L;l++){ if(s.Kc[l])cudaFree(s.Kc[l]); if(s.Vc[l])cudaFree(s.Vc[l]); }
    free(s.Kc); free(s.Vc); free(s.kv_cap); s.Kc=s.Vc=nullptr; s.kv_cap=nullptr;
}
void reset_g4_decode_state(G4DecodeState &s){ s.n_past=0; }

// ---- mm dispatch (gemma4.cu와 동일 규약) ----
static int8_t *gd_xq=nullptr; static float *gd_xs=nullptr; static int32_t *gd_xsum=nullptr; static size_t gd_cap=0;
static int gd_sdtype;
static inline void mm(const QMat &m, const float *X, float *Y, int IN, int OUT, int S, int use_q4){
    if(use_q4==2){ size_t n=(size_t)S*IN;
        if(gd_cap<n){ if(gd_xq)cudaFree(gd_xq); if(gd_xs)cudaFree(gd_xs); if(gd_xsum)cudaFree(gd_xsum);
            DK(cudaMalloc(&gd_xq,n)); DK(cudaMalloc(&gd_xs,(n/QBLK)*4)); DK(cudaMalloc(&gd_xsum,(n/QBLK)*4)); gd_cap=n; }
        launch_quantize_x_q8(X,gd_xq,gd_xs,gd_xsum,IN,S);
        launch_gemv_w4a8(gd_xq,gd_xs,gd_xsum,m.wq,m.sc,Y,IN,OUT,S,gd_sdtype);
    } else if(use_q4) launch_gemm_tiled_q4(X,m.wq,m.sc,Y,S,OUT,IN,gd_sdtype);
    else              launch_gemv_f32(X,m.wf,Y,IN,OUT,S);
}

void gemma4_decode(const ModelConfig &c, const G4Weights &w, G4DecodeState &st,
                   const int *d_tokens, int Sq,
                   const float *d_cos_f, const float *d_sin_f,
                   const float *d_cos_s, const float *d_sin_s,
                   float *d_logits, int commit, int use_q4, int last_only) {
    gd_sdtype=c.scale_dtype;
    const int H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, NQ=c.num_q_heads;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const float eps=c.rms_norm_eps, ascale=c.attn_scaling;
    const int base=st.n_past;
    const size_t SH=(size_t)Sq*H;
    const int hd_max=hd_f>hd_s?hd_f:hd_s;
    const size_t qmax=(size_t)Sq*NQ*hd_max;
    const size_t kvmax=(size_t)Sq*(nkv_s*hd_s>nkv_f*hd_f?nkv_s*hd_s:nkv_f*hd_f);

    float *x=g4d_dm(SH),*norm=g4d_dm(SH),*tmp=g4d_dm(SH),*after=g4d_dm(SH);
    float *qb=g4d_dm(qmax),*qn=g4d_dm(qmax);
    float *kb=g4d_dm(kvmax),*kn=g4d_dm(kvmax),*vb=g4d_dm(kvmax),*vn=g4d_dm(kvmax);
    float *attn=g4d_dm(qmax),*ob=g4d_dm(SH);
    float *g=g4d_dm((size_t)Sq*FFN),*u=g4d_dm((size_t)Sq*FFN),*act=g4d_dm((size_t)Sq*FFN),*dn=g4d_dm(SH);

    launch_embedding(d_tokens, w.embed, x, Sq, H);
    if(c.embed_scale>0.f) launch_scale(x,x,c.embed_scale,SH);

    for(int l=0;l<c.num_layers;l++){
        const G4Layer &L=w.layers[l];
        const int full=L.is_full;
        const int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        const int qdim=NQ*hd, kvdim=nkv*hd;
        const float *cosT=full?d_cos_f:d_cos_s, *sinT=full?d_sin_f:d_sin_s;
        const int window=full?0:c.sliding_window;

        launch_rmsnorm(x, L.input_ln, norm, Sq, H, eps);
        mm(L.q, norm, qb, H, qdim, Sq, use_q4);
        launch_rmsnorm(qb, L.q_norm, qn, Sq*NQ, hd, eps);
        launch_rope(qn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, Sq, NQ, hd);
        mm(L.k, norm, kb, H, kvdim, Sq, use_q4);
        if(full && c.attention_k_eq_v) launch_rmsnorm_noscale(kb, vn, Sq*nkv, hd, eps);  // V=v_norm(k_proj), rope 전
        else { mm(L.v, norm, vb, H, kvdim, Sq, use_q4); launch_rmsnorm_noscale(vb, vn, Sq*nkv, hd, eps); }
        launch_rmsnorm(kb, L.k_norm, kn, Sq*nkv, hd, eps);
        launch_rope(kn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, Sq, nkv, hd);
        // KV 캐시 append: 슬롯 base.. (A-1 선형; A-2 캡). roped kn, v_norm된 vn.
        DK(cudaMemcpy(st.Kc[l]+(size_t)base*kvdim, kn, (size_t)Sq*kvdim*4, cudaMemcpyDeviceToDevice));
        DK(cudaMemcpy(st.Vc[l]+(size_t)base*kvdim, vn, (size_t)Sq*kvdim*4, cudaMemcpyDeviceToDevice));
        launch_attn_decode(qn, st.Kc[l], st.Vc[l], attn, Sq, NQ, nkv, hd, base, 0, ascale, window);
        mm(L.o, attn, ob, qdim, H, Sq, use_q4);
        launch_rmsnorm(ob, L.post_attn_ln, tmp, Sq, H, eps);
        launch_add(after, x, tmp, (int)SH);   // after = x + post_attn_ln(o_proj)

        launch_rmsnorm(after, L.pre_ffn_ln, norm, Sq, H, eps);
        mm(L.gate, norm, g, H, FFN, Sq, use_q4);
        mm(L.up,   norm, u, H, FFN, Sq, use_q4);
        launch_geglu(g, u, act, (size_t)Sq*FFN);
        mm(L.down, act, dn, FFN, H, Sq, use_q4);
        launch_rmsnorm(dn, L.post_ffn_ln, tmp, Sq, H, eps);
        launch_add(x, after, tmp, (int)SH);
        if(L.layer_scalar!=1.0f) launch_scale(x, x, L.layer_scalar, SH);
    }

    launch_rmsnorm(x, w.final_norm, norm, Sq, H, eps);
    const float *hn = last_only ? norm+(size_t)(Sq-1)*H : norm;
    int rows = last_only ? 1 : Sq;
    mm(w.lm_head, hn, d_logits, H, V, rows, use_q4);
    if(c.final_logit_softcapping>0.f) launch_softcap(d_logits, d_logits, c.final_logit_softcapping, (size_t)rows*V);
    if(commit) st.n_past += Sq;
    DK(cudaDeviceSynchronize());

    float *all[]={x,norm,tmp,after,qb,qn,kb,kn,vb,vn,attn,ob,g,u,act,dn};
    for(float *p:all) cudaFree(p);
}
