#include "../runtime/decode_state_gemma4.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemm_batch.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"

#define BK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
static float *bdm(size_t n){ float*p; BK(cudaMalloc(&p,n*4)); return p; }

G4BatchState alloc_g4_batch_state(const ModelConfig &c, int B, int max_len){
    G4BatchState s{}; s.B=B; s.max_len=max_len; s.n_past=0; s.L=c.num_layers;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    s.kv_cap=(int*)calloc(c.num_layers,sizeof(int));
    s.Kc=(float**)calloc(c.num_layers,sizeof(float*)); s.Vc=(float**)calloc(c.num_layers,sizeof(float*));
    for(int l=0;l<c.num_layers;l++){
        int full=c.layer_is_full?c.layer_is_full[l]:1;
        int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        s.kv_cap[l]=max_len;   // 선형 캡(A-2 안전판). 링버퍼는 별도.
        s.Kc[l]=bdm((size_t)B*max_len*nkv*hd); s.Vc[l]=bdm((size_t)B*max_len*nkv*hd);
    }
    return s;
}
void free_g4_batch_state(G4BatchState &s){
    for(int l=0;l<s.L;l++){ if(s.Kc[l])cudaFree(s.Kc[l]); if(s.Vc[l])cudaFree(s.Vc[l]); }
    free(s.Kc); free(s.Vc); free(s.kv_cap); s.Kc=s.Vc=nullptr; s.kv_cap=nullptr;
}

// batch matmul: X[B,IN] × W[OUT,IN] → Y[B,OUT]. q4 weight × fp32 act (gemm_batch_q4, B-무관 canonical).
static int gb_sdtype;
static inline void mmB(const QMat &m, const float *X, float *Y, int IN, int OUT, int B){
    launch_gemm_batch_q4(X, m.wq, m.sc, Y, IN, OUT, B, gb_sdtype);
}

void gemma4_decode_batch(const ModelConfig &c, const G4Weights &w, G4BatchState &st,
                         const int *d_tokens, const float *d_cos_f, const float *d_sin_f,
                         const float *d_cos_s, const float *d_sin_s,
                         float *d_logits, int commit, int use_q4,
                         const int *d_pos_b, const int *h_pos_b) {
    gb_sdtype=c.scale_dtype;
    const int B=st.B, H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, NQ=c.num_q_heads;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const float eps=c.rms_norm_eps, ascale=c.attn_scaling;
    const bool ragged = (d_pos_b != nullptr);
    const int base = st.n_past;                       // same-length 공통 위치
    int base_max = base;                              // ragged: attn shmem 크기용 max(pos_b)+...
    if(ragged && h_pos_b) for(int b=0;b<B;b++) base_max = h_pos_b[b]>base_max ? h_pos_b[b] : base_max;
    const size_t BH=(size_t)B*H;
    const int hd_max=hd_f>hd_s?hd_f:hd_s;
    const size_t qmax=(size_t)B*NQ*hd_max;
    const size_t kvmax=(size_t)B*(nkv_s*hd_s>nkv_f*hd_f?nkv_s*hd_s:nkv_f*hd_f);

    float *x=bdm(BH),*norm=bdm(BH),*tmp=bdm(BH),*after=bdm(BH);
    float *qb=bdm(qmax),*qn=bdm(qmax);
    float *kb=bdm(kvmax),*kn=bdm(kvmax),*vb=bdm(kvmax),*vn=bdm(kvmax);
    float *attn=bdm(qmax),*ob=bdm(BH);
    float *g=bdm((size_t)B*FFN),*u=bdm((size_t)B*FFN),*act=bdm((size_t)B*FFN),*dn=bdm(BH);

    launch_embedding(d_tokens, w.embed, x, B, H);
    if(c.embed_scale>0.f) launch_scale(x,x,c.embed_scale,BH);

    for(int l=0;l<c.num_layers;l++){
        const G4Layer &L=w.layers[l];
        const int full=L.is_full;
        const int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        const int qdim=NQ*hd, kvdim=nkv*hd;
        const float *cosT=full?d_cos_f:d_cos_s, *sinT=full?d_sin_f:d_sin_s;
        const int window=full?0:c.sliding_window;
        const long kv_stride=(long)st.max_len*kvdim;

        launch_rmsnorm(x, L.input_ln, norm, B, H, eps);
        mmB(L.q, norm, qb, H, qdim, B);
        launch_rmsnorm(qb, L.q_norm, qn, B*NQ, hd, eps);
        // rope: ragged=슬롯별 위치(rope_partial_pos, 전체 테이블), same-length=공통 base(rope_partial_batch). rot=hd(full).
        if(ragged) launch_rope_partial_pos(qn, cosT, sinT, d_pos_b, B, NQ, hd, hd);
        else       launch_rope_partial_batch(qn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, B, NQ, hd, hd);
        mmB(L.k, norm, kb, H, kvdim, B);
        if(full && c.attention_k_eq_v) launch_rmsnorm_noscale(kb, vn, B*nkv, hd, eps);   // V=v_norm(k_proj)
        else { mmB(L.v, norm, vb, H, kvdim, B); launch_rmsnorm_noscale(vb, vn, B*nkv, hd, eps); }
        launch_rmsnorm(kb, L.k_norm, kn, B*nkv, hd, eps);
        if(ragged) launch_rope_partial_pos(kn, cosT, sinT, d_pos_b, B, nkv, hd, hd);
        else       launch_rope_partial_batch(kn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, B, nkv, hd, hd);
        // KV append: 슬롯 b의 위치(ragged=h_pos_b[b], same=base). Kc[l] 레이아웃 [B, max_len, NKV, hd].
        for(int b=0;b<B;b++){
            size_t pos = ragged ? (size_t)h_pos_b[b] : (size_t)base;
            BK(cudaMemcpy(st.Kc[l]+(size_t)b*kv_stride+pos*kvdim, kn+(size_t)b*kvdim, (size_t)kvdim*4, cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy(st.Vc[l]+(size_t)b*kv_stride+pos*kvdim, vn+(size_t)b*kvdim, (size_t)kvdim*4, cudaMemcpyDeviceToDevice));
        }
        launch_attn_decode_batch(qn, st.Kc[l], st.Vc[l], attn, B, NQ, nkv, hd, base_max, kv_stride, 0, ascale, window, ragged?d_pos_b:nullptr);
        mmB(L.o, attn, ob, qdim, H, B);
        launch_rmsnorm(ob, L.post_attn_ln, tmp, B, H, eps);
        launch_add(after, x, tmp, (int)BH);

        launch_rmsnorm(after, L.pre_ffn_ln, norm, B, H, eps);
        mmB(L.gate, norm, g, H, FFN, B);
        mmB(L.up,   norm, u, H, FFN, B);
        launch_geglu(g, u, act, (size_t)B*FFN);
        mmB(L.down, act, dn, FFN, H, B);
        launch_rmsnorm(dn, L.post_ffn_ln, tmp, B, H, eps);
        launch_add(x, after, tmp, (int)BH);
        if(L.layer_scalar!=1.0f) launch_scale(x, x, L.layer_scalar, BH);
    }

    launch_rmsnorm(x, w.final_norm, norm, B, H, eps);
    mmB(w.lm_head, norm, d_logits, H, V, B);
    if(c.final_logit_softcapping>0.f) launch_softcap(d_logits, d_logits, c.final_logit_softcapping, (size_t)B*V);
    if(commit && !ragged) st.n_past += 1;   // ragged면 per-slot 위치는 호출자(scheduler) 관리
    BK(cudaDeviceSynchronize());

    float *all[]={x,norm,tmp,after,qb,qn,kb,kn,vb,vn,attn,ob,g,u,act,dn};
    for(float *p:all) cudaFree(p);
}
