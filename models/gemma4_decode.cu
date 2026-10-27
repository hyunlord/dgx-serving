#include "../runtime/decode_state_gemma4.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_mmq_tmpl.cuh"
#include "../kernels/gemm_tiled.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/attention_flash.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"

#define DK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float *g4d_dm(size_t n){ float*p; DK(cudaMalloc(&p,n*4)); return p; }
static inline float* g4d_grow(float*&p,size_t&cap,size_t n){ if(cap<n){ if(p)cudaFree(p); DK(cudaMalloc(&p,n*4)); cap=n; } return p; }

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
        // 현 append=선형 base offset → 캡은 반드시 max_len(안전). sliding 링버퍼(min(max_len,w) modulo)는
        // attn_decode 물리-슬롯 재작성 + >w golden 필요 → A-2 별도. 캡<max_len + 선형 append = OOB이므로 금지.
        s.kv_cap[l]=max_len;
        s.Kc[l]=g4d_dm((size_t)max_len*nkv*hd); s.Vc[l]=g4d_dm((size_t)max_len*nkv*hd);
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
static int gd_amort=0;   // 1=프리필 MMQ/SB 매트멀. 기본0=plain gemv(golden 경로 무변경).
static inline void mm(const QMat &m, const float *X, float *Y, int IN, int OUT, int S, int use_q4){
    if(use_q4==2){ size_t n=(size_t)S*IN;
        if(gd_cap<n){ if(gd_xq)cudaFree(gd_xq); if(gd_xs)cudaFree(gd_xs); if(gd_xsum)cudaFree(gd_xsum);
            DK(cudaMalloc(&gd_xq,n)); DK(cudaMalloc(&gd_xs,(n/QBLK)*4)); DK(cudaMalloc(&gd_xsum,(n/QBLK)*4)); gd_cap=n; }
        launch_quantize_x_q8(X,gd_xq,gd_xs,gd_xsum,IN,S);
        static const bool mmq_prefill=[](){ const char* v=getenv("G4_MMQ_PREFILL"); return !v || strcmp(v,"0")!=0; }();
        static const int mmq_min_s=[](){
            int value=16;
            if(const char* v=getenv("G4_MMQ_MIN_S")){
                char* end=nullptr; errno=0;
                const long parsed=strtol(v,&end,10);
                if(errno==0 && end!=v && *end=='\0' && parsed>=INT_MIN && parsed<=INT_MAX)
                    value=parsed<16?16:(int)parsed;
                else fprintf(stderr,"Invalid G4_MMQ_MIN_S=%s; using 16\n",v);
            }
            fprintf(stderr,"G4_MMQ_MIN_S=%d\n",value);
            return value;
        }();
        if(mmq_prefill && gd_amort==1 && S>=mmq_min_s && IN%QBLK==0 && (gd_sdtype==0 || gd_sdtype==1)){
            launch_gemm_mmq_w4a8_auto(gd_xq,gd_xs,m.wq,m.sc,Y,S,OUT,IN,gd_sdtype);
        } else if(gd_amort && S>1){  // 프리필: SB 커널로 weight 1회/≤16행 amortize(비-amortized plain gemv의 S× 재적재 제거).
            //   SB per-row == plain gemv per-row bit-identical(test_gemma4_prefill_slot 게이트 확증) → golden 불변.
            for(int s0=0;s0<S;s0+=16){ int sq=(S-s0<16)?(S-s0):16;
                launch_gemv_w4a8_sb(gd_xq+(size_t)s0*IN, gd_xs+(size_t)s0*(IN/QBLK), m.wq, m.sc, Y+(size_t)s0*OUT, IN, OUT, sq, gd_sdtype); }
        } else launch_gemv_w4a8(gd_xq,gd_xs,gd_xsum,m.wq,m.sc,Y,IN,OUT,S,gd_sdtype);
    } else if(use_q4) launch_gemm_tiled_q4(X,m.wq,m.sc,Y,S,OUT,IN,gd_sdtype);
    else              launch_gemv_f32(X,m.wf,Y,IN,OUT,S);
}

void gemma4_decode(const ModelConfig &c, const G4Weights &w, G4DecodeState &st,
                   const int *d_tokens, int Sq,
                   const float *d_cos_f, const float *d_sin_f,
                   const float *d_cos_s, const float *d_sin_s,
                   float *d_logits, int commit, int use_q4, int last_only, int amort_mm,
                   float *d_prenorm) {
    gd_sdtype=c.scale_dtype; gd_amort=amort_mm;
    static const bool flash_prefill=[](){ const char* v=getenv("G4_FLASH_PREFILL"); return !v || strcmp(v,"0")!=0; }();
    static const int flash_min_s=[](){
        int value=16;
        if(const char* v=getenv("G4_FLASH_MIN_S")){
            char* end=nullptr; errno=0;
            const long parsed=strtol(v,&end,10);
            if(errno==0 && end!=v && *end=='\0' && parsed>=INT_MIN && parsed<=INT_MAX)
                value=parsed<16?16:(int)parsed;
            else fprintf(stderr,"Invalid G4_FLASH_MIN_S=%s; using 16\n",v);
        }
        return value;
    }();
    const int H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, NQ=c.num_q_heads;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const float eps=c.rms_norm_eps, ascale=c.attn_scaling;
    const int base=st.n_past;
    const size_t SH=(size_t)Sq*H;
    const int hd_max=hd_f>hd_s?hd_f:hd_s;
    const size_t qmax=(size_t)Sq*NQ*hd_max;
    const size_t kvmax=(size_t)Sq*(nkv_s*hd_s>nkv_f*hd_f?nkv_s*hd_s:nkv_f*hd_f);

    // Like the existing matmul globals: serialized, single-device, non-reentrant calls.
    // Each buffer grows independently; CUDA context/process teardown reclaims storage.
    static float *scratch[16]={};
    static size_t capacity[16]={};
    float *x=g4d_grow(scratch[0],capacity[0],SH);
    float *norm=g4d_grow(scratch[1],capacity[1],SH);
    float *tmp=g4d_grow(scratch[2],capacity[2],SH);
    float *after=g4d_grow(scratch[3],capacity[3],SH);
    float *qb=g4d_grow(scratch[4],capacity[4],qmax);
    float *qn=g4d_grow(scratch[5],capacity[5],qmax);
    float *kb=g4d_grow(scratch[6],capacity[6],kvmax);
    float *kn=g4d_grow(scratch[7],capacity[7],kvmax);
    float *vb=g4d_grow(scratch[8],capacity[8],kvmax);
    float *vn=g4d_grow(scratch[9],capacity[9],kvmax);
    float *attn=g4d_grow(scratch[10],capacity[10],qmax);
    float *ob=g4d_grow(scratch[11],capacity[11],SH);
    float *g=g4d_grow(scratch[12],capacity[12],(size_t)Sq*FFN);
    float *u=g4d_grow(scratch[13],capacity[13],(size_t)Sq*FFN);
    float *act=g4d_grow(scratch[14],capacity[14],(size_t)Sq*FFN);
    float *dn=g4d_grow(scratch[15],capacity[15],SH);

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
        if(flash_prefill && Sq>=flash_min_s)
            launch_flash_attn<float>(qn, st.Kc[l], st.Vc[l], attn, Sq, NQ, nkv, hd, base, 0, nullptr, ascale, window);
        else
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

    // pre-norm residual 노출(3관문 assistant hook): HF hidden_states[-1] == 마지막 층 출력(final-norm 전, 실측 확정)
    if(d_prenorm) DK(cudaMemcpy(d_prenorm, x, SH*4, cudaMemcpyDeviceToDevice));
    launch_rmsnorm(x, w.final_norm, norm, Sq, H, eps);
    const float *hn = last_only ? norm+(size_t)(Sq-1)*H : norm;
    int rows = last_only ? 1 : Sq;
    mm(w.lm_head, hn, d_logits, H, V, rows, use_q4);
    if(c.final_logit_softcapping>0.f) launch_softcap(d_logits, d_logits, c.final_logit_softcapping, (size_t)rows*V);
    if(commit) st.n_past += Sq;
    DK(cudaDeviceSynchronize());
}
