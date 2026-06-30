#include "qwen35_batch.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemm_batch.cuh"
#include "../kernels/gemv_w4a8.cuh"   // 챕터9: B≤8 batch decode에 SB 커널(odd-B 1적재)
#include "../kernels/gemv_f32.cuh"
#include "../kernels/quantize_x.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/attention_flash.cuh"  // FA.2: batch도 single과 같은 attention(정밀도 일치)
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/gdn.cuh"

#define BK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
static float *dmb(size_t n){ float*p; BK(cudaMalloc(&p,n*4)); return p; }
static __nv_bfloat16 *dmbh(size_t n){ __nv_bfloat16*p; BK(cudaMalloc(&p,n*sizeof(__nv_bfloat16))); return p; }  // bf16 KV cache (메모리 절반)

// ---- batch state alloc/free ----
BatchDecodeState alloc_batch_decode_state(const ModelConfig &c, int B, int max_len){
    BatchDecodeState s{}; s.B=B; s.max_len=max_len; s.n_past=0; s.L=c.num_layers;
    int L=c.num_layers; const int HD=c.head_dim, NKV=c.num_kv_heads;
    const int nh=c.gdn_num_v_heads, D=c.gdn_head_v_dim, convd=cfg_gdn_conv_dim(&c), K=c.gdn_conv_kernel;
    s.Kc=(__nv_bfloat16**)calloc(L,sizeof(__nv_bfloat16*)); s.Vc=(__nv_bfloat16**)calloc(L,sizeof(__nv_bfloat16*));
    s.S=(float**)calloc(L,sizeof(float*)); s.convctx=(float**)calloc(L,sizeof(float*));
    for(int l=0;l<L;l++){
        if(c.layer_is_full[l]){ s.Kc[l]=dmbh((size_t)B*max_len*NKV*HD); s.Vc[l]=dmbh((size_t)B*max_len*NKV*HD); }  // bf16
        else { s.S[l]=dmb((size_t)B*nh*D*D); s.convctx[l]=dmb((size_t)B*(K-1)*convd); }
    }
    reset_batch_decode_state(c,s); return s;
}
void reset_batch_decode_state(const ModelConfig &c, BatchDecodeState &s){
    s.n_past=0; int L=c.num_layers; const int nh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),K=c.gdn_conv_kernel;
    for(int l=0;l<L;l++) if(!c.layer_is_full[l]){ BK(cudaMemset(s.S[l],0,(size_t)s.B*nh*D*D*4)); BK(cudaMemset(s.convctx[l],0,(size_t)s.B*(K-1)*convd*4)); }
}
void free_batch_decode_state(BatchDecodeState &s){
    auto F=[](void*p){ if(p)cudaFree(p); };
    for(int l=0;l<s.L;l++){ F(s.Kc[l]);F(s.Vc[l]);F(s.S[l]);F(s.convctx[l]); }
    free(s.Kc);free(s.Vc);free(s.S);free(s.convctx); s.Kc=s.Vc=nullptr; s.S=s.convctx=nullptr;
}

// persistent batch scratch
static std::vector<float*> g_bws; static std::vector<size_t> g_bwscap;
static float *bws(int &idx, size_t n){
    if((int)g_bws.size()<=idx){ g_bws.push_back(nullptr); g_bwscap.push_back(0); }
    if(g_bwscap[idx]<n){ if(g_bws[idx])cudaFree(g_bws[idx]); BK(cudaMalloc(&g_bws[idx],n*4)); g_bwscap[idx]=n; }
    return g_bws[idx++];
}

void qwen35_decode_batch(const ModelConfig &c, const QwenWeights &w, BatchDecodeState &st,
                         const int *d_tokens, const float *d_cos, const float *d_sin, float *d_logits, int use_q4){
    const int H=c.hidden_size,FFN=c.intermediate_size,HD=c.head_dim,V=c.vocab_size;
    const int NQ=c.num_q_heads,NKV=c.num_kv_heads,rot=c.rotary_dim; const float eps=c.rms_norm_eps;
    const int qdim2=NQ*HD*(c.attn_output_gate?2:1), qdim=NQ*HD, kvdim=NKV*HD;
    const int convd=cfg_gdn_conv_dim(&c),vdim=cfg_gdn_value_dim(&c),kdim=cfg_gdn_key_dim(&c);
    const int nvh=c.gdn_num_v_heads,hv=c.gdn_head_v_dim,Kk=c.gdn_conv_kernel;
    const int B=st.B, base=st.n_past, ML=st.max_len;
    const size_t BH=(size_t)B*H;
    const long kvstride=(long)ML*kvdim;

    int wi=0;
    float *x=bws(wi,BH),*norm=bws(wi,BH),*after=bws(wi,BH),*n2=bws(wi,BH);
    float *qg=bws(wi,(size_t)B*qdim2),*qb=bws(wi,(size_t)B*qdim),*gate=bws(wi,(size_t)B*qdim),*qn=bws(wi,(size_t)B*qdim);
    float *kb=bws(wi,(size_t)B*kvdim),*kn=bws(wi,(size_t)B*kvdim),*vb=bws(wi,(size_t)B*kvdim),*attn=bws(wi,(size_t)B*qdim),*ob=bws(wi,BH);
    float *mixed=bws(wi,(size_t)B*convd),*mconv=bws(wi,(size_t)B*convd);
    float *qg2=bws(wi,(size_t)B*kdim),*kg2=bws(wi,(size_t)B*kdim),*vg2=bws(wi,(size_t)B*vdim);
    float *zb=bws(wi,(size_t)B*vdim),*ab=bws(wi,(size_t)B*nvh),*bb=bws(wi,(size_t)B*nvh),*gg=bws(wi,(size_t)B*nvh),*beta=bws(wi,(size_t)B*nvh);
    float *core=bws(wi,(size_t)B*vdim),*normed=bws(wi,(size_t)B*vdim),*gout=bws(wi,BH);
    float *gm=bws(wi,(size_t)B*FFN),*um=bws(wi,(size_t)B*FFN),*act=bws(wi,(size_t)B*FFN),*dbuf=bws(wi,BH),*fn=bws(wi,BH);
    float *convfull=bws(wi,(size_t)B*Kk*convd);
    // W4A8 batch quantize scratch (persistent, grow): xq[B*IN] int8 + xs/xsum[B*IN/QBLK]
    static int8_t *g_bxq=nullptr; static float *g_bxs=nullptr; static int32_t *g_bxsum=nullptr; static size_t g_bxcap=0;

    const int sdtype = c.scale_dtype;
    auto mmb=[&](const QMat&m,const float*X,float*Y,int IN,int OUT){
        if(use_q4==2){
            size_t n=(size_t)B*IN;
            if(g_bxcap<n){ if(g_bxq)cudaFree(g_bxq);if(g_bxs)cudaFree(g_bxs);if(g_bxsum)cudaFree(g_bxsum);
                BK(cudaMalloc(&g_bxq,n)); BK(cudaMalloc(&g_bxs,(n/QBLK)*4)); BK(cudaMalloc(&g_bxsum,(n/QBLK)*4)); g_bxcap=n; }
            launch_quantize_x_q8(X,g_bxq,g_bxs,g_bxsum,IN,B);          // B행 X 양자화(1 launch)
            // 챕터9 batch serving: B≤8은 SB 커널(런타임 S, 1적재 — gemm_batch의 odd-B default 루프[B× 재적재] 회피).
            //   gemv_w4a8_sb<SMAX=8>와 gemm_batch_w4a8<B>는 동일 구조·누적순서 → bit-exact. B>8은 기존 gemm_batch.
            if(B<=8) launch_gemv_w4a8_sb(g_bxq,g_bxs,m.wq,m.sc,Y,IN,OUT,B,sdtype);
            else     launch_gemm_batch_w4a8(g_bxq,g_bxs,m.wq,m.sc,Y,IN,OUT,B,sdtype);   // 챕터4: W4A8 lane-minor 최적
        } else if(use_q4){ if(!launch_gemm_mw_q4(X,m.wq,m.sc,Y,IN,OUT,B,sdtype)) launch_gemm_batch_q4(X,m.wq,m.sc,Y,IN,OUT,B,sdtype); }  // q4만 mw 이득(B≥4)
        else              launch_gemm_batch_f32(X,m.wf,Y,IN,OUT,B);      // fp32: bandwidth-bound, mw 무익
    };

    launch_embedding(d_tokens, w.embed, x, B, H);
    for(int l=0;l<c.num_layers;l++){
        const QLayer &L=w.layers[l];
        launch_rmsnorm_zc(x,L.input_ln,norm,B,H,eps);
        if(L.is_full){
            mmb(L.q,norm,qg,H,qdim2);
            BK(cudaMemcpy2D(qb,HD*4,qg,2*HD*4,HD*4,(size_t)B*NQ,cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy2D(gate,HD*4,qg+HD,2*HD*4,HD*4,(size_t)B*NQ,cudaMemcpyDeviceToDevice));
            launch_rmsnorm_zc(qb,L.q_norm,qn,B*NQ,HD,eps);
            mmb(L.k,norm,kb,H,kvdim); launch_rmsnorm_zc(kb,L.k_norm,kn,B*NKV,HD,eps);
            mmb(L.v,norm,vb,H,kvdim);
            launch_rope_partial_batch(qn,d_cos+(size_t)base*rot,d_sin+(size_t)base*rot,B,NQ,HD,rot);
            launch_rope_partial_batch(kn,d_cos+(size_t)base*rot,d_sin+(size_t)base*rot,B,NKV,HD,rot);
            // 각 시퀀스 b: K[b]/V[b]를 cache[b]의 위치 base에 append (시퀀스별 stride, fp32→bf16 변환)
            for (int b=0; b<B; b++) {
                launch_f32_to_bf16(kn+(size_t)b*kvdim, st.Kc[l]+(size_t)b*kvstride+(size_t)base*kvdim, (size_t)kvdim);
                launch_f32_to_bf16(vb+(size_t)b*kvdim, st.Vc[l]+(size_t)b*kvstride+(size_t)base*kvdim, (size_t)kvdim);
            }
            // FA.2 통일: 시퀀스별 KV(kvstride 간격)에 FA.2(single과 동일 attention). decode Sq=1.
            for (int b=0; b<B; b++)
                launch_flash_attn(qn+(size_t)b*qdim, st.Kc[l]+(size_t)b*kvstride, st.Vc[l]+(size_t)b*kvstride,
                                  attn+(size_t)b*qdim, 1, NQ, NKV, HD, base);
            launch_gate_sigmoid(attn, gate, (int)((size_t)B*qdim));
            mmb(L.o,attn,ob,qdim,H); launch_add(after,x,ob,(int)BH);
        } else {
            mmb(L.in_qkv,norm,mixed,H,convd);
            launch_gdn_conv1d_silu_decode_batch(mixed, st.convctx[l], L.conv1d, mconv, B, convd, Kk);
            BK(cudaMemcpy2D(qg2,kdim*4,mconv,convd*4,kdim*4,B,cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy2D(kg2,kdim*4,mconv+kdim,convd*4,kdim*4,B,cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy2D(vg2,vdim*4,mconv+2*kdim,convd*4,vdim*4,B,cudaMemcpyDeviceToDevice));
            mmb(L.in_z,norm,zb,H,vdim);
            launch_gemv_f32(norm,L.in_a,ab,H,nvh,B); launch_gemv_f32(norm,L.in_b,bb,H,nvh,B);
            launch_gdn_decay_beta(ab,bb,L.A_log,L.dt_bias,gg,beta,B,nvh);
            launch_gdn_step_batch(qg2,kg2,vg2,gg,beta, st.S[l], st.S[l], core, B, nvh, hv, c.gdn_num_k_heads);  // in-place state commit
            // conv ctx 갱신: [convctx[b] ; mixed[b]] 의 마지막 K-1
            BK(cudaMemcpy2D(convfull, Kk*convd*4, st.convctx[l], (Kk-1)*convd*4, (Kk-1)*convd*4, B, cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy2D(convfull+(size_t)(Kk-1)*convd, Kk*convd*4, mixed, convd*4, convd*4, B, cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy2D(st.convctx[l], (Kk-1)*convd*4, convfull+(size_t)convd, Kk*convd*4, (Kk-1)*convd*4, B, cudaMemcpyDeviceToDevice));
            launch_rms_norm_gated(core,zb,L.gdn_norm,normed,B*nvh,hv,eps);
            mmb(L.out_p,normed,gout,vdim,H); launch_add(after,x,gout,(int)BH);
        }
        launch_rmsnorm_zc(after,L.post_ln,n2,B,H,eps);
        mmb(L.gate,n2,gm,H,FFN); mmb(L.up,n2,um,H,FFN);
        launch_swiglu(gm,um,act,(size_t)B*FFN);
        mmb(L.down,act,dbuf,FFN,H); launch_add(x,after,dbuf,(int)BH);
    }
    launch_rmsnorm_zc(x,w.final_norm,fn,B,H,eps);
    mmb(w.lm_head,fn,d_logits,H,V);
    st.n_past += 1;
    BK(cudaDeviceSynchronize());
}
