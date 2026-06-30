#include "../runtime/decode_state.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_q4.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/quantize_x.cuh"
#include "../kernels/gemv_w4a8.cuh"
#include "../kernels/gemm_batch.cuh"
#include "../kernels/gemm_tiled.cuh"   // 챕터7 Phase A: software tiled q4 GEMM (prefill)
#include "../kernels/gemm_mma.cuh"     // 챕터7 Phase B: tensor-core MMA W4A8 (prefill)
#include "../kernels/gemm_mmq.cuh"     // 챕터7 STEP 0i: full-MMQ W4A8 (128×128+ldmatrix, L1/TEX 84→40%, 1.89×)
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/attention_flash.cuh"  // 챕터7 FA.1a: full-attn 16층 (attn_decode 교체 — 12K 벽 제거)
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/gdn.cuh"
#include "moe_block.cuh"   // moe_mlp_block (decode MoE 분기)

#define DK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float *dm(size_t n){ float*p; DK(cudaMalloc(&p,n*4)); return p; }
static __nv_bfloat16 *dmh(size_t n){ __nv_bfloat16*p; DK(cudaMalloc(&p,n*sizeof(__nv_bfloat16))); return p; }  // bf16 KV cache (메모리 절반)

// persistent scratch workspace: 호출 순서가 결정적이라 idx별 버퍼가 일관됨. 첫 호출시 할당, 이후 재사용
// (capacity 부족하면 grow). per-call cudaMalloc/Free churn 제거 (2d 성능).
#include <vector>
static std::vector<float*> g_ws; static std::vector<size_t> g_wscap;
static float *ws(int &idx, size_t n){
    if ((int)g_ws.size() <= idx) { g_ws.push_back(nullptr); g_wscap.push_back(0); }
    if (g_wscap[idx] < n) { if (g_ws[idx]) cudaFree(g_ws[idx]); DK(cudaMalloc(&g_ws[idx], n*4)); g_wscap[idx]=n; }
    return g_ws[idx++];
}
// int8 persistent ws (W4A8 shared-X quantize 버퍼용)
static std::vector<int8_t*> g_wsb; static std::vector<size_t> g_wsbcap;
static int8_t *wsi8(int &idx, size_t n){
    if ((int)g_wsb.size() <= idx) { g_wsb.push_back(nullptr); g_wsbcap.push_back(0); }
    if (g_wsbcap[idx] < n) { if (g_wsb[idx]) cudaFree(g_wsb[idx]); DK(cudaMalloc(&g_wsb[idx], n)); g_wsbcap[idx]=n; }
    return g_wsb[idx++];
}
// W4A8 per-call quantize 임시버퍼 (persistent, grow). STEP2b: 호출당 X 양자화(최악 호출수) — 측정용.
//   STEP2c에서 shared-X(norm/n2)는 1회 양자화로 호출수↓ 예정.
static int8_t *g_xq=nullptr; static float *g_xs=nullptr; static int32_t *g_xsum=nullptr; static size_t g_xqcap=0;
static int g_sdtype;  // scale_dtype cached from ModelConfig
static void mm(const QMat &m, const float *X, float *Y, int IN, int OUT, int S, int use_q4) {
    if (use_q4==2) {                                   // W4A8: activation int8 per-block-32 + int4 weight
        size_t n=(size_t)S*IN;
        if (g_xqcap < n) { if(g_xq)cudaFree(g_xq); if(g_xs)cudaFree(g_xs); if(g_xsum)cudaFree(g_xsum);
            DK(cudaMalloc(&g_xq,n)); DK(cudaMalloc(&g_xs,(n/QBLK)*4)); DK(cudaMalloc(&g_xsum,(n/QBLK)*4)); g_xqcap=n; }
        launch_quantize_x_q8(X, g_xq, g_xs, g_xsum, IN, S);
        if      (S == 1) launch_gemv_w4a8_mlp<2>(g_xq, g_xs, g_xsum, m.wq, m.sc, Y, IN, OUT, S, g_sdtype);  // 챕터9: MLP↑(outstanding load 2개로 long_scoreboard latency 숨김, bit-exact)
        else if (S <= 8) launch_gemv_w4a8_sb(g_xq, g_xs, m.wq, m.sc, Y, IN, OUT, S, g_sdtype);  // STEP D: verify 소-batch(2~8). (챕터9 SB-MLP 실험은 역효과 — S-accum이 이미 ILP, U-unroll이 occupancy만↓)
        else             launch_gemm_mmq_w4a8(g_xq, g_xs, m.wq, m.sc, Y, S, OUT, IN, g_sdtype); // prefill: full-MMQ(STEP 0i, e2e 1.51× vs gemm_mma)
    } else if (use_q4) launch_gemm_tiled_q4(X, m.wq, m.sc, Y, S, OUT, IN, g_sdtype);   // prefill tiled GEMM(Phase A); S=1은 gemv
    else               launch_gemv_f32(X, m.wf, Y, IN, OUT, S);
}

// ---- state alloc/free ----
DecodeState alloc_decode_state(const ModelConfig &c, int max_len) {
    DecodeState s{}; s.max_len = max_len; s.n_past = 0; s.L = c.num_layers;
    int L = c.num_layers; const int HD = c.head_dim, NKV = c.num_kv_heads;
    const int nh = c.gdn_num_v_heads, D = c.gdn_head_v_dim, convd = cfg_gdn_conv_dim(&c), K = c.gdn_conv_kernel;
    s.Kc = (__nv_bfloat16**)calloc(L,sizeof(__nv_bfloat16*)); s.Vc = (__nv_bfloat16**)calloc(L,sizeof(__nv_bfloat16*));
    s.S = (float**)calloc(L,sizeof(float*)); s.convctx = (float**)calloc(L,sizeof(float*));
    for (int l=0;l<L;l++){
        if (c.layer_is_full[l]) {
            s.Kc[l] = dmh((size_t)max_len*NKV*HD); s.Vc[l] = dmh((size_t)max_len*NKV*HD);  // bf16
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
    auto F=[](void*p){ if(p) cudaFree(p); };
    for(int l=0;l<s.L;l++){ F(s.Kc[l]); F(s.Vc[l]); F(s.S[l]); F(s.convctx[l]); }
    free(s.Kc); free(s.Vc); free(s.S); free(s.convctx);
    s.Kc=s.Vc=nullptr; s.S=s.convctx=nullptr;
}

VerifyArtifacts alloc_verify_artifacts(const ModelConfig &c, int max_Sq) {
    VerifyArtifacts v{}; v.Sq = max_Sq;
    int nl = 0; for (int l=0;l<c.num_layers;l++) if(!c.layer_is_full[l]) nl++;
    v.n_linear = nl; v.state_sz = c.gdn_num_v_heads * c.gdn_head_v_dim * c.gdn_head_v_dim;
    v.conv_dim = cfg_gdn_conv_dim(&c);
    v.gdn_snap = dm((size_t)max_Sq * nl * v.state_sz);
    v.conv_mixed = dm((size_t)max_Sq * nl * v.conv_dim);
    return v;
}
void free_verify_artifacts(VerifyArtifacts &v) { if(v.gdn_snap)cudaFree(v.gdn_snap); if(v.conv_mixed)cudaFree(v.conv_mixed); v.gdn_snap=v.conv_mixed=nullptr; }

void qwen35_decode(const ModelConfig &c, const QwenWeights &w, DecodeState &st,
                   const int *d_tokens, int Sq, const float *d_cos, const float *d_sin,
                   float *d_logits, float *d_prenorm, int commit, VerifyArtifacts *va, int use_q4,
                   float *d_postnorm, int last_logits_only,
                   const int *d_parent, const unsigned *d_anc, const int *d_pos) {
    g_sdtype = c.scale_dtype;
    const int H=c.hidden_size, FFN=c.intermediate_size, HD=c.head_dim, V=c.vocab_size;
    const int NQ=c.num_q_heads, NKV=c.num_kv_heads, rot=c.rotary_dim;
    const float eps=c.rms_norm_eps;
    const int qdim2=NQ*HD*(c.attn_output_gate?2:1), qdim=NQ*HD, kvdim=NKV*HD;
    const int convd=cfg_gdn_conv_dim(&c), vdim=cfg_gdn_value_dim(&c), kdim=cfg_gdn_key_dim(&c);
    const int nvh=c.gdn_num_v_heads, hv=c.gdn_head_v_dim, Kk=c.gdn_conv_kernel;
    const int base=st.n_past;
    const size_t SH=(size_t)Sq*H;

    int wi=0;  // persistent workspace index (호출마다 0부터, 순서 결정적)
    float *x=ws(wi,SH),*norm=ws(wi,SH),*after=ws(wi,SH),*n2=ws(wi,SH);
    float *qg=ws(wi,(size_t)Sq*qdim2),*qb=ws(wi,(size_t)Sq*qdim),*gate=ws(wi,(size_t)Sq*qdim),*qn=ws(wi,(size_t)Sq*qdim);
    float *kb=ws(wi,(size_t)Sq*kvdim),*kn=ws(wi,(size_t)Sq*kvdim),*vb=ws(wi,(size_t)Sq*kvdim),*attn=ws(wi,(size_t)Sq*qdim),*ob=ws(wi,SH);
    float *mixed=ws(wi,(size_t)Sq*convd),*mconv=ws(wi,(size_t)Sq*convd);
    float *qg2=ws(wi,(size_t)Sq*kdim),*kg2=ws(wi,(size_t)Sq*kdim),*vg2=ws(wi,(size_t)Sq*vdim);
    float *zb=ws(wi,(size_t)Sq*vdim),*ab=ws(wi,(size_t)Sq*nvh),*bb=ws(wi,(size_t)Sq*nvh),*gg=ws(wi,(size_t)Sq*nvh),*beta=ws(wi,(size_t)Sq*nvh);
    float *core=ws(wi,(size_t)Sq*vdim),*normed=ws(wi,(size_t)Sq*vdim),*gout=ws(wi,SH);
    float *gm=ws(wi,(size_t)Sq*FFN),*um=ws(wi,(size_t)Sq*FFN),*act=ws(wi,(size_t)Sq*FFN),*dbuf=ws(wi,SH),*fn=ws(wi,SH);
    // GDN snapshot scratch: verify(va)만 per-token([Sq,NH,D²]); plain prefill/decode는 최종 state 1개([NH,D²]).
    //   va 없을 때 Sq×NH×D² 폭발 방지(32K prefill에서 103GB→3MB — 시스템 크래시 근본 원인 제거).
    float *snap=ws(wi,(size_t)((va||d_parent)?Sq:1)*nvh*hv*hv);  // tree(d_parent)는 노드별 fork용 per-node snapshot 필수
    float *convfull=ws(wi,(size_t)(Kk-1+Sq)*convd);

    // W4A8 shared-X quantize 버퍼: norm/n2(IN=H)를 1회 양자화해 q/k/v(또는 in_qkv/in_z)·gate/up 재사용.
    int bi=0; int8_t *xqn=nullptr,*xq2=nullptr; float *xsn=nullptr,*xs2=nullptr,*xsmN=nullptr;
    if (use_q4==2) { xqn=wsi8(bi,SH); xq2=wsi8(bi,SH); xsn=ws(wi,SH/QBLK); xs2=ws(wi,SH/QBLK); xsmN=ws(wi,SH/QBLK); }
    // shared-X gemv: use_q4==2면 미리 양자화한 (xq,xs) 사용(재양자화 안 함), 아니면 mm() 경로.
    auto MMs=[&](const QMat&m,const float*X,const int8_t*xqs,const float*xss,float*Y,int IN,int OUT){
        if (use_q4==2) { if      (Sq==1) launch_gemv_w4a8_mlp<2>(xqs,xss,nullptr,m.wq,m.sc,Y,IN,OUT,Sq,g_sdtype);  // 챕터9 MLP↑
                         else if (Sq<=8) launch_gemv_w4a8_sb(xqs,xss,m.wq,m.sc,Y,IN,OUT,Sq,g_sdtype);   // STEP D (SB-MLP 역효과로 미적용)
                         else            launch_gemm_mmq_w4a8(xqs,xss,m.wq,m.sc,Y,Sq,OUT,IN,g_sdtype); } // prefill full-MMQ(STEP 0i)
        else           mm(m,X,Y,IN,OUT,Sq,use_q4);
    };

    launch_embedding(d_tokens, w.embed, x, Sq, H);
    int lin=0;  // linear-layer 카운터 (S_snap 인덱싱)
    for (int l=0;l<c.num_layers;l++){
        const QLayer &L=w.layers[l];
        launch_rmsnorm_zc(x, L.input_ln, norm, Sq, H, eps);
        if (use_q4==2) launch_quantize_x_q8(norm, xqn, xsn, (int32_t*)xsmN, H, Sq);  // norm 1회 양자화 → q/k/v(또는 in_qkv/in_z) 공유
        if (L.is_full) {
            MMs(L.q,norm,xqn,xsn,qg,H,qdim2);
            DK(cudaMemcpy2D(qb,HD*4,qg,2*HD*4,HD*4,(size_t)Sq*NQ,cudaMemcpyDeviceToDevice));
            DK(cudaMemcpy2D(gate,HD*4,qg+HD,2*HD*4,HD*4,(size_t)Sq*NQ,cudaMemcpyDeviceToDevice));
            launch_rmsnorm_zc(qb,L.q_norm,qn,Sq*NQ,HD,eps);
            MMs(L.k,norm,xqn,xsn,kb,H,kvdim); launch_rmsnorm_zc(kb,L.k_norm,kn,Sq*NKV,HD,eps);
            MMs(L.v,norm,xqn,xsn,vb,H,kvdim);
            // rope on q and on k (k는 저장 전 roped). tree(d_pos): 노드별 위치=base+depth, 아니면 contiguous base..base+Sq-1
            if (d_pos) { launch_rope_partial_pos(qn,d_cos,d_sin,d_pos,Sq,NQ,HD,rot);
                         launch_rope_partial_pos(kn,d_cos,d_sin,d_pos,Sq,NKV,HD,rot); }
            else       { launch_rope_partial(qn,d_cos+(size_t)base*rot,d_sin+(size_t)base*rot,Sq,NQ,HD,rot);
                         launch_rope_partial(kn,d_cos+(size_t)base*rot,d_sin+(size_t)base*rot,Sq,NKV,HD,rot); }
            // append roped K, V to cache at slot base.. (fp32 kn/vb → bf16 cache 변환 저장)
            launch_f32_to_bf16(kn, st.Kc[l]+(size_t)base*kvdim, (size_t)Sq*kvdim);
            launch_f32_to_bf16(vb, st.Vc[l]+(size_t)base*kvdim, (size_t)Sq*kvdim);
            launch_flash_attn(qn, st.Kc[l], st.Vc[l], attn, Sq, NQ, NKV, HD, base, 0, d_anc);  // FA.1a + tree mask(d_anc=조상 비트마스크, null=causal)
            launch_gate_sigmoid(attn, gate, (int)((size_t)Sq*qdim));
            mm(L.o,attn,ob,qdim,H,Sq,use_q4);
            launch_add(after,x,ob,(int)SH);
        } else {
            MMs(L.in_qkv,norm,xqn,xsn,mixed,H,convd);
            // conv: prev = convctx[l] (직전 K-1 토큰). tree(d_parent): window를 조상-path로(STEP 2)
            if (d_parent) launch_gdn_conv1d_silu_decode_tree(mixed, st.convctx[l], d_parent, L.conv1d, mconv, Sq, convd, Kk);
            else          launch_gdn_conv1d_silu_decode(mixed, st.convctx[l], L.conv1d, mconv, Sq, convd, Kk);
            DK(cudaMemcpy2D(qg2,kdim*4,mconv,convd*4,kdim*4,Sq,cudaMemcpyDeviceToDevice));
            DK(cudaMemcpy2D(kg2,kdim*4,mconv+kdim,convd*4,kdim*4,Sq,cudaMemcpyDeviceToDevice));
            DK(cudaMemcpy2D(vg2,vdim*4,mconv+2*kdim,convd*4,vdim*4,Sq,cudaMemcpyDeviceToDevice));
            MMs(L.in_z,norm,xqn,xsn,zb,H,vdim);
            launch_gemv_f32(norm,L.in_a,ab,H,nvh,Sq); launch_gemv_f32(norm,L.in_b,bb,H,nvh,Sq);
            launch_gdn_decay_beta(ab,bb,L.A_log,L.dt_bias,gg,beta,Sq,nvh);
            // gdn_step from committed S, Sq steps, snapshot
            if (hv==128) {  // STEP1 GDN k-split(8.75×): qk l2norm in-place(안전,자기원소) + occupancy↑ ksplit. Sq일관 보존(결합순서 Sq-무관)
                launch_gdn_qk_norm(qg2,kg2,qg2,kg2, Sq*c.gdn_num_k_heads, hv);
                if (d_parent) launch_gdn_step_ksplit_tree<128,2>(qg2,kg2,vg2,gg,beta, st.S[l], d_parent, snap, core, Sq, nvh, c.gdn_num_k_heads);  // STEP1 tree fork
                else          launch_gdn_step_ksplit<128,2>(qg2,kg2,vg2,gg,beta, st.S[l], snap, core, Sq, nvh, c.gdn_num_k_heads, va?1:0);
            } else {
                if (d_parent) { fprintf(stderr,"[qwen35_decode] tree-verify는 GDN head_v_dim=128만(현재 %d). 조용한 오답 대신 중단.\n",hv); abort(); }
                launch_gdn_step(qg2,kg2,vg2,gg,beta, st.S[l], snap, core, Sq, nvh, hv, c.gdn_num_k_heads, va?1:0);  // fallback(D≠128)
            }
            if (!commit && va) {  // verify: per-step GDN snapshot + conv mixed → va (승격용)
                size_t ssz=(size_t)nvh*hv*hv;
                for(int t=0;t<Sq;t++)
                    DK(cudaMemcpy(va->gdn_snap + ((size_t)t*va->n_linear + lin)*ssz,
                                  snap + (size_t)t*ssz, ssz*4, cudaMemcpyDeviceToDevice));
                for(int t=0;t<Sq;t++)
                    DK(cudaMemcpy(va->conv_mixed + ((size_t)t*va->n_linear + lin)*convd,
                                  mixed + (size_t)t*convd, (size_t)convd*4, cudaMemcpyDeviceToDevice));
            }
            if (commit) {
                // 최종 state: va면 snap[Sq-1](per-token), 아니면 snap[0](snapshot_all=0 → 최종만 기록됨)
                DK(cudaMemcpy(st.S[l], snap+(size_t)(va?(Sq-1):0)*nvh*hv*hv, (size_t)nvh*hv*hv*4, cudaMemcpyDeviceToDevice));
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
        if (c.num_experts>0) {                                  // MoE 분기 (prefill과 동일 블록 공유)
            moe_mlp_block(c, L, n2, dbuf, Sq);
        } else {
            if (use_q4==2) launch_quantize_x_q8(n2, xq2, xs2, (int32_t*)xsmN, H, Sq);  // n2 1회 양자화 → gate/up 공유
            MMs(L.gate,n2,xq2,xs2,gm,H,FFN); MMs(L.up,n2,xq2,xs2,um,H,FFN);
            launch_swiglu(gm,um,act,(size_t)Sq*FFN);
            mm(L.down,act,dbuf,FFN,H,Sq,use_q4);
        }
        launch_add(x,after,dbuf,(int)SH);
    }
    if (d_prenorm) DK(cudaMemcpy(d_prenorm,x,SH*4,cudaMemcpyDeviceToDevice));
    launch_rmsnorm_zc(x,w.final_norm,fn,Sq,H,eps);
    if (d_postnorm) DK(cudaMemcpy(d_postnorm,fn,SH*4,cudaMemcpyDeviceToDevice));  // h_nextn (MTP 입력)
    if (last_logits_only == 2) { /* lm_head 생략 — prefill 비-마지막 청크(logits 불필요, KV/state만) */ }
    else if (last_logits_only) mm(w.lm_head, fn+(size_t)(Sq-1)*H, d_logits, H, V, 1, use_q4);  // 마지막 토큰만(prefill TTFT)
    else                       mm(w.lm_head, fn, d_logits, H, V, Sq, use_q4);
    if (commit) st.n_past += Sq;
    DK(cudaDeviceSynchronize());

    // scratch는 persistent (g_ws) — free 안 함. 재사용으로 per-call malloc churn 제거.
    (void)lin;
}

// verify 후 앞 m 토큰 승격. KV는 verify 중 cache[n_past..] 에 이미 기록됨 → n_past 전진만.
//   GDN: S[l] = snapshot[m-1]. conv ctx: [old_ctx(K-1) ++ conv_mixed[0..m-1]] 의 마지막 K-1.
void promote_prefix(const ModelConfig &c, DecodeState &st, const VerifyArtifacts &va, int m) {
    const int nvh=c.gdn_num_v_heads, hv=c.gdn_head_v_dim, convd=cfg_gdn_conv_dim(&c), Kk=c.gdn_conv_kernel;
    const size_t ssz=(size_t)nvh*hv*hv;
    float *cf=dm((size_t)(Kk-1+m)*convd);   // [old K-1 ; m] 임시
    int j=0;
    for(int l=0;l<c.num_layers;l++){
        if(c.layer_is_full[l]) continue;
        // GDN state 승격
        DK(cudaMemcpy(st.S[l], va.gdn_snap + ((size_t)(m-1)*va.n_linear + j)*ssz, ssz*4, cudaMemcpyDeviceToDevice));
        // conv ctx 승격: old(K-1) ++ mixed[0..m-1][j]
        DK(cudaMemcpy(cf, st.convctx[l], (size_t)(Kk-1)*convd*4, cudaMemcpyDeviceToDevice));
        for(int t=0;t<m;t++)
            DK(cudaMemcpy(cf+(size_t)(Kk-1+t)*convd, va.conv_mixed + ((size_t)t*va.n_linear + j)*convd,
                          (size_t)convd*4, cudaMemcpyDeviceToDevice));
        DK(cudaMemcpy(st.convctx[l], cf+(size_t)m*convd, (size_t)(Kk-1)*convd*4, cudaMemcpyDeviceToDevice));
        j++;
    }
    st.n_past += m;
    cudaFree(cf);
}

// 챕터9 STEP4: tree verify 후 accept된 path(노드 인덱스 path[0..plen-1], path[0]=root) 승격.
//   path는 strictly increasing(BFS layout parent<child) — KV compaction 안전(src=base+path[i]≥dst=base+i, 증가 i서 미리 읽음).
//   GDN: S[l]=snap[path末]. conv ctx: old(K-1) ++ mixed[path[0..plen-1]] 의 마지막 K-1. KV(full): slot base+path[i]→base+i.
void promote_tree_path(const ModelConfig &c, DecodeState &st, const VerifyArtifacts &va, const int *path, int plen) {
    const int nvh=c.gdn_num_v_heads, hv=c.gdn_head_v_dim, convd=cfg_gdn_conv_dim(&c), Kk=c.gdn_conv_kernel;
    const int NKV=c.num_kv_heads, HD=c.head_dim, kvdim=NKV*HD;
    const size_t ssz=(size_t)nvh*hv*hv;
    const int base=st.n_past, last=path[plen-1];
    float *cf=dm((size_t)(Kk-1+plen)*convd);
    int j=0;
    for(int l=0;l<c.num_layers;l++){
        if(c.layer_is_full[l]){
            // KV compaction: accept된 path 노드를 contiguous base..base+plen-1로. (path 증가 → src≥dst, 안전)
            for(int i=1;i<plen;i++) if(path[i]!=i){
                DK(cudaMemcpy(st.Kc[l]+(size_t)(base+i)*kvdim, st.Kc[l]+(size_t)(base+path[i])*kvdim, (size_t)kvdim*sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
                DK(cudaMemcpy(st.Vc[l]+(size_t)(base+i)*kvdim, st.Vc[l]+(size_t)(base+path[i])*kvdim, (size_t)kvdim*sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
            }
            continue;
        }
        // GDN state = path末 노드 snapshot (tree fork된 누적 state)
        DK(cudaMemcpy(st.S[l], va.gdn_snap + ((size_t)last*va.n_linear + j)*ssz, ssz*4, cudaMemcpyDeviceToDevice));
        // conv ctx: old(K-1) ++ mixed[path[0..plen-1]][j] 의 마지막 K-1
        DK(cudaMemcpy(cf, st.convctx[l], (size_t)(Kk-1)*convd*4, cudaMemcpyDeviceToDevice));
        for(int t=0;t<plen;t++)
            DK(cudaMemcpy(cf+(size_t)(Kk-1+t)*convd, va.conv_mixed + ((size_t)path[t]*va.n_linear + j)*convd,
                          (size_t)convd*4, cudaMemcpyDeviceToDevice));
        DK(cudaMemcpy(st.convctx[l], cf+(size_t)plen*convd, (size_t)(Kk-1)*convd*4, cudaMemcpyDeviceToDevice));
        j++;
    }
    st.n_past += plen;
    cudaFree(cf);
}
