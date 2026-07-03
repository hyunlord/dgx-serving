#include "qwen35_batch.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/decode_state.h"   // 챕터9 chunked prefill: qwen35_decode(single Sq>1) + DecodeState view 재사용
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemm_batch.cuh"
#include "../kernels/gemv_w4a8.cuh"   // 챕터9: B≤8 batch decode에 SB 커널(odd-B 1적재)
#include "../kernels/gemm_mmq.cuh"    // 챕터9: BM=16 split-K 배치 GEMM(B=9~16 load-latency wall, stream-K few-blocks 해결)
#include "../kernels/gemm_tiled.cuh"  // 챕터10 verify_batch: use_q4==1(plain q4) row-독립 GEMM 경로
#include "../kernels/gemv_f32.cuh"
#include "../kernels/quantize_x.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/attention_flash.cuh"  // FA.2: batch도 single과 같은 attention(정밀도 일치)
#include "../kernels/swiglu.cuh"
#include "moe_block.cuh"   // MoE batch: moe_mlp_block(S=B) — prefill/single decode 공유 블록(naive S×top_k 루프)
#include "../kernels/elementwise.cuh"
#include "../kernels/gdn.cuh"

#define BK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
static float *dmb(size_t n){ float*p; BK(cudaMalloc(&p,n*4)); return p; }
static __nv_bfloat16 *dmbh(size_t n){ __nv_bfloat16*p; BK(cudaMalloc(&p,n*sizeof(__nv_bfloat16))); return p; }  // bf16 KV cache (메모리 절반)
static GdnStateT *dmbs(size_t n){ GdnStateT*p; BK(cudaMalloc(&p,n*sizeof(GdnStateT))); return p; }  // GDN state (fp32/bf16 by GDN_BF16_STATE)

// ---- batch state alloc/free ----
BatchDecodeState alloc_batch_decode_state(const ModelConfig &c, int B, int max_len){
    BatchDecodeState s{}; s.B=B; s.max_len=max_len; s.n_past=0; s.L=c.num_layers;
    int L=c.num_layers; const int HD=c.head_dim, NKV=c.num_kv_heads;
    const int nh=c.gdn_num_v_heads, D=c.gdn_head_v_dim, convd=cfg_gdn_conv_dim(&c), K=c.gdn_conv_kernel;
    s.Kc=(__nv_bfloat16**)calloc(L,sizeof(__nv_bfloat16*)); s.Vc=(__nv_bfloat16**)calloc(L,sizeof(__nv_bfloat16*));
    s.S=(GdnStateT**)calloc(L,sizeof(GdnStateT*)); s.convctx=(float**)calloc(L,sizeof(float*));
    for(int l=0;l<L;l++){
        if(c.layer_is_full[l]){ s.Kc[l]=dmbh((size_t)B*max_len*NKV*HD); s.Vc[l]=dmbh((size_t)B*max_len*NKV*HD); }  // bf16
        else { s.S[l]=dmbs((size_t)B*nh*D*D); s.convctx[l]=dmb((size_t)B*(K-1)*convd); }  // S=GdnStateT
    }
    reset_batch_decode_state(c,s); return s;
}
void reset_batch_decode_state(const ModelConfig &c, BatchDecodeState &s){
    s.n_past=0; int L=c.num_layers; const int nh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),K=c.gdn_conv_kernel;
    for(int l=0;l<L;l++) if(!c.layer_is_full[l]){ BK(cudaMemset(s.S[l],0,(size_t)s.B*nh*D*D*sizeof(GdnStateT))); BK(cudaMemset(s.convctx[l],0,(size_t)s.B*(K-1)*convd*4)); }
}
// 챕터9 연속배칭: slot 재사용 시 그 물리 slot의 GDN 잔재만 clear(새 요청이 이전 요청 state 안 읽음 — compaction 버그 교훈).
//   KV는 clear 불요(새 요청이 pos0부터 덮고 attention은 [0..pos]만 읽음). GDN S/convctx는 recurrence라 반드시 zero.
void reset_batch_slot(const ModelConfig &c, BatchDecodeState &s, int b){
    int L=c.num_layers; const int nh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),K=c.gdn_conv_kernel;
    const size_t ssz=(size_t)nh*D*D, csz=(size_t)(K-1)*convd;
    for(int l=0;l<L;l++) if(!c.layer_is_full[l]){
        BK(cudaMemset(s.S[l]+(size_t)b*ssz,0,ssz*sizeof(GdnStateT))); BK(cudaMemset(s.convctx[l]+(size_t)b*csz,0,csz*4)); }
}
// 챕터9 chunked prefill: slot b에 프롬프트 P토큰을 청크(≤chunk_max)로 병렬 prefill(1토큰/step 대신 — probe_chunked_prefill 62×).
//   batch slot b의 KV/GDN 서브영역([B,...]의 b번째=single과 동일 연속 레이아웃)에 alias한 DecodeState view로 기존
//   qwen35_decode(Sq>1=full-MMQ 병렬) 재사용. device 메모리 공유, host 포인터 배열만 임시(free_decode_state 호출 금지).
//   호출 후 slot b는 KV[0..P-1]+GDN state 채워짐 → pos=P부터 batch decode 합류. d_logit_out=마지막 청크 로짓([lastSq,V],
//   마지막 행=pos P-1=첫 gen 예측). 커널 재사용(새 커널 아님). mism: chunked==1토큰 prefill(같은 causal, fp32 bit-identical).
void prefill_batch_slot(const ModelConfig &c, const QwenWeights &w, BatchDecodeState &st, int b,
                        const int *d_tokens, int P, const float *d_cos, const float *d_sin,
                        float *d_logit_out, int use_q4, int chunk_max) {
    int L=c.num_layers; const int HD=c.head_dim, NKV=c.num_kv_heads;
    const int nh=c.gdn_num_v_heads, D=c.gdn_head_v_dim, convd=cfg_gdn_conv_dim(&c), K=c.gdn_conv_kernel;
    std::vector<__nv_bfloat16*> Kc(L,nullptr),Vc(L,nullptr); std::vector<GdnStateT*> S(L,nullptr); std::vector<float*> Cx(L,nullptr);
    for(int l=0;l<L;l++){
        if(c.layer_is_full[l]){ Kc[l]=st.Kc[l]+(size_t)b*st.max_len*NKV*HD; Vc[l]=st.Vc[l]+(size_t)b*st.max_len*NKV*HD; }
        else { S[l]=st.S[l]+(size_t)b*nh*D*D; Cx[l]=st.convctx[l]+(size_t)b*(K-1)*convd; }
    }
    DecodeState v{}; v.n_past=0; v.max_len=st.max_len; v.L=L; v.Kc=Kc.data(); v.Vc=Vc.data(); v.S=S.data(); v.convctx=Cx.data();
    if(chunk_max<=0) chunk_max=512;
    for(int off=0; off<P; off+=chunk_max){ int sq = (P-off<chunk_max)?(P-off):chunk_max;
        qwen35_decode(c,w,v, d_tokens+off, sq, d_cos, d_sin, d_logit_out, nullptr, 1, nullptr, use_q4); }
    // v(host 벡터)는 스코프 종료 시 해제 — device 메모리(batch slot)는 st 소유라 건드리지 않음.
}
void free_batch_decode_state(BatchDecodeState &s){
    auto F=[](void*p){ if(p)cudaFree(p); };
    for(int l=0;l<s.L;l++){ F(s.Kc[l]);F(s.Vc[l]);F(s.S[l]);F(s.convctx[l]); }
    free(s.Kc);free(s.Vc);free(s.S);free(s.convctx); s.Kc=s.Vc=nullptr; s.S=nullptr; s.convctx=nullptr;
}

// persistent batch scratch
static std::vector<float*> g_bws; static std::vector<size_t> g_bwscap;
static float *bws(int &idx, size_t n){
    if((int)g_bws.size()<=idx){ g_bws.push_back(nullptr); g_bwscap.push_back(0); }
    if(g_bwscap[idx]<n){ if(g_bws[idx])cudaFree(g_bws[idx]); BK(cudaMalloc(&g_bws[idx],n*4)); g_bwscap[idx]=n; }
    return g_bws[idx++];
}

void qwen35_decode_batch(const ModelConfig &c, const QwenWeights &w, BatchDecodeState &st,
                         const int *d_tokens, const float *d_cos, const float *d_sin, float *d_logits, int use_q4,
                         const int *h_pos_b){
    const int H=c.hidden_size,FFN=c.intermediate_size,HD=c.head_dim,V=c.vocab_size;
    const int NQ=c.num_q_heads,NKV=c.num_kv_heads,rot=c.rotary_dim; const float eps=c.rms_norm_eps;
    const int qdim2=NQ*HD*(c.attn_output_gate?2:1), qdim=NQ*HD, kvdim=NKV*HD;
    const int convd=cfg_gdn_conv_dim(&c),vdim=cfg_gdn_value_dim(&c),kdim=cfg_gdn_key_dim(&c);
    const int nvh=c.gdn_num_v_heads,hv=c.gdn_head_v_dim,Kk=c.gdn_conv_kernel;
    const int B=st.B, base=st.n_past, ML=st.max_len;
    const size_t BH=(size_t)B*H;
    const long kvstride=(long)ML*kvdim;
    // ragged(연속배칭): h_pos_b[b]=seq b 위치. d_pos에 업로드(rope per-seq). null이면 same-length(모두 base).
    static int *g_dpos=nullptr; static int g_dposcap=0;
    if(h_pos_b){ if(g_dposcap<B){ if(g_dpos)cudaFree(g_dpos); BK(cudaMalloc(&g_dpos,B*4)); g_dposcap=B; }
                 BK(cudaMemcpy(g_dpos,h_pos_b,B*4,cudaMemcpyHostToDevice)); }
    auto POS=[&](int b){ return h_pos_b ? h_pos_b[b] : base; };   // seq b의 현재 위치

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
            // 챕터9 batch serving: B≤15는 SB 커널(런타임 S, 1적재 — gemm_batch의 비특수화 default 루프[B× 재적재] 회피).
            //   SB는 SMAX dispatch(S≤8=SMAX8 occ100% / 9~15=SMAX16 occ83%, de-risk probe_sb16: S=12 gemm_batch 대비 5.85×).
            //   B=16은 gemm_batch<16> 특수화가 SB16보다 빠름(340 vs 398us) → 유지. bit-exact(SB16==gemm_batch 모든 S).
            // 챕터9 stream-K BM=16 split-K(few-blocks 해결): in-engine WIN(27B +13.9% tok/s·TPOT −34%, 0.8b +4.6%, argmax==single).
            //   단 opt-in(기본 off): split-K(MMA)는 sb(DP4A)와 다른 커널 → W4A8 borderline flip으로 SB 계열의 bit-identical-across-B
            //   속성 깸(test_cbatch_prod got==iso). argmax는 정확. env BM16=1로 켜서 throughput↑(borderline flip trade 감수). B≤8 sb8.
            static int USE_BM16 = getenv("BM16") ? atoi(getenv("BM16")) : 0;
            if(B<=8)                       launch_gemv_w4a8_sb(g_bxq,g_bxs,m.wq,m.sc,Y,IN,OUT,B,sdtype);
            else if(USE_BM16 && B>=14)     launch_gemm_bm16_splitk(g_bxq,g_bxs,m.wq,m.sc,Y,B,OUT,IN,sdtype);  // B14-16만: BM=16 타일 낭비(16−B)/16 < few-blocks 이득(측정: B≤13 SB가 빠름, B14 +4.7%/B16 +3.5%). M=B rows, split-K
            else if(B<=15)                 launch_gemv_w4a8_sb(g_bxq,g_bxs,m.wq,m.sc,Y,IN,OUT,B,sdtype);
            else                           launch_gemm_batch_w4a8(g_bxq,g_bxs,m.wq,m.sc,Y,IN,OUT,B,sdtype);   // 챕터4 W4A8 lane-minor(16 특수화)
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
            if(h_pos_b){  // ragged: seq별 위치 g_dpos (rope_partial_pos = per-row 위치, STEP3 자산). row=b.
                launch_rope_partial_pos(qn,d_cos,d_sin,g_dpos,B,NQ,HD,rot);
                launch_rope_partial_pos(kn,d_cos,d_sin,g_dpos,B,NKV,HD,rot);
            } else {
                launch_rope_partial_batch(qn,d_cos+(size_t)base*rot,d_sin+(size_t)base*rot,B,NQ,HD,rot);
                launch_rope_partial_batch(kn,d_cos+(size_t)base*rot,d_sin+(size_t)base*rot,B,NKV,HD,rot);
            }
            // 각 시퀀스 b: K[b]/V[b]를 cache[b]의 위치 POS(b)에 append (시퀀스별 stride, fp32→bf16 변환)
            for (int b=0; b<B; b++) {
                launch_f32_to_bf16(kn+(size_t)b*kvdim, st.Kc[l]+(size_t)b*kvstride+(size_t)POS(b)*kvdim, (size_t)kvdim);
                launch_f32_to_bf16(vb+(size_t)b*kvdim, st.Vc[l]+(size_t)b*kvstride+(size_t)POS(b)*kvdim, (size_t)kvdim);
            }
            // FA.2 통일: 시퀀스별 KV(kvstride 간격)에 FA.2(single과 동일). decode Sq=1, attend 0..POS(b).
            for (int b=0; b<B; b++)
                launch_flash_attn(qn+(size_t)b*qdim, st.Kc[l]+(size_t)b*kvstride, st.Vc[l]+(size_t)b*kvstride,
                                  attn+(size_t)b*qdim, 1, NQ, NKV, HD, POS(b));
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
        if(c.num_experts>0){                      // MoE 분기(qwen35.cu:122·qwen35_decode.cu:211과 동형, config 주도, ③=0)
            moe_mlp_block(c,L,n2,dbuf,B);          // naive S=B×top_k 루프(정확성 우선 — 그룹핑은 성능 판단 후)
        } else {
            mmb(L.gate,n2,gm,H,FFN); mmb(L.up,n2,um,H,FFN);
            launch_swiglu(gm,um,act,(size_t)B*FFN);
            mmb(L.down,act,dbuf,FFN,H);
        }
        launch_add(x,after,dbuf,(int)BH);
    }
    launch_rmsnorm_zc(x,w.final_norm,fn,B,H,eps);
    mmb(w.lm_head,fn,d_logits,H,V);
    if(!h_pos_b) st.n_past += 1;   // ragged면 per-seq 위치는 호출자(scheduler)가 관리
    BK(cudaDeviceSynchronize());
}

// ─────────────────────────────────────────────────────────────────────────────
// 챕터10 Phase 1: batched tree verify (qwen35_verify_batch)
//   qwen35_decode(single tree verify, models/qwen35_decode.cu:103) 본체를 슬롯 차원으로 확장.
//   행 레이아웃 concat: 행 r = b*Nnodes + i (슬롯 b 노드 i). 총 M = B*Nnodes.
//   row-독립(norm/rope/act/matmul)은 concat 행 그대로; attention/GDN은 슬롯별 루프(+tree mask/fork).
//   ★W4A8 bit-identical: row-독립 matmul을 슬롯별(M=Nnodes) 호출 — 단독 spec(Nnodes)과 동일 커널 선택
//     (sb S≤8). concat M=B*Nnodes 한 호출은 sb→mmq 커널을 바꿔 borderline flip → mism≠0(honest-perf 트랩).
//     fp32(launch_gemv_f32)는 M-무관 행-독립이라 concat 한 호출로 처리(타일 story). quantize_x_q8는 per-row라
//     concat/슬롯별 bit-identical.
static std::vector<float*> g_vbws; static std::vector<size_t> g_vbwscap;
static float *vbws(int &idx, size_t n){
    if((int)g_vbws.size()<=idx){ g_vbws.push_back(nullptr); g_vbwscap.push_back(0); }
    if(g_vbwscap[idx]<n){ if(g_vbws[idx])cudaFree(g_vbws[idx]); BK(cudaMalloc(&g_vbws[idx],n*4)); g_vbwscap[idx]=n; }
    return g_vbws[idx++];
}
static std::vector<int8_t*> g_vbwsb; static std::vector<size_t> g_vbwsbcap;
static int8_t *vbwsi8(int &idx, size_t n){
    if((int)g_vbwsb.size()<=idx){ g_vbwsb.push_back(nullptr); g_vbwsbcap.push_back(0); }
    if(g_vbwsbcap[idx]<n){ if(g_vbwsb[idx])cudaFree(g_vbwsb[idx]); BK(cudaMalloc(&g_vbwsb[idx],n)); g_vbwsbcap[idx]=n; }
    return g_vbwsb[idx++];
}

void qwen35_verify_batch(const ModelConfig &c, const QwenWeights &w, DecodeState **sts,
                         const int *d_tokens_b, int B, const int *Nnodes_b, const int *base_b,
                         const float *d_cos, const float *d_sin,
                         float *d_logits, float *d_hnextn, VerifyArtifacts *va, int use_q4,
                         const int *d_parent_b, const unsigned *d_anc_b, const int *d_pos_b) {
    const int H=c.hidden_size, FFN=c.intermediate_size, HD=c.head_dim, V=c.vocab_size;
    const int NQ=c.num_q_heads, NKV=c.num_kv_heads, rot=c.rotary_dim; const float eps=c.rms_norm_eps;
    const int qdim2=NQ*HD*(c.attn_output_gate?2:1), qdim=NQ*HD, kvdim=NKV*HD;
    const int convd=cfg_gdn_conv_dim(&c), vdim=cfg_gdn_value_dim(&c), kdim=cfg_gdn_key_dim(&c);
    const int nvh=c.gdn_num_v_heads, hv=c.gdn_head_v_dim, Kk=c.gdn_conv_kernel;
    // ragged(Phase3 축2): 슬롯별 Nnodes_b[b] + prefix-offset off[b]=ΣNnodes_b[0..b-1]. uniform은 off[b]=b*Nnodes의 일반화.
    std::vector<int> off(B+1,0);
    for(int b=0;b<B;b++) off[b+1]=off[b]+Nnodes_b[b];
    const int M=off[B];                   // concat 행수 = ΣNnodes_b
    // canonical 게이팅(축1 결정, ragged 일반화): 전 슬롯 같은 커널 family여야 concat이 단독 spec과 bit-identical.
    bool all_gt8=true, all_2to8=true;
    for(int b=0;b<B;b++){ if(Nnodes_b[b]<=8) all_gt8=false; if(Nnodes_b[b]<2||Nnodes_b[b]>8) all_2to8=false; }
    const size_t MH=(size_t)M*H;
    const int sdtype=c.scale_dtype;

    int wi=0;
    float *x=vbws(wi,MH),*norm=vbws(wi,MH),*after=vbws(wi,MH),*n2=vbws(wi,MH);
    float *qg=vbws(wi,(size_t)M*qdim2),*qb=vbws(wi,(size_t)M*qdim),*gate=vbws(wi,(size_t)M*qdim),*qn=vbws(wi,(size_t)M*qdim);
    float *kb=vbws(wi,(size_t)M*kvdim),*kn=vbws(wi,(size_t)M*kvdim),*vb=vbws(wi,(size_t)M*kvdim),*attn=vbws(wi,(size_t)M*qdim),*ob=vbws(wi,MH);
    float *mixed=vbws(wi,(size_t)M*convd),*mconv=vbws(wi,(size_t)M*convd);
    float *qg2=vbws(wi,(size_t)M*kdim),*kg2=vbws(wi,(size_t)M*kdim),*vg2=vbws(wi,(size_t)M*vdim);
    float *zb=vbws(wi,(size_t)M*vdim),*ab=vbws(wi,(size_t)M*nvh),*bb=vbws(wi,(size_t)M*nvh),*gg=vbws(wi,(size_t)M*nvh),*beta=vbws(wi,(size_t)M*nvh);
    float *core=vbws(wi,(size_t)M*vdim),*normed=vbws(wi,(size_t)M*vdim),*gout=vbws(wi,MH);
    float *gm=vbws(wi,(size_t)M*FFN),*um=vbws(wi,(size_t)M*FFN),*act=vbws(wi,(size_t)M*FFN),*dbuf=vbws(wi,MH),*fn=vbws(wi,MH);
    float *snap=vbws(wi,(size_t)M*nvh*hv*hv);          // per-node GDN snapshot (슬롯 b 노드 t → 행 b*Nnodes+t; 슬롯별 sub-buffer)

    // W4A8 shared-X quantize 버퍼: norm/n2([M,H])를 1회 양자화(per-row) → 슬롯별 gemv서 offset 재사용.
    int bi=0; int8_t *xqn=nullptr,*xq2=nullptr; float *xsn=nullptr,*xs2=nullptr; int32_t *xsmN=nullptr;
    if(use_q4==2){ xqn=vbwsi8(bi,MH); xq2=vbwsi8(bi,MH); xsn=vbws(wi,MH/QBLK); xs2=vbws(wi,MH/QBLK); xsmN=(int32_t*)vbws(wi,MH/QBLK); }

    // 행-독립 matmul. fp32=concat 한 호출(M). W4A8=슬롯별(M=Nnodes) 호출로 단독 spec 커널 선택 일치(bit-identical).
    //   xqs/xss: 미리 양자화한 [M,H] 버퍼(use_q4==2). X: fp32 소스.
    auto MMrow=[&](const QMat&m,const float*X,const int8_t*xqs,const float*xss,float*Y,int IN,int OUT){
        if(use_q4==2){
            if(all_gt8){
                // ★곱: 전 슬롯 Nnodes>8이면 단독 spec(S=Nnodes_b)도 mmq → concat mmq(M=ΣNnodes_b) 1콜이 bit-identical
                //   (mmq row-독립: tools/probe_mmq_rowdep + in-context golden dump로 concat[slot]==solo max|d|=0.0 확정).
                //   weight 1운반이 B슬롯 전부 서빙(per-slot은 슬롯마다 weight 재적재). matmul 배칭 이득 = 곱의 원천.
                launch_gemm_mmq_w4a8(xqs,xss,m.wq,m.sc,Y,M,OUT,IN,sdtype);
            } else if(all_2to8 && M<=16){
                // ★곱(≤8 트리, Phase3 축1): concat-sb(M) 1콜. sb는 grid=OUT warps(S무관, weight 1적재)+컬럼-독립
                //   → per-slot sb(B콜=B적재)와 bit-identical하되 weight B적재→1 = 곱. 단독 spec(2≤S≤8)도 sb라 canonical 일치
                //   (mism=0, borderline flip 없음). sb SMAX=16 한계라 M≤16.
                launch_gemv_w4a8_sb(xqs,xss,m.wq,m.sc,Y,IN,OUT,M,sdtype);
            } else {
                // 혼합(슬롯별 커널 family 다름)·M>16(sb 범위 밖)·Nnodes==1 포함: per-slot로 각 슬롯 단독 커널과 일치(bit-identical, 곱 부분 손실).
                for(int b=0;b<B;b++){
                    const int Nb=Nnodes_b[b];
                    const int8_t *xb=xqs+(size_t)off[b]*IN; const float *sb=xss+(size_t)off[b]*IN/QBLK;
                    float *Yb=Y+(size_t)off[b]*OUT;
                    if      (Nb==1) launch_gemv_w4a8_mlp<2>(xb,sb,nullptr,m.wq,m.sc,Yb,IN,OUT,Nb,sdtype);
                    else if (Nb<=8) launch_gemv_w4a8_sb(xb,sb,m.wq,m.sc,Yb,IN,OUT,Nb,sdtype);
                    else            launch_gemm_mmq_w4a8(xb,sb,m.wq,m.sc,Yb,Nb,OUT,IN,sdtype);
                }
            }
        } else if(use_q4){ launch_gemm_tiled_q4(X,m.wq,m.sc,Y,M,OUT,IN,sdtype); }
        else             { launch_gemv_f32(X,m.wf,Y,IN,OUT,M); }
    };
    // o_proj/out_p/lm_head도 행-독립. 단독 spec은 mm()(quantize+dispatch) — 여기선 슬롯별 quantize+dispatch로 동일.
    auto MMrow_q=[&](const QMat&m,const float*X,float*Y,int IN,int OUT){
        if(use_q4==2){
            int8_t *xq=vbwsi8(bi,(size_t)M*IN); float *xs=vbws(wi,(size_t)M*IN/QBLK); int32_t*xsm=(int32_t*)vbws(wi,(size_t)M*IN/QBLK);
            launch_quantize_x_q8(X,xq,xs,xsm,IN,M);            // per-row 양자화(M행) — 슬롯별 offset과 bit-identical
            if(all_gt8){
                launch_gemm_mmq_w4a8(xq,xs,m.wq,m.sc,Y,M,OUT,IN,sdtype);   // ★곱: concat mmq(M) — 단독 mmq(Nnodes_b)와 row-독립 bit-identical
            } else if(all_2to8 && M<=16){
                launch_gemv_w4a8_sb(xq,xs,m.wq,m.sc,Y,IN,OUT,M,sdtype);     // ★곱(≤8): concat-sb(M) — 단독 sb(Nnodes_b)와 컬럼-독립 bit-identical, weight 1적재
            } else {
                for(int b=0;b<B;b++){
                    const int Nb=Nnodes_b[b];
                    const int8_t *xb=xq+(size_t)off[b]*IN; const float *sb=xs+(size_t)off[b]*IN/QBLK;
                    float *Yb=Y+(size_t)off[b]*OUT;
                    if      (Nb==1) launch_gemv_w4a8_mlp<2>(xb,sb,xsm+(size_t)off[b]*IN/QBLK,m.wq,m.sc,Yb,IN,OUT,Nb,sdtype);
                    else if (Nb<=8) launch_gemv_w4a8_sb(xb,sb,m.wq,m.sc,Yb,IN,OUT,Nb,sdtype);
                    else            launch_gemm_mmq_w4a8(xb,sb,m.wq,m.sc,Yb,Nb,OUT,IN,sdtype);
                }
            }
        } else if(use_q4){ launch_gemm_tiled_q4(X,m.wq,m.sc,Y,M,OUT,IN,sdtype); }
        else             { launch_gemv_f32(X,m.wf,Y,IN,OUT,M); }
    };

    launch_embedding(d_tokens_b, w.embed, x, M, H);
    int lin=0;
    for(int l=0;l<c.num_layers;l++){
        const QLayer &L=w.layers[l];
        launch_rmsnorm_zc(x,L.input_ln,norm,M,H,eps);
        if(use_q4==2) launch_quantize_x_q8(norm,xqn,xsn,xsmN,H,M);   // norm 1회 per-row 양자화 → q/k/v(또는 in_qkv/in_z) 공유
        if(L.is_full){
            MMrow(L.q,norm,xqn,xsn,qg,H,qdim2);
            BK(cudaMemcpy2D(qb,HD*4,qg,2*HD*4,HD*4,(size_t)M*NQ,cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy2D(gate,HD*4,qg+HD,2*HD*4,HD*4,(size_t)M*NQ,cudaMemcpyDeviceToDevice));
            launch_rmsnorm_zc(qb,L.q_norm,qn,M*NQ,HD,eps);
            MMrow(L.k,norm,xqn,xsn,kb,H,kvdim); launch_rmsnorm_zc(kb,L.k_norm,kn,M*NKV,HD,eps);
            MMrow(L.v,norm,xqn,xsn,vb,H,kvdim);
            // rope: 노드별 위치 d_pos_b(base+depth) — 행-독립, concat 그대로.
            launch_rope_partial_pos(qn,d_cos,d_sin,d_pos_b,M,NQ,HD,rot);
            launch_rope_partial_pos(kn,d_cos,d_sin,d_pos_b,M,NKV,HD,rot);
            // 슬롯별: K/V[슬롯 b 노드 i] → sts[b]->Kc/Vc slot (base_b[b]+i). 단독 spec과 동일(node index, position 아님). ragged=off[b]행.
            for(int b=0;b<B;b++){
                launch_f32_to_bf16(kn+(size_t)off[b]*kvdim, sts[b]->Kc[l]+(size_t)base_b[b]*kvdim, (size_t)Nnodes_b[b]*kvdim);
                launch_f32_to_bf16(vb+(size_t)off[b]*kvdim, sts[b]->Vc[l]+(size_t)base_b[b]*kvdim, (size_t)Nnodes_b[b]*kvdim);
            }
            // 슬롯별 flash_attn: 슬롯 b 노드(Sq=Nnodes_b[b])는 sts[b]->Kc/Vc의 [0..base_b[b]+Nnodes_b[b]) + tree mask(d_anc_b[off[b]..]).
            for(int b=0;b<B;b++)
                launch_flash_attn(qn+(size_t)off[b]*qdim, sts[b]->Kc[l], sts[b]->Vc[l],
                                  attn+(size_t)off[b]*qdim, Nnodes_b[b], NQ, NKV, HD, base_b[b], 0, d_anc_b+(size_t)off[b]);
            launch_gate_sigmoid(attn, gate, (int)((size_t)M*qdim));
            MMrow_q(L.o,attn,ob,qdim,H);
            launch_add(after,x,ob,(int)MH);
        } else {
            MMrow(L.in_qkv,norm,xqn,xsn,mixed,H,convd);
            // conv tree-fork: 슬롯 b 노드는 슬롯-로컬 parent(d_parent_b[off[b]..]) + 슬롯 b convctx(sts[b]).
            for(int b=0;b<B;b++)
                launch_gdn_conv1d_silu_decode_tree(mixed+(size_t)off[b]*convd, sts[b]->convctx[l],
                                                   d_parent_b+(size_t)off[b], L.conv1d, mconv+(size_t)off[b]*convd, Nnodes_b[b], convd, Kk);
            BK(cudaMemcpy2D(qg2,kdim*4,mconv,convd*4,kdim*4,M,cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy2D(kg2,kdim*4,mconv+kdim,convd*4,kdim*4,M,cudaMemcpyDeviceToDevice));
            BK(cudaMemcpy2D(vg2,vdim*4,mconv+2*kdim,convd*4,vdim*4,M,cudaMemcpyDeviceToDevice));
            MMrow(L.in_z,norm,xqn,xsn,zb,H,vdim);
            launch_gemv_f32(norm,L.in_a,ab,H,nvh,M); launch_gemv_f32(norm,L.in_b,bb,H,nvh,M);
            launch_gdn_decay_beta(ab,bb,L.A_log,L.dt_bias,gg,beta,M,nvh);
            if(hv==128){
                launch_gdn_qk_norm(qg2,kg2,qg2,kg2, M*c.gdn_num_k_heads, hv);   // 행-독립(row=노드×k-head), concat 그대로
                // 슬롯별 tree-fork: 슬롯 b 노드는 sts[b]->S서 fork(parent 슬롯-로컬), snap도 슬롯별 sub-buffer(자식이 부모 read).
                for(int b=0;b<B;b++)
                    launch_gdn_step_ksplit_tree<128,2>(qg2+(size_t)off[b]*kdim, kg2+(size_t)off[b]*kdim, vg2+(size_t)off[b]*vdim,
                                                       gg+(size_t)off[b]*nvh, beta+(size_t)off[b]*nvh, sts[b]->S[l],
                                                       d_parent_b+(size_t)off[b], snap+(size_t)off[b]*nvh*hv*hv, core+(size_t)off[b]*vdim,
                                                       Nnodes_b[b], nvh, c.gdn_num_k_heads);
            } else {
                fprintf(stderr,"[qwen35_verify_batch] tree-verify는 GDN head_v_dim=128만(현재 %d). 조용한 오답 대신 중단.\n",hv); abort();
            }
            // verify snapshot → va[b] (슬롯별). 슬롯 b 노드 t = snap 행 off[b]+t → va[b].gdn_snap[(t*n_linear+lin)].
            if(va){
                size_t ssz=(size_t)nvh*hv*hv;
                for(int b=0;b<B;b++) for(int t=0;t<Nnodes_b[b];t++)
                    BK(cudaMemcpy(va[b].gdn_snap + ((size_t)t*va[b].n_linear + lin)*ssz,
                                  snap + (size_t)(off[b]+t)*ssz, ssz*4, cudaMemcpyDeviceToDevice));
                for(int b=0;b<B;b++) for(int t=0;t<Nnodes_b[b];t++)
                    BK(cudaMemcpy(va[b].conv_mixed + ((size_t)t*va[b].n_linear + lin)*convd,
                                  mixed + (size_t)(off[b]+t)*convd, (size_t)convd*4, cudaMemcpyDeviceToDevice));
            }
            launch_rms_norm_gated(core,zb,L.gdn_norm,normed,M*nvh,hv,eps);
            MMrow_q(L.out_p,normed,gout,vdim,H);
            launch_add(after,x,gout,(int)MH);
            lin++;
        }
        launch_rmsnorm_zc(after,L.post_ln,n2,M,H,eps);
        if(use_q4==2) launch_quantize_x_q8(n2,xq2,xs2,xsmN,H,M);   // n2 1회 per-row 양자화 → gate/up 공유
        MMrow(L.gate,n2,xq2,xs2,gm,H,FFN); MMrow(L.up,n2,xq2,xs2,um,H,FFN);
        launch_swiglu(gm,um,act,(size_t)M*FFN);
        MMrow_q(L.down,act,dbuf,FFN,H);
        launch_add(x,after,dbuf,(int)MH);
    }
    launch_rmsnorm_zc(x,w.final_norm,fn,M,H,eps);
    // d_hnextn = postnorm(final-norm 적용 후) = MTP draft 입력 h_nextn. single spec qwen35_decode의 d_postnorm과 동일 역할
    //   (test_treespec_e2e/mtp_step이 h_post로 쓰는 값). ★prenorm(x) 아님 — 초기 오작(prenorm 출력)이 draft 오염 → mism 원인.
    if(d_hnextn) BK(cudaMemcpy(d_hnextn,fn,MH*4,cudaMemcpyDeviceToDevice));
    MMrow_q(w.lm_head,fn,d_logits,H,V);
    // commit 안 함(verify) — n_past 불변. promote는 호스트가 슬롯별 promote_tree_path.
    BK(cudaDeviceSynchronize());
    (void)lin;
}
