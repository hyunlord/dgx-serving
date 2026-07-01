// 챕터10 Phase 1: 2-요청 batch+spec 결합 게이트 (test_batchspec2).
//   요청 2개(다른 프롬프트 A/B). 각각 tree spec 사이클(draft MTP → verify → longest-greedy accept → promote_tree_path).
//   ★verify를 qwen35_verify_batch로 batch(2슬롯 tree 한 forward, M=2×Nnodes). 각 요청 생성 토큰 ==
//     그 요청 단독 spec(test_treespec_e2e 로직)과 bit-identical(mism=0). A가 B와 batch돼도 A 불변
//     (block-diagonal 슬롯 격리 정확성). fp32(uq=0) → W4A8(uq=2) 순.
//   상태는 슬롯별 독립 DecodeState(single spec의 KV/GDN/convctx + promote_tree_path 자산 그대로). verify만 batch.
//   사용법: ./tests/test_batchspec2 [weights] [mtp_golden] [N] [K] [W] [uq]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#include "../models/qwen35_batch.h"
#include "../kernels/embedding.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/gemv_q4.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rfg(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static int* upi(const std::vector<int>&h){int*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

// ── MTP (test_treespec_e2e와 동일 스킴) ──────────────────────────────────────
struct MQ4 { uint8_t* wq=nullptr; float* sc=nullptr; };
struct MtpW { float *fc,*enorm,*hnorm,*norm,*iln,*pln,*qn,*kn,*Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd;
              int q4=0; MQ4 qfc,qWq,qWk,qWv,qWo,qWg,qWu,qWd; };
static MQ4 q4up(const std::vector<float>& w){
    size_t n=w.size(), nblk=n/32; std::vector<uint8_t> q(n); std::vector<float> scale(nblk);
    for(size_t b=0;b<nblk;b++){ float mx=0; for(int j=0;j<32;j++) mx=fmaxf(mx,fabsf(w[b*32+j]));
        float s=mx/7.0f; if(s==0.f)s=1e-8f; scale[b]=s;
        for(int j=0;j<32;j++){ long qq=lroundf(w[b*32+j]/s)+8; q[b*32+j]=(uint8_t)(qq<0?0:(qq>15?15:qq)); } }
    std::vector<uint8_t> packed(n/2);
    for(size_t i=0;i<n/2;i++) packed[i]=(uint8_t)(q[2*i] | (q[2*i+1]<<4));
    MQ4 m; CK(cudaMalloc(&m.wq,n/2)); CK(cudaMemcpy(m.wq,packed.data(),n/2,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&m.sc,nblk*4)); CK(cudaMemcpy(m.sc,scale.data(),nblk*4,cudaMemcpyHostToDevice)); return m;
}
static MtpW load_mtp(const ModelConfig&c, int mq4=0){
    int H=c.hidden_size,HD=c.head_dim,NQ=c.num_q_heads,NKV=c.num_kv_heads,FFN=c.intermediate_size;
    int qdim2=NQ*HD*2,kvdim=NKV*HD,adim=NQ*HD;
    MtpW m; m.q4=mq4;
    m.enorm=up(rfg("mtp_enorm.bin",H)); m.hnorm=up(rfg("mtp_hnorm.bin",H));
    m.norm=up(rfg("mtp_norm.bin",H)); m.iln=up(rfg("mtp_input_layernorm.bin",H)); m.pln=up(rfg("mtp_post_attention_layernorm.bin",H));
    m.qn=up(rfg("mtp_q_norm.bin",HD)); m.kn=up(rfg("mtp_k_norm.bin",HD));
    auto big=[&](const char* f, size_t OUT, size_t IN, float*& fp, MQ4& q){
        auto wv=rfg(f, OUT*IN); if(mq4) q=q4up(wv); else fp=up(wv); };
    big("mtp_fc_f32.bin",        H,     2*H,  m.fc, m.qfc);
    big("mtp_q_proj_f32.bin",    qdim2, H,    m.Wq, m.qWq);
    big("mtp_k_proj_f32.bin",    kvdim, H,    m.Wk, m.qWk);
    big("mtp_v_proj_f32.bin",    kvdim, H,    m.Wv, m.qWv);
    big("mtp_o_proj_f32.bin",    H,     adim, m.Wo, m.qWo);
    big("mtp_gate_proj_f32.bin", FFN,   H,    m.Wg, m.qWg);
    big("mtp_up_proj_f32.bin",   FFN,   H,    m.Wu, m.qWu);
    big("mtp_down_proj_f32.bin", H,     FFN,  m.Wd, m.qWd);
    return m;
}
struct MtpKV { float *Kc,*Vc; int n; };

static int mtp_step(const ModelConfig&c, const MtpW&m, MtpKV&kv, float*E,
                    int e_tok, const float*h_post, int rope_pos, bool append,
                    const float*d_cos,const float*d_sin, float*d_own, bool want_logits=true,
                    const QMat* lmq=nullptr, int dq4=0, std::vector<int>* topk=nullptr, int topw=1){
    const int H=c.hidden_size,HD=c.head_dim,NQ=c.num_q_heads,NKV=c.num_kv_heads,FFN=c.intermediate_size,rot=c.rotary_dim,V=c.vocab_size;
    const int qdim2=NQ*HD*2,qdim=NQ*HD,kvdim=NKV*HD; const float eps=c.rms_norm_eps;
    static float* g_ws=nullptr; static size_t g_cap=0; static int* g_de=nullptr;
    size_t need=(size_t)12*H + qdim2 + 4*(size_t)qdim + 3*(size_t)kvdim + 3*(size_t)FFN + V;
    if(g_cap<need){ if(g_ws)cudaFree(g_ws); CK(cudaMalloc(&g_ws,need*4)); g_cap=need; }
    if(!g_de) CK(cudaMalloc(&g_de,4));
    size_t off=0; auto B=[&](size_t n){ float* p=g_ws+off; off+=n; return p; };
    float *e=B(H),*en=B(H),*hn=B(H),*cat=B(2*H),*x=B(H),*cur=B(H);
    float *qg=B(qdim2),*qb=B(qdim),*gate=B(qdim),*qn=B(qdim),*k=B(kvdim),*kn=B(kvdim),*v=B(kvdim),*attn=B(qdim),*ob=B(H);
    float *cur2=B(H),*ff=B(H),*gm=B(FFN),*um=B(FFN),*act=B(FFN),*dnb=B(H),*cur3=B(H),*logits=B(V);
    int *de=g_de; CK(cudaMemcpy(de,&e_tok,4,cudaMemcpyHostToDevice));
    auto MM=[&](const float*X, float* fp, const MQ4& q, float*Y, int IN, int OUT){
        if(m.q4) launch_gemv_q4(X,q.wq,q.sc,Y,IN,OUT,1,0); else launch_gemv_f32(X,fp,Y,IN,OUT,1); };
    launch_embedding(de,E,e,1,H);
    launch_rmsnorm_zc(e,m.enorm,en,1,H,eps); launch_rmsnorm_zc(h_post,m.hnorm,hn,1,H,eps);
    CK(cudaMemcpy(cat,en,H*4,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(cat+H,hn,H*4,cudaMemcpyDeviceToDevice));
    MM(cat,m.fc,m.qfc,x,2*H,H);
    launch_rmsnorm_zc(x,m.iln,cur,1,H,eps);
    MM(cur,m.Wq,m.qWq,qg,H,qdim2);
    CK(cudaMemcpy2D(qb,HD*4,qg,2*HD*4,HD*4,NQ,cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy2D(gate,HD*4,qg+HD,2*HD*4,HD*4,NQ,cudaMemcpyDeviceToDevice));
    launch_rmsnorm_zc(qb,m.qn,qn,NQ,HD,eps);
    MM(cur,m.Wk,m.qWk,k,H,kvdim); launch_rmsnorm_zc(k,m.kn,kn,NKV,HD,eps);
    MM(cur,m.Wv,m.qWv,v,H,kvdim);
    launch_rope_partial(qn,d_cos+(size_t)rope_pos*rot,d_sin+(size_t)rope_pos*rot,1,NQ,HD,rot);
    launch_rope_partial(kn,d_cos+(size_t)rope_pos*rot,d_sin+(size_t)rope_pos*rot,1,NKV,HD,rot);
    CK(cudaMemcpy(kv.Kc+(size_t)kv.n*kvdim,kn,(size_t)kvdim*4,cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(kv.Vc+(size_t)kv.n*kvdim,v,(size_t)kvdim*4,cudaMemcpyDeviceToDevice));
    launch_attn_decode(qn,kv.Kc,kv.Vc,attn,1,NQ,NKV,HD,kv.n);
    launch_gate_sigmoid(attn,gate,qdim);
    MM(attn,m.Wo,m.qWo,ob,qdim,H); launch_add(cur2,x,ob,H);
    launch_rmsnorm_zc(cur2,m.pln,ff,1,H,eps);
    MM(ff,m.Wg,m.qWg,gm,H,FFN); MM(ff,m.Wu,m.qWu,um,H,FFN);
    launch_swiglu(gm,um,act,FFN); MM(act,m.Wd,m.qWd,dnb,FFN,H); launch_add(cur3,cur2,dnb,H);
    launch_rmsnorm_zc(cur3,m.norm,d_own,1,H,eps);
    int best=-1;
    if(want_logits){
        if(dq4&&lmq) launch_gemv_q4(d_own,lmq->wq,lmq->sc,logits,H,V,1,c.scale_dtype);
        else         launch_gemv_f32(d_own,E,logits,H,V,1);
        CK(cudaDeviceSynchronize());
        static std::vector<float> hl; hl.resize(V); CK(cudaMemcpy(hl.data(),logits,(size_t)V*4,cudaMemcpyDeviceToHost));
        best=0; float bv=hl[0]; for(int j=1;j<V;j++) if(hl[j]>bv){bv=hl[j];best=j;}
        if(topk){ static std::vector<int> idx; idx.resize(V); for(int j=0;j<V;j++)idx[j]=j;
            int wv=std::min(topw,V);
            std::partial_sort(idx.begin(),idx.begin()+wv,idx.end(),[&](int a,int b){return hl[a]>hl[b];});
            topk->assign(idx.begin(),idx.begin()+wv); }
    } else CK(cudaDeviceSynchronize());
    if(append) kv.n++;
    return best;
}

// 전역(테스트 컨텍스트) — draft/accept 로직 공유용.
static ModelConfig c;
static QwenWeights w; static MtpW mw; static float* E;
static float *d_cos,*d_sin;
static int H,V,rot,NKV,HD, N,K,W,Nnodes, bq4,dq4, maxlen;
static int argmaxV(const std::vector<float>&lg,int p){int b=0;float bv=lg[(size_t)p*V];for(int j=1;j<V;j++){float z=lg[(size_t)p*V+j];if(z>bv){bv=z;b=j;}}return b;}

// [곱 측정] verify forward 시간 누적(TIME=1). single(M=Nnodes 1슬롯)×2 vs batched(M=B*Nnodes) — matmul weight-load 배칭 이득.
//   warmup 첫 호출은 스킵(JIT/alloc). controlled A/B: 같은 시나리오·같은 사이클수(mism=0 보장)로 공정 비교.
static double g_t_single=0,g_t_batch=0; static long g_n_single=0,g_n_batch=0; static int g_time=0;
static cudaEvent_t g_e0,g_e1; static int g_ev_init=0;
static inline void tstart(){ if(!g_ev_init){cudaEventCreate(&g_e0);cudaEventCreate(&g_e1);g_ev_init=1;} cudaEventRecord(g_e0); }
static inline float tstop(){ cudaEventRecord(g_e1); cudaEventSynchronize(g_e1); float ms=0; cudaEventElapsedTime(&ms,g_e0,g_e1); return ms; }

// 슬롯의 draft tree(root=cur, h_cur, base) → tok/par/pos/anc(슬롯-로컬) + owns hidden. (solo/batch 공용, 동일 로직)
static void draft_tree(MtpKV& kv, int cur, float* h_cur, int base,
                       std::vector<int>&tok, std::vector<int>&par, std::vector<int>&pos, std::vector<unsigned>&anc,
                       std::vector<float*>&owns){
    tok.assign(Nnodes,0); par.assign(Nnodes,0); pos.assign(Nnodes,0); owns.clear();
    std::vector<int> top0;
    tok[0]=cur; par[0]=-1; pos[0]=base;
    { float*own; CK(cudaMalloc(&own,(size_t)H*4)); mtp_step(c,mw,kv,E,cur,h_cur,base,true,d_cos,d_sin,own,true,&w.lm_head,dq4,&top0,W); owns.push_back(own); }
    for(int j=0;j<W;j++){ tok[1+j]=top0[j]; par[1+j]=0; pos[1+j]=base+1; }
    int e_tok=top0[0]; float* h_in=owns[0]; int prev_idx=1;
    for(int i=1;i<K;i++){ float*own; CK(cudaMalloc(&own,(size_t)H*4));
        int d=mtp_step(c,mw,kv,E,e_tok,h_in,base+i,true,d_cos,d_sin,own,true,&w.lm_head,dq4); owns.push_back(own);
        int idx=W+i; tok[idx]=d; par[idx]=prev_idx; pos[idx]=base+1+i; prev_idx=idx; e_tok=d; h_in=own; }
    anc.assign(Nnodes,0);
    for(int j=0;j<Nnodes;j++){ unsigned mm=0; for(int x=j;x>=0;x=par[x]) mm|=(1u<<x); anc[j]=mm; }
}

// verify logits(vlh, [Nnodes,V]) + draft(tok/par) → longest-greedy accept path + plen + next_cur(끝 노드 argmax).
static void accept_path(const std::vector<float>&vlh, const std::vector<int>&tok, const std::vector<int>&par,
                        std::vector<int>&path, int&plen, int&node_out, int&next_cur){
    path.clear(); path.push_back(0); int node=0;
    while(true){ int g=argmaxV(vlh,node); int child=-1;
        for(int j=0;j<Nnodes;j++) if(par[j]==node && tok[j]==g){ child=j; break; }
        if(child<0) break; path.push_back(child); node=child; }
    plen=(int)path.size(); node_out=node; next_cur=argmaxV(vlh,node);
}

// accept 후 slot 승격 + MTP catch-up (solo/batch 공용). st 승격은 promote_tree_path, mtp는 catch-up.
static void promote_and_catchup(DecodeState& st, VerifyArtifacts& va, MtpKV& kv, int mn_committed, int base, int cur,
                                const std::vector<int>&tok, const std::vector<int>&path, int plen, float* h_cur,
                                const std::vector<float>&hpn, int node, const float* d_vpn, float* d_own){
    int a=plen-1;
    promote_tree_path(c,st,va,path.data(),plen);
    kv.n=mn_committed;
    mtp_step(c,mw,kv,E,cur,h_cur,base,true,d_cos,d_sin,d_own,false);
    for(int i=1;i<=a;i++){ float*ht=up(std::vector<float>(hpn.begin()+(size_t)path[i-1]*H,hpn.begin()+(size_t)(path[i-1]+1)*H));
        mtp_step(c,mw,kv,E,tok[path[i]],ht,base+i,true,d_cos,d_sin,d_own,false); cudaFree(ht); }
    CK(cudaMemcpy(h_cur,d_vpn+(size_t)node*H,(size_t)H*4,cudaMemcpyDeviceToDevice));   // 다음 root의 h_post
}

// ── 단독 spec 전체 (요청 하나) → 생성 시퀀스 (test_treespec_e2e B-루프와 동일) ──
static std::vector<int> solo_spec(const std::vector<int>& prompt){
    int P=prompt.size(); std::vector<int> out;
    DecodeState st=alloc_decode_state(c,maxlen);
    MtpKV kv; CK(cudaMalloc(&kv.Kc,(size_t)maxlen*NKV*HD*4)); CK(cudaMalloc(&kv.Vc,(size_t)maxlen*NKV*HD*4)); kv.n=0;
    int*d_tok; CK(cudaMalloc(&d_tok,maxlen*4)); CK(cudaMemcpy(d_tok,prompt.data(),P*4,cudaMemcpyHostToDevice));
    float *dl,*dpn; CK(cudaMalloc(&dl,(size_t)P*V*4)); CK(cudaMalloc(&dpn,(size_t)P*H*4));
    qwen35_decode(c,w,st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,dpn);
    std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
    int cur=argmaxV(lg,P-1);
    for(int q=1;q<P;q++){ float*ht; CK(cudaMalloc(&ht,(size_t)H*4)); mtp_step(c,mw,kv,E,prompt[q],dpn+(size_t)(q-1)*H,q,true,d_cos,d_sin,ht,false); cudaFree(ht); }
    float *h_cur; CK(cudaMalloc(&h_cur,(size_t)H*4)); CK(cudaMemcpy(h_cur,dpn+(size_t)(P-1)*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
    cudaFree(dl); cudaFree(dpn);
    VerifyArtifacts va=alloc_verify_artifacts(c,Nnodes);
    float *vl,*vpn; CK(cudaMalloc(&vl,(size_t)Nnodes*V*4)); CK(cudaMalloc(&vpn,(size_t)Nnodes*H*4));
    float *d_own; CK(cudaMalloc(&d_own,(size_t)H*4));
    std::vector<float> hpn((size_t)Nnodes*H);
    while((int)out.size()<N){
        int mn_committed=kv.n, base=st.n_past;
        std::vector<int> tok,par,pos; std::vector<unsigned> anc; std::vector<float*> owns;
        draft_tree(kv,cur,h_cur,base,tok,par,pos,anc,owns);
        CK(cudaMemcpy(d_tok,tok.data(),(size_t)Nnodes*4,cudaMemcpyHostToDevice));
        int*dpar=upi(par); int*dpos=upi(pos); unsigned*danc; CK(cudaMalloc(&danc,Nnodes*4)); CK(cudaMemcpy(danc,anc.data(),Nnodes*4,cudaMemcpyHostToDevice));
        static int wsingle=0;
        if(g_time) tstart();
        qwen35_decode(c,w,st,d_tok,Nnodes,d_cos,d_sin,vl,nullptr,0,&va,bq4,vpn,0,dpar,danc,dpos);
        if(g_time){ float ms=tstop(); if(wsingle++){ g_t_single+=ms; g_n_single++; } }   // 첫 호출 warmup 스킵
        std::vector<float> vlh((size_t)Nnodes*V); CK(cudaMemcpy(vlh.data(),vl,vlh.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hpn.data(),vpn,hpn.size()*4,cudaMemcpyDeviceToHost));
        cudaFree(dpar);cudaFree(danc);cudaFree(dpos);
        std::vector<int> path; int plen,node,next_cur; accept_path(vlh,tok,par,path,plen,node,next_cur);
        int a=plen-1;
        out.push_back(cur); for(int t=1;t<=a&&(int)out.size()<N;t++) out.push_back(tok[path[t]]);
        promote_and_catchup(st,va,kv,mn_committed,base,cur,tok,path,plen,h_cur,hpn,node,vpn,d_own);
        cur=next_cur;
        for(float*p:owns)cudaFree(p);
    }
    out.resize(N);
    free_verify_artifacts(va); cudaFree(vl);cudaFree(vpn);cudaFree(d_own);cudaFree(h_cur);cudaFree(d_tok);
    cudaFree(kv.Kc);cudaFree(kv.Vc); free_decode_state(st);
    return out;
}

// ── batch spec 전체 (2 요청) → 각 생성 시퀀스. verify만 qwen35_verify_batch(concat), 나머지 solo와 동일 ──
static std::pair<std::vector<int>,std::vector<int>> batch_spec(const std::vector<int>& pA, const std::vector<int>& pB){
    const int B=2;
    std::vector<std::vector<int>> prompt={pA,pB};
    // 슬롯별 상태
    std::vector<DecodeState> st(B); std::vector<MtpKV> kv(B); std::vector<float*> h_cur(B);
    std::vector<VerifyArtifacts> va(B); std::vector<std::vector<int>> out(B);
    std::vector<int> cur(B); std::vector<std::vector<float>> hpn(B);
    for(int b=0;b<B;b++){
        int P=prompt[b].size();
        st[b]=alloc_decode_state(c,maxlen);
        CK(cudaMalloc(&kv[b].Kc,(size_t)maxlen*NKV*HD*4)); CK(cudaMalloc(&kv[b].Vc,(size_t)maxlen*NKV*HD*4)); kv[b].n=0;
        va[b]=alloc_verify_artifacts(c,Nnodes); hpn[b].resize((size_t)Nnodes*H);
        int*d_tok; CK(cudaMalloc(&d_tok,maxlen*4)); CK(cudaMemcpy(d_tok,prompt[b].data(),P*4,cudaMemcpyHostToDevice));
        float *dl,*dpn; CK(cudaMalloc(&dl,(size_t)P*V*4)); CK(cudaMalloc(&dpn,(size_t)P*H*4));
        qwen35_decode(c,w,st[b],d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,dpn);
        std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
        cur[b]=argmaxV(lg,P-1);
        for(int q=1;q<P;q++){ float*ht; CK(cudaMalloc(&ht,(size_t)H*4)); mtp_step(c,mw,kv[b],E,prompt[b][q],dpn+(size_t)(q-1)*H,q,true,d_cos,d_sin,ht,false); cudaFree(ht); }
        CK(cudaMalloc(&h_cur[b],(size_t)H*4)); CK(cudaMemcpy(h_cur[b],dpn+(size_t)(P-1)*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
        cudaFree(dl);cudaFree(dpn);cudaFree(d_tok);
    }
    float *d_own; CK(cudaMalloc(&d_own,(size_t)H*4));
    float *vl,*vpn; CK(cudaMalloc(&vl,(size_t)B*Nnodes*V*4)); CK(cudaMalloc(&vpn,(size_t)B*Nnodes*H*4));
    int *d_tokb; CK(cudaMalloc(&d_tokb,(size_t)B*Nnodes*4));
    int *d_parb,*d_posb; unsigned *d_ancb;
    CK(cudaMalloc(&d_parb,(size_t)B*Nnodes*4)); CK(cudaMalloc(&d_posb,(size_t)B*Nnodes*4)); CK(cudaMalloc(&d_ancb,(size_t)B*Nnodes*4));
    DecodeState* sts[2]={ &st[0], &st[1] };

    // 사이클: 두 슬롯 모두 N 도달까지. (같은 Nnodes, 슬롯별 독립 전진 — Phase1은 매 사이클 두 슬롯 다 draft/verify.)
    // 슬롯 draft 상태 보관(사이클 내)
    std::vector<std::vector<int>> tokS(B),parS(B),posS(B); std::vector<std::vector<unsigned>> ancS(B);
    std::vector<std::vector<float*>> ownsS(B); std::vector<int> baseS(B), mnS(B);
    std::vector<char> done(B,0);   // out[b]>=N이면 freeze(더 이상 draft/promote 안 함 — n_past 고정, solo와 동일 종료).
    while(out[0].size()<(size_t)N || out[1].size()<(size_t)N){
        std::vector<int> tokc(B*Nnodes),parc(B*Nnodes),posc(B*Nnodes); std::vector<unsigned> ancc(B*Nnodes); std::vector<int> base_b(B);
        for(int b=0;b<B;b++){
            if(done[b]){
                // freeze: 직전 tree 재사용(base 고정, MtpKV 불변). 배치 2-wide 유지용 harmless 행. promote/collect 안 함.
                base_b[b]=baseS[b];   // 마지막 draft 시점 base 그대로(verify는 그 위치에 재기록 — 무해)
                for(int i=0;i<Nnodes;i++){ tokc[b*Nnodes+i]=tokS[b][i]; parc[b*Nnodes+i]=parS[b][i]; posc[b*Nnodes+i]=posS[b][i]; ancc[b*Nnodes+i]=ancS[b][i]; }
                continue;
            }
            mnS[b]=kv[b].n; baseS[b]=st[b].n_past; base_b[b]=st[b].n_past;
            draft_tree(kv[b],cur[b],h_cur[b],baseS[b], tokS[b],parS[b],posS[b],ancS[b], ownsS[b]);
            for(int i=0;i<Nnodes;i++){ tokc[b*Nnodes+i]=tokS[b][i]; parc[b*Nnodes+i]=parS[b][i]; posc[b*Nnodes+i]=posS[b][i]; ancc[b*Nnodes+i]=ancS[b][i]; }
        }
        CK(cudaMemcpy(d_tokb,tokc.data(),(size_t)B*Nnodes*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_parb,parc.data(),(size_t)B*Nnodes*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_posb,posc.data(),(size_t)B*Nnodes*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ancb,ancc.data(),(size_t)B*Nnodes*4,cudaMemcpyHostToDevice));
        VerifyArtifacts vab[2]={va[0],va[1]};
        // ★한 forward로 2슬롯 verify
        static int wbatch=0;
        if(g_time) tstart();
        qwen35_verify_batch(c,w,sts,d_tokb,B,Nnodes,base_b.data(),d_cos,d_sin,vl,vpn,vab,bq4,d_parb,d_ancb,d_posb);
        if(g_time){ float ms=tstop(); if(wbatch++){ g_t_batch+=ms; g_n_batch++; } }   // 첫 호출 warmup 스킵
        std::vector<float> vlh((size_t)B*Nnodes*V); CK(cudaMemcpy(vlh.data(),vl,vlh.size()*4,cudaMemcpyDeviceToHost));
        std::vector<float> vpnh((size_t)B*Nnodes*H); CK(cudaMemcpy(vpnh.data(),vpn,vpnh.size()*4,cudaMemcpyDeviceToHost));
        // 슬롯별 accept + promote + catchup (finished slot은 skip)
        for(int b=0;b<B;b++){
            if(done[b]) continue;
            // 슬롯 b logits/prenorm sub-view(행 b*Nnodes..)
            std::vector<float> vlh_b(vlh.begin()+(size_t)b*Nnodes*V, vlh.begin()+(size_t)(b+1)*Nnodes*V);
            std::copy(vpnh.begin()+(size_t)b*Nnodes*H, vpnh.begin()+(size_t)(b+1)*Nnodes*H, hpn[b].begin());
            std::vector<int> path; int plen,node,next_cur; accept_path(vlh_b,tokS[b],parS[b],path,plen,node,next_cur);
            int a=plen-1;
            out[b].push_back(cur[b]); for(int t=1;t<=a&&out[b].size()<(size_t)N;t++) out[b].push_back(tokS[b][path[t]]);
            // vpn 슬롯 b sub-buffer 포인터(h_cur 갱신용, node는 슬롯-로컬)
            float* d_vpn_b = vpn+(size_t)b*Nnodes*H;
            promote_and_catchup(st[b],va[b],kv[b],mnS[b],baseS[b],cur[b],tokS[b],path,plen,h_cur[b],hpn[b],node,d_vpn_b,d_own);
            cur[b]=next_cur;
            for(float*p:ownsS[b])cudaFree(p);
            if(out[b].size()>=(size_t)N) done[b]=1;   // N 도달 → freeze(다음 사이클부터 promote 안 함)
        }
    }
    out[0].resize(N); out[1].resize(N);
    for(int b=0;b<B;b++){ free_verify_artifacts(va[b]); cudaFree(kv[b].Kc);cudaFree(kv[b].Vc);cudaFree(h_cur[b]); free_decode_state(st[b]); }
    cudaFree(d_own);cudaFree(vl);cudaFree(vpn);cudaFree(d_tokb);cudaFree(d_parb);cudaFree(d_posb);cudaFree(d_ancb);
    return {out[0],out[1]};
}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    N=argc>3?atoi(argv[3]):24, K=argc>4?atoi(argv[4]):4, W=argc>5?atoi(argv[5]):3;
    int uq=argc>6?atoi(argv[6]):0;
    g_time=(getenv("TIME")&&atoi(getenv("TIME"))>0)?1:0;   // 곱 측정 모드
    bq4=uq; dq4=0; int mq4=0;             // body precision=uq(0 fp32/2 W4A8). draft/mtp fp32(정확성 게이트 목적).
    c=load_config_qwen((WD+"/manifest.json").c_str());
    H=c.hidden_size; V=c.vocab_size; rot=c.rotary_dim; NKV=c.num_kv_heads; HD=c.head_dim;
    std::vector<int> ids=ri("token_ids.bin");
    Nnodes=K+W;
    struct Scen{ const char*name; std::vector<int> A,B; };
    std::vector<Scen> scens;
    { std::vector<int> a(ids.begin(),ids.begin()+5), b(ids.begin()+2,ids.begin()+10);
      scens.push_back({"S1 A=5 B=8(다른내용)",a,b}); }
    { std::vector<int> a(ids.begin(),ids.begin()+7), b(ids.begin()+3,ids.begin()+7);
      scens.push_back({"S2 A=7 B=4(길이차)",a,b}); }
    { std::vector<int> a(ids.begin()+1,ids.begin()+6), b(ids.begin(),ids.begin()+6);
      scens.push_back({"S3 A=5 B=6(겹침)",a,b}); }

    printf("[batchspec2] uq=%d(0=fp32,2=W4A8) N=%d K=%d W=%d Nnodes=%d | 시나리오 %zu개\n",uq,N,K,W,Nnodes,scens.size());
    if(Nnodes>8) printf("  주의 nodes=%d>8 → W4A8 verify가 MMQ 경로(단독도 MMQ라 일치). ≤8 권장.\n",Nnodes);
    maxlen=0; for(auto&s:scens) maxlen=std::max(maxlen,(int)std::max(s.A.size(),s.B.size())); maxlen += N+Nnodes+8;
    if(maxlen>=128){ printf("need longer rope table (maxlen=%d)\n",maxlen); return 1; }

    std::vector<float> cl=rfg("rope_cos_long.bin",(size_t)128*rot), sl=rfg("rope_sin_long.bin",(size_t)128*rot);
    d_cos=up(cl); d_sin=up(sl);
    w=load_weights_qwen(WD.c_str(),c, 1, bq4==0?1:0);
    mw=load_mtp(c,mq4); E=w.embed;

    int total_mism=0;
    for(size_t si=0; si<scens.size(); si++){
        auto&s=scens[si];
        std::vector<int> soloA=solo_spec(s.A), soloB=solo_spec(s.B);
        auto pr=batch_spec(s.A,s.B);
        std::vector<int>&batA=pr.first,&batB=pr.second;
        int mmA=0,mmB=0;
        for(int i=0;i<N;i++){ if(i>=(int)batA.size()||batA[i]!=soloA[i]) mmA++; if(i>=(int)batB.size()||batB[i]!=soloB[i]) mmB++; }
        printf("\n--- %s ---\n",s.name);
        printf("  A solo : "); for(int i=0;i<12&&i<N;i++)printf("%d ",soloA[i]); printf("\n");
        printf("  A batch: "); for(int i=0;i<12&&i<N;i++)printf("%d ",batA[i]); printf("\n");
        printf("  B solo : "); for(int i=0;i<12&&i<N;i++)printf("%d ",soloB[i]); printf("\n");
        printf("  B batch: "); for(int i=0;i<12&&i<N;i++)printf("%d ",batB[i]); printf("\n");
        printf("  mism A=%d/%d  B=%d/%d\n",mmA,N,mmB,N);
        total_mism += mmA+mmB;
    }
    bool pass=(total_mism==0);
    printf("\n[test_batchspec2] uq=%d total_mism=%d → %s\n",uq,total_mism,pass?"PASS":"FAIL");
    if(g_time && g_n_single>0 && g_n_batch>0){
        const int Bg=2;                                            // Phase 1 = 고정 2요청
        double as=g_t_single/g_n_single, ab=g_t_batch/g_n_batch;   // single=1슬롯 forward, batch=B슬롯 1 forward
        double prod=(double)Bg*as/ab;                              // 곱: single B번 vs batched 1번의 forward 시간비
        printf("[곱 측정] verify forward 평균: single(M=%d,1슬롯)=%.3f ms(n=%ld), batched(M=%d,%d슬롯)=%.3f ms(n=%ld)\n",
               Nnodes,as,g_n_single,Bg*Nnodes,Bg,ab,g_n_batch);
        printf("[곱 측정] %d×single/batched = %.3f× (>1=곱 실현: weight 1운반이 %d슬롯 서빙, matmul 배칭 이득)\n",Bg,prod,Bg);
    }
    free_weights_qwen(w); free_config_qwen(c);
    cudaFree(d_cos);cudaFree(d_sin);
    return pass?0:1;
}
