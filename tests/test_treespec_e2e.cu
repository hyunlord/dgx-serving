// 챕터9 STEP 4: tree speculative 생성 루프 + e2e (payoff).
//   하이브리드 tree: root(cur) + MTP chain(d1..dK, top-1) + 첫 위치 top-W 대안(a2..aW = leaf, 추가 MTP forward 0).
//     → chain을 strictly dominate(body가 +1서 #2/#3 고르는 경우 복구; chain은 0 accept). W=1이면 chain spec과 동일(sanity).
//   1-forward verify(STEP1~3: GDN fork+conv fork+attn mask+rope-depth) → longest-greedy accept → promote_tree_path.
//   정확성: tree-spec 생성 == plain greedy bit-identical(mism=0). 측정: accept(chain 1.73→?)·t/s.
//   사용법: ./tests/test_treespec_e2e [weights] [mtp_golden] [N] [K] [W] [bq4] [dq4]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
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

struct MQ4 { uint8_t* wq=nullptr; float* sc=nullptr; };
struct MtpW { float *fc,*enorm,*hnorm,*norm,*iln,*pln,*qn,*kn,*Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd;
              int q4=0; MQ4 qfc,qWq,qWk,qWv,qWo,qWg,qWu,qWd; };
// STEP5: q4 host 양자화 — extract_weights.quantize_int4 복제(QBLK=32, scale=maxabs/7, q=clip(round(w/s)+8,0,15), pack 2/byte).
//   IN%32==0이라 블록이 row 내부 — gemv_q4(scale_dtype=0=fp32 scale)와 포맷 일치. body q4 경로와 동일 스킴.
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
    // 큰 matmul: mq4면 q4 양자화(draft 대역폭↓), 아니면 fp32
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

// MTP 1-스텝: (e_tok,h_post) → top-1 draft + own hidden(d_own). topk!=null이면 top-topw 후보(host).
static int mtp_step(const ModelConfig&c, const MtpW&m, MtpKV&kv, float*E,
                    int e_tok, const float*h_post, int rope_pos, bool append,
                    const float*d_cos,const float*d_sin, float*d_own, bool want_logits=true,
                    const QMat* lmq=nullptr, int dq4=0, std::vector<int>* topk=nullptr, int topw=1){
    const int H=c.hidden_size,HD=c.head_dim,NQ=c.num_q_heads,NKV=c.num_kv_heads,FFN=c.intermediate_size,rot=c.rotary_dim,V=c.vocab_size;
    const int qdim2=NQ*HD*2,qdim=NQ*HD,kvdim=NKV*HD; const float eps=c.rms_norm_eps;
    // STEP5: persistent scratch — 호출당 ~48 cudaMalloc/free 제거(body qwen35_decode와 동일 패턴). 출력 불변(순수 perf).
    static float* g_ws=nullptr; static size_t g_cap=0; static int* g_de=nullptr;
    size_t need=(size_t)12*H + qdim2 + 4*(size_t)qdim + 3*(size_t)kvdim + 3*(size_t)FFN + V;
    if(g_cap<need){ if(g_ws)cudaFree(g_ws); CK(cudaMalloc(&g_ws,need*4)); g_cap=need; }
    if(!g_de) CK(cudaMalloc(&g_de,4));
    size_t off=0; auto B=[&](size_t n){ float* p=g_ws+off; off+=n; return p; };
    float *e=B(H),*en=B(H),*hn=B(H),*cat=B(2*H),*x=B(H),*cur=B(H);
    float *qg=B(qdim2),*qb=B(qdim),*gate=B(qdim),*qn=B(qdim),*k=B(kvdim),*kn=B(kvdim),*v=B(kvdim),*attn=B(qdim),*ob=B(H);
    float *cur2=B(H),*ff=B(H),*gm=B(FFN),*um=B(FFN),*act=B(FFN),*dnb=B(H),*cur3=B(H),*logits=B(V);
    int *de=g_de; CK(cudaMemcpy(de,&e_tok,4,cudaMemcpyHostToDevice));
    // STEP5: q4면 gemv_q4(draft 대역폭↓, MTP body 1.7GB→215MB), 아니면 fp32. scale_dtype=0(fp32 scale).
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
    return best;   // persistent scratch — free 불요(다음 호출 재사용)
}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int N=argc>3?atoi(argv[3]):40, K=argc>4?atoi(argv[4]):4, W=argc>5?atoi(argv[5]):3, bq4=argc>6?atoi(argv[6]):0, dq4=argc>7?atoi(argv[7]):0, mq4=argc>8?atoi(argv[8]):0;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size,rot=c.rotary_dim,NKV=c.num_kv_heads,HD=c.head_dim;
    std::vector<int> prompt=ri("token_ids.bin"); int P=prompt.size();
    int Nnodes=K+W;                                  // root(1)+depth1(W)+chain(K-1) = K+W
    int maxlen=P+N+Nnodes+8;
    printf("[treespec] prompt=%d gen=%d K(chain)=%d W(width)=%d nodes=%d body_q4=%d draft_q4=%d mtp_q4=%d\n",P,N,K,W,Nnodes,bq4,dq4,mq4);
    if(Nnodes>8) printf("  ⚠ nodes=%d>8 → verify가 MMQ 경로(amortize 함정, STEP3.3). ≤8 권장.\n",Nnodes);
    std::vector<float> cl=rfg("rope_cos_long.bin",(size_t)128*rot), sl=rfg("rope_sin_long.bin",(size_t)128*rot);
    if(P+N+Nnodes>=128){printf("need longer rope table\n");return 1;}
    float *d_cos=up(cl),*d_sin=up(sl);
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1, bq4==0?1:0);
    MtpW mw=load_mtp(c,mq4); float*E=w.embed;
    int *d_tok; CK(cudaMalloc(&d_tok,maxlen*4));
    auto argmaxV=[&](const std::vector<float>&lg,int p){int b=0;float bv=lg[(size_t)p*V];for(int j=1;j<V;j++){float z=lg[(size_t)p*V+j];if(z>bv){bv=z;b=j;}}return b;};

    // ---------- (A) plain greedy ----------
    std::vector<int> out_plain;
    {
        DecodeState st=alloc_decode_state(c,maxlen);
        CK(cudaMemcpy(d_tok,prompt.data(),P*4,cudaMemcpyHostToDevice));
        float *dl,*dpn; CK(cudaMalloc(&dl,(size_t)P*V*4)); CK(cudaMalloc(&dpn,(size_t)P*H*4));
        qwen35_decode(c,w,st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,dpn);
        std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
        int cur=argmaxV(lg,P-1);
        clock_t t0=clock();
        for(int i=0;i<N;i++){ out_plain.push_back(cur);
            CK(cudaMemcpy(d_tok,&cur,4,cudaMemcpyHostToDevice));
            float*o; CK(cudaMalloc(&o,(size_t)V*4));
            qwen35_decode(c,w,st,d_tok,1,d_cos,d_sin,o,nullptr,1,nullptr,bq4,nullptr);
            std::vector<float> hl(V); CK(cudaMemcpy(hl.data(),o,(size_t)V*4,cudaMemcpyDeviceToHost)); cudaFree(o);
            int b=0;float bv=hl[0];for(int j=1;j<V;j++)if(hl[j]>bv){bv=hl[j];b=j;} cur=b;
        }
        double sec=(double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("[plain] %d tokens in %.3fs = %.1f t/s\n",N,sec,N/sec);
        cudaFree(dl);cudaFree(dpn);free_decode_state(st);
    }

    // ---------- (B) tree speculative ----------
    std::vector<int> out_spec; int total_acc=0, cycles=0, alt_hits=0;
    {
        DecodeState st=alloc_decode_state(c,maxlen);
        MtpKV kv; CK(cudaMalloc(&kv.Kc,(size_t)maxlen*NKV*HD*4)); CK(cudaMalloc(&kv.Vc,(size_t)maxlen*NKV*HD*4)); kv.n=0;
        CK(cudaMemcpy(d_tok,prompt.data(),P*4,cudaMemcpyHostToDevice));
        float *dl,*dpn; CK(cudaMalloc(&dl,(size_t)P*V*4)); CK(cudaMalloc(&dpn,(size_t)P*H*4));
        qwen35_decode(c,w,st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,dpn);
        std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
        int cur=argmaxV(lg,P-1);
        float *d_htmp; CK(cudaMalloc(&d_htmp,(size_t)H*4));
        for(int q=1;q<P;q++) mtp_step(c,mw,kv,E, prompt[q], dpn+(size_t)(q-1)*H, q, true, d_cos,d_sin, d_htmp, false);
        float *h_cur; CK(cudaMalloc(&h_cur,(size_t)H*4));
        CK(cudaMemcpy(h_cur,dpn+(size_t)(P-1)*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
        cudaFree(dl);cudaFree(dpn);
        VerifyArtifacts va=alloc_verify_artifacts(c,Nnodes);
        float *vl,*vpn; CK(cudaMalloc(&vl,(size_t)Nnodes*V*4)); CK(cudaMalloc(&vpn,(size_t)Nnodes*H*4));
        float *d_own; CK(cudaMalloc(&d_own,(size_t)H*4));
        std::vector<float> hpn((size_t)Nnodes*H);

        double t_draft=0,t_verify=0,t_catch=0;
        clock_t t0=clock();
        while((int)out_spec.size()<N){
            clock_t cs=clock();
            int mn_committed=kv.n, base=st.n_past;
            // ---- tree draft: chain(d1..dK,top-1) + 첫위치 top-W(대안 leaf) ----
            std::vector<int> tok(Nnodes), par(Nnodes), pos(Nnodes); std::vector<float*> owns;
            std::vector<int> top0;
            tok[0]=cur; par[0]=-1; pos[0]=base;
            { float* own; CK(cudaMalloc(&own,(size_t)H*4));
              mtp_step(c,mw,kv,E,cur,h_cur,base,true,d_cos,d_sin,own,true,&w.lm_head,dq4,&top0,W); owns.push_back(own); }
            for(int j=0;j<W;j++){ tok[1+j]=top0[j]; par[1+j]=0; pos[1+j]=base+1; }   // depth1: d1 + 대안
            int e_tok=top0[0]; float* h_in=owns[0]; int prev_idx=1;                  // d1=node1
            for(int i=1;i<K;i++){
                float* own; CK(cudaMalloc(&own,(size_t)H*4));
                int d=mtp_step(c,mw,kv,E,e_tok,h_in,base+i,true,d_cos,d_sin,own,true,&w.lm_head,dq4); owns.push_back(own);
                int idx=W+i; tok[idx]=d; par[idx]=prev_idx; pos[idx]=base+1+i; prev_idx=idx; e_tok=d; h_in=own;
            }
            std::vector<unsigned> anc(Nnodes,0);
            for(int j=0;j<Nnodes;j++){ unsigned mm=0; for(int x=j;x>=0;x=par[x]) mm|=(1u<<x); anc[j]=mm; }
            CK(cudaDeviceSynchronize()); t_draft+=(double)(clock()-cs)/CLOCKS_PER_SEC; cs=clock();

            // ---- 1-forward verify ----
            CK(cudaMemcpy(d_tok,tok.data(),(size_t)Nnodes*4,cudaMemcpyHostToDevice));
            int *dpar=upi(par); int *dpos=upi(pos);
            unsigned *danc; CK(cudaMalloc(&danc,Nnodes*4)); CK(cudaMemcpy(danc,anc.data(),Nnodes*4,cudaMemcpyHostToDevice));
            qwen35_decode(c,w,st,d_tok,Nnodes,d_cos,d_sin,vl,nullptr,0,&va,bq4,vpn, 0, dpar,danc,dpos);
            std::vector<float> vlh((size_t)Nnodes*V); CK(cudaMemcpy(vlh.data(),vl,vlh.size()*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hpn.data(),vpn,hpn.size()*4,cudaMemcpyDeviceToHost));
            cudaFree(dpar);cudaFree(danc);cudaFree(dpos);

            // ---- longest-greedy accept ----
            std::vector<int> path; path.push_back(0); int node=0;
            while(true){ int g=argmaxV(vlh,node); int child=-1;
                for(int j=0;j<Nnodes;j++) if(par[j]==node && tok[j]==g){ child=j; break; }
                if(child<0) break; path.push_back(child); node=child; }
            int a=(int)path.size()-1, plen=a+1;
            int next_cur=argmaxV(vlh,node);
            if(a>=1 && path[1]!=1) alt_hits++;
            out_spec.push_back(cur); for(int t=1;t<=a&&(int)out_spec.size()<N;t++) out_spec.push_back(tok[path[t]]);
            total_acc+=a; cycles++;
            CK(cudaDeviceSynchronize()); t_verify+=(double)(clock()-cs)/CLOCKS_PER_SEC; cs=clock();

            // ---- promote accepted path + MTP catch-up ----
            promote_tree_path(c,st,va,path.data(),plen);
            kv.n=mn_committed;
            mtp_step(c,mw,kv,E,cur,h_cur,base,true,d_cos,d_sin,d_own,false);
            for(int i=1;i<=a;i++){ float*ht=up(std::vector<float>(hpn.begin()+(size_t)path[i-1]*H,hpn.begin()+(size_t)(path[i-1]+1)*H));
                mtp_step(c,mw,kv,E,tok[path[i]],ht,base+i,true,d_cos,d_sin,d_own,false); cudaFree(ht); }
            cur=next_cur;
            CK(cudaMemcpy(h_cur,vpn+(size_t)node*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
            for(float*p:owns)cudaFree(p);
            CK(cudaDeviceSynchronize()); t_catch+=(double)(clock()-cs)/CLOCKS_PER_SEC;
        }
        double sec=(double)(clock()-t0)/CLOCKS_PER_SEC;
        out_spec.resize(N);
        printf("[tree]  %d tokens in %.3fs = %.1f t/s | cycles=%d avg_accept=%.2f alt_hits=%d (K=%d W=%d)\n",
               N,sec,N/sec,cycles,(double)total_acc/cycles,alt_hits,K,W);
        printf("[tree]  phase: draft=%.0f%% verify=%.0f%% catchup+promote=%.0f%%\n",100*t_draft/sec,100*t_verify/sec,100*t_catch/sec);
        free_verify_artifacts(va); cudaFree(vl);cudaFree(vpn);cudaFree(d_own);cudaFree(h_cur);cudaFree(d_htmp);
        cudaFree(kv.Kc);cudaFree(kv.Vc); free_decode_state(st);
    }

    // ---------- 정확성: tree-spec == plain ----------
    int mism=0; for(int i=0;i<N;i++) if(out_plain[i]!=out_spec[i]) mism++;
    printf("\n=== tree-spec vs plain greedy (deterministic) ===\n");
    printf("  first 12 plain: "); for(int i=0;i<12&&i<N;i++)printf("%d ",out_plain[i]); printf("\n");
    printf("  first 12 tree : "); for(int i=0;i<12&&i<N;i++)printf("%d ",out_spec[i]); printf("\n");
    printf("  mismatches: %d / %d\n",mism,N);
    free_weights_qwen(w); free_config_qwen(c);
    cudaFree(d_cos);cudaFree(d_sin);cudaFree(d_tok);
    bool pass=(mism==0);
    printf("\n[test_treespec_e2e] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
