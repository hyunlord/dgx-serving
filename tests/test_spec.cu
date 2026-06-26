// 2c STEP 4: speculative 생성 루프 + t/s 측정.
//   정확성: spec on 출력 == off(일반 greedy decode) 출력 (deterministic, 완전 일치).
//   draft = MTP head (자체 KV + prefix catch-up + chain). accept = draft가 본체 greedy와 일치한 비율.
//   2b 페어링: 토큰 T(직전) + hidden(=T를 만든 본체 post-norm) → T 다음 토큰 예측. h=post-norm(h_nextn).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <ctime>
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
#include "../kernels/argmax.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rfg(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

// ---- MTP weights (fp32, mtp_golden dir) ----
struct MtpW { float *fc,*enorm,*hnorm,*norm,*iln,*pln,*qn,*kn,*Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd; };
static MtpW load_mtp(const ModelConfig&c){
    int H=c.hidden_size,HD=c.head_dim,NQ=c.num_q_heads,NKV=c.num_kv_heads,FFN=c.intermediate_size;
    int qdim2=NQ*HD*2,kvdim=NKV*HD,adim=NQ*HD;
    MtpW m;
    m.fc=up(rfg("mtp_fc_f32.bin",(size_t)H*2*H)); m.enorm=up(rfg("mtp_enorm.bin",H)); m.hnorm=up(rfg("mtp_hnorm.bin",H));
    m.norm=up(rfg("mtp_norm.bin",H)); m.iln=up(rfg("mtp_input_layernorm.bin",H)); m.pln=up(rfg("mtp_post_attention_layernorm.bin",H));
    m.qn=up(rfg("mtp_q_norm.bin",HD)); m.kn=up(rfg("mtp_k_norm.bin",HD));
    m.Wq=up(rfg("mtp_q_proj_f32.bin",(size_t)qdim2*H)); m.Wk=up(rfg("mtp_k_proj_f32.bin",(size_t)kvdim*H));
    m.Wv=up(rfg("mtp_v_proj_f32.bin",(size_t)kvdim*H)); m.Wo=up(rfg("mtp_o_proj_f32.bin",(size_t)H*adim));
    m.Wg=up(rfg("mtp_gate_proj_f32.bin",(size_t)FFN*H)); m.Wu=up(rfg("mtp_up_proj_f32.bin",(size_t)FFN*H)); m.Wd=up(rfg("mtp_down_proj_f32.bin",(size_t)H*FFN));
    return m;
}
struct MtpKV { float *Kc,*Vc; int n; };

// MTP 1-스텝: (e_tok, h_post) → draft(argmax), own post-norm hidden(d_own[H]). append이면 KV에 추가.
// want_logits=false(catch-up): lm_head/argmax/host복사 생략 (KV+own hidden만 필요). draft 안 씀.
static int* g_dmax=nullptr;   // persistent argmax 출력 (per-call malloc 회피)
static int mtp_step(const ModelConfig&c, const MtpW&m, MtpKV&kv, float*E,
                    int e_tok, const float*h_post, int rope_pos, bool append,
                    const float*d_cos,const float*d_sin, float*d_own, bool want_logits=true,
                    const QMat* lmq=nullptr, int dq4=0){
    const int H=c.hidden_size,HD=c.head_dim,NQ=c.num_q_heads,NKV=c.num_kv_heads,FFN=c.intermediate_size,rot=c.rotary_dim,V=c.vocab_size;
    const int qdim2=NQ*HD*2,qdim=NQ*HD,kvdim=NKV*HD; const float eps=c.rms_norm_eps;
    auto B=[&](size_t n){float*d;CK(cudaMalloc(&d,n*4));return d;};
    float *e=B(H),*en=B(H),*hn=B(H),*cat=B(2*H),*x=B(H),*cur=B(H);
    float *qg=B(qdim2),*qb=B(qdim),*gate=B(qdim),*qn=B(qdim),*k=B(kvdim),*kn=B(kvdim),*v=B(kvdim),*attn=B(qdim),*ob=B(H);
    float *cur2=B(H),*ff=B(H),*gm=B(FFN),*um=B(FFN),*act=B(FFN),*dn=B(H),*cur3=B(H),*logits=B(V);
    int *de; CK(cudaMalloc(&de,4)); CK(cudaMemcpy(de,&e_tok,4,cudaMemcpyHostToDevice));
    launch_embedding(de,E,e,1,H);
    launch_rmsnorm_zc(e,m.enorm,en,1,H,eps); launch_rmsnorm_zc(h_post,m.hnorm,hn,1,H,eps);
    CK(cudaMemcpy(cat,en,H*4,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(cat+H,hn,H*4,cudaMemcpyDeviceToDevice));
    launch_gemv_f32(cat,m.fc,x,2*H,H,1);
    launch_rmsnorm_zc(x,m.iln,cur,1,H,eps);
    launch_gemv_f32(cur,m.Wq,qg,H,qdim2,1);
    CK(cudaMemcpy2D(qb,HD*4,qg,2*HD*4,HD*4,NQ,cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy2D(gate,HD*4,qg+HD,2*HD*4,HD*4,NQ,cudaMemcpyDeviceToDevice));
    launch_rmsnorm_zc(qb,m.qn,qn,NQ,HD,eps);
    launch_gemv_f32(cur,m.Wk,k,H,kvdim,1); launch_rmsnorm_zc(k,m.kn,kn,NKV,HD,eps);
    launch_gemv_f32(cur,m.Wv,v,H,kvdim,1);
    launch_rope_partial(qn,d_cos+(size_t)rope_pos*rot,d_sin+(size_t)rope_pos*rot,1,NQ,HD,rot);
    launch_rope_partial(kn,d_cos+(size_t)rope_pos*rot,d_sin+(size_t)rope_pos*rot,1,NKV,HD,rot);
    // append K/V at slot kv.n, attend 0..kv.n
    CK(cudaMemcpy(kv.Kc+(size_t)kv.n*kvdim,kn,(size_t)kvdim*4,cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(kv.Vc+(size_t)kv.n*kvdim,v,(size_t)kvdim*4,cudaMemcpyDeviceToDevice));
    launch_attn_decode(qn,kv.Kc,kv.Vc,attn,1,NQ,NKV,HD,kv.n);
    launch_gate_sigmoid(attn,gate,qdim);
    launch_gemv_f32(attn,m.Wo,ob,qdim,H,1); launch_add(cur2,x,ob,H);
    launch_rmsnorm_zc(cur2,m.pln,ff,1,H,eps);
    launch_gemv_f32(ff,m.Wg,gm,H,FFN,1); launch_gemv_f32(ff,m.Wu,um,H,FFN,1);
    launch_swiglu(gm,um,act,FFN); launch_gemv_f32(act,m.Wd,dn,FFN,H,1); launch_add(cur3,cur2,dn,H);
    launch_rmsnorm_zc(cur3,m.norm,d_own,1,H,eps);      // own post-norm (다음 step h)
    int best=-1;
    if(want_logits){
        if(dq4&&lmq) launch_gemv_q4(d_own,lmq->wq,lmq->sc,logits,H,V,1);  // draft lm_head q4 (tied=body lm_head q4, 2a 검증)
        else         launch_gemv_f32(d_own,E,logits,H,V,1);               // lm_head tie (fp32)
        // B1(device argmax) 측정 결과 역효과(-2%) — draft 비용은 lm_head gemv지 argmax 아님(nsys gemv 90%).
        // host argmax 유지 (DMA+CPU루프가 단일블록 GPU argmax보다 빠름).
        CK(cudaDeviceSynchronize());
        std::vector<float> hl(V); CK(cudaMemcpy(hl.data(),logits,(size_t)V*4,cudaMemcpyDeviceToHost));
        best=0; float bv=hl[0]; for(int j=1;j<V;j++) if(hl[j]>bv){bv=hl[j];best=j;}
    } else CK(cudaDeviceSynchronize());
    if(append) kv.n++;
    float* all[]={e,en,hn,cat,x,cur,qg,qb,gate,qn,k,kn,v,attn,ob,cur2,ff,gm,um,act,dn,cur3,logits};
    for(float*p:all)cudaFree(p); cudaFree(de);
    return best;
}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int N=argc>3?atoi(argv[3]):40, K=argc>4?atoi(argv[4]):4, bq4=argc>5?atoi(argv[5]):0, dq4=argc>6?atoi(argv[6]):0;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size,rot=c.rotary_dim,NKV=c.num_kv_heads,HD=c.head_dim;
    std::vector<int> prompt=ri("token_ids.bin"); int P=prompt.size();
    int maxlen=P+N+K+8;
    printf("[spec] prompt=%d gen=%d K=%d maxlen=%d body_q4=%d draft_q4=%d\n",P,N,K,maxlen,bq4,dq4);
    std::vector<float> cl=rfg("rope_cos_long.bin",(size_t)128*rot), sl=rfg("rope_sin_long.bin",(size_t)128*rot);
    if(P+N+K>=128){printf("need longer rope table\n");return 1;}
    float *d_cos=up(cl),*d_sin=up(sl);
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,1);  // q4+fp32 둘 다 (use_q4 토글)
    MtpW mw=load_mtp(c); float*E=w.embed;
    int *d_tok; CK(cudaMalloc(&d_tok,maxlen*4));

    auto argmaxV=[&](const std::vector<float>&lg,int p){int b=0;float bv=lg[(size_t)p*V];for(int j=1;j<V;j++){float z=lg[(size_t)p*V+j];if(z>bv){bv=z;b=j;}}return b;};

    // ---------- (A) plain greedy decode ----------
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

    // ---------- (B) speculative ----------
    std::vector<int> out_spec; int total_acc=0, cycles=0;
    {
        DecodeState st=alloc_decode_state(c,maxlen);
        MtpKV kv; CK(cudaMalloc(&kv.Kc,(size_t)maxlen*NKV*HD*4)); CK(cudaMalloc(&kv.Vc,(size_t)maxlen*NKV*HD*4)); kv.n=0;
        CK(cudaMemcpy(d_tok,prompt.data(),P*4,cudaMemcpyHostToDevice));
        float *dl,*dpn; CK(cudaMalloc(&dl,(size_t)P*V*4)); CK(cudaMalloc(&dpn,(size_t)P*H*4));
        qwen35_decode(c,w,st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,dpn);  // prefill, commit
        std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
        int cur=argmaxV(lg,P-1);
        // MTP catch-up over prompt: entry q (1..P-1): (e=token[q], h=Hpost[q-1], pos=q)
        float *d_htmp; CK(cudaMalloc(&d_htmp,(size_t)H*4));   // own-hidden sink (catchup 무시)
        for(int q=1;q<P;q++) mtp_step(c,mw,kv,E, prompt[q], dpn+(size_t)(q-1)*H, q, true, d_cos,d_sin, d_htmp, false);
        float *h_cur; CK(cudaMalloc(&h_cur,(size_t)H*4));
        CK(cudaMemcpy(h_cur,dpn+(size_t)(P-1)*H,(size_t)H*4,cudaMemcpyDeviceToDevice)); // hidden that produced cur
        cudaFree(dl);cudaFree(dpn);
        VerifyArtifacts va=alloc_verify_artifacts(c,K+1);
        float *vl,*vpn; CK(cudaMalloc(&vl,(size_t)(K+1)*V*4)); CK(cudaMalloc(&vpn,(size_t)(K+1)*H*4));
        float *d_own; CK(cudaMalloc(&d_own,(size_t)H*4));
        std::vector<float> hpn((size_t)(K+1)*H);

        double t_draft=0,t_verify=0,t_catch=0;
        clock_t t0=clock();
        while((int)out_spec.size()<N){
            clock_t cs=clock();
            int mn_committed=kv.n;
            // draft chain: e=cur, h=h_cur, pos=n_past
            std::vector<int> drafts; std::vector<float*> owns;
            int e_tok=cur; float* h_in=h_cur; int pos=st.n_past;
            for(int i=0;i<K;i++){
                float* own; CK(cudaMalloc(&own,(size_t)H*4));
                int d=mtp_step(c,mw,kv,E,e_tok,h_in,pos,true,d_cos,d_sin,own,true,&w.lm_head,dq4);
                drafts.push_back(d); owns.push_back(own);
                e_tok=d; h_in=own; pos++;
            }
            CK(cudaDeviceSynchronize()); t_draft+=(double)(clock()-cs)/CLOCKS_PER_SEC; cs=clock();
            // verify [cur, drafts] at positions n_past..n_past+K
            std::vector<int> vt; vt.push_back(cur); for(int d:drafts) vt.push_back(d);
            CK(cudaMemcpy(d_tok,vt.data(),(K+1)*4,cudaMemcpyHostToDevice));
            qwen35_decode(c,w,st,d_tok,K+1,d_cos,d_sin,vl,nullptr,/*commit*/0,&va,bq4,vpn);
            std::vector<float> vlh((size_t)(K+1)*V); CK(cudaMemcpy(vlh.data(),vl,vlh.size()*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hpn.data(),vpn,hpn.size()*4,cudaMemcpyDeviceToHost));
            // accept: drafts[i]==argmax(verify[i]) longest prefix
            int a=0; while(a<K){ int va_i=argmaxV(vlh,a); if(drafts[a]==va_i) a++; else break; }
            // emit cur + drafts[0..a-1], next cur=argmax(verify[a])
            out_spec.push_back(cur); for(int t=0;t<a&&(int)out_spec.size()<N;t++) out_spec.push_back(drafts[t]);
            total_acc+=a; cycles++;
            int next_cur=argmaxV(vlh,a);
            CK(cudaDeviceSynchronize()); t_verify+=(double)(clock()-cs)/CLOCKS_PER_SEC; cs=clock();
            // promote body state by a+1 tokens
            promote_prefix(c,st,va,a+1);
            // MTP catch-up: truncate to committed, rebuild a+1 committed entries with body hiddens
            kv.n=mn_committed;
            // token at pos (n_past_before=pos0): cur with h=h_cur(=Hpost prev) ; drafts[t] with h=verify Hpost[t]
            int pos0=st.n_past-(a+1);
            mtp_step(c,mw,kv,E,cur,h_cur,pos0,true,d_cos,d_sin,d_own,false);
            for(int t=0;t<a;t++){ float*ht=up(std::vector<float>(hpn.begin()+(size_t)t*H,hpn.begin()+(size_t)(t+1)*H));
                mtp_step(c,mw,kv,E,drafts[t],ht,pos0+1+t,true,d_cos,d_sin,d_own,false); cudaFree(ht); }
            // next cur/h
            cur=next_cur;
            CK(cudaMemcpy(h_cur,vpn+(size_t)a*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
            for(float*p:owns)cudaFree(p);
            CK(cudaDeviceSynchronize()); t_catch+=(double)(clock()-cs)/CLOCKS_PER_SEC;
        }
        double sec=(double)(clock()-t0)/CLOCKS_PER_SEC;
        out_spec.resize(N);
        printf("[spec]  %d tokens in %.3fs = %.1f t/s | cycles=%d avg_accept=%.2f (K=%d)\n",
               N,sec,N/sec,cycles,(double)total_acc/cycles,K);
        printf("[spec]  phase breakdown: draft(MTP chain)=%.0f%% verify(body batch)=%.0f%% catchup+promote=%.0f%%\n",
               100*t_draft/sec,100*t_verify/sec,100*t_catch/sec);
        free_verify_artifacts(va); cudaFree(vl);cudaFree(vpn);cudaFree(d_own);cudaFree(h_cur);cudaFree(d_htmp);
        cudaFree(kv.Kc);cudaFree(kv.Vc); free_decode_state(st);
    }

    // ---------- 정확성: spec == plain ----------
    int mism=0; for(int i=0;i<N;i++) if(out_plain[i]!=out_spec[i]) mism++;
    printf("\n=== spec vs plain greedy (deterministic) ===\n");
    printf("  first 12 plain: "); for(int i=0;i<12&&i<N;i++)printf("%d ",out_plain[i]); printf("\n");
    printf("  first 12 spec : "); for(int i=0;i<12&&i<N;i++)printf("%d ",out_spec[i]); printf("\n");
    printf("  mismatches: %d / %d\n",mism,N);
    printf("  PLAINTOK(bq4=%d):",bq4); for(int i=0;i<N;i++)printf(" %d",out_plain[i]); printf("\n");

    free_weights_qwen(w); free_config_qwen(c);
    cudaFree(d_cos);cudaFree(d_sin);cudaFree(d_tok);
    bool pass=(mism==0);
    printf("\n[test_spec] %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}
