// 챕터9 STEP 6.0a: full multi-level forking tree accept 천장 de-risk (per-depth rank).
//   de-risk(test_treespec)는 depth-1만(top-4 0.949=즉시 다음). full forking은 각 depth fork → per-depth rank 필요.
//   질문: MTP가 깊이 갈수록 자기-conditioning(own hidden)으로 degrade하나? per-depth P(rank<k) 측정 →
//     full top-w tree depth D accept = Σ_{d=1}^D Π_{i=1}^d P(rank<w @ depth i). ≤8(or 16/32)노드 내 천장 4-5 가능?
//   teacher-forced chain: true 토큰 먹이되 d≥2는 MTP own hidden(=tree 수용 path의 conditioning과 동일). 측정만(mism 무관).
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

struct MtpW { float *fc,*enorm,*hnorm,*norm,*iln,*pln,*qn,*kn,*Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd; };
static MtpW load_mtp(const ModelConfig&c){
    int H=c.hidden_size,HD=c.head_dim,NQ=c.num_q_heads,NKV=c.num_kv_heads,FFN=c.intermediate_size;
    int qdim2=NQ*HD*2,kvdim=NKV*HD,adim=NQ*HD; MtpW m;
    m.fc=up(rfg("mtp_fc_f32.bin",(size_t)H*2*H)); m.enorm=up(rfg("mtp_enorm.bin",H)); m.hnorm=up(rfg("mtp_hnorm.bin",H));
    m.norm=up(rfg("mtp_norm.bin",H)); m.iln=up(rfg("mtp_input_layernorm.bin",H)); m.pln=up(rfg("mtp_post_attention_layernorm.bin",H));
    m.qn=up(rfg("mtp_q_norm.bin",HD)); m.kn=up(rfg("mtp_k_norm.bin",HD));
    m.Wq=up(rfg("mtp_q_proj_f32.bin",(size_t)qdim2*H)); m.Wk=up(rfg("mtp_k_proj_f32.bin",(size_t)kvdim*H));
    m.Wv=up(rfg("mtp_v_proj_f32.bin",(size_t)kvdim*H)); m.Wo=up(rfg("mtp_o_proj_f32.bin",(size_t)H*adim));
    m.Wg=up(rfg("mtp_gate_proj_f32.bin",(size_t)FFN*H)); m.Wu=up(rfg("mtp_up_proj_f32.bin",(size_t)FFN*H)); m.Wd=up(rfg("mtp_down_proj_f32.bin",(size_t)H*FFN));
    return m;
}
struct MtpKV { float *Kc,*Vc; int n; };
// MTP 1-스텝: append K/V at kv.n, predict next → logits(host). own hidden(d_own).
static void mtp_logits(const ModelConfig&c, const MtpW&m, MtpKV&kv, float*E,
                       int e_tok, const float*h_post, int rope_pos,
                       const float*d_cos,const float*d_sin, float*d_own, std::vector<float>&hl_out,
                       const QMat& lmq, int sdt){
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
    launch_gemv_f32(cat,m.fc,x,2*H,H,1); launch_rmsnorm_zc(x,m.iln,cur,1,H,eps);
    launch_gemv_f32(cur,m.Wq,qg,H,qdim2,1);
    CK(cudaMemcpy2D(qb,HD*4,qg,2*HD*4,HD*4,NQ,cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy2D(gate,HD*4,qg+HD,2*HD*4,HD*4,NQ,cudaMemcpyDeviceToDevice));
    launch_rmsnorm_zc(qb,m.qn,qn,NQ,HD,eps);
    launch_gemv_f32(cur,m.Wk,k,H,kvdim,1); launch_rmsnorm_zc(k,m.kn,kn,NKV,HD,eps);
    launch_gemv_f32(cur,m.Wv,v,H,kvdim,1);
    launch_rope_partial(qn,d_cos+(size_t)rope_pos*rot,d_sin+(size_t)rope_pos*rot,1,NQ,HD,rot);
    launch_rope_partial(kn,d_cos+(size_t)rope_pos*rot,d_sin+(size_t)rope_pos*rot,1,NKV,HD,rot);
    CK(cudaMemcpy(kv.Kc+(size_t)kv.n*kvdim,kn,(size_t)kvdim*4,cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(kv.Vc+(size_t)kv.n*kvdim,v,(size_t)kvdim*4,cudaMemcpyDeviceToDevice));
    launch_attn_decode(qn,kv.Kc,kv.Vc,attn,1,NQ,NKV,HD,kv.n);
    launch_gate_sigmoid(attn,gate,qdim);
    launch_gemv_f32(attn,m.Wo,ob,qdim,H,1); launch_add(cur2,x,ob,H);
    launch_rmsnorm_zc(cur2,m.pln,ff,1,H,eps);
    launch_gemv_f32(ff,m.Wg,gm,H,FFN,1); launch_gemv_f32(ff,m.Wu,um,H,FFN,1);
    launch_swiglu(gm,um,act,FFN); launch_gemv_f32(act,m.Wd,dn,FFN,H,1); launch_add(cur3,cur2,dn,H);
    launch_rmsnorm_zc(cur3,m.norm,d_own,1,H,eps);
    launch_gemv_q4(d_own,lmq.wq,lmq.sc,logits,H,V,1,sdt);
    CK(cudaDeviceSynchronize());
    hl_out.resize(V); CK(cudaMemcpy(hl_out.data(),logits,(size_t)V*4,cudaMemcpyDeviceToHost));
    kv.n++;
    float* all[]={e,en,hn,cat,x,cur,qg,qb,gate,qn,k,kn,v,attn,ob,cur2,ff,gm,um,act,dn,cur3,logits};
    for(float*p:all)cudaFree(p); cudaFree(de);
}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-27b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-27b/mtp_golden")+std::string("/");
    int N=argc>3?atoi(argv[3]):48, bq4=argc>4?atoi(argv[4]):2, D=argc>5?atoi(argv[5]):6;  // D=측정 최대 depth
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int H=c.hidden_size,V=c.vocab_size,rot=c.rotary_dim;
    std::vector<int> prompt=ri("token_ids.bin"); int P=prompt.size();
    int maxlen=P+N+D+8;
    printf("[treespec_depth] prompt=%d gen=%d body_q4=%d maxdepth=%d (per-depth rank de-risk)\n",P,N,bq4,D);
    std::vector<float> cl=rfg("rope_cos_long.bin",(size_t)128*rot), sl=rfg("rope_sin_long.bin",(size_t)128*rot);
    if(P+N+D>=128){printf("need longer rope\n");return 1;}
    float *d_cos=up(cl),*d_sin=up(sl);
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,bq4==0?1:0); MtpW mw=load_mtp(c); float*E=w.embed;
    int *d_tok; CK(cudaMalloc(&d_tok,maxlen*4));
    auto argmaxV=[&](const std::vector<float>&lg,int p){int b=0;float bv=lg[(size_t)p*V];for(int j=1;j<V;j++){float z=lg[(size_t)p*V+j];if(z>bv){bv=z;b=j;}}return b;};

    // ---- greedy → tru[], hbuf[t]=tru[t] 만든 hidden ----
    DecodeState st=alloc_decode_state(c,maxlen);
    CK(cudaMemcpy(d_tok,prompt.data(),P*4,cudaMemcpyHostToDevice));
    float *dl,*dpn; CK(cudaMalloc(&dl,(size_t)P*V*4)); CK(cudaMalloc(&dpn,(size_t)P*H*4));
    qwen35_decode(c,w,st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,bq4,dpn);
    std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
    int cur=argmaxV(lg,P-1);
    std::vector<float> dpn_h((size_t)P*H); CK(cudaMemcpy(dpn_h.data(),dpn,(size_t)P*H*4,cudaMemcpyDeviceToHost));
    std::vector<int> tru; std::vector<std::vector<float>> hbuf;
    { std::vector<float> h0(dpn_h.begin()+(size_t)(P-1)*H, dpn_h.begin()+(size_t)P*H); hbuf.push_back(h0); }
    cudaFree(dl);cudaFree(dpn);
    for(int i=0;i<N+D;i++){ tru.push_back(cur);
        CK(cudaMemcpy(d_tok,&cur,4,cudaMemcpyHostToDevice));
        float*o,*pn; CK(cudaMalloc(&o,(size_t)V*4)); CK(cudaMalloc(&pn,(size_t)H*4));
        qwen35_decode(c,w,st,d_tok,1,d_cos,d_sin,o,nullptr,1,nullptr,bq4,pn);
        std::vector<float> hl(V); CK(cudaMemcpy(hl.data(),o,(size_t)V*4,cudaMemcpyDeviceToHost));
        int b=0;float bv=hl[0];for(int j=1;j<V;j++)if(hl[j]>bv){bv=hl[j];b=j;}
        std::vector<float> hh(H); CK(cudaMemcpy(hh.data(),pn,H*4,cudaMemcpyDeviceToHost)); hbuf.push_back(hh);
        cudaFree(o);cudaFree(pn); cur=b;
    }
    free_decode_state(st);

    // ---- MTP KV를 true seq로 따라가며, 각 위치 t서 depth-D teacher-forced chain rank 측정 ----
    MtpKV kv; CK(cudaMalloc(&kv.Kc,(size_t)maxlen*c.num_kv_heads*c.head_dim*4)); CK(cudaMalloc(&kv.Vc,(size_t)maxlen*c.num_kv_heads*c.head_dim*4)); kv.n=0;
    std::vector<float*> downs(D+1); for(int d=1;d<=D;d++) CK(cudaMalloc(&downs[d],(size_t)H*4));
    // prompt catch-up
    for(int q=1;q<P;q++){ std::vector<float> hl; float* hd=up(std::vector<float>(dpn_h.begin()+(size_t)(q-1)*H,dpn_h.begin()+(size_t)q*H));
        mtp_logits(c,mw,kv,E, prompt[q], hd, q, d_cos,d_sin, downs[1], hl, w.lm_head, c.scale_dtype); cudaFree(hd); }
    std::vector<int> ks={1,2,4,8,16,32};
    std::vector<std::vector<long>> intopk(D+1, std::vector<long>(ks.size(),0));  // [depth][k]
    std::vector<long> cnt(D+1,0);
    int pos0=P;
    for(int t=0;t+1<(int)tru.size() && t<N;t++){
        int saved=kv.n;
        float* hdev=up(hbuf[t]);
        const float* hin=hdev;
        for(int d=1;d<=D;d++){
            if(t+d>=(int)tru.size()) break;
            std::vector<float> hl;
            mtp_logits(c,mw,kv,E, tru[t+d-1], hin, pos0+t+d-1, d_cos,d_sin, downs[d], hl, w.lm_head, c.scale_dtype);
            int truenext=tru[t+d]; float tv=hl[truenext];
            long rank=0; for(int j=0;j<V;j++) if(hl[j]>tv) rank++;
            for(size_t ki=0;ki<ks.size();ki++) if(rank<ks[ki]) intopk[d][ki]++;
            cnt[d]++; hin=downs[d];
        }
        cudaFree(hdev);
        kv.n=saved+1;  // tru[t]만 commit(depth-1 append), 깊은 speculative roll back
    }

    // ---- per-depth P(rank<k) ----
    printf("\n=== per-depth MTP rank: P(true[t+d] rank < k) (27B %s, N=%d) ===\n", bq4==2?"W4A8":"fp32", N);
    printf("  depth |  top-1  top-2  top-4  top-8 top-16 (top-1=chain, top-k=fork 천장)\n");
    std::vector<std::vector<double>> p(D+1, std::vector<double>(ks.size(),0));
    for(int d=1;d<=D;d++){ printf("   d=%-2d  |",d);
        for(size_t ki=0;ki<ks.size();ki++){ p[d][ki]= cnt[d]? (double)intopk[d][ki]/cnt[d]:0; if(ki<5)printf("  %.3f",p[d][ki]); }
        printf("   (n=%ld)\n",cnt[d]); }

    // ---- 트리 모양별 accept 천장 (Σ_{d=1}^{D} Π_{i=1}^{d} P(rank<w @ i)) + 노드수 ----
    auto kidx=[&](int ww){ for(size_t i=0;i<ks.size();i++) if(ks[i]==ww) return (int)i; return 0; };
    auto full_accept=[&](int ww,int dep){ double acc=0,prod=1; for(int d=1;d<=dep;d++){ if(d>D)break; prod*=p[d][kidx(ww)]; acc+=prod; } return acc; };
    auto full_nodes=[&](int ww,int dep){ long nodes=1,lvl=1; for(int d=1;d<=dep;d++){ lvl*=ww; nodes+=lvl; } return nodes; };
    printf("\n=== full top-w fork tree accept 천장 (depth×width, 노드수) ===\n");
    printf("  shape           accept   nodes  (<=8 amortize / >8 SB확장 필요)\n");
    for(int ww=1; ww<=4; ww++) for(int dep=2; dep<=D; dep++){
        long nd=full_nodes(ww,dep); if(ww==1 && dep>7) continue; if(ww>1 && nd>40) continue;
        printf("  top-%d depth-%d   %.2f     %-4ld  %s\n", ww,dep, full_accept(ww,dep), nd, nd<=8?"<=8":(nd<=32?"SB(>8)":"big"));
    }
    printf("\n해석: full top-w가 chain(top-1)보다 accept 크게↑면 forking 유효. 단 노드수(=verify Sq) 봐야 — ≤8 amortize,\n");
    printf("      >8은 SB확장. 천장 4-5가 ≤8(or ≤32 SB)서 나오면 STEP6 빌드 정당. per-depth가 깊이서 급락하면 forking 한계.\n");
    free_weights_qwen(w); free_config_qwen(c);
    return 0;
}
