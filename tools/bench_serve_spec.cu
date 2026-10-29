// 챕터10 Phase3 축2 STEP B: spec-serving bench — 연속 배칭 + 슬롯별 spec 사이클 (ragged speculative continuous batching).
//   bench_serve(plain 1토큰/step)와 동일 프로토콜(Poisson 도착, admit/slot 재사용, seed 고정 trace, warm-up 20% 제외)
//   위에 decode 스텝을 spec 사이클로 교체: 슬롯별 MTP draft(tree K,W) → qwen35_verify_batch(ragged concat, 활성 슬롯만)
//   → 슬롯별 accept/promote(plen 슬롯별 = ragged accept). 슬롯 상태 = per-slot DecodeState + MtpKV(수명=요청).
//   ★canonical(축1): 기본 트리 K6W4(Nnodes=10>8) → 모든 활성 B서 all_gt8=concat-mmq = 곱 + 단독 spec과 bit-identical.
//     (Nnodes≤8 트리는 B_act≥3부터 M>16 → per-slot 폴백 = mism=0 유지되나 곱 없음 — 가변 B 딜레마, STEP C서 결정.)
//   모드: 기본=부하 스윕(bench_serve와 controlled A/B — 같은 seed/λ/Bmax/trace 생성기).
//         ISO=1: 서빙 각 요청 출력 == 단독 spec(같은 프롬프트/gen/K/W) bit-identical 게이트(iso mism=0).
//         TIME=1: verify forward를 B_act별 버킷 계측(곱 B-scaling: 곱(B)=B·t(1)/t(B), STEP C).
//   사용법: ./tools/bench_serve_spec [weights] [mtp_golden] [uq=2] [Bmax=8] [nreq=200] [seed=42] [K=6] [W=4] [λ...]
#include "runtime/knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <deque>
#include <string>
#include <random>
#include <chrono>
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
#include "../kernels/gemm_batch.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
using clk=std::chrono::steady_clock;
static std::string WD,GD;
static std::vector<float> rfg(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static int* upi(const std::vector<int>&h){int*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}
static double pct(std::vector<double>&v,double p){ if(v.empty())return 0; std::sort(v.begin(),v.end()); size_t i=(size_t)ceil(p*v.size()); if(i>0)i--; if(i>=v.size())i=v.size()-1; return v[i]; }

// ── MTP (test_batchspec2/test_treespec_e2e와 동일 스킴 — 테스트 관례상 복제) ─────────────
struct MQ4 { uint8_t* wq=nullptr; float* sc=nullptr; };
struct MtpW { float *fc,*enorm,*hnorm,*norm,*iln,*pln,*qn,*kn,*Wq,*Wk,*Wv,*Wo,*Wg,*Wu,*Wd;
              int q4=0; MQ4 qfc,qWq,qWk,qWv,qWo,qWg,qWu,qWd; };
// P5 ②: MTP body W4(런타임 host 양자화) — 본체 추출과 동일 스킴(QBLK=32, scale=absmax/7,
//   q=clip(round(w/s)+8,0,15), 2/byte low=even). act는 fp32 유지(gemm_batch_q4 — weight bytes 4×↓는
//   동일, act 양자화 오차 없음 = accept 위험 최소). MTP_Q4=1로 켬(기본 fp32 — 단계 실측).
static int g_mtp_q4=-1;
static inline int mtp_q4_on(){ if(g_mtp_q4<0) g_mtp_q4=(knob_raw("MTP_Q4")&&atoi(knob_raw("MTP_Q4"))>0)?1:0; return g_mtp_q4; }
static MQ4 quant_q4_host(const std::vector<float>&w, size_t OUT, size_t IN){
    if(IN%QBLK){ fprintf(stderr,"quant_q4_host: IN %zu %% QBLK\n",IN); abort(); }
    size_t nb=IN/QBLK;
    std::vector<uint8_t> pk(OUT*IN/2); std::vector<float> sc(OUT*nb);
    for(size_t o=0;o<OUT;o++) for(size_t b=0;b<nb;b++){
        const float*wr=&w[o*IN+b*QBLK];
        float a=0; for(int i=0;i<QBLK;i++){ float f=std::fabs(wr[i]); if(f>a)a=f; }
        float s=a/7.0f; if(s==0.f)s=1.f; sc[o*nb+b]=s;
        for(int i=0;i<QBLK;i+=2){
            int q0=(int)std::lround(wr[i]/s)+8;   q0=q0<0?0:(q0>15?15:q0);
            int q1=(int)std::lround(wr[i+1]/s)+8; q1=q1<0?0:(q1>15?15:q1);
            pk[(o*IN+b*QBLK+i)>>1]=(uint8_t)(q0|(q1<<4));
        }
    }
    MQ4 r; CK(cudaMalloc(&r.wq,pk.size())); CK(cudaMemcpy(r.wq,pk.data(),pk.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&r.sc,sc.size()*4)); CK(cudaMemcpy(r.sc,sc.data(),sc.size()*4,cudaMemcpyHostToDevice));
    return r;
}
static MtpW load_mtp(const ModelConfig&c){
    int H=c.hidden_size,HD=c.head_dim,NQ=c.num_q_heads,NKV=c.num_kv_heads,FFN=c.intermediate_size;
    int qdim2=NQ*HD*2,kvdim=NKV*HD;
    MtpW m; m.q4=mtp_q4_on();
    m.enorm=up(rfg("mtp_enorm.bin",H)); m.hnorm=up(rfg("mtp_hnorm.bin",H));
    m.norm=up(rfg("mtp_norm.bin",H)); m.iln=up(rfg("mtp_input_layernorm.bin",H)); m.pln=up(rfg("mtp_post_attention_layernorm.bin",H));
    m.qn=up(rfg("mtp_q_norm.bin",HD)); m.kn=up(rfg("mtp_k_norm.bin",HD));
    auto big=[&](const char* f, size_t OUT, size_t IN, float*& fp, MQ4& q){
        auto wv=rfg(f, OUT*IN);
        if(m.q4){ q=quant_q4_host(wv,OUT,IN); fp=nullptr; }   // q4만 상주(fp32 1.7GB 미탑재)
        else fp=up(wv);
    };
    big("mtp_fc_f32.bin",        H,     2*H,  m.fc, m.qfc);
    big("mtp_q_proj_f32.bin",    qdim2, H,    m.Wq, m.qWq);
    big("mtp_k_proj_f32.bin",    kvdim, H,    m.Wk, m.qWk);
    big("mtp_v_proj_f32.bin",    kvdim, H,    m.Wv, m.qWv);
    big("mtp_o_proj_f32.bin",    H,     NQ*HD,m.Wo, m.qWo);
    big("mtp_gate_proj_f32.bin", FFN,   H,    m.Wg, m.qWg);
    big("mtp_up_proj_f32.bin",   FFN,   H,    m.Wu, m.qWu);
    big("mtp_down_proj_f32.bin", H,     FFN,  m.Wd, m.qWd);
    return m;
}
struct MtpKV { float *Kc,*Vc; int n; };

// ── P5 ①: draft matmul M-재사용 커널화 ──────────────────────────────────────
//   GEMV(warp당 (s,o), weight 재사용 0 — P4 CYC로 "배칭해도 M×운반" 실증)를 gemm_batch(weight row
//   1로드→B행 재사용, 챕터3 자산)로 교체. gemm_batch는 행별 acc가 B-template과 무관하게 같은 i 순서로
//   누적 → 같은 행은 B=1(순차)과 B=Mr(배칭)이 bit-identical = draft canonical ISO가 커널 구조로 성립.
//   단 기존 gemv(float4 mo)와는 합산 순서가 다름 → draft 토큰 flip 가능(출력은 verify가 보장, accept로 관찰).
//   DRAFT_GEMV=1: 기존 gemv 경로 폴백(P4 재현·A/B). gemm_batch 템플릿은 B∈{1,2,4,8,16} → Mr=round-up
//   패딩(gemm만 Mr, 나머지 커널은 M — row-독립이라 패딩 행 쓰레기 무해).
static int g_draft_gemv=-1;
static inline int draft_gemv_on(){ if(g_draft_gemv<0) g_draft_gemv=(knob_raw("DRAFT_GEMV")&&atoi(knob_raw("DRAFT_GEMV"))>0)?1:0; return g_draft_gemv; }
static inline int mroundup(int M){ return M<=1?1:M<=2?2:M<=4?4:M<=8?8:16; }
static inline void draft_mm_f32(const float*X,const float*W,float*Y,int IN,int OUT,int M){
    if(draft_gemv_on()) launch_gemv_f32(X,W,Y,IN,OUT,M);
    else                launch_gemm_batch_f32(X,W,Y,IN,OUT,M);
}
// P2b variant dispatch: DRAFT_MW{0=gemm_batch(기존),1=mw 타일1(NW↑행2),2=mw 타일2(행4)} — draft q4 matmul만.
// draft 스킴 변경은 제안 토큰만 바꿀 수 있고(accept 변동 가능) 출력은 verify가 보증(spec==greedy) → ISO 게이트 그대로 유효.
static int g_draft_mw=-1;
static inline int draft_mw_on(){ if(g_draft_mw<0) g_draft_mw=knob_raw("DRAFT_MW")?atoi(knob_raw("DRAFT_MW")):0; return g_draft_mw; }
static inline void draft_mm_q4(const float*X,const uint8_t*wq,const void*sc,float*Y,int IN,int OUT,int M,int sdt){
    if(draft_gemv_on()) launch_gemv_q4(X,wq,sc,Y,IN,OUT,M,sdt);
    else if(draft_mw_on() && launch_gemm_mw_q4(X,wq,sc,Y,IN,OUT,M,sdt,0,draft_mw_on())) {}
    else                launch_gemm_batch_q4(X,wq,sc,Y,IN,OUT,M,sdt);
}

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
    // P5 canonical(gemm_batch B=1); m.q4면 W4(fp32 act × q4 weight — 순차·배칭 같은 커널)
    auto MM=[&](const float*X, const float* fp, const MQ4& q, float*Y, int IN, int OUT){
        if(m.q4) draft_mm_q4(X,q.wq,q.sc,Y,IN,OUT,1,0);
        else     draft_mm_f32(X,fp,Y,IN,OUT,1); };
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
        // ★draft lm_head: dq4=1이면 진짜 lm_head(q4) — 27b는 embed 미공유(untied)라 E-as-lmhead는 draft 붕괴(accept→1.0).
        if(dq4&&lmq) draft_mm_q4(d_own,lmq->wq,lmq->sc,logits,H,V,1,c.scale_dtype);
        else         draft_mm_f32(d_own,E,logits,H,V,1);
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

// ── 전역 커널 자산 ──────────────────────────────────────────────────────────
static ModelConfig c; static QwenWeights w; static MtpW mw; static float* E;
static float *d_cos,*d_sin;
static int H,V,rot,NKV,HD, K,W,Nn, uq, maxlen, dq4;   // dq4: draft lm_head=q4(uq!=0 기본 — 27b untied embed 필수)
static const std::vector<int>* IDS;
static int argmaxV(const std::vector<float>&lg,int p){int b=0;float bv=lg[(size_t)p*V];for(int j=1;j<V;j++){float z=lg[(size_t)p*V+j];if(z>bv){bv=z;b=j;}}return b;}

static void draft_tree(MtpKV& kv, int cur, float* h_cur, int base,
                       std::vector<int>&tok, std::vector<int>&par, std::vector<int>&pos, std::vector<unsigned>&anc,
                       std::vector<float*>&owns){
    tok.assign(Nn,0); par.assign(Nn,0); pos.assign(Nn,0); owns.clear();
    std::vector<int> top0;
    tok[0]=cur; par[0]=-1; pos[0]=base;
    { float*own; CK(cudaMalloc(&own,(size_t)H*4)); mtp_step(c,mw,kv,E,cur,h_cur,base,true,d_cos,d_sin,own,true,&w.lm_head,dq4,&top0,W); owns.push_back(own); }
    for(int j=0;j<W;j++){ tok[1+j]=top0[j]; par[1+j]=0; pos[1+j]=base+1; }
    int e_tok=top0[0]; float* h_in=owns[0]; int prev_idx=1;
    for(int i=1;i<K;i++){ float*own; CK(cudaMalloc(&own,(size_t)H*4));
        int d=mtp_step(c,mw,kv,E,e_tok,h_in,base+i,true,d_cos,d_sin,own,true,&w.lm_head,dq4); owns.push_back(own);
        int idx=W+i; tok[idx]=d; par[idx]=prev_idx; pos[idx]=base+1+i; prev_idx=idx; e_tok=d; h_in=own; }
    anc.assign(Nn,0);
    for(int j=0;j<Nn;j++){ unsigned mm=0; for(int x=j;x>=0;x=par[x]) mm|=(1u<<x); anc[j]=mm; }
}
static void accept_path(const std::vector<float>&vlh, const std::vector<int>&tok, const std::vector<int>&par,
                        std::vector<int>&path, int&plen, int&node_out, int&next_cur){
    path.clear(); path.push_back(0); int node=0;
    while(true){ int g=argmaxV(vlh,node); int child=-1;
        for(int j=0;j<(int)par.size();j++) if(par[j]==node && tok[j]==g){ child=j; break; }
        if(child<0) break; path.push_back(child); node=child; }
    plen=(int)path.size(); node_out=node; next_cur=argmaxV(vlh,node);
}
// ── P4 draft 배칭: mtp_step의 슬롯 배치판(M행 1 forward) ──────────────────────
//   bit-identity 근거(P5 갱신): matmul은 draft_mm_*(canonical=gemm_batch — 행별 acc가 B-template 무관 같은
//   i 순서 누적 → 행 b는 B=1(순차 mtp_step)과 B=Mr(배칭) bit-identical. DRAFT_GEMV=1이면 기존 gemv —
//   그것도 warp당 (s,o) row-독립·S-무관이라 동일 성질). rmsnorm/embedding/swiglu/add/gate row-독립.
//   rope는 rope_partial_pos(per-row pos — verify_batch서 순차 rope_partial과 mism=0 검증). attn/KV append는 per-slot
//   루프(순차와 동일 커널·동일 kv.n·동일 순서). top-k는 host 동일 코드(M×V logits 1회 D2H) → sync B×K→K.
static float* g_bws=nullptr; static size_t g_bcap=0; static int* g_betok=nullptr; static int* g_bpos=nullptr; static int g_bicap=0;
static float* bws_at(size_t off){ return g_bws+off; }
static void mtp_step_batch(int M, MtpKV**kvs, const int*e_toks /*host[M]*/, const float*Hpost /*[M,H] dev*/,
                           const int*h_pos /*host[M]*/, bool append, float*Down /*[M,H] dev(caller)*/,
                           bool want_logits, std::vector<float>*hlog /*[M,V] host out*/){
    const int qdim2=c.num_q_heads*HD*2, qdim=c.num_q_heads*HD, kvdim=NKV*HD, FFN=c.intermediate_size;
    const int NQ=c.num_q_heads, rot_=c.rotary_dim; const float eps=c.rms_norm_eps;
    // P5: gemm_batch 템플릿 B∈{1,2,4,8,16} → Mr=round-up. 버퍼는 Mr행, gemm만 Mr로(패딩 행은 uninit이나
    //     모든 커널이 row-독립이라 실제 M행 결과 무영향), 나머지 커널·D2H·per-slot 루프는 M.
    const int Mr = draft_gemv_on()? M : mroundup(M);
    size_t need=(size_t)Mr*( (size_t)8*H /*e,en,hn,cat(2H),x,cur_,ob*/ + qdim2 + 4*(size_t)qdim + 3*(size_t)kvdim + 3*(size_t)FFN + (want_logits?(size_t)V:0) );
    if(g_bcap<need){ if(g_bws)cudaFree(g_bws); CK(cudaMalloc(&g_bws,need*4)); g_bcap=need; }
    if(g_bicap<M){ if(g_betok)cudaFree(g_betok); if(g_bpos)cudaFree(g_bpos); CK(cudaMalloc(&g_betok,M*4)); CK(cudaMalloc(&g_bpos,M*4)); g_bicap=M; }
    size_t off=0; auto B_=[&](size_t n){ float*p=bws_at(off); off+=n; return p; };
    float *e=B_((size_t)Mr*H),*en=B_((size_t)Mr*H),*hn=B_((size_t)Mr*H),*cat=B_((size_t)Mr*2*H),*x=B_((size_t)Mr*H),*cur_=B_((size_t)Mr*H);
    float *qg=B_((size_t)Mr*qdim2),*qb=B_((size_t)Mr*qdim),*gate=B_((size_t)Mr*qdim),*qn=B_((size_t)Mr*qdim);
    float *k=B_((size_t)Mr*kvdim),*kn=B_((size_t)Mr*kvdim),*v=B_((size_t)Mr*kvdim),*attn=B_((size_t)Mr*qdim);
    float *ob=B_((size_t)Mr*H),*gm=B_((size_t)Mr*FFN),*um=B_((size_t)Mr*FFN),*act_=B_((size_t)Mr*FFN);
    float *logits = want_logits ? B_((size_t)Mr*V) : nullptr;
    CK(cudaMemcpy(g_betok,e_toks,M*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_bpos,h_pos,M*4,cudaMemcpyHostToDevice));
    launch_embedding(g_betok,E,e,M,H);
    launch_rmsnorm_zc(e,mw.enorm,en,M,H,eps); launch_rmsnorm_zc(Hpost,mw.hnorm,hn,M,H,eps);
    CK(cudaMemcpy2D(cat,    (size_t)2*H*4, en,(size_t)H*4,(size_t)H*4,M,cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy2D(cat+H,  (size_t)2*H*4, hn,(size_t)H*4,(size_t)H*4,M,cudaMemcpyDeviceToDevice));
    auto MMB=[&](const float*X,const float*fp,const MQ4&q,float*Y,int IN,int OUT){
        if(mw.q4) draft_mm_q4(X,q.wq,q.sc,Y,IN,OUT,Mr,0);
        else      draft_mm_f32(X,fp,Y,IN,OUT,Mr); };
    MMB(cat,mw.fc,mw.qfc,x,2*H,H);
    launch_rmsnorm_zc(x,mw.iln,cur_,M,H,eps);
    MMB(cur_,mw.Wq,mw.qWq,qg,H,qdim2);
    CK(cudaMemcpy2D(qb,HD*4,qg,2*HD*4,HD*4,(size_t)M*NQ,cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy2D(gate,HD*4,qg+HD,2*HD*4,HD*4,(size_t)M*NQ,cudaMemcpyDeviceToDevice));
    launch_rmsnorm_zc(qb,mw.qn,qn,M*NQ,HD,eps);
    MMB(cur_,mw.Wk,mw.qWk,k,H,kvdim); launch_rmsnorm_zc(k,mw.kn,kn,M*NKV,HD,eps);
    MMB(cur_,mw.Wv,mw.qWv,v,H,kvdim);
    launch_rope_partial_pos(qn,d_cos,d_sin,g_bpos,M,NQ,HD,rot_);
    launch_rope_partial_pos(kn,d_cos,d_sin,g_bpos,M,NKV,HD,rot_);
    for(int b=0;b<M;b++){   // per-slot: KV append(kv.n 위치) + attn(길이 kv.n) — 순차와 동일 커널·순서
        MtpKV&kv=*kvs[b];
        CK(cudaMemcpy(kv.Kc+(size_t)kv.n*kvdim,kn+(size_t)b*kvdim,(size_t)kvdim*4,cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(kv.Vc+(size_t)kv.n*kvdim,v +(size_t)b*kvdim,(size_t)kvdim*4,cudaMemcpyDeviceToDevice));
        launch_attn_decode(qn+(size_t)b*qdim,kv.Kc,kv.Vc,attn+(size_t)b*qdim,1,NQ,NKV,HD,kv.n);
    }
    launch_gate_sigmoid(attn,gate,M*qdim);
    MMB(attn,mw.Wo,mw.qWo,ob,qdim,H); launch_add(x,x,ob,M*H);                 // cur2 = x+ob (x 재사용)
    launch_rmsnorm_zc(x,mw.pln,cur_,M,H,eps);
    MMB(cur_,mw.Wg,mw.qWg,gm,H,FFN); MMB(cur_,mw.Wu,mw.qWu,um,H,FFN);
    launch_swiglu(gm,um,act_,(size_t)M*FFN); MMB(act_,mw.Wd,mw.qWd,ob,FFN,H); launch_add(x,x,ob,M*H);  // cur3
    launch_rmsnorm_zc(x,mw.norm,Down,M,H,eps);
    if(want_logits){
        // lm_head 입력 Down은 caller 버퍼 — gemm이 Mr행 읽으므로 caller는 Mr*H 할당(g_bown).
        if(dq4) draft_mm_q4(Down,w.lm_head.wq,w.lm_head.sc,logits,H,V,Mr,c.scale_dtype);
        else    draft_mm_f32(Down,E,logits,H,V,Mr);
        hlog->resize((size_t)M*V); CK(cudaMemcpy(hlog->data(),logits,(size_t)M*V*4,cudaMemcpyDeviceToHost));
    } else CK(cudaDeviceSynchronize());
    if(append) for(int b=0;b<M;b++) kvs[b]->n++;
}

// P4: draft_tree의 슬롯 배치판 — depth 루프 유지(순차 본질), 슬롯 방향 배치. top-k는 순차와 동일 host 코드(partial_sort).
//   own 히든은 depth별 [M,H] 디바이스 버퍼(g_bown). 출력 tokS/parS/posS/ancS는 순차와 동일 레이아웃.
static std::vector<float*> g_bown; static int g_bown_M=0;
static void draft_tree_batch(int M, MtpKV**kvs, const int*curs, float*const*h_curs /*dev ptr per slot*/, const int*bases,
                             std::vector<std::vector<int>>&tok, std::vector<std::vector<int>>&par,
                             std::vector<std::vector<int>>&pos, std::vector<std::vector<unsigned>>&anc,
                             const int*slot_ids){
    int Mown=mroundup(M);   // P5: lm_head gemm이 Down을 Mr행 읽음 → own 버퍼는 Mr 할당
    if((int)g_bown.size()<K || g_bown_M<Mown){ for(float*p:g_bown)cudaFree(p); g_bown.assign(K,nullptr);
        for(int i=0;i<K;i++) CK(cudaMalloc(&g_bown[i],(size_t)Mown*H*4)); g_bown_M=Mown; }
    static float* Hroot=nullptr; static int hcap=0;
    if(hcap<M){ if(Hroot)cudaFree(Hroot); CK(cudaMalloc(&Hroot,(size_t)M*H*4)); hcap=M; }
    for(int b=0;b<M;b++) CK(cudaMemcpy(Hroot+(size_t)b*H,h_curs[b],(size_t)H*4,cudaMemcpyDeviceToDevice));
    std::vector<int> etok(M),posv(M);
    for(int b=0;b<M;b++){ int s=slot_ids[b];
        tok[s].assign(Nn,0); par[s].assign(Nn,0); pos[s].assign(Nn,0);
        tok[s][0]=curs[b]; par[s][0]=-1; pos[s][0]=bases[b];
        etok[b]=curs[b]; posv[b]=bases[b]; }
    std::vector<float> hlog;
    mtp_step_batch(M,kvs,etok.data(),Hroot,posv.data(),true,g_bown[0],true,&hlog);
    std::vector<int> prev_tok(M);
    for(int b=0;b<M;b++){ int s=slot_ids[b];
        // 순차 draft_tree와 동일 top-W(partial_sort, 동일 비교) — 행 b logits는 순차와 bit-identical이라 토큰 동일
        const float*hl=hlog.data()+(size_t)b*V;
        static std::vector<int> idx; idx.resize(V); for(int j=0;j<V;j++)idx[j]=j;
        int wv=std::min(W,V);
        std::partial_sort(idx.begin(),idx.begin()+wv,idx.end(),[&](int a2,int b2){return hl[a2]>hl[b2];});
        for(int j=0;j<W;j++){ tok[s][1+j]=idx[j]; par[s][1+j]=0; pos[s][1+j]=bases[b]+1; }
        prev_tok[b]=idx[0]; }
    int prev_idx=1;
    for(int i=1;i<K;i++){
        for(int b=0;b<M;b++) posv[b]=bases[b]+i;
        mtp_step_batch(M,kvs,prev_tok.data(),g_bown[i-1],posv.data(),true,g_bown[i],true,&hlog);
        for(int b=0;b<M;b++){ int s=slot_ids[b];
            const float*hl=hlog.data()+(size_t)b*V;
            int best=0; float bv=hl[0]; for(int j=1;j<V;j++) if(hl[j]>bv){bv=hl[j];best=j;}   // 순차와 동일 argmax
            int idx2=W+i; tok[s][idx2]=best; par[s][idx2]=prev_idx; pos[s][idx2]=bases[b]+1+i; prev_tok[b]=best; }
        prev_idx=W+i;
    }
    for(int b=0;b<M;b++){ int s=slot_ids[b]; anc[s].assign(Nn,0);
        for(int j=0;j<Nn;j++){ unsigned mm=0; for(int x2=j;x2>=0;x2=par[s][x2]) mm|=(1u<<x2); anc[s][j]=mm; } }
}

static void promote_and_catchup(DecodeState& st, VerifyArtifacts& va, MtpKV& kv, int mn_committed, int base, int cur,
                                const std::vector<int>&tok, const std::vector<int>&path, int plen, float* h_cur,
                                const std::vector<float>&hpn, int node, const float* d_vpn, float* d_own){
    int a=plen-1;
    promote_tree_path(c,st,va,path.data(),plen);
    kv.n=mn_committed;
    mtp_step(c,mw,kv,E,cur,h_cur,base,true,d_cos,d_sin,d_own,false);
    for(int i=1;i<=a;i++){ float*ht=up(std::vector<float>(hpn.begin()+(size_t)path[i-1]*H,hpn.begin()+(size_t)(path[i-1]+1)*H));
        mtp_step(c,mw,kv,E,tok[path[i]],ht,base+i,true,d_cos,d_sin,d_own,false); cudaFree(ht); }
    CK(cudaMemcpy(h_cur,d_vpn+(size_t)node*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
}

// ── spec 슬롯 ────────────────────────────────────────────────────────────────
struct SpecSlot {
    int req=-1; int P=0,gen=0,g=0;
    DecodeState st; MtpKV kv; float* h_cur=nullptr; VerifyArtifacts va;
    int cur=0; std::vector<float> hpn;
    std::vector<int> out;   // ISO 게이트용(요청별 생성 토큰 기록)
};

struct Req{ int P, gen; double arrival; double admit=-1, first=-1, done=-1; };
struct Stats{ double thru_req=0,thru_tok=0; double ttft50=0,ttft95=0,e2e50=0,e2e95=0,e2e99=0,tpot50=0,tpot95=0;
              double avgB=0,maxq=0,makespan=0; double acc_sum=0; long acc_n=0; int completed=0; };

// TIME=1: verify forward B_act 버킷(곱 B-scaling)
static int g_time=0; static double g_tv[64]; static long g_nv[64];
static cudaEvent_t g_e0,g_e1; static int g_ev=0;
// CYC=1: 사이클 구간 분해(draft / pack+verify / D2H+accept·promote(host) / 배치 catchup / tail).
//        경계마다 deviceSync 삽입 — 계측 전용 모드(속도 수치는 CYC=0으로).
static int g_cyc=0; static double g_cd[5];

// prefill(단독 qwen35_decode, commit) + MTP catch-up → cur/h_cur. (test_batchspec2 solo_spec 앞부분과 동일)
static void slot_prefill(SpecSlot&s, const std::vector<int>&prompt){
    int P=prompt.size();
    int*d_tok=upi(prompt);
    float *dl,*dpn; CK(cudaMalloc(&dl,(size_t)P*V*4)); CK(cudaMalloc(&dpn,(size_t)P*H*4));
    qwen35_decode(c,w,s.st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,uq,dpn);
    std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
    s.cur=argmaxV(lg,P-1);
    for(int q=1;q<P;q++){ float*ht; CK(cudaMalloc(&ht,(size_t)H*4)); mtp_step(c,mw,s.kv,E,prompt[q],dpn+(size_t)(q-1)*H,q,true,d_cos,d_sin,ht,false); cudaFree(ht); }
    CK(cudaMemcpy(s.h_cur,dpn+(size_t)(P-1)*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
    cudaFree(dl);cudaFree(dpn);cudaFree(d_tok);
}

// ── spec-serving continuous scheduler ───────────────────────────────────────
// P2b 런타임 전환 최소판: HOTSWITCH="K1,W1,K2,W2,thr" — B_act 이동평균(EMA)<thr → 형상1, ≥thr → 형상2.
//   전환은 사이클 경계(트리는 사이클마다 새로 생성, 슬롯 상태=KV/plen뿐 — 형상 무관)라 안전.
//   출력은 spec==greedy 성질로 형상 무관 불변 → 기존 ISO 게이트가 전환 하에서도 유효.
static int hs_on=0; static int hsK[2],hsW[2]; static double hs_thr=0;
static inline void hs_init(){ static int done=0; if(done)return; done=1;
    const char*h=knob_raw("HOTSWITCH");
    if(h&&sscanf(h,"%d,%d,%d,%d,%lf",&hsK[0],&hsW[0],&hsK[1],&hsW[1],&hs_thr)==5) hs_on=1; }
static Stats run_spec_continuous(std::vector<Req> reqs,int Bmax,std::vector<std::vector<int>>*iso_out=nullptr){
    int R=reqs.size();
    hs_init();
    int NnH = hs_on ? std::max(hsK[0]+hsW[0], hsK[1]+hsW[1]) : Nn;   // 할당은 최악 형상 기준(전환 OOB 방지)
    std::vector<SpecSlot> slot(Bmax);
    for(int b=0;b<Bmax;b++){
        slot[b].st=alloc_decode_state(c,maxlen);
        CK(cudaMalloc(&slot[b].kv.Kc,(size_t)maxlen*NKV*HD*4)); CK(cudaMalloc(&slot[b].kv.Vc,(size_t)maxlen*NKV*HD*4)); slot[b].kv.n=0;
        CK(cudaMalloc(&slot[b].h_cur,(size_t)H*4));
        slot[b].va=alloc_verify_artifacts(c,NnH); slot[b].hpn.resize((size_t)NnH*H);
    }
    float *d_own; CK(cudaMalloc(&d_own,(size_t)H*4));
    float *vl,*vpn; CK(cudaMalloc(&vl,(size_t)Bmax*NnH*V*4)); CK(cudaMalloc(&vpn,(size_t)Bmax*NnH*H*4));
    int *d_tokb,*d_parb,*d_posb; unsigned *d_ancb;
    CK(cudaMalloc(&d_tokb,(size_t)Bmax*NnH*4)); CK(cudaMalloc(&d_parb,(size_t)Bmax*NnH*4));
    CK(cudaMalloc(&d_posb,(size_t)Bmax*NnH*4)); CK(cudaMalloc(&d_ancb,(size_t)Bmax*NnH*4));
    double hs_ema=0; long hs_sw=0;
    std::deque<int> queue; int nextArr=0, completed=0; double simt=0; long stepN=0; double sumB=0,maxq=0;
    double acc_sum=0; long acc_n=0;
    // 사이클 임시(슬롯별 draft 결과)
    std::vector<std::vector<int>> tokS(Bmax),parS(Bmax),posS(Bmax); std::vector<std::vector<unsigned>> ancS(Bmax);
    std::vector<std::vector<float*>> ownsS(Bmax); std::vector<int> baseS(Bmax), mnS(Bmax);
    while(completed<R){
        while(nextArr<R && reqs[nextArr].arrival<=simt) queue.push_back(nextArr++);
        // admit: free slot 재사용 — reset_decode_state(GDN/KV 잔재 clear) + prefill(+MTP catchup)
        for(int b=0;b<Bmax;b++) if(slot[b].req<0 && !queue.empty()){
            int r=queue.front();
            if(reqs[r].P+reqs[r].gen+K+Nn+8>maxlen){queue.pop_front();continue;}   // 길이 가드(spec overshoot 포함)
            queue.pop_front();
            SpecSlot&s=slot[b]; s.req=r; s.P=reqs[r].P; s.gen=reqs[r].gen; s.g=0; s.out.clear();
            reset_decode_state(c,s.st); s.kv.n=0; reqs[r].admit=simt;
            std::vector<int> prompt(s.P); for(int i=0;i<s.P;i++) prompt[i]=(*IDS)[i%IDS->size()];
            auto p0=clk::now(); slot_prefill(s,prompt); CK(cudaDeviceSynchronize()); auto p1=clk::now();
            simt+=std::chrono::duration<double>(p1-p0).count();
            reqs[r].first=simt; s.g=1; s.out.push_back(s.cur);   // 첫 토큰 = prefill 로짓 argmax(=spec root)
            if(s.g>=s.gen){ reqs[r].done=simt; completed++; if(iso_out)(*iso_out)[r]=s.out; s.req=-1; }
        }
        // 활성 수집(packing order = 물리 slot 순)
        std::vector<int> act; for(int b=0;b<Bmax;b++) if(slot[b].req>=0) act.push_back(b);
        int Ba=(int)act.size();
        if(Ba==0){ if(nextArr<R){ simt=reqs[nextArr].arrival; continue; } else break; }
        if((double)queue.size()>maxq) maxq=queue.size();
        if(hs_on){   // 사이클 경계 핫스위치: B_act EMA → regime 형상 룩업(전역 K/W/Nn 갱신)
            hs_ema = 0.7*hs_ema + 0.3*Ba;
            int i = hs_ema>=hs_thr ? 1 : 0;
            if(hsK[i]!=K||hsW[i]!=W){ K=hsK[i]; W=hsW[i]; Nn=K+W; hs_sw++; }
        }
        // ── spec 사이클(활성 슬롯) ──
        auto t0=clk::now();
        // 1) MTP draft — P4: 슬롯 배치(1 forward=M행, sync K회). DRAFT_SEQ=1이면 순차(A/B·게이트용).
        static int g_draft_seq=(knob_raw("DRAFT_SEQ")&&atoi(knob_raw("DRAFT_SEQ"))>0)?1:0;
        if(g_draft_seq){
            for(int a=0;a<Ba;a++){ int b=act[a]; SpecSlot&s=slot[b];
                mnS[b]=s.kv.n; baseS[b]=s.st.n_past;
                draft_tree(s.kv,s.cur,s.h_cur,baseS[b], tokS[b],parS[b],posS[b],ancS[b], ownsS[b]); }
        } else {
            std::vector<MtpKV*> kvs(Ba); std::vector<int> curs(Ba),bases(Ba),sids(Ba); std::vector<float*> hcs(Ba);
            for(int a=0;a<Ba;a++){ int b=act[a]; SpecSlot&s=slot[b];
                mnS[b]=s.kv.n; baseS[b]=s.st.n_past;
                kvs[a]=&s.kv; curs[a]=s.cur; bases[a]=baseS[b]; hcs[a]=s.h_cur; sids[a]=b; }
            draft_tree_batch(Ba,kvs.data(),curs.data(),hcs.data(),bases.data(),tokS,parS,posS,ancS,sids.data());
        }
        auto tb=t0;
        if(g_cyc){ CK(cudaDeviceSynchronize()); auto x=clk::now(); g_cd[0]+=std::chrono::duration<double>(x-tb).count(); tb=x; }
        // 2) ragged concat pack + batched verify(canonical: Nn>8 → concat-mmq 모든 Ba서 곱)
        {
            std::vector<int> tokc((size_t)Ba*Nn),parc((size_t)Ba*Nn),posc((size_t)Ba*Nn),Nb(Ba,Nn),base_b(Ba);
            std::vector<unsigned> ancc((size_t)Ba*Nn);
            std::vector<DecodeState*> sts(Ba); std::vector<VerifyArtifacts> vab(Ba);
            for(int a=0;a<Ba;a++){ int b=act[a];
                for(int i=0;i<Nn;i++){ tokc[(size_t)a*Nn+i]=tokS[b][i]; parc[(size_t)a*Nn+i]=parS[b][i];
                                       posc[(size_t)a*Nn+i]=posS[b][i]; ancc[(size_t)a*Nn+i]=ancS[b][i]; }
                base_b[a]=baseS[b]; sts[a]=&slot[b].st; vab[a]=slot[b].va; }
            CK(cudaMemcpy(d_tokb,tokc.data(),(size_t)Ba*Nn*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_parb,parc.data(),(size_t)Ba*Nn*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_posb,posc.data(),(size_t)Ba*Nn*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_ancb,ancc.data(),(size_t)Ba*Nn*4,cudaMemcpyHostToDevice));
            if(g_time){ if(!g_ev){cudaEventCreate(&g_e0);cudaEventCreate(&g_e1);g_ev=1;} cudaEventRecord(g_e0); }
            qwen35_verify_batch(c,w,sts.data(),d_tokb,Ba,Nb.data(),base_b.data(),d_cos,d_sin,vl,vpn,vab.data(),uq,d_parb,d_ancb,d_posb);
            if(g_time){ cudaEventRecord(g_e1); cudaEventSynchronize(g_e1); float ms=0; cudaEventElapsedTime(&ms,g_e0,g_e1);
                        if(Ba<64){ g_tv[Ba]+=ms; g_nv[Ba]++; } }
        }
        if(g_cyc){ CK(cudaDeviceSynchronize()); auto x=clk::now(); g_cd[1]+=std::chrono::duration<double>(x-tb).count(); tb=x; }
        // 3) 슬롯별 accept + promote + catchup (ragged accept: plen 슬롯별). P4: catchup도 슬롯 배치(depth별 active 부분집합).
        std::vector<float> vlh((size_t)Ba*Nn*V); CK(cudaMemcpy(vlh.data(),vl,vlh.size()*4,cudaMemcpyDeviceToHost));
        std::vector<float> vpnh((size_t)Ba*Nn*H); CK(cudaMemcpy(vpnh.data(),vpn,vpnh.size()*4,cudaMemcpyDeviceToHost));
        if(g_draft_seq){
            for(int a=0;a<Ba;a++){ int b=act[a]; SpecSlot&s=slot[b];
                std::vector<float> vlh_b(vlh.begin()+(size_t)a*Nn*V, vlh.begin()+(size_t)(a+1)*Nn*V);
                std::copy(vpnh.begin()+(size_t)a*Nn*H, vpnh.begin()+(size_t)(a+1)*Nn*H, s.hpn.begin());
                std::vector<int> path; int plen,node,next_cur; accept_path(vlh_b,tokS[b],parS[b],path,plen,node,next_cur);
                acc_sum+=plen; acc_n++;
                for(int t=1;t<plen && s.g<s.gen;t++){ s.out.push_back(tokS[b][path[t]]); s.g++; }
                float* d_vpn_b=vpn+(size_t)a*Nn*H;
                promote_and_catchup(s.st,s.va,s.kv,mnS[b],baseS[b],s.cur,tokS[b],path,plen,s.h_cur,s.hpn,node,d_vpn_b,d_own);
                s.cur=next_cur;
                if(s.g<s.gen){ s.out.push_back(s.cur); s.g++; }   // 다음 root(=이번 verify가 생성한 bonus 토큰)
                for(float*p:ownsS[b])cudaFree(p);
            }
            if(g_cyc){ CK(cudaDeviceSynchronize()); auto x=clk::now(); g_cd[2]+=std::chrono::duration<double>(x-tb).count(); tb=x; }   // 순차 경로: accept+promote+catchup 통짜
        } else {
            // (a) accept + promote(tree path compaction)만 슬롯별 — catchup은 아래 배치.
            std::vector<std::vector<int>> pathS(Ba); std::vector<int> plenS(Ba),nodeS(Ba),ncurS(Ba);
            int max_a=0;
            for(int a=0;a<Ba;a++){ int b=act[a]; SpecSlot&s=slot[b];
                std::vector<float> vlh_b(vlh.begin()+(size_t)a*Nn*V, vlh.begin()+(size_t)(a+1)*Nn*V);
                std::copy(vpnh.begin()+(size_t)a*Nn*H, vpnh.begin()+(size_t)(a+1)*Nn*H, s.hpn.begin());
                accept_path(vlh_b,tokS[b],parS[b],pathS[a],plenS[a],nodeS[a],ncurS[a]);
                acc_sum+=plenS[a]; acc_n++;
                for(int t=1;t<plenS[a] && s.g<s.gen;t++){ s.out.push_back(tokS[b][pathS[a][t]]); s.g++; }
                promote_tree_path(c,s.st,s.va,pathS[a].data(),plenS[a]);
                s.kv.n=mnS[b];                                    // 순차와 동일: catchup 전 MtpKV 되감기
                if(plenS[a]-1>max_a) max_a=plenS[a]-1;
            }
            if(g_cyc){ CK(cudaDeviceSynchronize()); auto x=clk::now(); g_cd[2]+=std::chrono::duration<double>(x-tb).count(); tb=x; }
            // (b) 배치 catchup: j=0(root, h_cur) 전 슬롯 + j=1..max_a(hpn[path[j-1]]) active 부분집합. 순차와 동일 스텝/입력.
            static float* Hcat=nullptr; static int hccap=0;
            if(hccap<Ba){ if(Hcat)cudaFree(Hcat); CK(cudaMalloc(&Hcat,(size_t)Ba*H*4)); hccap=Ba; }
            static float* Dscratch=nullptr; static int dscap=0;
            if(dscap<Ba){ if(Dscratch)cudaFree(Dscratch); CK(cudaMalloc(&Dscratch,(size_t)Ba*H*4)); dscap=Ba; }
            std::vector<float> hupload((size_t)Ba*H);
            for(int j=0;j<=max_a;j++){
                std::vector<MtpKV*> kvs; std::vector<int> etok,posv; kvs.reserve(Ba);
                int m=0;
                for(int a=0;a<Ba;a++){ int b=act[a]; SpecSlot&s=slot[b];
                    int a_b=plenS[a]-1; if(j>a_b) continue;
                    kvs.push_back(&s.kv);
                    if(j==0){ etok.push_back(s.cur); CK(cudaMemcpy(Hcat+(size_t)m*H,s.h_cur,(size_t)H*4,cudaMemcpyDeviceToDevice)); }
                    else    { etok.push_back(tokS[b][pathS[a][j]]);
                              std::copy(s.hpn.begin()+(size_t)pathS[a][j-1]*H, s.hpn.begin()+(size_t)(pathS[a][j-1]+1)*H,
                                        hupload.begin()+(size_t)m*H); }
                    posv.push_back(baseS[b]+j); m++;
                }
                if(m==0) break;
                if(j>0) CK(cudaMemcpy(Hcat,hupload.data(),(size_t)m*H*4,cudaMemcpyHostToDevice));
                mtp_step_batch(m,kvs.data(),etok.data(),Hcat,posv.data(),true,Dscratch,false,nullptr);
            }
            if(g_cyc){ CK(cudaDeviceSynchronize()); auto x=clk::now(); g_cd[3]+=std::chrono::duration<double>(x-tb).count(); tb=x; }
            // (c) h_cur 갱신(vpn[node]) + cur/out 마감 — 순차와 동일 순서(승격 후).
            for(int a=0;a<Ba;a++){ int b=act[a]; SpecSlot&s=slot[b];
                CK(cudaMemcpy(s.h_cur,vpn+(size_t)a*Nn*H+(size_t)nodeS[a]*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
                s.cur=ncurS[a];
                if(s.g<s.gen){ s.out.push_back(s.cur); s.g++; }
            }
        }
        CK(cudaDeviceSynchronize());
        auto t1=clk::now(); double dt=std::chrono::duration<double>(t1-t0).count();
        if(g_cyc){ g_cd[4]+=std::chrono::duration<double>(t1-tb).count(); }
        simt+=dt; sumB+=Ba; stepN++;
        for(int a=0;a<Ba;a++){ int b=act[a]; SpecSlot&s=slot[b];
            if(s.req>=0 && s.g>=s.gen){ reqs[s.req].done=simt; completed++; if(iso_out)(*iso_out)[s.req]=s.out; s.req=-1; } }
    }
    for(int b=0;b<Bmax;b++){ free_verify_artifacts(slot[b].va); cudaFree(slot[b].kv.Kc);cudaFree(slot[b].kv.Vc);
                             cudaFree(slot[b].h_cur); free_decode_state(slot[b].st); }
    cudaFree(d_own);cudaFree(vl);cudaFree(vpn);cudaFree(d_tokb);cudaFree(d_parb);cudaFree(d_posb);cudaFree(d_ancb);
    if(g_cyc){
        double tot=g_cd[0]+g_cd[1]+g_cd[2]+g_cd[3]+g_cd[4];
        if(tot>0) printf("  [CYC] draft %.1f%%(%.2fs) | verify %.1f%%(%.2fs) | accept·host %.1f%%(%.2fs) | catchup %.1f%%(%.2fs) | tail %.1f%%(%.2fs) — 사이클합 %.2fs, %ld스텝\n",
            100*g_cd[0]/tot,g_cd[0],100*g_cd[1]/tot,g_cd[1],100*g_cd[2]/tot,g_cd[2],
            100*g_cd[3]/tot,g_cd[3],100*g_cd[4]/tot,g_cd[4],tot,stepN);
        for(int i=0;i<5;i++)g_cd[i]=0;
    }
    // 통계(warm-up 제외 20%, bench_serve와 동일)
    std::vector<int> order; for(int r=0;r<R;r++) if(reqs[r].done>=0) order.push_back(r);
    std::sort(order.begin(),order.end(),[&](int a,int b){return reqs[a].done<reqs[b].done;});
    int warm=(int)(order.size()*0.2); Stats S; std::vector<double> ttft,e2e,tpot;
    double t_lo=1e18,t_hi=0; long toks_steady=0;
    for(size_t k=warm;k<order.size();k++){ int r=order[k]; double a=reqs[r].arrival,f=reqs[r].first,d=reqs[r].done;
        ttft.push_back(f-a); e2e.push_back(d-a); if(reqs[r].gen>1) tpot.push_back((d-f)/(reqs[r].gen-1));
        t_lo=std::min(t_lo,a); t_hi=std::max(t_hi,d); toks_steady+=reqs[r].gen; }
    S.completed=order.size(); int nst=(int)order.size()-warm; double win=t_hi-t_lo;
    S.thru_req=win>0?nst/win:0; S.thru_tok=win>0?toks_steady/win:0;
    S.ttft50=pct(ttft,.5);S.ttft95=pct(ttft,.95);
    S.e2e50=pct(e2e,.5);S.e2e95=pct(e2e,.95);S.e2e99=pct(e2e,.99);
    S.tpot50=pct(tpot,.5);S.tpot95=pct(tpot,.95);
    S.avgB=stepN?sumB/stepN:0; S.maxq=maxq; S.makespan=simt; S.acc_sum=acc_sum; S.acc_n=acc_n;
    return S;
}

// ── ISO 게이트: 단독 spec(같은 프롬프트/gen/K/W) — 서빙 출력과 bit-identical? ──
static std::vector<int> solo_spec(const std::vector<int>& prompt, int gen){
    std::vector<int> out;
    SpecSlot s; s.st=alloc_decode_state(c,maxlen);
    CK(cudaMalloc(&s.kv.Kc,(size_t)maxlen*NKV*HD*4)); CK(cudaMalloc(&s.kv.Vc,(size_t)maxlen*NKV*HD*4)); s.kv.n=0;
    CK(cudaMalloc(&s.h_cur,(size_t)H*4)); s.va=alloc_verify_artifacts(c,Nn); s.hpn.resize((size_t)Nn*H);
    float *d_own; CK(cudaMalloc(&d_own,(size_t)H*4));
    float *vl,*vpn; CK(cudaMalloc(&vl,(size_t)Nn*V*4)); CK(cudaMalloc(&vpn,(size_t)Nn*H*4));
    slot_prefill(s,prompt); out.push_back(s.cur);
    std::vector<int> tok,par,pos; std::vector<unsigned> anc; std::vector<float*> owns;
    while((int)out.size()<gen){
        int mn=s.kv.n, base=s.st.n_past;
        draft_tree(s.kv,s.cur,s.h_cur,base,tok,par,pos,anc,owns);
        int*dt=upi(tok); int*dp=upi(pos); int*dpar=upi(par);
        unsigned*da; CK(cudaMalloc(&da,Nn*4)); CK(cudaMemcpy(da,anc.data(),Nn*4,cudaMemcpyHostToDevice));
        // ★단독도 verify_batch(B=1) — 서빙과 같은 경로(canonical 자동 일치; qwen35_decode 단독 verify와도 동일 커널)
        DecodeState* sts1[1]={&s.st}; VerifyArtifacts vab1[1]={s.va}; int Nb1[1]={Nn}; int b1[1]={base};
        qwen35_verify_batch(c,w,sts1,dt,1,Nb1,b1,d_cos,d_sin,vl,vpn,vab1,uq,dpar,da,dp);
        std::vector<float> vlh((size_t)Nn*V); CK(cudaMemcpy(vlh.data(),vl,vlh.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s.hpn.data(),vpn,s.hpn.size()*4,cudaMemcpyDeviceToHost));
        cudaFree(dt);cudaFree(dp);cudaFree(dpar);cudaFree(da);
        std::vector<int> path; int plen,node,next_cur; accept_path(vlh,tok,par,path,plen,node,next_cur);
        for(int t=1;t<plen && (int)out.size()<gen;t++) out.push_back(tok[path[t]]);
        promote_and_catchup(s.st,s.va,s.kv,mn,base,s.cur,tok,path,plen,s.h_cur,s.hpn,node,vpn,d_own);
        s.cur=next_cur;
        if((int)out.size()<gen) out.push_back(s.cur);
        for(float*p:owns)cudaFree(p);
    }
    free_verify_artifacts(s.va); cudaFree(s.kv.Kc);cudaFree(s.kv.Vc);cudaFree(s.h_cur); free_decode_state(s.st);
    cudaFree(d_own);cudaFree(vl);cudaFree(vpn);
    return out;
}

static std::vector<Req> gen_trace(double lambda,int nreq,unsigned seed,int pmax){
    std::mt19937 rng(seed); std::exponential_distribution<double> ia(lambda);
    std::uniform_int_distribution<int> pP(8,pmax), gS(16,48), gL(96,192); std::uniform_real_distribution<double> u(0,1);
    // P2b regime 축: SHORTP=short 요청 비율(기본 0.7). short-heavy 0.9 / long-heavy 0.3 등 — trace 형상만 바꿈(스킴 불변).
    static double shortp=-1; if(shortp<0){ shortp=knob_raw("SHORTP")?atof(knob_raw("SHORTP")):0.7; }
    std::vector<Req> v(nreq); double t=0;
    for(int i=0;i<nreq;i++){ t+=ia(rng); v[i].arrival=t; v[i].P=pP(rng); v[i].gen=(u(rng)<shortp)?gS(rng):gL(rng); }
    return v;
}

int main(int argc,char**argv){
    knobs_init("bench_serve_spec");
    setvbuf(stdout,NULL,_IONBF,0);
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    uq=argc>3?atoi(argv[3]):2; int Bmax=argc>4?atoi(argv[4]):8; int nreq=argc>5?atoi(argv[5]):200; unsigned seed=argc>6?atoi(argv[6]):42;
    K=argc>7?atoi(argv[7]):6; W=argc>8?atoi(argv[8]):4; Nn=K+W;
    std::vector<double> lambdas; for(int i=9;i<argc;i++) lambdas.push_back(atof(argv[i]));
    if(lambdas.empty()) lambdas={1,2,4,8};
    g_time=(knob_raw("TIME")&&atoi(knob_raw("TIME"))>0)?1:0;
    g_cyc=(knob_raw("CYC")&&atoi(knob_raw("CYC"))>0)?1:0;
    int iso=(knob_raw("ISO")&&atoi(knob_raw("ISO"))>0)?1:0;
    dq4=(uq!=0)?1:0; if(knob_raw("DQ4")) dq4=atoi(knob_raw("DQ4"));   // draft lm_head=q4 기본(uq!=0) — 27b untied embed서 필수(E-as-lmhead=accept 붕괴)
    int pmax=knob_raw("PMAX")?atoi(knob_raw("PMAX")):32;
    int trials=(knob_raw("TRIALS")&&atoi(knob_raw("TRIALS"))>0)?1:0;   // warm 서버 모드(오토튜닝): 모델 1회 로드, stdin에서 trial 반복
    c=load_config_qwen((WD+"/manifest.json").c_str());
    H=c.hidden_size; V=c.vocab_size; rot=c.rotary_dim; NKV=c.num_kv_heads; HD=c.head_dim;
    maxlen=pmax+192+((trials||knob_raw("HOTSWITCH"))?8+13:K+Nn)+16; if(maxlen<288) maxlen=288;   // TRIALS/HOTSWITCH: 최악 형상(K8,Nn13) 기준 고정(형상 갱신에도 rope/KV 안전)
    static std::vector<int> ids=ri("token_ids.bin"); IDS=&ids;
    // rope: ext 테이블 우선(bench_serve와 동일)
    std::vector<float> cl,sl;
    { FILE*e=fopen((GD+"rope_cos_ext.bin").c_str(),"rb");
      if(e){ fseek(e,0,SEEK_END); long er=ftell(e)/4/rot; fclose(e);
             int use=(er>=maxlen)?maxlen:(int)er; cl=rfg("rope_cos_ext.bin",(size_t)use*rot); sl=rfg("rope_sin_ext.bin",(size_t)use*rot);
             if(use<maxlen){ printf("  rope ext %d행<maxlen %d — maxlen 축소\n",use,maxlen); maxlen=use; } }
      else { auto c0=rfg("rope_cos_long.bin",(size_t)128*rot), s0=rfg("rope_sin_long.bin",(size_t)128*rot);
             cl.resize((size_t)maxlen*rot); sl.resize((size_t)maxlen*rot);
             for(int i=0;i<maxlen;i++)for(int j=0;j<rot;j++){cl[(size_t)i*rot+j]=c0[(size_t)(i%128)*rot+j];sl[(size_t)i*rot+j]=s0[(size_t)(i%128)*rot+j];} } }
    d_cos=up(cl); d_sin=up(sl);
    w=load_weights_qwen(WD.c_str(),c,1,uq==0?1:0);
    mw=load_mtp(c); E=w.embed;
    // admission 예산(unified 메모리 가드): per-slot = 본체 KV(bf16) + GDN + MtpKV(fp32) + VerifyArtifacts(gdn_snap 지배)
    int nfull=0,ngdn=0; for(int l=0;l<c.num_layers;l++) c.layer_is_full[l]?nfull++:ngdn++;
    int nh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),Kk=c.gdn_conv_kernel;
    size_t per_kv=(size_t)nfull*maxlen*NKV*HD*2*2 + (size_t)ngdn*((size_t)nh*D*D+(size_t)(Kk-1)*convd)*4;
    size_t per_mtp=(size_t)2*maxlen*NKV*HD*4;
    size_t per_va=(size_t)Nn*ngdn*((size_t)nh*D*D+convd)*4;
    size_t perslot=per_kv+per_mtp+per_va;
    size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB);
    // GB10 unified: cudaMemGetInfo free는 리눅스 페이지캐시(메모리 압박 시 회수 가능)를 가용으로 안 쳐서
    // 과소보고 — /proc/meminfo MemAvailable(회수 가능 포함)이 실제 가용. 둘 중 큰 값(가드 정확화, 완화 아님).
    { FILE*f=fopen("/proc/meminfo","r"); if(f){ char line[128];
        while(fgets(line,sizeof line,f)){ long kb;
            if(sscanf(line,"MemAvailable: %ld",&kb)==1){ size_t avail=(size_t)kb*1024; if(avail>freeB)freeB=avail; break; } }
        fclose(f);} }
    double budget=0.7*(double)freeB;
    printf("[bench_serve_spec] uq=%d dq4=%d Bmax=%d nreq=%d seed=%u K=%d W=%d Nn=%d(%s) maxlen=%d\n",
           uq,dq4,Bmax,nreq,seed,K,W,Nn, Nn>8?"concat-mmq 곱(전 B)":"소트리(B_act≥3 per-slot, 곱 부분)", maxlen);
    printf("  admission 예산: per-slot %.1fMB(KV %.1f + MTP %.1f + VA %.1f) × Bmax %d = %.2fGB vs 0.7·free %.2fGB — %s\n",
           perslot/1e6,per_kv/1e6,per_mtp/1e6,per_va/1e6,Bmax,(double)Bmax*perslot/1e9,budget/1e9,
           (double)Bmax*perslot<=budget?"OK":"초과(중단)");
    if((double)Bmax*perslot>budget){ printf("  admission guard: 예산 초과 → 중단(시스템 다운 방지).\n"); return 1; }

    if(iso){
        // ISO 게이트: 작은 trace 서빙 → 각 요청 vs 단독 spec bit-identical
        int Riso=nreq<16?nreq:16;
        auto tr=gen_trace(lambdas[0],Riso,seed,pmax);
        std::vector<std::vector<int>> got(Riso);
        run_spec_continuous(tr,Bmax,&got);
        int bad=0;
        for(int r=0;r<Riso;r++){
            std::vector<int> prompt(tr[r].P); for(int i=0;i<tr[r].P;i++) prompt[i]=ids[i%ids.size()];
            std::vector<int> ref=solo_spec(prompt,tr[r].gen);
            int mism=0; for(int i=0;i<tr[r].gen;i++){ if(i>=(int)got[r].size()||i>=(int)ref.size()||got[r][i]!=ref[i]) mism++; }
            if(mism){ bad++; printf("  req%d P=%d gen=%d mism=%d/%d\n",r,tr[r].P,tr[r].gen,mism,tr[r].gen); }
        }
        printf("[ISO] %d/%d 요청 bit-identical → %s\n",Riso-bad,Riso,bad==0?"PASS":"FAIL");
        return bad==0?0:1;
    }

    { auto tr=gen_trace(4.0,12,seed,pmax); run_spec_continuous(tr,Bmax); }   // warm-up
    if(trials){
        // warm 서버: stdin 한 줄 = "Bmax K W lam nreq seed" → 측정 1회 → "TRIAL tok/s tpot95 accept avgB" 출력.
        // 상태: DecodeState/MtpKV/VA는 요청 수명(run 내 할당·해제), 워크스페이스는 용량-단조 재사용(row-독립) —
        //   trial 간 오염 없음(같은 라인 재실행 bit-exact 동일로 게이트). 형상(K/W/Nn)은 전역 갱신, maxlen은 최악 고정.
        printf("TRIALS_READY\n");
        char line[256];
        while(fgets(line,sizeof line,stdin)){
            int b,k,ww,nr; unsigned sd; double lm;
            if(sscanf(line,"%d %d %d %lf %d %u",&b,&k,&ww,&lm,&nr,&sd)!=6){ printf("TRIAL_ERR parse\n"); continue; }
            K=k; W=ww; Nn=K+W;
            auto tr=gen_trace(lm,nr,sd,pmax);
            Stats S=run_spec_continuous(tr,b);
            printf("TRIAL %.1f %.4f %.2f %.2f\n",S.thru_tok,S.tpot95,S.acc_n?S.acc_sum/S.acc_n:0,S.avgB);
        }
        return 0;
    }
    printf("\n  === spec-serving 부하 스윕 (warm-up 20%% 제외, seed 고정 trace — bench_serve와 동일 프로토콜) ===\n");
    printf("  %-6s %-4s | %8s %8s | %7s %7s %7s | %7s %7s | %5s %5s | %6s\n",
           "mode","λ","req/s","tok/s","e2e50","e2e95","e2e99","ttft95","tpot95","avgB","maxQ","accept");
    printf("  %s\n",std::string(104,'-').c_str());
    for(double lam:lambdas){
        auto tr=gen_trace(lam,nreq,seed,pmax);
        Stats S=run_spec_continuous(tr,Bmax);
        printf("  %-6s %-4g | %8.3f %8.1f | %7.3f %7.3f %7.3f | %7.3f %7.4f | %5.2f %5.0f | %6.2f\n",
               "c-spec",lam,S.thru_req,S.thru_tok,S.e2e50,S.e2e95,S.e2e99,S.ttft95,S.tpot95,S.avgB,S.maxq,
               S.acc_n?S.acc_sum/S.acc_n:0);
    }
    if(g_time){
        printf("\n  [TIME] verify forward B_act 버킷(곱 B-scaling: 곱(B)=B·t(1)/t(B)):\n");
        for(int b=1;b<64;b++) if(g_nv[b]) printf("    B=%d: %.3f ms (n=%ld)%s\n",b,g_tv[b]/g_nv[b],g_nv[b],
            (g_nv[1]&&b>1)?(std::string("  곱=")+std::to_string((double)b*(g_tv[1]/g_nv[1])/(g_tv[b]/g_nv[b]))+"×").c_str():"");
    }
    printf("\n  (tok/s=생성 토큰. accept=평균 plen(사이클당 생성). draft=MTP 순차(호스트 동기 포함) — 사이클 시간에 포함=정직 e2e.)\n");
    free_weights_qwen(w); free_config_qwen(c); return 0;
}
