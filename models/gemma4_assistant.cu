// Gemma4 assistant(0.4B MTP drafter) graph 구현 — 헤더(models/gemma4_assistant.h) 참조.
//   ★신규 커널 0: rmsnorm·gemv_f32·rope·attn_decode·geglu·add·scale 전부 기존 커널 재사용.
//   fp32 전용(골든 게이트 우선 — W4A8/q4 양자화는 ③ perf에서 판단). Sq=1 고정(single-position draft).
#include "gemma4_assistant.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/manifest.h"      // mf_* 헬퍼(manifest.json 파싱)
#include "../kernels/rmsnorm.cuh"
#include "../kernels/gemv_f32.cuh"
#include "../kernels/quantize_x.cuh"  // W4A8: 활성화 int8 양자화(Lever1)
#include "../kernels/gemv_w4a8.cuh"   // W4A8: q4 weight gemv(target 스킴 동일·신규 커널 0)
#include "../kernels/rope.cuh"
#include "../kernels/attention.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"

#define AK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

AstConfig load_config_assistant(const char *manifest_path){
    char *s = mf_read_file(manifest_path);
    AstConfig c{};
    c.num_layers          = mf_int(s,"num_layers");
    c.hidden_size         = mf_int(s,"hidden_size");
    c.intermediate_size   = mf_int(s,"intermediate_size");
    c.num_q_heads         = mf_int(s,"num_q_heads");
    c.head_dim            = mf_int(s,"head_dim");
    c.global_head_dim     = mf_int(s,"global_head_dim");
    c.num_kv_heads        = mf_int(s,"num_kv_heads");
    c.num_global_kv_heads = mf_int(s,"num_global_kv_heads");
    c.sliding_window      = mf_int(s,"sliding_window");
    c.rms_norm_eps        = (float)mf_num(s,"rms_norm_eps");
    c.vocab_size          = mf_int(s,"vocab_size");
    c.backbone            = mf_int(s,"backbone_hidden_size");
    c.attn_scaling        = (float)mf_num(s,"attn_scaling");   // extract가 항상 기록(1.0 — 코드 실사)
    if(c.num_layers>64){ fprintf(stderr,"[ast] num_layers %d > 64\n",c.num_layers); exit(1); }
    // "layer_is_full": [0, 0, 0, 1] 파싱
    const char *p = strstr(s,"\"layer_is_full\"");
    if(!p){ fprintf(stderr,"[ast] manifest에 layer_is_full 없음\n"); exit(1); }
    p = strchr(p,'['); if(!p) exit(1);
    for(int l=0;l<c.num_layers;l++){
        while(*p && *p!='0' && *p!='1') p++;
        c.layer_is_full[l] = (*p=='1'); p++;
    }
    free(s);
    return c;
}

static void* ast_load_raw(const std::string &p, size_t bytes){
    FILE *f = fopen(p.c_str(),"rb");
    if(!f){ fprintf(stderr,"[ast] weight 없음: %s\n",p.c_str()); exit(1); }
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET);
    if((size_t)b != bytes){ fprintf(stderr,"[ast] %s 크기 %ld != 기대 %zu — 형상 불일치, 중단.\n",p.c_str(),b,bytes); exit(1); }
    void *h=malloc(bytes);
    if(fread(h,1,bytes,f)!=bytes){ fprintf(stderr,"[ast] short read: %s\n",p.c_str()); exit(1); }
    fclose(f);
    void *d; AK(cudaMalloc(&d,bytes)); AK(cudaMemcpy(d,h,bytes,cudaMemcpyHostToDevice)); free(h);
    return d;
}
static float* ast_load(const std::string &dir, const std::string &name, size_t n){
    return (float*)ast_load_raw(dir+"/"+name+".bin", n*4);
}
// 매트: use_q4=0→wf(fp32) / 2→wq(_q4.bin 팩드 [n/2 uint8])+sc(_scale.bin fp32 [n/QBLK]) — extract 스킴 그대로
static QMat ast_load_mat(const std::string &dir, const std::string &name, size_t rows, size_t cols, int use_q4){
    QMat m{}; size_t n=rows*cols;
    if(use_q4==2){
        m.wq=(uint8_t*)ast_load_raw(dir+"/"+name+"_q4.bin", n/2);
        m.sc=ast_load_raw(dir+"/"+name+"_scale.bin", (n/QBLK)*4);
    } else {
        m.wf=ast_load(dir,name,n);
    }
    return m;
}

AstWeights load_weights_assistant(const char *dir_, const AstConfig &c, int use_q4){
    std::string dir(dir_);
    AstWeights w{};
    const int H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, B=c.backbone;
    w.embed      = ast_load_mat(dir,"model_embed_tokens",(size_t)V,H,use_q4);
    w.final_norm = ast_load(dir,"model_norm",H);
    w.pre_proj   = ast_load_mat(dir,"pre_projection",(size_t)H,2*(size_t)B,use_q4);
    w.post_proj  = ast_load_mat(dir,"post_projection",(size_t)B,H,use_q4);
    for(int l=0;l<c.num_layers;l++){
        AstLayer &L=w.layers[l];
        L.is_full = c.layer_is_full[l];
        const int hd = L.is_full ? c.global_head_dim : c.head_dim;
        const int qdim = c.num_q_heads*hd;
        char pre[16]; snprintf(pre,sizeof(pre),"L%d_",l);
        std::string P(pre);
        L.input_ln     = ast_load(dir,P+"input_layernorm",H);
        L.q_norm       = ast_load(dir,P+"self_attn_q_norm",hd);
        L.post_attn_ln = ast_load(dir,P+"post_attention_layernorm",H);
        L.pre_ffn_ln   = ast_load(dir,P+"pre_feedforward_layernorm",H);
        L.post_ffn_ln  = ast_load(dir,P+"post_feedforward_layernorm",H);
        L.q    = ast_load_mat(dir,P+"self_attn_q_proj",(size_t)qdim,H,use_q4);
        L.o    = ast_load_mat(dir,P+"self_attn_o_proj",(size_t)H,qdim,use_q4);
        L.gate = ast_load_mat(dir,P+"mlp_gate_proj",(size_t)FFN,H,use_q4);
        L.up   = ast_load_mat(dir,P+"mlp_up_proj",(size_t)FFN,H,use_q4);
        L.down = ast_load_mat(dir,P+"mlp_down_proj",(size_t)H,FFN,use_q4);
        // layer_scalar: 1 float — host로 읽어 launch_scale에 사용
        std::string sp = dir+"/"+P+"layer_scalar.bin";
        FILE *f=fopen(sp.c_str(),"rb");
        if(!f){ fprintf(stderr,"[ast] %s 없음\n",sp.c_str()); exit(1); }
        if(fread(&L.layer_scalar,4,1,f)!=1) exit(1);
        fclose(f);
    }
    return w;
}

static void free_mat(QMat &m){ if(m.wf)cudaFree(m.wf); if(m.wq)cudaFree(m.wq); if(m.sc)cudaFree(m.sc); m=QMat{}; }
void free_weights_assistant(AstWeights &w, const AstConfig &c){
    free_mat(w.embed); cudaFree(w.final_norm); free_mat(w.pre_proj); free_mat(w.post_proj);
    for(int l=0;l<c.num_layers;l++){
        AstLayer &L=w.layers[l];
        float *ps[]={L.input_ln,L.q_norm,L.post_attn_ln,L.pre_ffn_ln,L.post_ffn_ln};
        for(float *p:ps) cudaFree(p);
        QMat *ms[]={&L.q,&L.o,&L.gate,&L.up,&L.down};
        for(QMat *m:ms) free_mat(*m);
    }
}

// scratch: static grow-on-demand(매 호출 전량 덮어씀 = 수학 불변) — draft는 K회/사이클 호출이라 재할당 회피.
static float *g_ab[10]={}; static size_t g_abcap[10]={};
static float* ab(int i, size_t n){
    if(g_abcap[i]<n){ if(g_ab[i])cudaFree(g_ab[i]); AK(cudaMalloc(&g_ab[i],n*4)); g_abcap[i]=n; }
    return g_ab[i];
}
// W4A8 활성화 양자화 버퍼(S=1) — gemma4_decode mm()과 동일 규약(quantize_x_q8 → gemv_w4a8, scale fp32)
static int8_t *g_axq=nullptr; static float *g_axs=nullptr; static int32_t *g_axsum=nullptr; static size_t g_axcap=0;
static inline void amm(const QMat &m, const float *X, float *Y, int IN, int OUT, int use_q4){
    if(use_q4==2){
        size_t n=(size_t)IN;
        if(g_axcap<n){ if(g_axq)cudaFree(g_axq); if(g_axs)cudaFree(g_axs); if(g_axsum)cudaFree(g_axsum);
            AK(cudaMalloc(&g_axq,n)); AK(cudaMalloc(&g_axs,(n/QBLK)*4)); AK(cudaMalloc(&g_axsum,(n/QBLK)*4)); g_axcap=n; }
        launch_quantize_x_q8(X,g_axq,g_axs,g_axsum,IN,1);
        launch_gemv_w4a8(g_axq,g_axs,g_axsum,m.wq,m.sc,Y,IN,OUT,1,0);   // sdtype=0(scale fp32)
    } else {
        launch_gemv_f32(X,m.wf,Y,IN,OUT,1);
    }
}

void gemma4_assistant_step(const AstConfig &c, const AstWeights &w,
                           const float *d_in7680,
                           const float *d_kf, const float *d_vf, int len_f,
                           const float *d_ks, const float *d_vs, int len_s,
                           const float *d_cos_f1, const float *d_sin_f1,
                           const float *d_cos_s1, const float *d_sin_s1,
                           float *d_logits, float *d_hout, int use_q4){
    const int H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, NQ=c.num_q_heads;
    const float eps=c.rms_norm_eps;
    const int hd_max = c.global_head_dim>c.head_dim ? c.global_head_dim : c.head_dim;
    float *x=ab(0,H), *norm=ab(1,H), *tmp=ab(2,H), *after=ab(3,H);
    float *qb=ab(4,(size_t)NQ*hd_max), *qn=ab(5,(size_t)NQ*hd_max), *attn=ab(6,(size_t)NQ*hd_max);
    float *g=ab(7,FFN), *u=ab(8,FFN), *act=ab(9,FFN);

    amm(w.pre_proj, d_in7680, x, 2*c.backbone, H, use_q4);   // pre-projection(concat → 트렁크 입력·스케일 없음)
    for(int l=0;l<c.num_layers;l++){
        const AstLayer &L=w.layers[l];
        const int hd   = L.is_full ? c.global_head_dim      : c.head_dim;
        const int NKV  = L.is_full ? c.num_global_kv_heads  : c.num_kv_heads;
        const int len  = L.is_full ? len_f                  : len_s;
        const float *K = L.is_full ? d_kf : d_ks, *Vv = L.is_full ? d_vf : d_vs;
        const float *cosR = L.is_full ? d_cos_f1 : d_cos_s1, *sinR = L.is_full ? d_sin_f1 : d_sin_s1;
        const int window = L.is_full ? 0 : c.sliding_window;
        const int qdim = NQ*hd;

        launch_rmsnorm(x, L.input_ln, norm, 1, H, eps);
        amm(L.q, norm, qb, H, qdim, use_q4);
        launch_rmsnorm(qb, L.q_norm, qn, NQ, hd, eps);              // per-head q_norm
        launch_rope(qn, cosR, sinR, 1, NQ, hd);                     // 고정 위치 행(호출자 준비)
        // ★cross-attn: target KV 직독. base=len-1 → attendlen=len(전체)·sliding lo=len-window
        //   (HF bidirectional-flip 마스크의 q_len=1 의미와 정확히 일치 — KV 끝 기준 마지막 w개).
        launch_attn_decode(qn, K, Vv, attn, 1, NQ, NKV, hd, len-1, 0, c.attn_scaling, window);
        amm(L.o, attn, tmp, qdim, H, use_q4);
        launch_rmsnorm(tmp, L.post_attn_ln, norm, 1, H, eps);
        launch_add(after, x, norm, H);

        launch_rmsnorm(after, L.pre_ffn_ln, norm, 1, H, eps);
        amm(L.gate, norm, g, H, FFN, use_q4);
        amm(L.up,   norm, u, H, FFN, use_q4);
        launch_geglu(g, u, act, (size_t)FFN);
        amm(L.down, act, tmp, FFN, H, use_q4);
        launch_rmsnorm(tmp, L.post_ffn_ln, norm, 1, H, eps);
        launch_add(x, after, norm, H);
        if(L.layer_scalar!=1.0f) launch_scale(x, x, L.layer_scalar, H);
    }
    launch_rmsnorm(x, w.final_norm, norm, 1, H, eps);
    amm(w.embed, norm, d_logits, H, V, use_q4);                     // tied lm_head(softcap 없음 — 코드 실사)
    amm(w.post_proj, norm, d_hout, H, c.backbone, use_q4);          // 재귀 hidden(다음 스텝 h_prev)
    AK(cudaDeviceSynchronize());
}

// ---- draft concat(배치) — 헤더 주석 참조. amm의 B행 판: B==1은 기존 gemv(경로 불변), B 2~16은 SB 1적재. ----
static inline void ammB(const QMat &m, const float *X, float *Y, int IN, int OUT, int B, int use_q4){
    if(use_q4==2){
        size_t n=(size_t)B*IN;
        if(g_axcap<n){ if(g_axq)cudaFree(g_axq); if(g_axs)cudaFree(g_axs); if(g_axsum)cudaFree(g_axsum);
            AK(cudaMalloc(&g_axq,n)); AK(cudaMalloc(&g_axs,(n/QBLK)*4)); AK(cudaMalloc(&g_axsum,(n/QBLK)*4)); g_axcap=n; }
        launch_quantize_x_q8(X,g_axq,g_axs,g_axsum,IN,B);
        if(B==1) launch_gemv_w4a8(g_axq,g_axs,g_axsum,m.wq,m.sc,Y,IN,OUT,1,0);
        else     launch_gemv_w4a8_sb(g_axq,g_axs,m.wq,m.sc,Y,IN,OUT,B,0);   // ≤16행 1적재(SB==gemv per-row 기게이트)
    } else {
        launch_gemv_f32(X,m.wf,Y,IN,OUT,B);
    }
}

void gemma4_assistant_step_batch(const AstConfig &c, const AstWeights &w, int B,
                                 const float *d_in_B,
                                 const float* const* kf, const float* const* vf,
                                 const float* const* ks, const float* const* vs,
                                 const int *len_b,
                                 const float *d_cos_f, const float *d_sin_f,
                                 const float *d_cos_s, const float *d_sin_s,
                                 const int *d_pos,
                                 float *d_logits_B, float *d_hout_B, int use_q4){
    if(B<1 || B>16){ fprintf(stderr,"[ast_batch] B=%d 미지원(1~16 — SB 단일그룹·Bmax 규약)\n",B); exit(1); }
    const int H=c.hidden_size, FFN=c.intermediate_size, V=c.vocab_size, NQ=c.num_q_heads;
    const float eps=c.rms_norm_eps;
    const int hd_max = c.global_head_dim>c.head_dim ? c.global_head_dim : c.head_dim;
    float *x=ab(0,(size_t)B*H), *norm=ab(1,(size_t)B*H), *tmp=ab(2,(size_t)B*H), *after=ab(3,(size_t)B*H);
    float *qb=ab(4,(size_t)B*NQ*hd_max), *qn=ab(5,(size_t)B*NQ*hd_max), *attn=ab(6,(size_t)B*NQ*hd_max);
    float *g=ab(7,(size_t)B*FFN), *u=ab(8,(size_t)B*FFN), *act=ab(9,(size_t)B*FFN);

    ammB(w.pre_proj, d_in_B, x, 2*c.backbone, H, B, use_q4);
    for(int l=0;l<c.num_layers;l++){
        const AstLayer &L=w.layers[l];
        const int hd   = L.is_full ? c.global_head_dim      : c.head_dim;
        const int NKV  = L.is_full ? c.num_global_kv_heads  : c.num_kv_heads;
        const float *cosT = L.is_full ? d_cos_f : d_cos_s, *sinT = L.is_full ? d_sin_f : d_sin_s;
        const int window = L.is_full ? 0 : c.sliding_window;
        const int qdim = NQ*hd;

        launch_rmsnorm(x, L.input_ln, norm, B, H, eps);
        ammB(L.q, norm, qb, H, qdim, B, use_q4);
        launch_rmsnorm(qb, L.q_norm, qn, B*NQ, hd, eps);
        launch_rope_partial_pos(qn, cosT, sinT, d_pos, B, NQ, hd, hd);     // 행별 위치(=len[b])
        for(int b=0;b<B;b++){                                              // ★cross-attn: 슬롯별 target KV 직독
            const float *K = L.is_full ? kf[b] : ks[b], *Vv = L.is_full ? vf[b] : vs[b];
            launch_attn_decode(qn+(size_t)b*qdim, K, Vv, attn+(size_t)b*qdim,
                               1, NQ, NKV, hd, len_b[b]-1, 0, c.attn_scaling, window);
        }
        ammB(L.o, attn, tmp, qdim, H, B, use_q4);
        launch_rmsnorm(tmp, L.post_attn_ln, norm, B, H, eps);
        launch_add(after, x, norm, B*H);

        launch_rmsnorm(after, L.pre_ffn_ln, norm, B, H, eps);
        ammB(L.gate, norm, g, H, FFN, B, use_q4);
        ammB(L.up,   norm, u, H, FFN, B, use_q4);
        launch_geglu(g, u, act, (size_t)B*FFN);
        ammB(L.down, act, tmp, FFN, H, B, use_q4);
        launch_rmsnorm(tmp, L.post_ffn_ln, norm, B, H, eps);
        launch_add(x, after, norm, B*H);
        if(L.layer_scalar!=1.0f) launch_scale(x, x, L.layer_scalar, (size_t)B*H);
    }
    launch_rmsnorm(x, w.final_norm, norm, B, H, eps);
    ammB(w.embed, norm, d_logits_B, H, V, B, use_q4);                      // tied lm_head — B행 1적재(지배항 분할상환)
    ammB(w.post_proj, norm, d_hout_B, H, c.backbone, B, use_q4);           // 재귀 hidden(행별 — 슬롯 독립)
    AK(cudaDeviceSynchronize());
}
