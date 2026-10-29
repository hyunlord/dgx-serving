#include "../runtime/decode_state_gemma4.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <chrono>
#include <functional>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
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
#include "../kernels/attention_decode_split.cuh"
#include "../kernels/kv_store.cuh"
#include "../kernels/swiglu.cuh"
#include "../kernels/elementwise.cuh"

#define DK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static float *g4d_dm(size_t n){ float*p; DK(cudaMalloc(&p,n*4)); return p; }
static inline float* g4d_grow(float*&p,size_t&cap,size_t n){ if(cap<n){ if(p)cudaFree(p); DK(cudaMalloc(&p,n*4)); cap=n; } return p; }

// ---- KV 캐시 alloc/free (A-1: 캡=max_len 선형; A-2에서 sliding 캡) ----
G4DecodeState alloc_g4_decode_state(const ModelConfig &c, int max_len) {
    if(c.num_experts>0) g4_kv_force_fp32("MoE model");
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
        // Storage dtype G4_KV_DTYPE (kv_store.cuh): bytes = elements * g4_kv_elem(); handles stay float*.
        DK(cudaMalloc(&s.Kc[l],(size_t)max_len*nkv*hd*g4_kv_elem())); DK(cudaMalloc(&s.Vc[l],(size_t)max_len*nkv*hd*g4_kv_elem()));
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
static void g4d_reserve_quant(size_t n){
    if(gd_cap<n){ if(gd_xq)cudaFree(gd_xq); if(gd_xs)cudaFree(gd_xs); if(gd_xsum)cudaFree(gd_xsum);
        DK(cudaMalloc(&gd_xq,n)); DK(cudaMalloc(&gd_xs,(n/QBLK)*4)); DK(cudaMalloc(&gd_xsum,(n/QBLK)*4)); gd_cap=n; }
}
static bool g4d_mmq_prefill(){
    static const bool enabled=[](){ const char* v=getenv("G4_MMQ_PREFILL"); return !v || strcmp(v,"0")!=0; }();
    return enabled;
}
static int g4d_mmq_min_s(){
    static const int min_s=[](){
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
    return min_s;
}
static inline void mm(const QMat &m, const float *X, float *Y, int IN, int OUT, int S, int use_q4, cudaStream_t stream){
    if(use_q4==2){
        launch_quantize_x_q8(X,gd_xq,gd_xs,gd_xsum,IN,S,stream);
        if(g4d_mmq_prefill() && gd_amort==1 && S>=g4d_mmq_min_s() && IN%QBLK==0 && (gd_sdtype==0 || gd_sdtype==1)){
            launch_gemm_mmq_w4a8_auto(gd_xq,gd_xs,m.wq,m.sc,Y,S,OUT,IN,gd_sdtype,stream);
        } else if(gd_amort && S>1){  // 프리필: SB 커널로 weight 1회/≤16행 amortize(비-amortized plain gemv의 S× 재적재 제거).
            //   SB per-row == plain gemv per-row bit-identical(test_gemma4_prefill_slot 게이트 확증) → golden 불변.
            for(int s0=0;s0<S;s0+=16){ int sq=(S-s0<16)?(S-s0):16;
                launch_gemv_w4a8_sb(gd_xq+(size_t)s0*IN, gd_xs+(size_t)s0*(IN/QBLK), m.wq, m.sc, Y+(size_t)s0*OUT, IN, OUT, sq, gd_sdtype,stream); }
        } else launch_gemv_w4a8(gd_xq,gd_xs,gd_xsum,m.wq,m.sc,Y,IN,OUT,S,gd_sdtype,stream);
    } else if(use_q4) launch_gemm_tiled_q4(X,m.wq,m.sc,Y,S,OUT,IN,gd_sdtype,stream);
    else              launch_gemv_f32(X,m.wf,Y,IN,OUT,S,stream);
}

// Dense decode-type attention (S=1, verify, concat, batch) uses the split-KV kernel by default (R15);
// G4_ATTN_SPLIT=0 restores the original per-query-head kernels.
static bool g4_attn_split(){ static const bool on=[](){ const char* v=getenv("G4_ATTN_SPLIT"); return !(v && strcmp(v,"0")==0); }(); return on; }

struct G4DecodeGraph {
    cudaStream_t stream=nullptr;
    cudaGraphExec_t executable=nullptr;
    unsigned long long steps=0, updates=0, update_failures=0, instantiations=0;

    ~G4DecodeGraph(){
        if(executable) cudaGraphExecDestroy(executable);
        if(stream) cudaStreamDestroy(stream);
    }
};

static bool g4d_graph_enabled(){
    static const bool enabled=[](){ const char* v=getenv("G4_CUDA_GRAPH"); return v && strcmp(v,"1")==0; }();
    return enabled;
}

static G4GenerationStats g4d_graph_totals;

template<class Enqueue>
static void g4d_run_graph(Enqueue &&enqueue, int base){
    // Same serialized, single-device lifetime as the decode scratch buffers.
    static G4DecodeGraph graph;
    static const bool stats=[](){ const char* v=getenv("G4_CUDA_GRAPH_STATS"); return v && strcmp(v,"1")==0; }();
    using Clock=std::chrono::steady_clock;
    const auto elapsed_us=[](Clock::time_point a, Clock::time_point b){
        return std::chrono::duration<double,std::micro>(b-a).count();
    };
    if(!graph.stream) DK(cudaStreamCreateWithFlags(&graph.stream,cudaStreamNonBlocking));
    const auto input_begin=Clock::now();
    // Device inputs and benchmark timing events may have been queued on stream 0.
    DK(cudaStreamSynchronize(0));
    const auto capture_begin=Clock::now();
    DK(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeGlobal));
    const cudaError_t enqueue_status=enqueue(graph.stream);
    const cudaError_t launch_status=cudaGetLastError();
    cudaGraph_t captured=nullptr;
    const cudaError_t capture_status=cudaStreamEndCapture(graph.stream,&captured);
    const auto capture_end=Clock::now();
    if(enqueue_status!=cudaSuccess || launch_status!=cudaSuccess || capture_status!=cudaSuccess || !captured){
        fprintf(stderr,"G4_CUDA_GRAPH capture failed base=%d enqueue=%s launch=%s capture=%s graph=%p\n",
                base,cudaGetErrorString(enqueue_status),cudaGetErrorString(launch_status),
                cudaGetErrorString(capture_status),(void*)captured);
        if(captured) cudaGraphDestroy(captured);
        exit(1);
    }

    if(graph.executable){
        cudaGraphExecUpdateResultInfo info{};
        const cudaError_t status=cudaGraphExecUpdate(graph.executable,captured,&info);
        if(status==cudaSuccess && info.result==cudaGraphExecUpdateSuccess){
            ++graph.updates;
        } else {
            ++graph.update_failures;
            fprintf(stderr,"G4_CUDA_GRAPH update failed base=%d status=%s result=%d node=%p from=%p; reinstantiating\n",
                    base,cudaGetErrorString(status),(int)info.result,(void*)info.errorNode,(void*)info.errorFromNode);
            if(status!=cudaSuccess && status!=cudaErrorGraphExecUpdateFailure){
                cudaGraphDestroy(captured);
                DK(status);
            }
            // A supported update fallback must not poison the next launch check.
            const cudaError_t last_error=cudaGetLastError();
            if(last_error!=cudaSuccess && last_error!=cudaErrorGraphExecUpdateFailure){
                cudaGraphDestroy(captured);
                DK(last_error);
            }
            DK(cudaGraphExecDestroy(graph.executable));
            graph.executable=nullptr;
        }
    }
    if(!graph.executable){
        DK(cudaGraphInstantiate(&graph.executable,captured,0));
        ++graph.instantiations;
    }
    DK(cudaGraphDestroy(captured));
    const auto update_end=Clock::now();
    DK(cudaGraphLaunch(graph.executable,graph.stream));
    const auto launch_end=Clock::now();
    DK(cudaStreamSynchronize(graph.stream));
    const auto sync_end=Clock::now();
    ++graph.steps;
    g4d_graph_totals.planned_captures=graph.steps;
    g4d_graph_totals.graph_updates=graph.updates;
    g4d_graph_totals.update_failures=graph.update_failures;
    g4d_graph_totals.instantiations=graph.instantiations;
    g4d_graph_totals.reinstantiations=graph.instantiations?graph.instantiations-1:0;
    g4d_graph_totals.capture_us+=elapsed_us(capture_begin,capture_end);
    g4d_graph_totals.update_us+=elapsed_us(capture_end,update_end);
    if(stats)
        fprintf(stderr,"G4_CUDA_GRAPH_STATS base=%d step=%llu input_sync_us=%.6f capture_us=%.6f update_us=%.6f launch_us=%.6f sync_us=%.6f updates=%llu update_failures=%llu instantiations=%llu reinstantiations=%llu\n",
                base,graph.steps,elapsed_us(input_begin,capture_begin),elapsed_us(capture_begin,capture_end),
                elapsed_us(capture_end,update_end),elapsed_us(update_end,launch_end),elapsed_us(launch_end,sync_end),
                graph.updates,graph.update_failures,graph.instantiations,graph.instantiations-1);
}

using G4PreparedStep = std::function<cudaError_t(cudaStream_t,int,const int*,float*)>;

static void gemma4_decode_impl(const ModelConfig &c, const G4Weights &w, G4DecodeState &st,
                   const int *d_tokens, int Sq,
                   const float *d_cos_f, const float *d_sin_f,
                   const float *d_cos_s, const float *d_sin_s,
                   float *d_logits, int commit, int use_q4, int last_only, int amort_mm,
                   float *d_prenorm, bool graph_request, G4PreparedStep *prepared=nullptr) {
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

    if(use_q4==2){
        size_t max_in=(size_t)H;
        if((size_t)FFN>max_in) max_in=FFN;
        if((size_t)NQ*hd_max>max_in) max_in=(size_t)NQ*hd_max;
        g4d_reserve_quant((size_t)Sq*max_in);
        (void)g4d_mmq_prefill();
        (void)g4d_mmq_min_s();
    }
    const bool attn_split=g4_attn_split() && !(flash_prefill && Sq>=flash_min_s);
    if(attn_split) attn_split_reserve(Sq,NQ,st.max_len,hd_max);   // before any capture; max_len bound
    if(Sq==1 && c.num_layers>0 && !attn_split)
        attn_smem_guard((const void *)attn_decode_kernel,
                        (size_t)(hd_max+base+Sq)*sizeof(float), "attn_decode", base+Sq);

    // Reservation and host state changes remain outside the enqueue-only body.
    auto enqueue=[=,&c,&w,&st](cudaStream_t stream,int base,const int *d_tokens,float *d_logits)->cudaError_t{
    launch_embedding(d_tokens, w.embed, x, Sq, H, stream);
    if(c.embed_scale>0.f) launch_scale(x,x,c.embed_scale,SH,stream);

    for(int l=0;l<c.num_layers;l++){
        const G4Layer &L=w.layers[l];
        const int full=L.is_full;
        const int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        const int qdim=NQ*hd, kvdim=nkv*hd;
        const float *cosT=full?d_cos_f:d_cos_s, *sinT=full?d_sin_f:d_sin_s;
        const int window=full?0:c.sliding_window;

        launch_rmsnorm(x, L.input_ln, norm, Sq, H, eps, stream);
        mm(L.q, norm, qb, H, qdim, Sq, use_q4, stream);
        launch_rmsnorm(qb, L.q_norm, qn, Sq*NQ, hd, eps, stream);
        launch_rope(qn, cosT+(size_t)base*hd, sinT+(size_t)base*hd, Sq, NQ, hd, stream);
        mm(L.k, norm, kb, H, kvdim, Sq, use_q4, stream);
        // One token writes its cache slice directly; kb remains raw for K-equals-V.
        // fp32 KV: one token writes its cache slice directly. 16-bit KV: scratch, then kv_store.
        const bool kv32=g4_kv_dtype()==KV_FP32;
        float *kn_out=(Sq==1 && kv32)?st.Kc[l]+(size_t)base*kvdim:kn;
        float *vn_out=(Sq==1 && kv32)?st.Vc[l]+(size_t)base*kvdim:vn;
        if(full && c.attention_k_eq_v) launch_rmsnorm_noscale(kb, vn_out, Sq*nkv, hd, eps, stream);  // V=v_norm(k_proj), rope 전
        else { mm(L.v, norm, vb, H, kvdim, Sq, use_q4, stream); launch_rmsnorm_noscale(vb, vn_out, Sq*nkv, hd, eps, stream); }
        launch_rmsnorm(kb, L.k_norm, kn_out, Sq*nkv, hd, eps, stream);
        launch_rope(kn_out, cosT+(size_t)base*hd, sinT+(size_t)base*hd, Sq, nkv, hd, stream);
        // Multi-token calls retain the scratch-to-cache append path.
        if(Sq!=1 && kv32){
            DK(cudaMemcpy(st.Kc[l]+(size_t)base*kvdim, kn, (size_t)Sq*kvdim*4, cudaMemcpyDeviceToDevice));
            DK(cudaMemcpy(st.Vc[l]+(size_t)base*kvdim, vn, (size_t)Sq*kvdim*4, cudaMemcpyDeviceToDevice));
        } else if(!kv32){
            kv_store(kn, kv_at(st.Kc[l],(size_t)base*kvdim), (size_t)Sq*kvdim, 0, stream);
            kv_store(vn, kv_at(st.Vc[l],(size_t)base*kvdim), (size_t)Sq*kvdim, 1, stream);
        }
        if(flash_prefill && Sq>=flash_min_s){
            if(kv32) launch_flash_attn<float>(qn, st.Kc[l], st.Vc[l], attn, Sq, NQ, nkv, hd, base, stream, nullptr, ascale, window);
            else if(g4_kv_dtype()==KV_F16) launch_flash_attn<__half>(qn, (const __half*)st.Kc[l], (const __half*)st.Vc[l], attn, Sq, NQ, nkv, hd, base, stream, nullptr, ascale, window);
            else launch_flash_attn<__nv_bfloat16>(qn, (const __nv_bfloat16*)st.Kc[l], (const __nv_bfloat16*)st.Vc[l], attn, Sq, NQ, nkv, hd, base, stream, nullptr, ascale, window);
        }
        else if(attn_split)
            launch_attn_decode_split(qn, st.Kc[l], st.Vc[l], attn, Sq, NQ, nkv, hd, base, 1, nullptr, base+Sq-1, 0,
                                     attn_split_ns_cap(st.max_len), stream, ascale, window);
        else {
            if(!kv32){ fprintf(stderr,"G4_KV_DTYPE other than fp32 requires G4_ATTN_SPLIT=1 for decode attention\n"); exit(1); }
            launch_attn_decode(qn, st.Kc[l], st.Vc[l], attn, Sq, NQ, nkv, hd, base, stream, ascale, window);
        }
        mm(L.o, attn, ob, qdim, H, Sq, use_q4, stream);
        launch_rmsnorm(ob, L.post_attn_ln, tmp, Sq, H, eps, stream);
        launch_add(after, x, tmp, (int)SH, stream);   // after = x + post_attn_ln(o_proj)

        launch_rmsnorm(after, L.pre_ffn_ln, norm, Sq, H, eps, stream);
        mm(L.gate, norm, g, H, FFN, Sq, use_q4, stream);
        mm(L.up,   norm, u, H, FFN, Sq, use_q4, stream);
        launch_geglu(g, u, act, (size_t)Sq*FFN, stream);
        mm(L.down, act, dn, FFN, H, Sq, use_q4, stream);
        launch_rmsnorm(dn, L.post_ffn_ln, tmp, Sq, H, eps, stream);
        launch_add(x, after, tmp, (int)SH, stream);
        if(L.layer_scalar!=1.0f) launch_scale(x, x, L.layer_scalar, SH, stream);
    }

    // pre-norm residual 노출(3관문 assistant hook): HF hidden_states[-1] == 마지막 층 출력(final-norm 전, 실측 확정)
    if(d_prenorm){
        if(stream){
            const cudaError_t status=cudaMemcpyAsync(d_prenorm,x,SH*4,cudaMemcpyDeviceToDevice,stream);
            if(status!=cudaSuccess) return status;
        }
        else DK(cudaMemcpy(d_prenorm, x, SH*4, cudaMemcpyDeviceToDevice));
    }
    launch_rmsnorm(x, w.final_norm, norm, Sq, H, eps, stream);
    const float *hn = last_only ? norm+(size_t)(Sq-1)*H : norm;
    int rows = last_only ? 1 : Sq;
    mm(w.lm_head, hn, d_logits, H, V, rows, use_q4, stream);
    if(c.final_logit_softcapping>0.f) launch_softcap(d_logits, d_logits, c.final_logit_softcapping, (size_t)rows*V, stream);
    return cudaSuccess;
    };
    if(prepared){ *prepared=enqueue; return; }
    if(graph_request && Sq==1 && commit!=0 && amort_mm==0 && g4d_graph_enabled()){
        g4d_run_graph([&](cudaStream_t stream){return enqueue(stream,base,d_tokens,d_logits);},base);
        st.n_past += Sq;
    } else {
        DK(enqueue(0,base,d_tokens,d_logits));
        if(commit) st.n_past += Sq;
        DK(cudaDeviceSynchronize());
    }
}

void gemma4_decode(const ModelConfig &c, const G4Weights &w, G4DecodeState &st,
                   const int *d_tokens, int Sq,
                   const float *d_cos_f, const float *d_sin_f,
                   const float *d_cos_s, const float *d_sin_s,
                   float *d_logits, int commit, int use_q4, int last_only, int amort_mm,
                   float *d_prenorm){
    gemma4_decode_impl(c,w,st,d_tokens,Sq,d_cos_f,d_sin_f,d_cos_s,d_sin_s,
                      d_logits,commit,use_q4,last_only,amort_mm,d_prenorm,false);
}

void gemma4_decode_step(const ModelConfig &c, const G4Weights &w, G4DecodeState &st,
                   const int *d_tokens, int Sq,
                   const float *d_cos_f, const float *d_sin_f,
                   const float *d_cos_s, const float *d_sin_s,
                   float *d_logits, int commit, int use_q4, int last_only, int amort_mm,
                   float *d_prenorm){
    gemma4_decode_impl(c,w,st,d_tokens,Sq,d_cos_f,d_sin_f,d_cos_s,d_sin_s,
                      d_logits,commit,use_q4,last_only,amort_mm,d_prenorm,true);
}

#include "gemma4_decode_graph.cuh"
