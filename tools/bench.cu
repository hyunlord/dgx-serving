// 챕터6 STEP 2: 측정 하네스 — TTFT(prefill) + TPOT(per-token decode) + throughput, context 스윕.
//   cudaEvent 타이밍(clock() 아님). rope 엔진 생성(32K native). 합성 토큰(속도만, 정확성은 별도 증명됨).
//   B=1: qwen35_decode (prefill Sq=C 1콜 + decode 루프). B>1: qwen35_decode_batch (dense만; MoE batch 미구현).
//   사용법: ./bench <weights_dir> [use_q4=1] [ctx,...] [ngen,...] [B,...]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/engine.h"
#include "../runtime/decode_state.h"
#include "../models/qwen35.h"
#include "../kernels/rope.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

static float ev(cudaEvent_t a, cudaEvent_t b){ float ms; cudaEventElapsedTime(&ms,a,b); return ms; }

// qwen35_decode workspace 예산 추정 (Sq=prefill 길이). GB10 unified 크래시 방지용 — O(Sq) 버퍼 합.
//   주: snap 수정 후에도 gm/um/act(3×FFN), mixed/mconv 등 O(Sq) 존재 → 가드 필수(메모리 예산 사전계산 원칙).
static size_t est_workspace_bytes(const ModelConfig &c, int Sq){
    size_t H=c.hidden_size, FFN=c.intermediate_size, HD=c.head_dim;
    size_t qdim2=(size_t)c.num_q_heads*HD*(c.attn_output_gate?2:1), qdim=(size_t)c.num_q_heads*HD, kvdim=(size_t)c.num_kv_heads*HD;
    size_t convd=cfg_gdn_conv_dim(&c), vdim=cfg_gdn_value_dim(&c), kdim=cfg_gdn_key_dim(&c), nvh=c.gdn_num_v_heads, hv=c.gdn_head_v_dim;
    // O(Sq) 폭: dense backbone
    size_t perTok = 8*H + qdim2 + 4*qdim + 3*kvdim + 3*convd + 2*kdim + 4*vdim + 4*nvh;
    if (c.num_experts>0) perTok += (size_t)c.num_experts + 3*(size_t)c.shared_expert_intermediate_size + H; // MoE moe_mlp O(Sq)
    else                 perTok += 3*FFN;                                                                   // dense gm/um/act
    size_t snap = (size_t)nvh*hv*hv;   // snapshot_all=0(plain) → Sq 비례 아님(수정됨)
    return ((size_t)Sq*perTok + snap) * 4;
}
// KV 캐시 예산 (full-attn 레이어만, 길이 len). [len, NKV, HD] × (K+V) × n_full.
static size_t est_kv_bytes(const ModelConfig &c, int len){
    int nfull=0; for(int i=0;i<c.num_layers;i++) if(c.layer_is_full?c.layer_is_full[i]:1) nfull++;
    return (size_t)nfull*(size_t)len*c.num_kv_heads*c.head_dim*2*4;
}

int main(int argc,char**argv){
    setvbuf(stdout, NULL, _IOLBF, 0);   // line-buffered: 행별 즉시 flush (백그라운드 로그 가시화)
    if(argc<2){ fprintf(stderr,"usage: %s <weights_dir> [use_q4=1]\n",argv[0]); return 1; }
    std::string WD=argv[1];
    int use_q4 = argc>2 ? atoi(argv[2]) : 1;
    Engine e=engine_load(WD.c_str(), /*q4*/use_q4>=1?1:0, /*fp32*/use_q4==0?1:0);
    const int H=e.c.hidden_size, V=e.c.vocab_size, rot=e.c.rotary_dim, L=e.c.num_layers;
    const bool moe = e.c.num_experts>0;
    printf("# model=%s H=%d L=%d V=%d %s use_q4=%d (1=q4,2=W4A8)\n",
           WD.c_str(),H,L,V, moe?"MoE":"dense", use_q4);

    std::vector<int> ctxs={128,2048,8192,32768};
    std::vector<int> ngens={128,512};
    int MAXLEN=32768+512+8;

    // rope 테이블 엔진 생성 (theta from config — 골든 의존 없음)
    float *d_cos,*d_sin; CK(cudaMalloc(&d_cos,(size_t)MAXLEN*rot*4)); CK(cudaMalloc(&d_sin,(size_t)MAXLEN*rot*4));
    launch_gen_rope(d_cos,d_sin,MAXLEN,rot,e.c.rope_theta); CK(cudaDeviceSynchronize());

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float *d_logits; CK(cudaMalloc(&d_logits,(size_t)32*V*4));   // B<=32

    // warmup: CUDA context/JIT 초기화 제외 (첫 TTFT 오염 방지)
    {
        int *d_wt; CK(cudaMalloc(&d_wt,16*4)); int wt[16]; for(int i=0;i<16;i++)wt[i]=i;
        CK(cudaMemcpy(d_wt,wt,16*4,cudaMemcpyHostToDevice));
        DecodeState ws=alloc_decode_state(e.c, 32);
        qwen35_decode(e.c,e.w,ws,d_wt,8,d_cos,d_sin,d_logits,nullptr,1,nullptr,use_q4,nullptr,1);
        for(int i=0;i<3;i++) qwen35_decode(e.c,e.w,ws,d_wt+8+i,1,d_cos,d_sin,d_logits,nullptr,1,nullptr,use_q4);
        CK(cudaDeviceSynchronize()); free_decode_state(ws); cudaFree(d_wt);
    }
    // 합성 토큰 (vocab 내 임의)
    std::vector<int> htok(MAXLEN); for(int i=0;i<MAXLEN;i++) htok[i]=(i*131+7)%V;
    int *d_tok; CK(cudaMalloc(&d_tok,(size_t)MAXLEN*4)); CK(cudaMemcpy(d_tok,htok.data(),(size_t)MAXLEN*4,cudaMemcpyHostToDevice));

    printf("# %-4s %-6s %-7s | %-10s %-10s %-12s %-12s\n","B","ctx","ngen","TTFT(ms)","TPOT(ms)","decode tok/s","prefill tok/s");
    // ★ 단발 prefill Sq 상한 (컴퓨트-행 방지): decode 커널(gemv_q4=untiled GEMM, O(Sq²) attn, 순차 GDN)은
    //   짧은 Sq 전제 — 32K 단발 prefill은 ~825 TFLOP/71M블록 = 수십 분 GPU 점유 → GB10 행/크래시(2차 크래시 원인).
    //   메모리 가드로는 못 막음(메모리는 안전). 청크 prefill(2번) 구현 전까지 단발 상한으로 차단.
    const int PREFILL_SQ_CAP = 2048;
    // ---- B=1 (qwen35_decode: prefill Sq=C 1콜 + decode 루프) ----
    for(int C: ctxs) for(int N: ngens){
        if(C > PREFILL_SQ_CAP){
            printf("  %-4d %-6d %-7d | SKIP — 단발 prefill Sq=%d > 상한 %d (청크 prefill 필요, 컴퓨트-행 방지)\n",
                   1,C,N,C,PREFILL_SQ_CAP);
            continue;
        }
        // 메모리 가드 (GB10 unified 크래시 방지): 예상 workspace+KV가 가용 초과면 skip+경고.
        size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB);
        size_t need=est_workspace_bytes(e.c,C)+est_kv_bytes(e.c,C+N+1);
        if(need > (size_t)(0.75*(double)freeB)){
            printf("  %-4d %-6d %-7d | SKIP — 예산 %.1fGB > 가용 %.1fGB×0.75 (메모리 가드)\n",
                   1,C,N, need/1e9, freeB/1e9);
            continue;
        }
        DecodeState st=alloc_decode_state(e.c, C+N+1);
        // prefill: C 토큰 한 번에 (TTFT)
        CK(cudaDeviceSynchronize()); cudaEventRecord(t0);
        qwen35_decode(e.c,e.w,st,d_tok,C,d_cos,d_sin,d_logits,nullptr,1,nullptr,use_q4,nullptr,/*last_logits_only*/1);
        cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
        float ttft=ev(t0,t1);
        // decode N 토큰 (per-token)
        CK(cudaDeviceSynchronize()); cudaEventRecord(t0);
        for(int i=0;i<N;i++) qwen35_decode(e.c,e.w,st,d_tok+C+i,1,d_cos,d_sin,d_logits,nullptr,1,nullptr,use_q4);
        cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
        float dec=ev(t0,t1);
        printf("  %-4d %-6d %-7d | %-10.2f %-10.3f %-12.1f %-12.1f\n",
               1,C,N,ttft, dec/N, N*1000.0/dec, C*1000.0/ttft);
        free_decode_state(st);
    }
    // ---- B>1 throughput (dense만; batch 경로) ----
    if(!moe){
        std::vector<int> Bs={8};
        for(int B: Bs) for(int C: {2048}) for(int N: ngens){
            size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB);
            size_t need=(size_t)B*est_kv_bytes(e.c,C+N+1) + (size_t)B*est_workspace_bytes(e.c,1);
            if(need > (size_t)(0.75*(double)freeB)){
                printf("  %-4d %-6d %-7d | SKIP — KV예산 %.1fGB > 가용 %.1fGB×0.75 (메모리 가드)\n",
                       B,C,N, need/1e9, freeB/1e9); continue;
            }
            BatchDecodeState bst=alloc_batch_decode_state(e.c,B,C+N+1);
            std::vector<int> bt(B); for(int j=0;j<B;j++) bt[j]=htok[j];
            int *dbt; CK(cudaMalloc(&dbt,B*4));
            // batch prefill = C 스텝(1토큰/스텝), 그 후 N decode (throughput만)
            CK(cudaDeviceSynchronize()); cudaEventRecord(t0);
            for(int p=0;p<C;p++){ for(int j=0;j<B;j++)bt[j]=htok[(p+j)%V]; CK(cudaMemcpy(dbt,bt.data(),B*4,cudaMemcpyHostToDevice));
                qwen35_decode_batch(e.c,e.w,bst,dbt,d_cos,d_sin,d_logits,use_q4); }
            cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); float pf=ev(t0,t1);
            CK(cudaDeviceSynchronize()); cudaEventRecord(t0);
            for(int i=0;i<N;i++){ CK(cudaMemcpy(dbt,bt.data(),B*4,cudaMemcpyHostToDevice));
                qwen35_decode_batch(e.c,e.w,bst,dbt,d_cos,d_sin,d_logits,use_q4); }
            cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); float dec=ev(t0,t1);
            printf("  %-4d %-6d %-7d | %-10s %-10.3f %-12.1f %-12s (batch, total %.1f tok/s)\n",
                   B,C,N,"-", dec/N, B*N*1000.0/dec, "-", B*N*1000.0/dec);
            free_batch_decode_state(bst); cudaFree(dbt);
        }
    } else printf("# (MoE B>1 batch decode 미구현 — B=1만. batch MoE는 다음 STEP)\n");

    engine_free(e);
    return 0;
}
