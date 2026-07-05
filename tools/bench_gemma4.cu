// Gemma4 31B 성능 벤치 (STEP B): single/batch decode t/s + prefill t/s. roofline 대조용.
//   cudaEvent 타이밍. rope=엔진 생성 dummy(타이밍은 rope 값 무관 — 연산량은 위치 n_past가 결정, 골든 17행 제약 우회).
//   decode t/s = matmul/대역폭 바운드(roofline 17.4 관련): n_past=0부터 ngen 스텝(attention over ≤ngen keys=미미) 타이밍.
//   ★warm 프로토콜: 고정 워밍 런(throwaway) 후 본 측정. B 스윕 전 메모리 예산 가드(unified 크래시 방지).
//   사용법: ./bench_gemma4 <weights_dir> [uq=1] [ctx=256] [ngen=64] [B,...]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../models/gemma4_moe.h"   // 26B-A4B MoE: gemma4_moe_decode / gemma4_moe_decode_batch (dense∥MoE)
#include "../kernels/rope.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static float ev(cudaEvent_t a,cudaEvent_t b){ float ms; cudaEventElapsedTime(&ms,a,b); return ms; }

// 26B-A4B MoE single-decode roofline 상한(t/s) — active-bytes÷대역폭. weight 이동만 계상(상한, 예측 아님).
//   active = lm_head(tied·매토큰 logits) + Σlayer(attn q/k/[v]/o + 병렬 dense FFN gate·up·down + router + top_k·expert gate_up·down).
//   ★plan.py active_params와 달리 26B 고유의 병렬 dense FFN(intermediate_size)을 포함 → 더 정확(낮은) 천장.
//   bytes/param = 0.5×1.0625 (w4a8, plan.py 규약). read BW=255.8 GB/s(GB10 probe 실측).
static double moe_single_roofline_ts(const ModelConfig &c, double bw_gbps){
    const double H=c.hidden_size, V=c.vocab_size, FFN=c.intermediate_size;
    const int NQ=c.num_q_heads, nkv_s=c.num_kv_heads, hd_s=c.head_dim;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s;
    const int nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const double E=c.num_experts, K=c.num_experts_per_tok, MI=c.moe_intermediate_size;
    double active = V*H;   // lm_head (tied — logits는 매 토큰 읽음)
    for(int l=0;l<c.num_layers;l++){
        int full=c.layer_is_full?c.layer_is_full[l]:1;
        double hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
        active += H*(NQ*hd);                                   // q_proj
        active += H*(nkv*hd);                                  // k_proj
        if(!(full && c.attention_k_eq_v)) active += H*(nkv*hd);// v_proj (full k_eq_v면 공유·미계상)
        active += (NQ*hd)*H;                                   // o_proj
        active += 3.0*H*FFN;                                   // 병렬 dense FFN (gate·up·down)
        active += H*E;                                         // router.proj
        active += K*(2.0*MI*H + H*MI);                         // top_k active expert (gate_up·down)
    }
    const double bytes_per=0.5*1.0625;                         // w4a8 (4bit weight + scale)
    return (bw_gbps*1e9)/(active*bytes_per);
}

// per-seq KV bytes(선형 캡 max_len): full=hd_f·nkv_f, sliding=hd_s·nkv_s, ×2(K,V)×4(fp32).
static size_t per_seq_bytes(const ModelConfig&c,int ml){
    int hd_s=c.head_dim,nkv_s=c.num_kv_heads;
    int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    size_t b=0; for(int l=0;l<c.num_layers;l++){ int full=c.layer_is_full?c.layer_is_full[l]:1;
        int hd=full?hd_f:hd_s,nkv=full?nkv_f:nkv_s; b+=(size_t)ml*nkv*hd*2*4; } return b;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IOLBF,0);
    if(argc<2){ fprintf(stderr,"usage: %s <weights_dir> [uq=1] [ctx=256] [ngen=64] [B,...]\n",argv[0]); return 1; }
    std::string WD=argv[1];
    int uq=argc>2?atoi(argv[2]):1;
    int ctx=argc>3?atoi(argv[3]):256;
    int ngen=argc>4?atoi(argv[4]):64;
    std::vector<int> Bs; for(int i=5;i<argc;i++) Bs.push_back(atoi(argv[i]));
    if(Bs.empty()) Bs={1,2,4,8,16};
    ModelConfig c=load_config_gemma4((WD+"/manifest.json").c_str());
    const int V=c.vocab_size;
    const int hd_s=c.head_dim, hd_f=c.global_head_dim?c.global_head_dim:hd_s;
    int Bmax=0; for(int b:Bs) if(b>Bmax)Bmax=b;
    int MAXLEN=ctx+ngen+8;
    printf("# Gemma4 bench: L=%d V=%d hd_f=%d hd_s=%d uq=%d(1=q4) ctx=%d ngen=%d MAXLEN=%d\n",
           c.num_layers,V,hd_f,hd_s,uq,ctx,ngen,MAXLEN);
    // dummy dual rope(타이밍용, 값 무관) — 올바른 모양 [MAXLEN, hd]
    float *cf,*sf,*cs,*ss;
    CK(cudaMalloc(&cf,(size_t)MAXLEN*hd_f*4)); CK(cudaMalloc(&sf,(size_t)MAXLEN*hd_f*4));
    CK(cudaMalloc(&cs,(size_t)MAXLEN*hd_s*4)); CK(cudaMalloc(&ss,(size_t)MAXLEN*hd_s*4));
    launch_gen_rope(cf,sf,MAXLEN,hd_f,1e4f); launch_gen_rope(cs,ss,MAXLEN,hd_s,1e4f); CK(cudaDeviceSynchronize());
    G4Weights w=load_weights_gemma4(WD.c_str(),c,uq!=0?1:0,uq==0?1:0);
    size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB);
    // ★ cudaMemGetInfo free는 unified에서 page cache 제외(과소보고) — /proc/meminfo MemAvailable(회수 가능 포함)이
    //   실제 가용. 둘 중 큰 값(가드 정확화, 완화 아님 — bench_serve_spec.cu:628 idiom).
    { FILE*f=fopen("/proc/meminfo","r"); if(f){ char line[128]; long kb;
        while(fgets(line,sizeof(line),f)){ if(sscanf(line,"MemAvailable: %ld",&kb)==1){ size_t a=(size_t)kb*1024; if(a>freeB)freeB=a; break; } } fclose(f); } }
    size_t perseq=per_seq_bytes(c,MAXLEN);
    printf("# per-seq KV(max_len 선형)=%.1fMB, free=%.1fGB (0.7·free=%.1fGB 예산)\n",perseq/1e6,freeB/1e9,0.7*freeB/1e9);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    int *d_tok,*d_pos; float *d_logit;
    CK(cudaMalloc(&d_tok,Bmax*4)); CK(cudaMalloc(&d_pos,Bmax*4)); CK(cudaMalloc(&d_logit,(size_t)Bmax*V*4));

    const bool is_moe = c.num_experts>0;   // 26B-A4B: dense∥MoE 경로(gemma4_moe_*) 사용
    if(is_moe) printf("# ★26B-A4B MoE 모드: E=%d top_k=%d moe_ffn=%d — gemma4_moe_decode(dense∥MoE) 경로\n",
                      c.num_experts,c.num_experts_per_tok,c.moe_intermediate_size);
    printf("\n  %-4s | %10s %10s | %9s | %s\n","B","dec/stream","dec/aggr","prefill","note (t/s)");
    printf("  %s\n",std::string(60,'-').c_str());
    double prefill_ts=0; (void)prefill_ts;
    for(int B:Bs){
        if((double)B*perseq > 0.7*(double)freeB){ printf("  %-4d | %10s %10s | %9s | 예산 초과(스킵, 시스템 다운 방지)\n",B,"-","-","-"); continue; }
        G4BatchState st=alloc_g4_batch_state(c,B,MAXLEN); st.B=B; st.n_past=0;
        std::vector<int> tk(B,1),pos(B,0);
        CK(cudaMemcpy(d_tok,tk.data(),B*4,cudaMemcpyHostToDevice));
        // warm-up: 8스텝(클럭 ramp) — n_past 전진 후 리셋. MoE는 gemma4_moe_decode_batch(same-length=null pos).
        for(int s=0;s<8;s++){ if(is_moe) gemma4_moe_decode_batch(c,w,st,d_tok,cf,sf,cs,ss,d_logit,1,uq,nullptr,nullptr);
                              else       gemma4_decode_batch(c,w,st,d_tok,cf,sf,cs,ss,d_logit,1,uq); }
        CK(cudaDeviceSynchronize()); st.n_past=0;   // 타이밍은 n_past=0부터(attention 미미=matmul 바운드)
        // 타이밍: ngen 스텝(same-length, commit=1)
        CK(cudaEventRecord(t0));
        for(int s=0;s<ngen;s++){ if(is_moe) gemma4_moe_decode_batch(c,w,st,d_tok,cf,sf,cs,ss,d_logit,1,uq,nullptr,nullptr);
                                 else       gemma4_decode_batch(c,w,st,d_tok,cf,sf,cs,ss,d_logit,1,uq); }
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        double ms=ev(t0,t1); double per_stream=ngen/(ms/1000.0); double aggr=(double)B*ngen/(ms/1000.0);
        printf("  %-4d | %10.2f %10.2f | %9s | %.3f ms/%d스텝\n",B,per_stream,aggr,"-",ms,ngen);
        free_g4_batch_state(st);
    }

    // ── single-path decode (gemma4_decode, gemv 경로) — roofline 공정 대조 헤드라인 ──
    //   batch-path(gemm_batch M=1)와 달리 single은 gemv_w4a8(W4A8)/gemm_tiled(q4). 27b 11.9/62.5%는 W4A8 gemv.
    //   q4=검증된 경로. W4A8=27b와 동일 gemv족(perf-preview — gemma4 W4A8 correctness golden 미검증, 성능만).
    printf("\n  === single-path decode (n_past=0 시작 matmul 바운드) ===\n");
    // 26B roofline 상한(active-bytes÷255.8GB/s) — 헤드라인 달성률 기준. 31B(dense)는 17.4.
    const double single_roof = is_moe ? moe_single_roofline_ts(c,255.8) : 17.4;
    if(is_moe) printf("  ★26B MoE roofline 상한: %.1f t/s (active-bytes÷255.8GB/s, 병렬 dense FFN 포함)\n", single_roof);
    for(int sq : std::vector<int>{1,2}){   // 1=q4(검증), 2=W4A8(perf-preview)
        G4DecodeState st=alloc_g4_decode_state(c,MAXLEN); reset_g4_decode_state(st);
        int tok=1; int* d1; CK(cudaMalloc(&d1,4)); CK(cudaMemcpy(d1,&tok,4,cudaMemcpyHostToDevice));
        for(int s=0;s<8;s++){ if(is_moe) gemma4_moe_decode(c,w,st,d1,1,cf,sf,cs,ss,d_logit,1,sq,1);
                              else       gemma4_decode(c,w,st,d1,1,cf,sf,cs,ss,d_logit,1,sq,1); }  // 워밍
        CK(cudaDeviceSynchronize()); reset_g4_decode_state(st);
        CK(cudaEventRecord(t0));
        for(int s=0;s<ngen;s++){ if(is_moe) gemma4_moe_decode(c,w,st,d1,1,cf,sf,cs,ss,d_logit,1,sq,1);
                                 else       gemma4_decode(c,w,st,d1,1,cf,sf,cs,ss,d_logit,1,sq,1); }
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        double ms=ev(t0,t1); double ts=ngen/(ms/1000.0);
        printf("  %-6s single decode: %.2f t/s (%.3f ms/%d스텝) — roofline %.1f 대비 %.1f%%  [검증 경로]\n",
               sq==2?"W4A8":"q4", ts, ms, ngen, single_roof, 100.0*ts/single_roof);
        cudaFree(d1); free_g4_decode_state(st);
    }

    // 프리필 t/s (B=1 통짜) — 별도 G4DecodeState
    { G4DecodeState st=alloc_g4_decode_state(c,MAXLEN);
      std::vector<int> tk(ctx,1); int* d_p; CK(cudaMalloc(&d_p,ctx*4)); CK(cudaMemcpy(d_p,tk.data(),ctx*4,cudaMemcpyHostToDevice));
      reset_g4_decode_state(st);
      if(is_moe) gemma4_moe_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1); else gemma4_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1);   // 워밍
      CK(cudaDeviceSynchronize()); reset_g4_decode_state(st);
      CK(cudaEventRecord(t0));
      if(is_moe) gemma4_moe_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1); else gemma4_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1);
      CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
      double ms=ev(t0,t1); prefill_ts=ctx/(ms/1000.0);
      printf("\n  프리필(B=1 통짜 %d토큰): %.2f tok/s (%.3f ms) — 참고치(chunked 아님, TTFT 주장 아님)\n",ctx,prefill_ts,ms);
      cudaFree(d_p); free_g4_decode_state(st);
    }

    printf("\n  (decode/stream=단일 시퀀스 t/s=roofline 대조 수치. decode/aggr=B×/stream=총처리량. n_past=0 시작=matmul 바운드.)\n");
    cudaFree(d_tok);cudaFree(d_pos);cudaFree(d_logit);cudaFree(cf);cudaFree(sf);cudaFree(cs);cudaFree(ss);
    free_weights_gemma4(w); free_config_gemma4(c); return 0;
}
