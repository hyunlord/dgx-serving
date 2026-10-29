// Gemma4 31B 성능 벤치 (STEP B): single/batch decode t/s + prefill t/s. roofline 대조용.
//   cudaEvent 타이밍. rope=엔진 생성 dummy(타이밍은 rope 값 무관 — 연산량은 위치 n_past가 결정, 골든 17행 제약 우회).
//   decode t/s = matmul/대역폭 바운드(roofline 17.4 관련): n_past=0부터 ngen 스텝(attention over ≤ngen keys=미미) 타이밍.
//   ★warm 프로토콜: 고정 워밍 런(throwaway) 후 본 측정. B 스윕 전 메모리 예산 가드(unified 크래시 방지).
//   사용법: ./bench_gemma4 <weights_dir> [uq=1] [ctx=256] [ngen=64] [B,...]
//   single decode 입력(env): BENCH_G4_DECODE_INPUT=fixed(기본: 토큰1 반복)|greedy(argmax 피드백),
//     BENCH_G4_DEPTH=d(기본0: 프리필로 KV를 d까지 채운 뒤 n_past=d부터 ngen 스텝), BENCH_G4_SINGLE_UQ=2(해당 uq만).
//     G4_CUDA_GRAPH=0/1: fixed는 기존 gemma4_decode_step 루프(cudaEvent 구간), greedy는 gemma4_generate(호스트 argmax).
//     G4_CUDA_GRAPH=2(미지정 기본): gemma4_generate v2(fixed는 G4_GENERATE_FIXED_INPUT — 매 스텝 입력 토큰1, argmax는 계산만).
//     gemma4_generate 측정 구간 = 호출 전체(스트림·버퍼 준비 포함, R13 하네스와 같은 회계).
//     BENCH_G4_DUMP_LOGITS=<prefix>: 측정 스텝마다 전체 logits를 <prefix>.uq<N>.f32에 기록(정확성 게이트 전용 —
//     D2H·파일 쓰기가 들어가므로 이 실행의 시간은 속도 결과에 쓰지 않는다).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <climits>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../models/gemma4_moe.h"   // 26B-A4B MoE: gemma4_moe_decode / gemma4_moe_decode_batch (dense∥MoE)
#include "../kernels/rope.cuh"
#include "../kernels/kv_store.cuh"
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

// per-seq KV bytes(선형 캡 max_len): full=hd_f·nkv_f, sliding=hd_s·nkv_s, ×2(K,V)×G4_KV_DTYPE 바이트.
static size_t per_seq_bytes(const ModelConfig&c,int ml){
    int hd_s=c.head_dim,nkv_s=c.num_kv_heads;
    int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    size_t b=0; for(int l=0;l<c.num_layers;l++){ int full=c.layer_is_full?c.layer_is_full[l]:1;
        int hd=full?hd_f:hd_s,nkv=full?nkv_f:nkv_s; b+=(size_t)ml*nkv*hd*2*g4_kv_elem(); } return b;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IOLBF,0);
    const char *min_sec_env=getenv("BENCH_G4_PREFILL_MIN_SEC");
    double min_prefill_sec=0;
    if(min_sec_env){
        char *end=nullptr; errno=0; min_prefill_sec=strtod(min_sec_env,&end);
        if(errno || end==min_sec_env || *end || !std::isfinite(min_prefill_sec) || min_prefill_sec<=0 || !std::isfinite(min_prefill_sec*1000.0)){
            fprintf(stderr,"BENCH_G4_PREFILL_MIN_SEC must be finite and positive\n"); return 1;
        }
    }
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
    const char* input_env=getenv("BENCH_G4_DECODE_INPUT");
    const bool greedy_input=input_env && strcmp(input_env,"greedy")==0;
    if(input_env && !greedy_input && strcmp(input_env,"fixed")!=0){ fprintf(stderr,"BENCH_G4_DECODE_INPUT must be fixed or greedy\n"); return 1; }
    const int depth=getenv("BENCH_G4_DEPTH")?atoi(getenv("BENCH_G4_DEPTH")):0;
    const int single_uq=getenv("BENCH_G4_SINGLE_UQ")?atoi(getenv("BENCH_G4_SINGLE_UQ")):0;
    const char* graph_env=getenv("G4_CUDA_GRAPH");
    const int graph_mode=graph_env?atoi(graph_env):2;   // gemma4_generate 기본 = v2(G4_CUDA_GRAPH 미지정)
    if(depth<0){ fprintf(stderr,"BENCH_G4_DEPTH must be >= 0\n"); return 1; }
    int MAXLEN=(ctx>depth?ctx:depth)+ngen+8;
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
    // roofline 상한(active-bytes÷255.8GB/s) — 헤드라인 달성률 기준. config 주도(dense=E/K=0이면 헬퍼가
    //   lm_head+attn+3·H·FFN로 자동 축소 → dense weight-per-token 정확). 하드코딩 금지(모델별 자동 산출).
    const double single_roof = moe_single_roofline_ts(c,255.8);
    printf("  ★%s roofline 상한: %.1f t/s (active-bytes÷255.8GB/s%s)\n",
           is_moe?"MoE":"dense", single_roof, is_moe?", 병렬 dense FFN 포함":"");
    const char* decode_interval_env=getenv("BENCH_G4_DECODE_INTERVAL");
    const bool decode_interval=decode_interval_env && strcmp(decode_interval_env,"1")==0;
    auto decode_epoch=[](){ return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); };
    printf("SINGLE_DECODE_MODE input=%s depth=%d graph=%d ngen=%d\n",greedy_input?"greedy":"fixed",depth,graph_mode,ngen);
    for(int sq : std::vector<int>{1,2}){   // 1=q4(검증), 2=W4A8(perf-preview)
        if(single_uq && sq!=single_uq) continue;
        G4DecodeState st=alloc_g4_decode_state(c,MAXLEN); reset_g4_decode_state(st);
        int tok=1; int* d1; CK(cudaMalloc(&d1,4)); CK(cudaMemcpy(d1,&tok,4,cudaMemcpyHostToDevice));
        // 깊이 d까지 KV 채우기: 토큰1 반복 프리필(2048 청크, MMQ·flash 프리필 경로), n_past=d.
        auto fill_depth=[&](){
            reset_g4_decode_state(st);
            if(depth>0){
                std::vector<int> ones(depth,1); int* dp; CK(cudaMalloc(&dp,(size_t)depth*4));
                CK(cudaMemcpy(dp,ones.data(),(size_t)depth*4,cudaMemcpyHostToDevice));
                for(int o=0;o<depth;o+=2048){ int n=depth-o<2048?depth-o:2048;
                    gemma4_decode(c,w,st,dp+o,n,cf,sf,cs,ss,d_logit,1,sq,1,1); }
                CK(cudaDeviceSynchronize()); cudaFree(dp);
            }
        };
        const char* dump_env=getenv("BENCH_G4_DUMP_LOGITS");
        FILE* dump=nullptr; std::vector<float> dump_row(V);
        if(dump_env){ std::string dp=std::string(dump_env)+".uq"+std::to_string(sq)+".f32";
            dump=fopen(dp.c_str(),"wbx"); if(!dump){ fprintf(stderr,"cannot create %s\n",dp.c_str()); return 1; } }
        auto dump_step=[&](const float* d){ if(!dump) return;
            CK(cudaMemcpy(dump_row.data(),d,(size_t)V*4,cudaMemcpyDeviceToHost));
            if(fwrite(dump_row.data(),4,V,dump)!=(size_t)V){ fprintf(stderr,"dump write failed\n"); exit(1); } };
        if(!is_moe && (greedy_input || graph_mode==2)){
            // gemma4_generate 경로: 워밍 1회(같은 구성) 후 본 측정. 구간=호출 전체.
            struct Dump { decltype(dump_step)* f; static int call(const float* d,int,int,void* o){ (*((Dump*)o)->f)(d); return 0; } };
            Dump dctx{&dump_step};
            auto noop=[](const float*,int,int,void*)->int{ return 0; };
            const unsigned flags=greedy_input?0u:G4_GENERATE_FIXED_INPUT;
            fill_depth(); { G4GenerationStats ws; gemma4_generate(c,w,st,1,8,-1,cf,sf,cs,ss,sq,noop,nullptr,&ws,flags); }
            fill_depth();
            G4GenerationStats gs;
            const double gstart=decode_epoch(); const auto c0=std::chrono::steady_clock::now();
            const int acc=dump?gemma4_generate(c,w,st,1,ngen,-1,cf,sf,cs,ss,sq,&Dump::call,&dctx,&gs,flags)
                              :gemma4_generate(c,w,st,1,ngen,-1,cf,sf,cs,ss,sq,noop,nullptr,&gs,flags);
            const auto c1=std::chrono::steady_clock::now(); const double gend=decode_epoch();
            const double ms=std::chrono::duration<double,std::milli>(c1-c0).count(); const double ts=acc/(ms/1000.0);
            printf("DECODE_INTERVAL uq=%d start=%.17g end=%.17g steps=%d milliseconds=%.17g token_per_second=%.17g input=%s depth=%d graph=%d accounting=gemma4_generate_wall executed=%llu planned_captures=%llu instantiations=%llu update_failures=%llu\n",
                   sq,gstart,gend,acc,ms,ts,greedy_input?"greedy":"fixed",depth,graph_mode,gs.executed_steps,gs.planned_captures,gs.instantiations,gs.update_failures);
            printf("  %-6s single decode: %.2f t/s (%.3f ms/%d스텝) — roofline %.1f 대비 %.1f%%  [generate %s]\n",
                   sq==2?"W4A8":"q4", ts, ms, acc, single_roof, 100.0*ts/single_roof, greedy_input?"greedy":"fixed");
            if(dump) fclose(dump);
            cudaFree(d1); free_g4_decode_state(st);
            continue;
        }
        fill_depth();
        for(int s=0;s<8;s++){ if(is_moe) gemma4_moe_decode(c,w,st,d1,1,cf,sf,cs,ss,d_logit,1,sq,1);
                              else       gemma4_decode_step(c,w,st,d1,1,cf,sf,cs,ss,d_logit,1,sq,1); }  // 워밍
        CK(cudaDeviceSynchronize()); fill_depth();
        const double decode_start=decode_interval?decode_epoch():0;
        CK(cudaEventRecord(t0));
        for(int s=0;s<ngen;s++){ if(is_moe) gemma4_moe_decode(c,w,st,d1,1,cf,sf,cs,ss,d_logit,1,sq,1);
                                 else       gemma4_decode_step(c,w,st,d1,1,cf,sf,cs,ss,d_logit,1,sq,1);
                                 dump_step(d_logit); }
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        const double decode_end=decode_interval?decode_epoch():0;
        double ms=ev(t0,t1); double ts=ngen/(ms/1000.0);
        if(decode_interval) printf("DECODE_INTERVAL uq=%d start=%.17g end=%.17g steps=%d milliseconds=%.17g token_per_second=%.17g input=fixed depth=%d graph=%d accounting=cuda_event\n",
                                  sq,decode_start,decode_end,ngen,ms,ts,depth,graph_mode);
        printf("  %-6s single decode: %.2f t/s (%.3f ms/%d스텝) — roofline %.1f 대비 %.1f%%  [검증 경로]\n",
               sq==2?"W4A8":"q4", ts, ms, ngen, single_roof, 100.0*ts/single_roof);
        if(dump) fclose(dump);
        cudaFree(d1); free_g4_decode_state(st);
    }

    // 프리필 t/s (B=1 통짜) — 별도 G4DecodeState
    { G4DecodeState st=alloc_g4_decode_state(c,MAXLEN);
      const char* amort_env=getenv("BENCH_G4_AMORT");
      const int amort=!amort_env || strcmp(amort_env,"0")!=0;
      const char* mmq_env=getenv("G4_MMQ_PREFILL");
      const bool mmq_enabled=!mmq_env || strcmp(mmq_env,"0")!=0;
      int mmq_min_s=16;
      if(const char* v=getenv("G4_MMQ_MIN_S")){
          char* end=nullptr; errno=0;
          const long parsed=strtol(v,&end,10);
          if(errno==0 && end!=v && *end=='\0' && parsed>=INT_MIN && parsed<=INT_MAX)
              mmq_min_s=parsed<16?16:(int)parsed;
      }
      const bool aligned=c.hidden_size%32==0 && c.intermediate_size%32==0 &&
          ((size_t)c.num_q_heads*c.head_dim)%32==0 &&
          ((size_t)c.num_q_heads*(c.global_head_dim?c.global_head_dim:c.head_dim))%32==0;
      const char* prefill_path=is_moe?"MOE":uq!=0 && uq!=2?"TILED_Q4":uq==0?"GEMV":
          !amort || ctx<=1?"GEMV":
          mmq_enabled && ctx>=mmq_min_s && (c.scale_dtype==0 || c.scale_dtype==1)?
              (aligned?"MMQ":"MMQ_SB_MIXED"):"SB";
      printf("PREFILL_PATH=%s BENCH_G4_AMORT=%d G4_MMQ_MIN_S=%d coverage=layer_matmul_dispatch lm_head_last_only=1\n",
             prefill_path,amort,mmq_min_s);
      std::vector<int> tk(ctx,1); int* d_p; CK(cudaMalloc(&d_p,ctx*4)); CK(cudaMemcpy(d_p,tk.data(),ctx*4,cudaMemcpyHostToDevice));
      reset_g4_decode_state(st);
      if(is_moe) gemma4_moe_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1); else gemma4_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1,amort);   // 워밍
      CK(cudaDeviceSynchronize()); reset_g4_decode_state(st);
      if(!min_sec_env){
      CK(cudaEventRecord(t0));
      if(is_moe) gemma4_moe_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1); else gemma4_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1,amort);
      CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
      double ms=ev(t0,t1); prefill_ts=ctx/(ms/1000.0);
      printf("\n  프리필(B=1 통짜 %d토큰): %.2f tok/s (%.3f ms) — 참고치(chunked 아님, TTFT 주장 아님)\n",ctx,prefill_ts,ms);
      } else {
          auto epoch=[](){ return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); };
          std::vector<float> first(V),last(V);
          size_t reps=0; float total_ms=0; double end_epoch=0;
          CK(cudaDeviceSynchronize());
          CK(cudaEventRecord(t0));
          const double start_epoch=epoch();
          do {
              reset_g4_decode_state(st);
              if(is_moe) gemma4_moe_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1);
              else gemma4_decode(c,w,st,d_p,ctx,cf,sf,cs,ss,d_logit,0,uq,1,amort);
              if(reps==0) CK(cudaMemcpy(first.data(),d_logit,(size_t)V*4,cudaMemcpyDeviceToHost));
              ++reps;
              CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
              end_epoch=epoch();
              CK(cudaEventElapsedTime(&total_ms,t0,t1));
          } while(total_ms<min_prefill_sec*1000.0);
          printf("PREFILL_INTERVAL start=%.9f end=%.9f reps=%zu\n",start_epoch,end_epoch,reps);
          CK(cudaMemcpy(last.data(),d_logit,(size_t)V*4,cudaMemcpyDeviceToHost));
          bool finite=true; int first_argmax=0,last_argmax=0;
          for(int i=0;i<V;++i){
              finite=finite && std::isfinite(first[i]) && std::isfinite(last[i]);
              if(first[i]>first[first_argmax]) first_argmax=i;
              if(last[i]>last[last_argmax]) last_argmax=i;
          }
          const bool valid=finite && first_argmax==last_argmax;
          const double mean_ms=(double)total_ms/reps;
          prefill_ts=(double)ctx*reps/(total_ms/1000.0);
          printf("PREFILL_REPEAT ctx=%d reps=%zu total_ms=%.9f mean_ms=%.9f pp_tok_s=%.9f min_sec=%.9f first_argmax=%d last_argmax=%d finite=%d valid=%d accounting=CUDA_event_whole_block_including_host_gaps_and_first_logit_copy\n",ctx,reps,(double)total_ms,mean_ms,prefill_ts,min_prefill_sec,first_argmax,last_argmax,finite,valid);
          if(!valid){ fprintf(stderr,"INVALID repeated prefill first/last logits argmax or finite check\n"); exit(1); }
      }
      cudaFree(d_p); free_g4_decode_state(st);
    }

    printf("\n  (decode/stream=단일 시퀀스 t/s=roofline 대조 수치. decode/aggr=B×/stream=총처리량. n_past=0 시작=matmul 바운드.)\n");
    cudaFree(d_tok);cudaFree(d_pos);cudaFree(d_logit);cudaFree(cf);cudaFree(sf);cudaFree(cs);cudaFree(ss);
    free_weights_gemma4(w); free_config_gemma4(c); return 0;
}
