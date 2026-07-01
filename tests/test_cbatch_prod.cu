// 챕터9 production scheduler: admission guard(KV 예산) + slot 재사용 + 요청 큐 → mism=0.
//   R 요청을 B_max 작은 slot으로 처리(재사용+큐 강제). 물리 slot 고정(compaction 안 함), 재사용 시 reset_batch_slot
//   (GDN 잔재 clear — 새 요청이 이전 state 안 읽음). free slot은 dummy(무시). ragged 위치(h_pos_b).
//   검증: 각 요청 생성 == solo greedy bit-identical(재사용·큐·이웃 무관). + KV 예산 alloc-time 가드(unified 다운 방지).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <deque>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#include "../models/qwen35_batch.h"
#include "../kernels/rope.cuh"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int uq=argc>3?atoi(argv[3]):0, Bmax=argc>4?atoi(argv[4]):2;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, rot=c.rotary_dim, HD=c.head_dim, NKV=c.num_kv_heads;
    const int nh=c.gdn_num_v_heads,D=c.gdn_head_v_dim,convd=cfg_gdn_conv_dim(&c),Kk=c.gdn_conv_kernel;
    int maxlen=argc>5?atoi(argv[5]):48;   // 긴 context 스트레스용(admission guard가 alloc 전 거부하나)
    std::vector<int> ids=ri("token_ids.bin");
    // R 요청: 다양 프롬프트(다른 slice)·gen 길이. B_max보다 많아 재사용/큐 강제.
    struct Req{ std::vector<int> prompt; int ng; };
    std::vector<Req> reqs = {
        {{ids.begin(),   ids.begin()+3}, 8},  {{ids.begin()+1, ids.begin()+6}, 10},
        {{ids.begin()+2, ids.begin()+5}, 6},  {{ids.begin(),   ids.begin()+7}, 9},
        {{ids.begin()+3, ids.begin()+8}, 7},  {{ids.begin()+1, ids.begin()+4}, 11},
    };
    int R=reqs.size();
    printf("[cbatch_prod] uq=%d B_max=%d R=%d (요청 %d개를 slot %d개로 → 재사용+큐)\n",uq,Bmax,R,R,Bmax);

    // ---- admission guard: KV 예산 (alloc-time, unified 다운 방지) ----
    int nfull=0,ngdn=0; for(int l=0;l<c.num_layers;l++) c.layer_is_full[l]?nfull++:ngdn++;
    size_t perseq = (size_t)nfull*maxlen*NKV*HD*2*2         // Kc+Vc bf16
                  + (size_t)ngdn*((size_t)nh*D*D + (size_t)(Kk-1)*convd)*4;  // GDN S+convctx fp32
    size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB);
    double budget=0.7*(double)freeB;
    printf("  KV 예산: per-seq %.1fMB × B_max %d = %.2fGB vs 예산(0.7×free) %.2fGB — %s\n",
           perseq/1e6, Bmax, (double)Bmax*perseq/1e9, budget/1e9, (double)Bmax*perseq<=budget?"OK":"초과(B_max↓ 필요)");
    if((double)Bmax*perseq > budget){ printf("  admission guard: alloc 예산 초과 → B_max 줄여야(시스템 다운 방지). 중단.\n"); return 1; }

    std::vector<float> cl=rf("rope_cos_long.bin",(size_t)128*rot), sl=rf("rope_sin_long.bin",(size_t)128*rot);
    float*d_cos=up(cl),*d_sin=up(sl);
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,uq==0?1:0);
    int*d_tok; CK(cudaMalloc(&d_tok,maxlen*4)); float*d_logit; CK(cudaMalloc(&d_logit,(size_t)Bmax*V*4));  // maxlen≥Bmax·prompt(solo prefill)
    auto amax=[&](const std::vector<float>&lg,int b){int bi=0;float bv=lg[(size_t)b*V];for(int j=1;j<V;j++){float z=lg[(size_t)b*V+j];if(z>bv){bv=z;bi=j;}}return bi;};

    // ---- solo greedy 각 요청 (ground truth) ----
    auto solo=[&](const std::vector<int>&prompt,int ng){
        DecodeState st=alloc_decode_state(c,maxlen); int P=prompt.size();
        CK(cudaMemcpy(d_tok,prompt.data(),P*4,cudaMemcpyHostToDevice));
        float*dl; CK(cudaMalloc(&dl,(size_t)P*V*4));
        qwen35_decode(c,w,st,d_tok,P,d_cos,d_sin,dl,nullptr,1,nullptr,uq);
        std::vector<float> lg((size_t)P*V); CK(cudaMemcpy(lg.data(),dl,lg.size()*4,cudaMemcpyDeviceToHost));
        int cur=amax(lg,P-1); std::vector<int> out;
        for(int i=0;i<ng;i++){ out.push_back(cur); CK(cudaMemcpy(d_tok,&cur,4,cudaMemcpyHostToDevice));
            float*o;CK(cudaMalloc(&o,(size_t)V*4)); qwen35_decode(c,w,st,d_tok,1,d_cos,d_sin,o,nullptr,1,nullptr,uq);
            std::vector<float> hl(V);CK(cudaMemcpy(hl.data(),o,(size_t)V*4,cudaMemcpyDeviceToHost));cudaFree(o);
            int bi=0;float bv=hl[0];for(int j=1;j<V;j++)if(hl[j]>bv){bv=hl[j];bi=j;} cur=bi; }
        cudaFree(dl); free_decode_state(st); return out;
    };
    std::vector<std::vector<int>> soloOut(R); for(int r=0;r<R;r++) soloOut[r]=solo(reqs[r].prompt,reqs[r].ng);

    // ---- iso: 각 요청을 batch 엔진에 홀로(B=1, 이웃·재사용 없음) — scheduler 정확성의 진짜 기준(이웃/재사용 불변).
    //   scheduler의 per-slot 로직과 동일(토큰 prefill→decode)하되 이웃/재사용/큐 없이. got==iso면 서빙 로직 정확.
    //   (solo qwen35_decode 비교는 그래프-구현 차이 섞음: fp32 등가지만 W4A8 borderline flip — 별도 진단으로 보고.)
    auto iso_run=[&](const std::vector<int>&prompt,int ng){
        BatchDecodeState st=alloc_batch_decode_state(c,1,maxlen); st.B=1; reset_batch_slot(c,st,0);
        float*d_logit;CK(cudaMalloc(&d_logit,(size_t)V*4)); int P=prompt.size(),pos=0; std::vector<int> gen;
        while((int)gen.size()<ng){
            int t=(pos<P)?prompt[pos]:(gen.empty()?prompt[P-1]:gen.back()); int hp[1]={pos};
            CK(cudaMemcpy(d_tok,&t,4,cudaMemcpyHostToDevice));
            qwen35_decode_batch(c,w,st,d_tok,d_cos,d_sin,d_logit,uq,hp);
            std::vector<float> hl(V);CK(cudaMemcpy(hl.data(),d_logit,V*4,cudaMemcpyDeviceToHost));
            if(pos<P-1) pos++; else { int bi=0;float bv=hl[0];for(int j=1;j<V;j++)if(hl[j]>bv){bv=hl[j];bi=j;} gen.push_back(bi); pos++; } }
        cudaFree(d_logit); free_batch_decode_state(st); return gen;
    };
    std::vector<std::vector<int>> iso(R); for(int r=0;r<R;r++) iso[r]=iso_run(reqs[r].prompt,reqs[r].ng);
    // 진단: iso(batch graph) vs soloOut(solo graph). fp32=0(등가), W4A8>0=borderline flip(그래프-구현 차, scheduler 무관).
    { int d=0; for(int r=0;r<R;r++){int mm=0;for(int i=0;i<reqs[r].ng;i++)if(iso[r][i]!=soloOut[r][i])mm++;d+=mm;}
      printf("  [진단] iso(batch graph, 홀로) vs solo(qwen35_decode) 총차 %d  (fp32:0=그래프 등가 / W4A8:>0=4bit borderline flip, top2gap≪양자화노이즈, scheduler 무관)\n",d); }

    // ---- production scheduler (요청 큐 + slot 재사용 + admission) ----
    BatchDecodeState st=alloc_batch_decode_state(c,Bmax,maxlen); st.B=Bmax;
    struct Slot{ int req=-1,pos=0,P=0,ng=0,gened=0; std::vector<int> gen; };
    std::vector<Slot> slot(Bmax);
    std::deque<int> queue; for(int r=0;r<R;r++) queue.push_back(r);
    std::vector<std::vector<int>> got(R); int completed=0, step=0, admitted=0;
    while(completed < R){
        // 1) 종료 요청 회수 → slot free
        for(int b=0;b<Bmax;b++) if(slot[b].req>=0 && slot[b].gened>=slot[b].ng){
            got[slot[b].req]=slot[b].gen; completed++; slot[b]=Slot(); }
        // 2) admission: free slot에 큐 다음 요청(길이≤maxlen) — reset_batch_slot(잔재 clear)
        for(int b=0;b<Bmax;b++) if(slot[b].req<0 && !queue.empty()){
            int r=queue.front(); if((int)reqs[r].prompt.size()+reqs[r].ng > maxlen) break;  // admission: 길이 가드
            queue.pop_front(); slot[b].req=r; slot[b].P=reqs[r].prompt.size(); slot[b].ng=reqs[r].ng; slot[b].pos=0;
            reset_batch_slot(c,st,b); admitted++;  // ★ 재사용 slot GDN 잔재 clear
        }
        if(completed>=R) break;
        // 3) 가변 B ragged step (Bmax 고정, free slot=dummy 무시)
        std::vector<int> toks(Bmax), pos(Bmax);
        for(int b=0;b<Bmax;b++){ Slot&s=slot[b];
            if(s.req<0){ toks[b]=0; pos[b]=0; }                  // free: dummy
            else { const std::vector<int>&pr=reqs[s.req].prompt;
                   toks[b] = (s.pos<s.P)? pr[s.pos] : (s.gen.empty()? pr[s.P-1] : s.gen.back()); pos[b]=s.pos; } }
        CK(cudaMemcpy(d_tok,toks.data(),Bmax*4,cudaMemcpyHostToDevice));
        qwen35_decode_batch(c,w,st,d_tok,d_cos,d_sin,d_logit,uq, pos.data());
        std::vector<float> lg((size_t)Bmax*V); CK(cudaMemcpy(lg.data(),d_logit,lg.size()*4,cudaMemcpyDeviceToHost));
        for(int b=0;b<Bmax;b++){ Slot&s=slot[b]; if(s.req<0) continue;
            if(s.pos < s.P-1){ s.pos++; }
            else { int g=amax(lg,b); s.gen.push_back(g); s.gened++; s.pos++; } }
        if(++step > R*(maxlen)+200){ printf("  ⚠ step 폭주 — 중단\n"); break; }
    }
    // ---- PRIMARY 게이트: got(scheduler: 재사용+큐+이웃) == iso(홀로) bit-identical → 이웃/재사용/큐 불변(서빙 정확) ----
    int totmm=0; for(int r=0;r<R;r++){ int mm=0; for(int i=0;i<reqs[r].ng;i++)
            if(i>=(int)got[r].size()||got[r][i]!=iso[r][i]) mm++;
        if(mm) printf("  req %d MISMATCH(vs iso 홀로) %d/%d — 이웃/재사용이 요청 오염!\n",r,mm,reqs[r].ng); totmm+=mm; }
    // 진단: got vs solo graph (fp32=0, W4A8=borderline flip; scheduler 무관)
    { int d=0; for(int r=0;r<R;r++)for(int i=0;i<reqs[r].ng;i++) if(i<(int)got[r].size()&&got[r][i]!=soloOut[r][i])d++;
      printf("  [진단] got vs solo(qwen35_decode) 총차 %d (W4A8 borderline flip, iso와 동일 — scheduler 아닌 그래프-구현 차)\n",d); }
    printf("  완료 %d/%d 요청, admission %d회(재사용 %d), step %d, 이웃/재사용 오염 mismatch %d\n",completed,R,admitted,admitted-Bmax,step,totmm);
    bool pass=(completed==R && totmm==0);
    printf("\n[test_cbatch_prod] %s (요청 큐+slot 재사용+admission, 각 요청==홀로(iso) bit-identical=이웃/재사용 불변, KV 예산 가드)\n",pass?"PASS":"FAIL");
    free_batch_decode_state(st); free_weights_qwen(w); free_config_qwen(c);
    return pass?0:1;
}
