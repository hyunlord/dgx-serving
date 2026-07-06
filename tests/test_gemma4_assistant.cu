// gemma4_assistant graph HF 골든 게이트 (draft-spec 2관문) — ★정확성 전부.
//   골든: tools/make_reference_assistant.py (HF assistant fp32·target 12B bf16 입력 덤프).
//   [A] 스텝별 teacher-forced: 덤프된 (emb, h_in) 주입 → 로짓 MEAN-rel + argmax==HF draft + h_out MEAN-rel
//       (스텝 단위 정확성 격리 — cross-attn KV 직독·pre/post proj·트렁크 전부 이 게이트가 잡음)
//   [B] 재귀: h를 엔진 자신의 post_proj 출력으로 이어감(EAGLE 재귀) → K스텝 draft 토큰 전부 == HF
//   [C] 결정성 30회: 같은 스텝 로짓 bit-동일
// 사용: ./tests/test_gemma4_assistant [ast_weights] [ast_golden] [uq=0]
//   uq=0: fp32 vs fp32(HF GPU) — 합산 순서만 다름 → MEAN-rel 게이트(1e-4)+argmax ALL. max-abs는 보고만(비척도).
//   uq=2: 엔진 W4A8 vs HF q4-라운드트립 골든(golden_q4·make_reference_assistant --q4) — 활성화 int8 차이만
//         남음 → 로짓 MEAN-rel<2e-2(측정 1.0e-2)·h_out<5e-2(측정 2.7e-2 — 양자화 2단 경로)+argmax ALL+재귀.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../models/gemma4_assistant.h"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static std::string GD;
static long fsz(const std::string &p){ FILE*f=fopen((GD+p).c_str(),"rb"); if(!f)return -1; fseek(f,0,SEEK_END); long b=ftell(f); fclose(f); return b; }
static std::vector<float> rf(const std::string &p, size_t n){
    FILE*f=fopen((GD+p).c_str(),"rb"); if(!f){fprintf(stderr,"골든 없음: %s%s\n",GD.c_str(),p.c_str());exit(1);}
    std::vector<float>v(n); if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v;
}
static float* up(const std::vector<float>&h){ float*d; CK(cudaMalloc(&d,h.size()*4)); CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static int amax(const float *x, size_t n){ size_t bi=0; float bv=x[0]; for(size_t i=1;i<n;i++) if(x[i]>bv){bv=x[i];bi=i;} return (int)bi; }
static void relstat(const float *a, const float *ref, size_t n, double &mrel, double &mabs){
    double sd=0, sr=0, mx=0;
    for(size_t i=0;i<n;i++){ double d=fabs((double)a[i]-ref[i]); sd+=d; sr+=fabs((double)ref[i]); if(d>mx)mx=d; }
    mrel=sd/(sr>0?sr:1); mabs=mx;
}
static long ndiff(const float *a, const float *b, size_t n){ long d=0; for(size_t i=0;i<n;i++) if(memcmp(&a[i],&b[i],4)) d++; return d; }

int main(int argc,char**argv){
    std::string WD = argc>1?argv[1]:"artifacts/gemma4-12b-assistant/weights";
    GD = std::string(argc>2?argv[2]:"artifacts/gemma4-12b-assistant/golden")+"/";
    const int uq = argc>3?atoi(argv[3]):0;
    // 스킴별 톨러런스: fp32=1e-4(합산 순서만). W4A8=로짓 2e-2(target op-level 1.44e-2 tier·측정 1.0e-2),
    //   h_out 5e-2(측정 2.7e-2×여유 — post_projection은 활성화 양자화 2단[입력 h·출력] 경로라 로짓보다 큼.
    //   기능 정확성은 argmax 8/8 + ★재귀 게이트[엔진 자신의 h_out으로 8스텝 draft==HF]가 직접 보증).
    const double TOL_L = (uq==2) ? 2e-2 : 1e-4;
    const double TOL_H = (uq==2) ? 5e-2 : 1e-4;

    AstConfig c = load_config_assistant((WD+"/manifest.json").c_str());
    const int H=c.hidden_size, V=c.vocab_size, B=c.backbone;
    const int hd_f=c.global_head_dim, hd_s=c.head_dim, NKVf=c.num_global_kv_heads, NKVs=c.num_kv_heads;
    const int len_f=(int)(fsz("ast_kf.bin")/4/NKVf/hd_f), len_s=(int)(fsz("ast_ks.bin")/4/NKVs/hd_s);
    const int K=(int)(fsz("ast_drafts.bin")/4);
    printf("[test_gemma4_assistant] L=%d H=%d FFN=%d hd_s/f=%d/%d NKVs/f=%d/%d V=%d | KV len f/s=%d/%d K=%d\n",
           c.num_layers,H,c.intermediate_size,hd_s,hd_f,NKVs,NKVf,V,len_f,len_s,K);
    if(len_f<=0||len_s<=0||K<=0){ fprintf(stderr,"골든 파일 불완전 — make_reference_assistant.py 먼저.\n"); return 1; }

    AstWeights w = load_weights_assistant(WD.c_str(), c, uq);
    float *d_kf=up(rf("ast_kf.bin",(size_t)len_f*NKVf*hd_f)), *d_vf=up(rf("ast_vf.bin",(size_t)len_f*NKVf*hd_f));
    float *d_ks=up(rf("ast_ks.bin",(size_t)len_s*NKVs*hd_s)), *d_vs=up(rf("ast_vs.bin",(size_t)len_s*NKVs*hd_s));
    float *d_cf=up(rf("ast_rope_cos_f.bin",hd_f)), *d_sf=up(rf("ast_rope_sin_f.bin",hd_f));
    float *d_cs=up(rf("ast_rope_cos_s.bin",hd_s)), *d_ss=up(rf("ast_rope_sin_s.bin",hd_s));
    std::vector<int> drafts(K);
    { FILE*f=fopen((GD+"ast_drafts.bin").c_str(),"rb"); if(fread(drafts.data(),4,K,f)!=(size_t)K)exit(1); fclose(f); }

    float *d_in,*d_lg,*d_ho; CK(cudaMalloc(&d_in,(size_t)2*B*4)); CK(cudaMalloc(&d_lg,(size_t)V*4)); CK(cudaMalloc(&d_ho,(size_t)B*4));
    std::vector<float> hlg(V), hho(B), in(2*(size_t)B);
    int fails=0;

    // ---------- [A] 스텝별 teacher-forced (HF 덤프 입력 주입 — 스텝 정확성 격리) ----------
    double worst_lrel=0, worst_hrel=0; int am_bad=0;
    for(int j=0;j<K;j++){
        char nm[64];
        snprintf(nm,sizeof(nm),"step%d_emb.bin",j);  auto emb=rf(nm,B);
        snprintf(nm,sizeof(nm),"step%d_hin.bin",j);  auto hin=rf(nm,B);
        memcpy(in.data(),emb.data(),(size_t)B*4); memcpy(in.data()+B,hin.data(),(size_t)B*4);   // concat[emb|h]
        CK(cudaMemcpy(d_in,in.data(),(size_t)2*B*4,cudaMemcpyHostToDevice));
        gemma4_assistant_step(c,w,d_in,d_kf,d_vf,len_f,d_ks,d_vs,len_s,d_cf,d_sf,d_cs,d_ss,d_lg,d_ho,uq);
        CK(cudaMemcpy(hlg.data(),d_lg,(size_t)V*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hho.data(),d_ho,(size_t)B*4,cudaMemcpyDeviceToHost));
        snprintf(nm,sizeof(nm),"step%d_logits.bin",j); auto rlg=rf(nm,V);
        snprintf(nm,sizeof(nm),"step%d_hout.bin",j);   auto rho=rf(nm,B);
        double lrel,labs,hrel,habs; relstat(hlg.data(),rlg.data(),V,lrel,labs); relstat(hho.data(),rho.data(),B,hrel,habs);
        int am=amax(hlg.data(),V);
        if(am!=drafts[j]) am_bad++;
        if(lrel>worst_lrel)worst_lrel=lrel; if(hrel>worst_hrel)worst_hrel=hrel;
        printf("  [A step%d] logits MEAN-rel=%.3e (max-abs %.3e) h_out MEAN-rel=%.3e | argmax=%d HF=%d %s\n",
               j,lrel,labs,hrel,am,drafts[j],am==drafts[j]?"OK":"MISMATCH");
    }
    bool okA = (am_bad==0 && worst_lrel<TOL_L && worst_hrel<TOL_H);
    printf("  [Gate A] argmax %d/%d 일치 · worst logits rel %.3e · worst h rel %.3e → %s\n",
           K-am_bad,K,worst_lrel,worst_hrel,okA?"PASS":"FAIL");
    if(!okA) fails++;

    // ---------- [B] 재귀 (엔진 자신의 h_out 재귀 — EAGLE 구조) ----------
    {
        auto h = rf("step0_hin.bin",B);                       // 시작 h = target hidden(사이클 시작)
        int ok=1; double final_lrel=0;
        for(int j=0;j<K;j++){
            char nm[64]; snprintf(nm,sizeof(nm),"step%d_emb.bin",j); auto emb=rf(nm,B);
            memcpy(in.data(),emb.data(),(size_t)B*4); memcpy(in.data()+B,h.data(),(size_t)B*4);
            CK(cudaMemcpy(d_in,in.data(),(size_t)2*B*4,cudaMemcpyHostToDevice));
            gemma4_assistant_step(c,w,d_in,d_kf,d_vf,len_f,d_ks,d_vs,len_s,d_cf,d_sf,d_cs,d_ss,d_lg,d_ho,uq);
            CK(cudaMemcpy(hlg.data(),d_lg,(size_t)V*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h.data(),d_ho,(size_t)B*4,cudaMemcpyDeviceToHost));   // ★재귀: 엔진 h_out
            int am=amax(hlg.data(),V);
            if(am!=drafts[j]){ ok=0; printf("  [B step%d] draft %d != HF %d — 재귀 발산\n",j,am,drafts[j]); break; }
            if(j==K-1){ char n2[64]; snprintf(n2,sizeof(n2),"step%d_logits.bin",j); auto rlg=rf(n2,V);
                        double la; relstat(hlg.data(),rlg.data(),V,final_lrel,la); }
        }
        printf("  [Gate B 재귀 %d스텝] draft 토큰 %s == HF · 최종 스텝 로짓 rel %.3e → %s\n",
               K, ok?"전부":"불일치", final_lrel, ok?"PASS":"FAIL");
        if(!ok) fails++;
    }

    // ---------- [C] 결정성 30회 ----------
    {
        auto emb=rf("step0_emb.bin",B); auto hin=rf("step0_hin.bin",B);
        memcpy(in.data(),emb.data(),(size_t)B*4); memcpy(in.data()+B,hin.data(),(size_t)B*4);
        CK(cudaMemcpy(d_in,in.data(),(size_t)2*B*4,cudaMemcpyHostToDevice));
        std::vector<float> first(V); long dsum=0;
        for(int r=0;r<30;r++){
            gemma4_assistant_step(c,w,d_in,d_kf,d_vf,len_f,d_ks,d_vs,len_s,d_cf,d_sf,d_cs,d_ss,d_lg,d_ho,uq);
            if(r==0) CK(cudaMemcpy(first.data(),d_lg,(size_t)V*4,cudaMemcpyDeviceToHost));
            else { CK(cudaMemcpy(hlg.data(),d_lg,(size_t)V*4,cudaMemcpyDeviceToHost)); dsum+=ndiff(hlg.data(),first.data(),V); }
        }
        printf("  [Gate C 결정성 30회] diff-원소 합 %ld → %s\n",dsum,dsum==0?"PASS":"FAIL");
        if(dsum!=0) fails++;
    }

    free_weights_assistant(w,c);
    printf("[test_gemma4_assistant] %s\n", fails?"FAIL":"PASS (HF 골든 스텝별+재귀+결정성)");
    return fails?1:0;
}
