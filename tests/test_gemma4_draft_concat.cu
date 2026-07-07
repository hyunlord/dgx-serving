// gemma4_assistant_step_batch 게이트 (draft concat — verify concat의 대칭) — ★배치==순차 bit-id가 전부.
//   함정 검증: ①재귀(출력 h가 다음 입력 — K스텝 연쇄에서 비트 유지) ②슬롯별 위치 다름(ragged rope 행)
//   ③cross-attn target KV 포인터 매핑(슬롯 뒤섞임 = 조용한 오류 — 슬롯별 로짓 대조가 잡음).
// 게이트: [1] B_act∈{2,3,4} × K=6 재귀 — 배치 로짓/h_out == 슬롯별 순차(기존 gemma4_assistant_step,
//            daemon draft 루프 그대로) **bit-id(Δ=0)** — 매 스텝·매 슬롯. draft 토큰도 일치.
//        [2] 결정성(배치 반복 == 같은 reference 재대조로 함의) ③ 타이밍 — 순차 vs 배치 draft 페이즈(K=4·warm).
// 사용: ./tests/test_gemma4_draft_concat [tgt_w] [tgt_golden] [ast_w] [wl_dir] [uq=2] [ast_uq=2]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../models/gemma4_assistant.h"
#include "../kernels/embedding.cuh"
#include "../kernels/elementwise.cuh"
#include "../kernels/argmax.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
using clk=std::chrono::steady_clock;
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }

static std::vector<float> rff(const std::string &p, size_t n){
    FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"없음: %s\n",p.c_str());exit(1);}
    std::vector<float>v(n); if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v;
}
static long fsz(const std::string &p){ FILE*f=fopen(p.c_str(),"rb"); if(!f)return -1; fseek(f,0,SEEK_END); long b=ftell(f); fclose(f); return b; }
static std::vector<int> rii(const std::string &p){
    long b=fsz(p); if(b<0){fprintf(stderr,"없음: %s\n",p.c_str());exit(1);}
    std::vector<int>v(b/4); FILE*f=fopen(p.c_str(),"rb"); if(fread(v.data(),4,v.size(),f)!=v.size())exit(1); fclose(f); return v;
}
static float* up(const std::vector<float>&h){ float*d; CK(cudaMalloc(&d,h.size()*4)); CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static long ndiff(const float *a, const float *b, size_t n){ long d=0; for(size_t i=0;i<n;i++) if(memcmp(&a[i],&b[i],4)!=0) d++; return d; }

int main(int argc,char**argv){
    std::string TW = argc>1?argv[1]:"artifacts/gemma4-12b/weights";
    std::string TG = std::string(argc>2?argv[2]:"artifacts/gemma4-12b/golden")+"/";
    std::string AW = argc>3?argv[3]:"artifacts/gemma4-12b-assistant/weights";
    std::string WL = std::string(argc>4?argv[4]:"artifacts/gemma4-12b-assistant/golden")+"/";
    const int uq     = argc>5?atoi(argv[5]):2;
    const int ast_uq = argc>6?atoi(argv[6]):2;
    setvbuf(stdout,nullptr,_IOLBF,0);

    ModelConfig c = load_config_gemma4((TW+"/manifest.json").c_str());
    AstConfig  ca = load_config_assistant((AW+"/manifest.json").c_str());
    const int V=c.vocab_size, H=c.hidden_size, hd_f=c.global_head_dim, hd_s=c.head_dim;
    const int nkv_s=c.num_kv_heads, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    const int total=(int)(fsz(TG+"gen_rope_full_cos.bin")/4/hd_f);
    int lf=-1, ls=-1;
    for(int l=0;l<c.num_layers;l++){ if(c.layer_is_full[l]) lf=l; else ls=l; }
    printf("[draft_concat] uq=%d ast_uq=%d total(rope)=%d V=%d KV소스 L%d/L%d\n",uq,ast_uq,total,V,lf,ls);

    float *d_cf=up(rff(TG+"gen_rope_full_cos.bin",(size_t)total*hd_f)), *d_sf=up(rff(TG+"gen_rope_full_sin.bin",(size_t)total*hd_f));
    float *d_cs=up(rff(TG+"gen_rope_sliding_cos.bin",(size_t)total*hd_s)), *d_ss=up(rff(TG+"gen_rope_sliding_sin.bin",(size_t)total*hd_s));
    G4Weights w = load_weights_gemma4(TW.c_str(), c, uq!=0?1:0, uq==0?1:0);
    AstWeights wa = load_weights_assistant(AW.c_str(), ca, ast_uq);

    // 슬롯 4개 프리필(daemon admit 미러: 슬롯-뷰 decode·amort=1·prenorm → h0·첫 토큰)
    std::vector<std::vector<int>> pref;
    pref.push_back(rii(WL+"wl_chat_ids.bin"));
    pref.push_back(rii(WL+"wl_code_ids.bin"));
    pref.push_back(rii(WL+"wl_prose_ids.bin"));
    { std::vector<int> t=pref[0]; t.resize(t.size()>9?t.size()-9:1); pref.push_back(t); }
    const int B=4, KC=6, KT=4;
    G4BatchState bs = alloc_g4_batch_state(c, B, total);
    int maxP=0; for(auto&p:pref) if((int)p.size()>maxP) maxP=(int)p.size();
    int *d_tok; CK(cudaMalloc(&d_tok,(size_t)maxP*4));
    float *d_lgP; CK(cudaMalloc(&d_lgP,(size_t)maxP*V*4));
    float *d_pnP; CK(cudaMalloc(&d_pnP,(size_t)maxP*H*4));
    int *d_am; CK(cudaMalloc(&d_am,(size_t)B*4));

    int pos[B], cur[B];
    float *d_h0; CK(cudaMalloc(&d_h0,(size_t)B*H*4));
    std::vector<std::vector<float*>> vK(B), vV(B);
    for(int b=0;b<B;b++){
        vK[b].resize(c.num_layers); vV[b].resize(c.num_layers);
        for(int l=0;l<c.num_layers;l++){
            int full=c.layer_is_full?c.layer_is_full[l]:1;
            int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
            size_t stride=(size_t)total*nkv*hd;
            vK[b][l]=bs.Kc[l]+(size_t)b*stride; vV[b][l]=bs.Vc[l]+(size_t)b*stride;
        }
    }
    for(int b=0;b<B;b++){
        pos[b]=(int)pref[b].size();
        G4DecodeState view{}; view.n_past=0; view.max_len=total; view.L=c.num_layers; view.kv_cap=bs.kv_cap;
        view.Kc=vK[b].data(); view.Vc=vV[b].data();
        CK(cudaMemcpy(d_tok,pref[b].data(),(size_t)pos[b]*4,cudaMemcpyHostToDevice));
        gemma4_decode(c,w,view,d_tok,pos[b],d_cf,d_sf,d_cs,d_ss,d_lgP,0,uq,1,1,d_pnP);
        launch_argmax(d_lgP,V,d_am); CK(cudaMemcpy(&cur[b],d_am,4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(d_h0+(size_t)b*H, d_pnP+(size_t)(pos[b]-1)*H, (size_t)H*4, cudaMemcpyDeviceToDevice));
    }
    printf("[draft_concat] 슬롯 프리필: pos = %d %d %d %d (ragged) · cur = %d %d %d %d\n",
           pos[0],pos[1],pos[2],pos[3],cur[0],cur[1],cur[2],cur[3]);

    // ---------- 순차 reference(daemon draft 루프 그대로): 슬롯별 KC스텝, 로짓/h_out/토큰 기록 ----------
    float *d_in; CK(cudaMalloc(&d_in,(size_t)2*ca.backbone*4));
    float *d_alg; CK(cudaMalloc(&d_alg,(size_t)V*4));
    float *d_ho; CK(cudaMalloc(&d_ho,(size_t)ca.backbone*4));
    std::vector<std::vector<float>> ref_lg(B), ref_h(B);      // [b] = KC×V / KC×backbone
    std::vector<std::vector<int>> ref_tok(B);
    for(int b=0;b<B;b++){
        ref_lg[b].resize((size_t)KC*V); ref_h[b].resize((size_t)KC*ca.backbone); ref_tok[b].resize(KC);
        int dt=cur[b];
        for(int j=0;j<KC;j++){
            CK(cudaMemcpy(d_tok,&dt,4,cudaMemcpyHostToDevice));
            launch_embedding(d_tok, w.embed, d_in, 1, H);
            if(c.embed_scale>0.f) launch_scale(d_in,d_in,c.embed_scale,(size_t)H);
            CK(cudaMemcpy(d_in+ca.backbone, j==0? d_h0+(size_t)b*H : d_ho, (size_t)H*4, cudaMemcpyDeviceToDevice));
            gemma4_assistant_step(ca,wa,d_in, vK[b][lf],vV[b][lf],pos[b], vK[b][ls],vV[b][ls],pos[b],
                                  d_cf+(size_t)pos[b]*hd_f, d_sf+(size_t)pos[b]*hd_f,
                                  d_cs+(size_t)pos[b]*hd_s, d_ss+(size_t)pos[b]*hd_s, d_alg, d_ho, ast_uq);
            launch_argmax(d_alg,V,d_am); CK(cudaMemcpy(&dt,d_am,4,cudaMemcpyDeviceToHost));
            ref_tok[b][j]=dt;
            CK(cudaMemcpy(ref_lg[b].data()+(size_t)j*V, d_alg,(size_t)V*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ref_h[b].data()+(size_t)j*ca.backbone, d_ho,(size_t)ca.backbone*4,cudaMemcpyDeviceToHost));
        }
    }

    // ---------- 배치: B_act∈{2,3,4} — 매 스텝 로짓/h_out/토큰 == reference bit-id ----------
    float *d_inB; CK(cudaMalloc(&d_inB,(size_t)B*2*ca.backbone*4));
    float *d_algB; CK(cudaMalloc(&d_algB,(size_t)B*V*4));
    float *d_hoB; CK(cudaMalloc(&d_hoB,(size_t)B*ca.backbone*4));
    float *d_emb; CK(cudaMalloc(&d_emb,(size_t)B*H*4));
    int *d_tokB; CK(cudaMalloc(&d_tokB,(size_t)B*4));
    int *d_pos; CK(cudaMalloc(&d_pos,(size_t)B*4));
    std::vector<float> h_lg((size_t)B*V), h_h((size_t)B*ca.backbone);
    int fails=0;
    for(int Bact : {2,3,4}){
        const float *kf[4],*vf[4],*ks[4],*vs[4]; int lenb[4], dtb[4];
        for(int a=0;a<Bact;a++){ kf[a]=vK[a][lf]; vf[a]=vV[a][lf]; ks[a]=vK[a][ls]; vs[a]=vV[a][ls];
                                 lenb[a]=pos[a]; dtb[a]=cur[a]; }
        CK(cudaMemcpy(d_pos,lenb,(size_t)Bact*4,cudaMemcpyHostToDevice));
        long dl=0, dh=0; int dtok=0;
        for(int j=0;j<KC;j++){
            CK(cudaMemcpy(d_tokB,dtb,(size_t)Bact*4,cudaMemcpyHostToDevice));
            launch_embedding(d_tokB, w.embed, d_emb, Bact, H);
            if(c.embed_scale>0.f) launch_scale(d_emb,d_emb,c.embed_scale,(size_t)Bact*H);
            for(int a=0;a<Bact;a++){
                CK(cudaMemcpy(d_inB+(size_t)a*2*ca.backbone, d_emb+(size_t)a*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(d_inB+(size_t)a*2*ca.backbone+ca.backbone,
                              j==0? d_h0+(size_t)a*H : d_hoB+(size_t)a*ca.backbone,(size_t)H*4,cudaMemcpyDeviceToDevice));
            }
            gemma4_assistant_step_batch(ca,wa,Bact,d_inB,kf,vf,ks,vs,lenb,d_cf,d_sf,d_cs,d_ss,d_pos,d_algB,d_hoB,ast_uq);
            for(int a=0;a<Bact;a++) launch_argmax(d_algB+(size_t)a*V,V,d_am+a);
            int am[4]; CK(cudaMemcpy(am,d_am,(size_t)Bact*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_lg.data(),d_algB,(size_t)Bact*V*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h_h.data(),d_hoB,(size_t)Bact*ca.backbone*4,cudaMemcpyDeviceToHost));
            for(int a=0;a<Bact;a++){
                dl+=ndiff(h_lg.data()+(size_t)a*V, ref_lg[a].data()+(size_t)j*V, V);
                dh+=ndiff(h_h.data()+(size_t)a*ca.backbone, ref_h[a].data()+(size_t)j*ca.backbone, ca.backbone);
                if(am[a]!=ref_tok[a][j]) dtok++;
                dtb[a]=am[a];
            }
        }
        printf("  [1] B_act=%d K=%d 재귀: 로짓Δ=%ld · h_outΔ=%ld · 토큰 불일치=%d %s\n",
               Bact,KC,dl,dh,dtok,(dl==0&&dh==0&&dtok==0)?"PASS":"★FAIL");
        if(dl||dh||dtok) fails++;
    }

    // ---------- 타이밍(K=4·warm 2·5회 평균): 순차 4슬롯 draft 페이즈 vs 배치 ----------
    auto seq_draft=[&](){
        for(int b=0;b<B;b++){
            int dt=cur[b];
            for(int j=0;j<KT;j++){
                CK(cudaMemcpy(d_tok,&dt,4,cudaMemcpyHostToDevice));
                launch_embedding(d_tok, w.embed, d_in, 1, H);
                if(c.embed_scale>0.f) launch_scale(d_in,d_in,c.embed_scale,(size_t)H);
                CK(cudaMemcpy(d_in+ca.backbone, j==0? d_h0+(size_t)b*H : d_ho,(size_t)H*4,cudaMemcpyDeviceToDevice));
                gemma4_assistant_step(ca,wa,d_in, vK[b][lf],vV[b][lf],pos[b], vK[b][ls],vV[b][ls],pos[b],
                                      d_cf+(size_t)pos[b]*hd_f, d_sf+(size_t)pos[b]*hd_f,
                                      d_cs+(size_t)pos[b]*hd_s, d_ss+(size_t)pos[b]*hd_s, d_alg, d_ho, ast_uq);
                launch_argmax(d_alg,V,d_am); CK(cudaMemcpy(&dt,d_am,4,cudaMemcpyDeviceToHost));
            }
        }
    };
    auto bat_draft=[&](int Bact){
        const float *kf[4],*vf[4],*ks[4],*vs[4]; int lenb[4], dtb[4];
        for(int a=0;a<Bact;a++){ kf[a]=vK[a][lf]; vf[a]=vV[a][lf]; ks[a]=vK[a][ls]; vs[a]=vV[a][ls];
                                 lenb[a]=pos[a]; dtb[a]=cur[a]; }
        CK(cudaMemcpy(d_pos,lenb,(size_t)Bact*4,cudaMemcpyHostToDevice));
        for(int j=0;j<KT;j++){
            CK(cudaMemcpy(d_tokB,dtb,(size_t)Bact*4,cudaMemcpyHostToDevice));
            launch_embedding(d_tokB, w.embed, d_emb, Bact, H);
            if(c.embed_scale>0.f) launch_scale(d_emb,d_emb,c.embed_scale,(size_t)Bact*H);
            for(int a=0;a<Bact;a++){
                CK(cudaMemcpy(d_inB+(size_t)a*2*ca.backbone, d_emb+(size_t)a*H,(size_t)H*4,cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy(d_inB+(size_t)a*2*ca.backbone+ca.backbone,
                              j==0? d_h0+(size_t)a*H : d_hoB+(size_t)a*ca.backbone,(size_t)H*4,cudaMemcpyDeviceToDevice));
            }
            gemma4_assistant_step_batch(ca,wa,Bact,d_inB,kf,vf,ks,vs,lenb,d_cf,d_sf,d_cs,d_ss,d_pos,d_algB,d_hoB,ast_uq);
            for(int a=0;a<Bact;a++) launch_argmax(d_algB+(size_t)a*V,V,d_am+a);
            int am[4]; CK(cudaMemcpy(am,d_am,(size_t)Bact*4,cudaMemcpyDeviceToHost));
            for(int a=0;a<Bact;a++) dtb[a]=am[a];
        }
    };
    for(int r=0;r<2;r++){ seq_draft(); bat_draft(4); }        // warm
    auto t0=clk::now(); for(int r=0;r<5;r++) seq_draft(); double ms_sq=ms_since(t0)/5;
    t0=clk::now(); for(int r=0;r<5;r++) bat_draft(4); double ms_b4=ms_since(t0)/5;
    t0=clk::now(); for(int r=0;r<5;r++) bat_draft(2); double ms_b2=ms_since(t0)/5;
    printf("  [3] 타이밍(K=%d draft 페이즈·5회 평균): 순차 4슬롯 %.1fms → 배치4 %.1fms = %.2fx (−%.0f%%) · 배치2 %.1fms\n",
           KT,ms_sq,ms_b4,ms_sq/ms_b4,(1.0-ms_b4/ms_sq)*100,ms_b2);

    printf("\n[test_gemma4_draft_concat] %s\n", fails?"FAIL":"PASS (배치==순차 bit-id·재귀·ragged·KV 매핑 + 타이밍)");
    return fails?1:0;
}
