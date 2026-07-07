// gemma4_verify_concat 게이트 (draft-spec verify concat 분할상환) — ★스킴 경계 flip 차단이 전부.
//   concat = 큰 matmul만 M=ΣNb 행으로 묶고(SB ≤16행 그룹 = gemma4_decode amort와 동일 분해),
//   rope는 행별 위치·attention/KV append는 슬롯별 기존 커널 → per-slot 경로와 bit-identical이 구성적 목표.
// 게이트: [1] concat 로짓 == per-slot canonical(gemma4_verify_batch, amort=0) **bit-id(Δ=0)** — M 경계 전수
//            스윕(M=5·8·9·10·15·16·17·20·36: SMAX8/16 경계·16행 그룹 분할 경계 포함, ragged 혼합 포함)
//        [2] prenorm(EAGLE h 재귀용) == per-slot amort=1(daemon 경로) prenorm bit-id
//        [3] KV 불침범 — concat 후 canonical 재실행 로짓 == concat 전(다른 슬롯/기존 위치 KV 오염 없음)
//        [4] 결정성 5회 — concat 로짓 bit-동일
//        [5] STEP 0 타이밍 — per-slot(amort=1, daemon 경로) vs concat, B_act 2/3/4 (warm·참고 수치)
// 사용: ./tests/test_gemma4_verify_concat [weights] [golden] [uq=2] [wl_dir=artifacts/gemma4-12b-assistant/golden]
//   wl_dir의 wl_{chat,code,prose}_ids.bin(e2e와 동일 자산)을 슬롯 프리필 접두사로 사용(길이 상이=자연 ragged).
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

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
using clk=std::chrono::steady_clock;
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }

static std::string GD;
static std::vector<float> rf(const std::string &p, size_t n){
    FILE*f=fopen((GD+p).c_str(),"rb"); if(!f){fprintf(stderr,"파일 없음: %s%s\n",GD.c_str(),p.c_str());exit(1);}
    std::vector<float>v(n); if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v;
}
static long fsize(const std::string &p){ FILE*f=fopen((GD+p).c_str(),"rb"); if(!f)return -1; fseek(f,0,SEEK_END); long b=ftell(f); fclose(f); return b; }
static std::vector<int> ri_abs(const std::string &p){
    FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"파일 없음: %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<int>v(b/4); if(fread(v.data(),4,v.size(),f)!=v.size())exit(1); fclose(f); return v;
}
static float* up(const std::vector<float>&h){ float*d; CK(cudaMalloc(&d,h.size()*4)); CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static long ndiff(const float *a, const float *b, size_t n){ long d=0; for(size_t i=0;i<n;i++) if(memcmp(&a[i],&b[i],4)!=0) d++; return d; }

int main(int argc, char**argv){
    std::string WD = argc>1?argv[1]:"artifacts/gemma4-12b/weights";
    GD = std::string(argc>2?argv[2]:"artifacts/gemma4-12b/golden")+"/";
    const int uq = argc>3?atoi(argv[3]):2;

    ModelConfig c = load_config_gemma4((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, H=c.hidden_size, hd_s=c.head_dim;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s;
    const int nkv_s=c.num_kv_heads, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;

    std::string WL = std::string(argc>4?argv[4]:"artifacts/gemma4-12b-assistant/golden")+"/";

    long rows_f=fsize("gen_rope_full_cos.bin")/4/hd_f;
    const int total=(int)rows_f;
    float *d_cf=up(rf("gen_rope_full_cos.bin",(size_t)total*hd_f)), *d_sf=up(rf("gen_rope_full_sin.bin",(size_t)total*hd_f));
    float *d_cs=up(rf("gen_rope_sliding_cos.bin",(size_t)total*hd_s)), *d_ss=up(rf("gen_rope_sliding_sin.bin",(size_t)total*hd_s));

    // 슬롯 프리필 접두사: e2e 워크로드(길이 상이=자연 ragged) + chat 절단(4번째)
    std::vector<std::vector<int>> pref;
    pref.push_back(ri_abs(WL+"wl_chat_ids.bin"));
    pref.push_back(ri_abs(WL+"wl_code_ids.bin"));
    pref.push_back(ri_abs(WL+"wl_prose_ids.bin"));
    { std::vector<int> t=pref[0]; t.resize(t.size()>9?t.size()-9:1); pref.push_back(t); }
    printf("[verify_concat] uq=%d total(rope)=%d V=%d\n", uq, total, V);

    G4Weights w = load_weights_gemma4(WD.c_str(), c, uq!=0?1:0, uq==0?1:0);
    const int B=4, MMAX=40;
    G4BatchState bs = alloc_g4_batch_state(c, B, total);
    int maxP=0; for(auto &p:pref) if((int)p.size()>maxP) maxP=(int)p.size();
    int *d_tok; CK(cudaMalloc(&d_tok,(size_t)(maxP>MMAX?maxP:MMAX)*4));
    float *d_lg;  CK(cudaMalloc(&d_lg,(size_t)MMAX*V*4));      // concat 로짓
    float *d_lgr; CK(cudaMalloc(&d_lgr,(size_t)MMAX*V*4));     // canonical 로짓
    float *d_pn;  CK(cudaMalloc(&d_pn,(size_t)MMAX*H*4));      // concat prenorm
    float *d_pnr; CK(cudaMalloc(&d_pnr,(size_t)MMAX*H*4));     // per-slot prenorm(ref)
    float *d_sc;  CK(cudaMalloc(&d_sc,(size_t)V*4));           // prefill 로짓 scratch

    int pos[B];
    for(int b=0;b<B;b++){
        pos[b]=(int)pref[b].size();
        if(pos[b]+10>=total){ fprintf(stderr,"접두사 %d 너무 김(pos %d + 청크 > rope %d)\n",b,pos[b],total); return 1; }
        CK(cudaMemcpy(d_tok,pref[b].data(),(size_t)pos[b]*4,cudaMemcpyHostToDevice));
        gemma4_prefill_batch_slot(c,w,bs,b,d_tok,pos[b],d_cf,d_sf,d_cs,d_ss,d_sc,uq);
    }
    printf("[verify_concat] 슬롯 프리필: pos = %d %d %d %d (ragged)\n",pos[0],pos[1],pos[2],pos[3]);

    // 청크 토큰(내용은 임의 — 수학은 내용 무관·결정적이게 각 슬롯 접두사에서 재사용)
    auto mk_chunk=[&](int a, int n, std::vector<int> &out){
        const std::vector<int> &src=pref[a]; int S=(int)src.size();
        for(int j=0;j<n;j++) out.push_back(src[(a*7+j*3)%S]);
    };
    // per-slot 뷰(amort=1 — daemon 경로) verify: 로짓+prenorm을 cat 버퍼에 기록
    auto per_slot_amort=[&](int B_act, const int *slot_ids, const int *Nb, const int *pos_b,
                            const int *h_tok_cat, float *lg_cat, float *pn_cat){
        long off=0;
        for(int a=0;a<B_act;a++){
            const int b=slot_ids[a], Sq=Nb[a];
            G4DecodeState view{}; view.n_past=pos_b[a]; view.max_len=bs.max_len; view.L=bs.L; view.kv_cap=bs.kv_cap;
            std::vector<float*> Kc(bs.L), Vc(bs.L);
            for(int l=0;l<bs.L;l++){
                int full=c.layer_is_full?c.layer_is_full[l]:1;
                int hd=full?hd_f:hd_s, nkv=full?nkv_f:nkv_s;
                size_t stride=(size_t)bs.max_len*nkv*hd;
                Kc[l]=bs.Kc[l]+(size_t)b*stride; Vc[l]=bs.Vc[l]+(size_t)b*stride;
            }
            view.Kc=Kc.data(); view.Vc=Vc.data();
            CK(cudaMemcpy(d_tok,h_tok_cat+off,(size_t)Sq*4,cudaMemcpyHostToDevice));
            gemma4_decode(c,w,view,d_tok,Sq,d_cf,d_sf,d_cs,d_ss,lg_cat+(size_t)off*V,0,uq,0,1,pn_cat+(size_t)off*H);
            off+=Sq;
        }
    };

    struct Cfg { int B_act; int Nb[4]; const char* note; };
    Cfg cfgs[]={
        {1,{5,0,0,0},      "M=5 단일(함수 일반성)"},
        {2,{5,3,0,0},      "M=8 (SMAX8 상단)"},
        {2,{5,4,0,0},      "M=9 (8→9 SMAX 경계)"},
        {2,{5,5,0,0},      "M=10 (K4 2슬롯 — 서빙 주사례)"},
        {3,{5,5,5,0},      "M=15 (SB 단일그룹 상단)"},
        {3,{5,5,6,0},      "M=16 (그룹 경계 정확)"},
        {3,{9,5,3,0},      "M=17 (16→17 그룹 분할·ragged)"},
        {4,{5,5,5,5},      "M=20 (K4 4슬롯 — 그룹 16+4)"},
        {4,{9,9,9,9},      "M=36 (K8 4슬롯 — 그룹 16+16+4)"},
    };
    int fails=0;
    setvbuf(stdout,nullptr,_IOLBF,0);                      // 리다이렉트 시에도 라인 flush(크래시 가시성)
    std::vector<float> h_a((size_t)MMAX*V), h_b((size_t)MMAX*V), h_c((size_t)MMAX*V);
    std::vector<float> h_pa((size_t)MMAX*H), h_pb((size_t)MMAX*H);
    std::vector<float> kv_probe_ref;                       // [3] KV 불침범 기준(첫 canonical)
    int slot_ids[4]={0,1,2,3};

    for(auto &cf : cfgs){
        int M=0; std::vector<int> tok;
        for(int a=0;a<cf.B_act;a++){ mk_chunk(a,cf.Nb[a],tok); M+=cf.Nb[a]; }
        CK(cudaMemcpy(d_tok,tok.data(),(size_t)M*4,cudaMemcpyHostToDevice));
        // canonical(amort=0 per-slot) — d_tok는 아래 재사용되므로 device cat 사본을 따로
        int *d_cat; CK(cudaMalloc(&d_cat,(size_t)M*4));
        CK(cudaMemcpy(d_cat,tok.data(),(size_t)M*4,cudaMemcpyHostToDevice));
        gemma4_verify_batch(c,w,bs,cf.B_act,slot_ids,cf.Nb,pos,d_cat,d_cf,d_sf,d_cs,d_ss,d_lgr,uq);
        CK(cudaMemcpy(h_a.data(),d_lgr,(size_t)M*V*4,cudaMemcpyDeviceToHost));
        // concat
        gemma4_verify_concat(c,w,bs,cf.B_act,slot_ids,cf.Nb,pos,d_cat,d_cf,d_sf,d_cs,d_ss,d_lg,uq,d_pn);
        CK(cudaMemcpy(h_b.data(),d_lg,(size_t)M*V*4,cudaMemcpyDeviceToHost));
        long dl=ndiff(h_a.data(),h_b.data(),(size_t)M*V);
        // prenorm ref(per-slot amort=1 daemon 경로)
        per_slot_amort(cf.B_act,slot_ids,cf.Nb,pos,tok.data(),d_lgr,d_pnr);
        CK(cudaMemcpy(h_pa.data(),d_pnr,(size_t)M*H*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_pb.data(),d_pn,(size_t)M*H*4,cudaMemcpyDeviceToHost));
        long dp=ndiff(h_pa.data(),h_pb.data(),(size_t)M*H);
        // per-slot amort=1 로짓도 canonical과 일치했나(전이 확인)
        CK(cudaMemcpy(h_c.data(),d_lgr,(size_t)M*V*4,cudaMemcpyDeviceToHost));
        long da=ndiff(h_a.data(),h_c.data(),(size_t)M*V);
        printf("  [1/2] B_act=%d %-28s 로짓Δ(concat vs canon)=%ld · prenormΔ=%ld · (amort-slot vs canon Δ=%ld) %s\n",
               cf.B_act,cf.note,dl,dp,da,(dl==0&&dp==0)?"PASS":"★FAIL");
        if(dl!=0||dp!=0) fails++;
        if(kv_probe_ref.empty()){ kv_probe_ref.assign(h_a.begin(),h_a.begin()+(size_t)M*V); }
        cudaFree(d_cat);
    }

    // [3] KV 불침범: 전 스윕 후 첫 config canonical 재실행 == 처음
    {
        auto &cf=cfgs[0]; int M=cf.Nb[0]; std::vector<int> tok; mk_chunk(0,M,tok);
        int *d_cat; CK(cudaMalloc(&d_cat,(size_t)M*4));
        CK(cudaMemcpy(d_cat,tok.data(),(size_t)M*4,cudaMemcpyHostToDevice));
        gemma4_verify_batch(c,w,bs,1,slot_ids,cf.Nb,pos,d_cat,d_cf,d_sf,d_cs,d_ss,d_lgr,uq);
        CK(cudaMemcpy(h_a.data(),d_lgr,(size_t)M*V*4,cudaMemcpyDeviceToHost));
        long d3=ndiff(h_a.data(),kv_probe_ref.data(),(size_t)M*V);
        printf("  [3] KV 불침범(스윕 후 canonical 재실행 Δ)=%ld %s\n",d3,d3==0?"PASS":"★FAIL");
        if(d3!=0) fails++;
        cudaFree(d_cat);
    }

    // [4] 결정성: M=17 ragged config 5회 bit-동일
    {
        Cfg cf={3,{9,5,3,0},""}; int M=17; std::vector<int> tok;
        for(int a=0;a<cf.B_act;a++) mk_chunk(a,cf.Nb[a],tok);
        int *d_cat; CK(cudaMalloc(&d_cat,(size_t)M*4));
        CK(cudaMemcpy(d_cat,tok.data(),(size_t)M*4,cudaMemcpyHostToDevice));
        gemma4_verify_concat(c,w,bs,cf.B_act,slot_ids,cf.Nb,pos,d_cat,d_cf,d_sf,d_cs,d_ss,d_lg,uq,d_pn);
        CK(cudaMemcpy(h_a.data(),d_lg,(size_t)M*V*4,cudaMemcpyDeviceToHost));
        long dmax=0;
        for(int r=1;r<5;r++){
            gemma4_verify_concat(c,w,bs,cf.B_act,slot_ids,cf.Nb,pos,d_cat,d_cf,d_sf,d_cs,d_ss,d_lg,uq,d_pn);
            CK(cudaMemcpy(h_b.data(),d_lg,(size_t)M*V*4,cudaMemcpyDeviceToHost));
            long d4=ndiff(h_a.data(),h_b.data(),(size_t)M*V); if(d4>dmax)dmax=d4;
        }
        printf("  [4] 결정성 5회(M=17 ragged) Δmax=%ld %s\n",dmax,dmax==0?"PASS":"★FAIL");
        if(dmax!=0) fails++;
        cudaFree(d_cat);
    }

    // [5] STEP 0 타이밍(참고): per-slot(amort=1 daemon 경로) vs concat — warm 2 + 5회 평균
    printf("  [5] 타이밍(warm·5회 평균, K4 청크=Sq5):\n");
    for(int Bact : {2,3,4}){
        int Nb[4]={5,5,5,5}; int M=5*Bact; std::vector<int> tok;
        for(int a=0;a<Bact;a++) mk_chunk(a,Nb[a],tok);
        int *d_cat; CK(cudaMalloc(&d_cat,(size_t)M*4));
        CK(cudaMemcpy(d_cat,tok.data(),(size_t)M*4,cudaMemcpyHostToDevice));
        for(int r=0;r<2;r++){ per_slot_amort(Bact,slot_ids,Nb,pos,tok.data(),d_lgr,d_pnr);
                              gemma4_verify_concat(c,w,bs,Bact,slot_ids,Nb,pos,d_cat,d_cf,d_sf,d_cs,d_ss,d_lg,uq,d_pn); }
        auto t0=clk::now();
        for(int r=0;r<5;r++) per_slot_amort(Bact,slot_ids,Nb,pos,tok.data(),d_lgr,d_pnr);
        double ms_ps=ms_since(t0)/5;
        t0=clk::now();
        for(int r=0;r<5;r++) gemma4_verify_concat(c,w,bs,Bact,slot_ids,Nb,pos,d_cat,d_cf,d_sf,d_cs,d_ss,d_lg,uq,d_pn);
        double ms_cc=ms_since(t0)/5;
        printf("      B_act=%d (M=%2d): per-slot %6.1fms → concat %6.1fms = %.2fx (−%.0f%%)\n",
               Bact,M,ms_ps,ms_cc,ms_ps/ms_cc,(1.0-ms_cc/ms_ps)*100);
        cudaFree(d_cat);
    }

    printf("\n[test_gemma4_verify_concat] %s\n", fails?"FAIL":"PASS (concat==per-slot bit-id·M 경계 스윕·KV 불침범·결정성 + 타이밍)");
    return fails?1:0;
}
