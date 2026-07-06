// gemma4_verify_batch POC 게이트 (draft-spec 1관문) — ★spec==greedy bit-id가 전부.
//   불변식: verify 청크(Sq=K+1)가 각 위치서 낸 로짓 == plain 1토큰 greedy가 같은 위치서 낸 로짓, **bit-identical(Δ=0)**.
//   근거: gemma4_decode 경로는 전 구간 행-독립(mm gemv/quantize per-row·rope per-row·attn block=행·lm_head row) —
//   청크 prefill Δ=0 게이트(955cffe)와 같은 성질. 27b의 M-버킷 커널 전환(sb/mmq flip)이 gemma4 decode 경로엔 없음.
//   로짓 bit-id ⟹ 어떤 결정적 argmax든 plain==verify 동일 선택 = HF서 실측한 tie-flip(완전 동률·1ULP) 부류 원천 차단.
// 게이트: [1] perfect draft K∈{2,3,4,8}(비배수 Sq=3·5 포함) — 전 위치 로짓 Δ=0·전부 accept·bonus==plain(시퀀스 재현)
//        [2] corrupted draft(K=4, 거부 위치 회전) — 정확한 지점 거부·교정 토큰==plain·재개 후에도 일치(KV 덮어쓰기 롤백 증명)
//        [3] 결정성 30회 — 같은 verify 호출 로짓 bit-동일
//        [4] batch(B=2, ragged Nb 5/9)==solo — gemma4_verify_batch 각 슬롯 로짓 == 단독 gemma4_decode Δ=0
// 사용: ./tests/test_gemma4_verify [weights] [golden] [uq=2] [N=64]
//   golden엔 gen_rope_{full,sliding}_{cos,sin}.bin(전 위치 테이블, make_reference_gemma4_gen)+token_ids.bin 필요.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static std::string WD, GD;
static std::vector<float> rf(const std::string &p, size_t n){
    FILE*f=fopen((GD+p).c_str(),"rb"); if(!f){fprintf(stderr,"파일 없음: %s%s\n",GD.c_str(),p.c_str());exit(1);}
    std::vector<float>v(n); if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read: %s\n",p.c_str());exit(1);} fclose(f); return v;
}
static long fsize(const std::string &p){ FILE*f=fopen((GD+p).c_str(),"rb"); if(!f)return -1; fseek(f,0,SEEK_END); long b=ftell(f); fclose(f); return b; }
static std::vector<int> ri(const std::string &p){
    FILE*f=fopen((GD+p).c_str(),"rb"); if(!f){fprintf(stderr,"파일 없음: %s%s\n",GD.c_str(),p.c_str());exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<int>v(b/4); if(fread(v.data(),4,v.size(),f)!=v.size())exit(1); fclose(f); return v;
}
static float* up(const std::vector<float>&h){ float*d; CK(cudaMalloc(&d,h.size()*4)); CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }

// 호스트 argmax — 결정적 tiebreak(동률=낮은 인덱스). plain/verify 양쪽에 같은 규칙 적용(단일 소스).
static int amax(const float *x, int n){ int bi=0; float bv=x[0]; for(int i=1;i<n;i++) if(x[i]>bv){bv=x[i];bi=i;} return bi; }
// 로짓 행 bit 비교: 다른 원소 수(0=bit-identical)
static long ndiff(const float *a, const float *b, int n){ long d=0; for(int i=0;i<n;i++) if(memcmp(&a[i],&b[i],4)!=0) d++; return d; }

int main(int argc, char**argv){
    WD = argc>1?argv[1]:"artifacts/gemma4-12b/weights";
    GD = std::string(argc>2?argv[2]:"artifacts/gemma4-12b/golden")+"/";
    const int uq = argc>3?atoi(argv[3]):2;
    int N = argc>4?atoi(argv[4]):64;

    ModelConfig c = load_config_gemma4((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, hd_s=c.head_dim;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s;

    // rope: gen_* 전 위치 테이블(행수=파일크기/hd). full/sliding 행수 동일해야(같은 total).
    long rows_f=fsize("gen_rope_full_cos.bin")/4/hd_f, rows_s=fsize("gen_rope_sliding_cos.bin")/4/hd_s;
    if(rows_f<=0||rows_s<=0){ fprintf(stderr,"gen_rope_* 없음 — make_reference_gemma4_gen.py로 생성 필요.\n"); exit(1); }
    if(rows_f!=rows_s){ fprintf(stderr,"gen rope full/sliding 행수 불일치 %ld vs %ld\n",rows_f,rows_s); exit(1); }
    const int total=(int)rows_f;
    float *d_cf=up(rf("gen_rope_full_cos.bin",(size_t)total*hd_f)), *d_sf=up(rf("gen_rope_full_sin.bin",(size_t)total*hd_f));
    float *d_cs=up(rf("gen_rope_sliding_cos.bin",(size_t)total*hd_s)), *d_ss=up(rf("gen_rope_sliding_sin.bin",(size_t)total*hd_s));

    std::vector<int> ids = ri("token_ids.bin"); const int S=(int)ids.size();
    const int KMAX=8;
    if(N > total-S-KMAX-2) N = total-S-KMAX-2;             // rope 테이블 범위 내
    printf("[test_gemma4_verify] uq=%d S=%d N=%d total(rope)=%d V=%d\n", uq, S, N, total, V);

    G4Weights w = load_weights_gemma4(WD.c_str(), c, uq!=0?1:0, uq==0?1:0);
    int *d_tok; CK(cudaMalloc(&d_tok,(size_t)(S>KMAX+1?S:KMAX+1)*4));
    float *d_lg; CK(cudaMalloc(&d_lg,(size_t)(S>KMAX+1?S:KMAX+1)*V*4));

    // ---------- plain greedy 기준(레퍼런스): 프리필 + N스텝, 각 스텝 로짓 보관 ----------
    auto prefill=[&](G4DecodeState &st, std::vector<float> *last_row){
        CK(cudaMemcpy(d_tok,ids.data(),(size_t)S*4,cudaMemcpyHostToDevice));
        gemma4_decode(c,w,st,d_tok,S,d_cf,d_sf,d_cs,d_ss,d_lg,1,uq,0,0);   // commit=1, 전 위치 로짓
        if(last_row){ last_row->resize(V); CK(cudaMemcpy(last_row->data(),d_lg+(size_t)(S-1)*V,(size_t)V*4,cudaMemcpyDeviceToHost)); }
    };
    G4DecodeState sp = alloc_g4_decode_state(c,total);
    std::vector<float> plog_pre; prefill(sp,&plog_pre);
    std::vector<int> ptok(N+1); ptok[0]=amax(plog_pre.data(),V);
    std::vector<std::vector<float>> hlog(N);                 // hlog[t] = plain이 토큰 ptok[t]를 위치 S+t서 먹은 뒤 로짓
    for(int t=0;t<N;t++){
        CK(cudaMemcpy(d_tok,&ptok[t],4,cudaMemcpyHostToDevice));
        gemma4_decode(c,w,sp,d_tok,1,d_cf,d_sf,d_cs,d_ss,d_lg,1,uq,0,0);
        hlog[t].resize(V); CK(cudaMemcpy(hlog[t].data(),d_lg,(size_t)V*4,cudaMemcpyDeviceToHost));
        ptok[t+1]=amax(hlog[t].data(),V);
    }
    printf("  plain greedy: t0=%d 마지막 5토큰 =",ptok[0]);
    for(int t=N-4;t<=N;t++) printf(" %d",ptok[t]); printf("\n");

    int fails=0;
    std::vector<float> vrow((size_t)(KMAX+1)*V);

    // ---------- Gate 1: perfect draft — 전 위치 로짓 Δ=0 + 전부 accept + bonus==plain ----------
    for(int K : {2,3,4,8}){
        G4DecodeState s1 = alloc_g4_decode_state(c,total);
        std::vector<float> pre1; prefill(s1,&pre1);
        long d0=ndiff(pre1.data(),plog_pre.data(),V);        // 교차-alloc 결정성(프리필 로짓 bit-id)
        int g=0, cur=ptok[0]; long rows=0, rowdiff=0; int acc_bad=0, bonus_bad=0;
        while(g+K+1<=N){
            std::vector<int> chunk(K+1); chunk[0]=cur; for(int i=1;i<=K;i++) chunk[i]=ptok[g+i];
            CK(cudaMemcpy(d_tok,chunk.data(),(size_t)(K+1)*4,cudaMemcpyHostToDevice));
            s1.n_past=S+g;                                    // 호스트가 위치 관리(수락 산술) — verify는 commit=0
            gemma4_decode(c,w,s1,d_tok,K+1,d_cf,d_sf,d_cs,d_ss,d_lg,0,uq,0,0);
            CK(cudaMemcpy(vrow.data(),d_lg,(size_t)(K+1)*V*4,cudaMemcpyDeviceToHost));
            for(int i=0;i<=K;i++){ rows++; rowdiff+=ndiff(&vrow[(size_t)i*V],hlog[g+i].data(),V); }
            int m=0; for(int i=1;i<=K;i++){ if(amax(&vrow[(size_t)(i-1)*V],V)==chunk[i]) m++; else break; }
            if(m!=K) acc_bad++;
            int bonus=amax(&vrow[(size_t)m*V],V);
            if(bonus!=ptok[g+m+1]) bonus_bad++;
            g+=m+1; cur=ptok[g];
        }
        bool ok = (d0==0 && rowdiff==0 && !acc_bad && !bonus_bad);
        printf("  [Gate1 K=%d] verify로짓 vs plain: %ld행 diff-원소 %ld · full-accept %s · bonus %s · 프리필Δ %ld → %s\n",
               K, rows, rowdiff, acc_bad?"FAIL":"ALL", bonus_bad?"FAIL":"ALL", d0, ok?"PASS":"FAIL");
        if(!ok) fails++;
        free_g4_decode_state(s1);
    }

    // ---------- Gate 2: corrupted draft — 거부 지점·교정 토큰·재개 후 일치(롤백) ----------
    {
        const int K=4;
        G4DecodeState s2 = alloc_g4_decode_state(c,total);
        prefill(s2,nullptr);
        int g=0, cur=ptok[0], cyc=0; long rows=0, rowdiff=0; int rej_bad=0, corr_bad=0;
        while(g+K+1<=N){
            int j=(cyc%K)+1;                                  // 이번 사이클 오염 위치(1..K 회전)
            std::vector<int> chunk(K+1); chunk[0]=cur; for(int i=1;i<=K;i++) chunk[i]=ptok[g+i];
            chunk[j]=(ptok[g+j]+12345)%V; if(chunk[j]==ptok[g+j]) chunk[j]=(chunk[j]+1)%V;
            CK(cudaMemcpy(d_tok,chunk.data(),(size_t)(K+1)*4,cudaMemcpyHostToDevice));
            s2.n_past=S+g;
            gemma4_decode(c,w,s2,d_tok,K+1,d_cf,d_sf,d_cs,d_ss,d_lg,0,uq,0,0);
            CK(cudaMemcpy(vrow.data(),d_lg,(size_t)(K+1)*V*4,cudaMemcpyDeviceToHost));
            for(int i=0;i<j;i++){ rows++; rowdiff+=ndiff(&vrow[(size_t)i*V],hlog[g+i].data(),V); }  // 오염 이전 행만 비교(causal)
            int m=0; for(int i=1;i<=K;i++){ if(amax(&vrow[(size_t)(i-1)*V],V)==chunk[i]) m++; else break; }
            if(m!=j-1) rej_bad++;                             // 정확히 오염 직전까지 수락해야
            int corr=amax(&vrow[(size_t)m*V],V);
            if(corr!=ptok[g+m+1]) corr_bad++;                 // 교정 토큰 == plain
            g+=m+1; cur=ptok[g]; cyc++;
        }
        bool ok=(rowdiff==0 && !rej_bad && !corr_bad);
        printf("  [Gate2 K=4 오염회전] 청정행 diff %ld/%ld행 · 거부지점 %s · 교정==plain %s → %s (사이클 %d — 미수락 KV 덮어쓰기 롤백 실증)\n",
               rowdiff, rows, rej_bad?"FAIL":"ALL", corr_bad?"FAIL":"ALL", ok?"PASS":"FAIL", cyc);
        if(!ok) fails++;
        free_g4_decode_state(s2);
    }

    // ---------- Gate 3: 결정성 — 같은 verify 30회 로짓 bit-동일 ----------
    {
        const int K=4;
        G4DecodeState s3 = alloc_g4_decode_state(c,total);
        prefill(s3,nullptr);
        std::vector<int> chunk(K+1); chunk[0]=ptok[0]; for(int i=1;i<=K;i++) chunk[i]=ptok[i];
        CK(cudaMemcpy(d_tok,chunk.data(),(size_t)(K+1)*4,cudaMemcpyHostToDevice));
        std::vector<float> first((size_t)(K+1)*V); long dsum=0;
        for(int r=0;r<30;r++){
            s3.n_past=S;                                       // commit=0이라 불변이나 명시
            gemma4_decode(c,w,s3,d_tok,K+1,d_cf,d_sf,d_cs,d_ss,d_lg,0,uq,0,0);
            if(r==0) CK(cudaMemcpy(first.data(),d_lg,first.size()*4,cudaMemcpyDeviceToHost));
            else { CK(cudaMemcpy(vrow.data(),d_lg,first.size()*4,cudaMemcpyDeviceToHost));
                   dsum+=ndiff(vrow.data(),first.data(),(int)first.size()); }
        }
        printf("  [Gate3 결정성 30회] diff-원소 합 %ld → %s\n", dsum, dsum==0?"PASS":"FAIL");
        if(dsum!=0) fails++;
        free_g4_decode_state(s3);
    }

    // ---------- Gate 4: gemma4_verify_batch(B=2 ragged 5/9) == solo Δ=0 ----------
    {
        G4BatchState bs = alloc_g4_batch_state(c,2,total);
        float *d_pl; CK(cudaMalloc(&d_pl,(size_t)V*4));
        CK(cudaMemcpy(d_tok,ids.data(),(size_t)S*4,cudaMemcpyHostToDevice));
        gemma4_prefill_batch_slot(c,w,bs,0,d_tok,S,d_cf,d_sf,d_cs,d_ss,d_pl,uq);
        gemma4_prefill_batch_slot(c,w,bs,1,d_tok,S,d_cf,d_sf,d_cs,d_ss,d_pl,uq);
        const int Nb[2]={5,9}; const int slot_ids[2]={0,1}; const int pos_b[2]={S,S};
        std::vector<int> cat; cat.push_back(ptok[0]); for(int i=1;i<=4;i++) cat.push_back(ptok[i]);
        cat.push_back(ptok[0]); for(int i=1;i<=8;i++) cat.push_back(ptok[i]);
        int *d_cat; CK(cudaMalloc(&d_cat,(size_t)cat.size()*4));
        CK(cudaMemcpy(d_cat,cat.data(),cat.size()*4,cudaMemcpyHostToDevice));
        float *d_vcat; CK(cudaMalloc(&d_vcat,(size_t)14*V*4));
        gemma4_verify_batch(c,w,bs,2,slot_ids,Nb,pos_b,d_cat,d_cf,d_sf,d_cs,d_ss,d_vcat,uq);
        std::vector<float> bat((size_t)14*V); CK(cudaMemcpy(bat.data(),d_vcat,bat.size()*4,cudaMemcpyDeviceToHost));
        long dsum=0;
        for(int a=0;a<2;a++){                                  // solo 대조: 신선 상태 프리필 + 같은 청크 verify
            G4DecodeState ss = alloc_g4_decode_state(c,total); prefill(ss,nullptr);
            int Sq=Nb[a]; long off=(a==0)?0:5;
            CK(cudaMemcpy(d_tok,cat.data()+off,(size_t)Sq*4,cudaMemcpyHostToDevice));
            ss.n_past=S;
            gemma4_decode(c,w,ss,d_tok,Sq,d_cf,d_sf,d_cs,d_ss,d_lg,0,uq,0,0);
            CK(cudaMemcpy(vrow.data(),d_lg,(size_t)Sq*V*4,cudaMemcpyDeviceToHost));
            dsum+=ndiff(vrow.data(),&bat[(size_t)off*V],Sq*V);
            free_g4_decode_state(ss);
        }
        printf("  [Gate4 batch(B=2, Nb=5/9 ragged)==solo] diff-원소 %ld → %s\n", dsum, dsum==0?"PASS":"FAIL");
        if(dsum!=0) fails++;
        free_g4_batch_state(bs); cudaFree(d_cat); cudaFree(d_vcat); cudaFree(d_pl);
    }

    free_g4_decode_state(sp);
    printf("[test_gemma4_verify] %s\n", fails?"FAIL":"PASS (spec==greedy bit-id·거부/롤백·결정성·batch==solo)");
    return fails?1:0;
}
