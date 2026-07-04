// Gemma4 chunked prefill 게이트 (STEP A). gemma4_prefill_batch_slot(어댑터) 정확성 검증.
//   ★핵심: chunked prefill(P토큰 1샷)이 c1tok(1-tok씩 ingest, 검증필 서빙 경로)와 동일 KV를 만드는가.
//   KV 내용(roped K/v_norm V)은 위치별·attn 무관 → 논리상 동일해야. 단 prefill=gemma4_decode(mm) vs
//   decode=gemma4_decode_batch(mmB)는 W4A8 커널이 달라(gemv_w4a8 vs _sb) bit-id는 경험적(측정 먼저).
//   격리(헌법): KV 층별 대조로 어느 층/연산에서 갈리는지 즉시 규명 + decode N토큰 전파 확인.
//   Path A(c1tok 기준): ragged 1-tok ingest(B=1) → KV[0..P-1] + decode N.  Path B: 어댑터 prefill → decode N.
//   사용법: ./test_gemma4_prefill_slot [weights] [golden] [use_q4=2] [P=8] [N=4]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../kernels/rope.cuh"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static std::string WD,GD;
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f){fprintf(stderr,"못엶 %s\n",p.c_str());exit(1);}std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/gemma4-31b/weights"; GD=(argc>2?argv[2]:"artifacts/gemma4-31b/golden")+std::string("/");
    int use_q4=argc>3?atoi(argv[3]):2;   // 서빙 모드 기본 W4A8(uq=2)
    ModelConfig c=load_config_gemma4((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, L=c.num_layers;
    const int hd_s=c.head_dim, nkv_s=c.num_kv_heads;
    const int hd_f=c.global_head_dim?c.global_head_dim:hd_s, nkv_f=c.num_global_kv_heads?c.num_global_kv_heads:nkv_s;
    int P=argc>4?atoi(argv[4]):8; int N=argc>5?atoi(argv[5]):4; int synth=argc>6?atoi(argv[6]):0;
    auto up=[&](std::vector<float>&h){float*d;TCK(cudaMalloc(&d,h.size()*4));TCK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;};
    std::vector<int> full; int total; float *d_cf,*d_sf,*d_cs,*d_ss;
    if(synth){   // 합성 모드: dummy rope(엔진 생성) + 합성 토큰 → P 임의화(멀티-청크 P>16 검증). 두 엔진 경로 비교라 골든 불요.
        total=P+N+2; full.resize(total); for(int i=0;i<total;i++) full[i]=(i*131+7)%V;
        TCK(cudaMalloc(&d_cf,(size_t)total*hd_f*4)); TCK(cudaMalloc(&d_sf,(size_t)total*hd_f*4));
        TCK(cudaMalloc(&d_cs,(size_t)total*hd_s*4)); TCK(cudaMalloc(&d_ss,(size_t)total*hd_s*4));
        launch_gen_rope(d_cf,d_sf,total,hd_f,1e4f); launch_gen_rope(d_cs,d_ss,total,hd_s,1e4f); TCK(cudaDeviceSynchronize());
    } else {     // 골든 모드: 실제 위치 rope + gen_tokens (rope 테이블 범위 0..total-1 안).
        full=ri("gen_tokens.bin"); total=full.size();
        if(P+N>total){ N=total-P; if(N<1){P=total-2;N=1;} }
        auto cf=rf("gen_rope_full_cos.bin",(size_t)total*hd_f), sf=rf("gen_rope_full_sin.bin",(size_t)total*hd_f);
        auto cs=rf("gen_rope_sliding_cos.bin",(size_t)total*hd_s), ss=rf("gen_rope_sliding_sin.bin",(size_t)total*hd_s);
        d_cf=up(cf);d_sf=up(sf);d_cs=up(cs);d_ss=up(ss);
    }
    G4Weights w=load_weights_gemma4(WD.c_str(),c,use_q4!=0?1:0,use_q4==0?1:0);
    auto am=[&](const float*lg){int b=0;float bv=lg[0];for(int j=1;j<V;j++)if(lg[j]>bv){bv=lg[j];b=j;}return b;};
    printf("[cfg] total=%d P=%d N=%d use_q4=%d(%s) L=%d V=%d\n",total,P,N,use_q4,use_q4==2?"W4A8":(use_q4?"q4":"fp32"),L,V);
    printf("  Path A=c1tok(ragged 1-tok ingest, 검증필 서빙 경로), Path B=chunked(gemma4_prefill_batch_slot 어댑터)\n");

    int ml=total+4; int *d_t; TCK(cudaMalloc(&d_t,ml*4)); int *d_pos; TCK(cudaMalloc(&d_pos,4));
    float *d_l; TCK(cudaMalloc(&d_l,(size_t)V*4));

    // ── Path A: c1tok ingest (위치 0..P-1 각각 ragged 1-tok) + decode N ──
    G4BatchState sA=alloc_g4_batch_state(c,1,ml);
    for(int p=0;p<P;p++){ int hp[1]={p}; TCK(cudaMemcpy(d_t,&full[p],4,cudaMemcpyHostToDevice)); TCK(cudaMemcpy(d_pos,hp,4,cudaMemcpyHostToDevice));
        gemma4_decode_batch(c,w,sA,d_t,d_cf,d_sf,d_cs,d_ss,d_l,0,use_q4,d_pos,hp); }
    std::vector<std::vector<float>> outA(N,std::vector<float>(V));
    for(int t=0;t<N;t++){ int p=P+t; int hp[1]={p}; TCK(cudaMemcpy(d_t,&full[p],4,cudaMemcpyHostToDevice)); TCK(cudaMemcpy(d_pos,hp,4,cudaMemcpyHostToDevice));
        gemma4_decode_batch(c,w,sA,d_t,d_cf,d_sf,d_cs,d_ss,d_l,0,use_q4,d_pos,hp);
        TCK(cudaMemcpy(outA[t].data(),d_l,V*4,cudaMemcpyDeviceToHost)); }

    // ── Path B: chunked prefill(어댑터, P토큰 1샷) + decode N ──
    G4BatchState sB=alloc_g4_batch_state(c,1,ml);
    { std::vector<int> pt(full.begin(),full.begin()+P); int *d_pt; TCK(cudaMalloc(&d_pt,P*4)); TCK(cudaMemcpy(d_pt,pt.data(),P*4,cudaMemcpyHostToDevice));
      gemma4_prefill_batch_slot(c,w,sB,0,d_pt,P,d_cf,d_sf,d_cs,d_ss,d_l,use_q4); cudaFree(d_pt); }
    std::vector<std::vector<float>> outB(N,std::vector<float>(V));
    for(int t=0;t<N;t++){ int p=P+t; int hp[1]={p}; TCK(cudaMemcpy(d_t,&full[p],4,cudaMemcpyHostToDevice)); TCK(cudaMemcpy(d_pos,hp,4,cudaMemcpyHostToDevice));
        gemma4_decode_batch(c,w,sB,d_t,d_cf,d_sf,d_cs,d_ss,d_l,0,use_q4,d_pos,hp);
        TCK(cudaMemcpy(outB[t].data(),d_l,V*4,cudaMemcpyDeviceToHost)); }

    // ── (1) KV 층별 격리 대조: slot0 위치 0..P-1 (어느 층/연산에서 갈리는지 규명) ──
    double kv_worst=0; int kv_wl=-1; int kv_div_layers=0;
    for(int l=0;l<L;l++){ int fullL=c.layer_is_full?c.layer_is_full[l]:1; int hd=fullL?hd_f:hd_s, nkv=fullL?nkv_f:nkv_s;
        size_t nel=(size_t)P*nkv*hd;   // slot0(offset 0)의 위치 0..P-1
        std::vector<float> kA(nel),kB(nel),vA(nel),vB(nel);
        TCK(cudaMemcpy(kA.data(),sA.Kc[l],nel*4,cudaMemcpyDeviceToHost)); TCK(cudaMemcpy(kB.data(),sB.Kc[l],nel*4,cudaMemcpyDeviceToHost));
        TCK(cudaMemcpy(vA.data(),sA.Vc[l],nel*4,cudaMemcpyDeviceToHost)); TCK(cudaMemcpy(vB.data(),sB.Vc[l],nel*4,cudaMemcpyDeviceToHost));
        double e=0; for(size_t i=0;i<nel;i++){ e=fmax(e,fabs((double)kA[i]-kB[i])); e=fmax(e,fabs((double)vA[i]-vB[i])); }
        if(e>0) kv_div_layers++;
        if(e>kv_worst){kv_worst=e;kv_wl=l;}
        if(l<4||l==L-1||e>0) printf("  KV L%-2d(%s) max|Δ|=%.3e\n", l, fullL?"full":"slid", e);
    }
    // ── (2) decode 전파 + (3) argmax ──
    double out_worst=0; int am_ok=0;
    for(int t=0;t<N;t++){ double e=0; for(int j=0;j<V;j++) e=fmax(e,fabs((double)outA[t][j]-outB[t][j])); out_worst=fmax(out_worst,e);
        if(am(outA[t].data())==am(outB[t].data())) am_ok++; }
    printf("\n(1) KV chunked vs c1tok: worst max|Δ|=%.3e @L%d | 갈린 층=%d/%d\n", kv_worst,kv_wl,kv_div_layers,L);
    printf("(2) decode %d토큰 logits vs c1tok: worst max|Δ|=%.3e\n", N, out_worst);
    printf("(3) decode argmax 일치: %d/%d\n", am_ok, N);
    bool bitid=(kv_worst==0.0)&&(out_worst==0.0);
    bool scheme_ok=(am_ok==N)&&(kv_worst<5e-2)&&(out_worst<1e-1);   // scheme-internal(W4A8 커널경계): argmax + 좁은 노이즈볼
    printf("[test_gemma4_prefill_slot] %s\n",
        bitid? "PASS(bit-id — chunked==c1tok 완전 동일 KV+출력)" :
        (scheme_ok? "PASS(scheme-internal — argmax 일치+W4A8 커널경계 노이즈볼 내. bit-id 아님=prefill mm vs decode mmB 커널차)"
                  : "FAIL(KV/출력 발산 — 어댑터 로직 조사, 위 층별 Δ로 규명)"));
    cudaFree(d_t);cudaFree(d_pos);cudaFree(d_l); free_g4_batch_state(sA); free_g4_batch_state(sB);
    free_weights_gemma4(w); free_config_gemma4(c);
    return (bitid||scheme_ok)?0:1;
}
