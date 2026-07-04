// 26B-A4B MoE decode 게이트 — gemma4_moe_decode(KV) == gemma4_moe_forward(full-seq) 엔진-내부 정합.
//   ★크래시-위험 HF 26B 재실행 회피: forward는 이미 HF-골든 PASS(bf16_hs, M4) → decode==forward ⟹ decode==HF(전이적).
//   Gate A(decode graph): decode(Sq=S, base=0, prefill-mode) 전위치 == forward(S) — argmax 일치 + rel.
//   Gate B(연속성): decode step-by-step(Sq=1×S, KV 이어받기) 각 위치 logit == forward(S) 같은 위치 — argmax 일치.
//   scheme-internal(q4/W4A8) — bit-id 아닌 argmax+rel(attn full vs decode fp-order 차). 짧은 S(크래시無).
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
#include "../models/gemma4_moe.h"
#include "../kernels/rope.cuh"

#define TCK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

int main(int argc,char**argv){
    std::string WD = argc>1?argv[1]:"artifacts/gemma4-26b-a4b/weights";
    int uq = argc>2?atoi(argv[2]):1;   // 1=q4 / 2=W4A8 (26B weights=q4)
    ModelConfig c = load_config_gemma4((WD+"/manifest.json").c_str());
    const int V=c.vocab_size;
    const int hd_s=c.head_dim, hd_f=c.global_head_dim?c.global_head_dim:hd_s;
    std::vector<int> ids = {2, 651, 1596, 573, 1001, 235292, 611, 573};   // 임의 유효 토큰(엔진 정합 게이트)
    int S=(int)ids.size(), maxlen=S+8;
    int *d_ids; TCK(cudaMalloc(&d_ids,S*4)); TCK(cudaMemcpy(d_ids,ids.data(),S*4,cudaMemcpyHostToDevice));

    // rope(maxlen 행 — decode base offset 대비). full θ=1e6 [maxlen,hd_f] / sliding θ=1e4 [maxlen,hd_s].
    float *d_cf,*d_sf,*d_cs,*d_ss;
    TCK(cudaMalloc(&d_cf,(size_t)maxlen*hd_f*4)); TCK(cudaMalloc(&d_sf,(size_t)maxlen*hd_f*4));
    TCK(cudaMalloc(&d_cs,(size_t)maxlen*hd_s*4)); TCK(cudaMalloc(&d_ss,(size_t)maxlen*hd_s*4));
    launch_gen_rope(d_cf,d_sf,maxlen,hd_f,1e6f); launch_gen_rope(d_cs,d_ss,maxlen,hd_s,1e4f);

    printf("[load] 26B-A4B weights (q4) ...\n");
    G4Weights w = load_weights_gemma4(WD.c_str(),c,1,0);

    // ── forward(S) 기준 ──
    float *d_fwd; TCK(cudaMalloc(&d_fwd,(size_t)S*V*4));
    gemma4_moe_forward(c,w,d_ids,S,d_cf,d_sf,d_cs,d_ss,d_fwd,nullptr,uq);
    std::vector<float> fwd((size_t)S*V); TCK(cudaMemcpy(fwd.data(),d_fwd,fwd.size()*4,cudaMemcpyDeviceToHost));

    auto amax=[&](const std::vector<float>&L,size_t pos){ int b=0; float bv=L[pos*V]; for(int j=1;j<V;j++){float x=L[pos*V+j]; if(x>bv){bv=x;b=j;}} return b; };
    auto relmax=[&](const std::vector<float>&A,const std::vector<float>&B,size_t pos){ double m=0; for(int j=0;j<V;j++){ double a=A[pos*V+j],b=B[pos*V+j],d=fabs(a-b),den=fabs(a)>1e-6?fabs(a):1e-6; m=fmax(m,d/den);} return m; };
    // near-zero rel 아티팩트 구분: max-abs-err(전체) + rel(|fwd|>1 유의미 logit만)
    auto abserr=[&](const std::vector<float>&A,const std::vector<float>&B,size_t pos){ double m=0; for(int j=0;j<V;j++) m=fmax(m,fabs((double)A[pos*V+j]-B[pos*V+j])); return m; };
    auto relsig=[&](const std::vector<float>&A,const std::vector<float>&B,size_t pos){ double m=0; for(int j=0;j<V;j++){ double a=A[pos*V+j]; if(fabs(a)>1.0){ m=fmax(m,fabs(a-B[pos*V+j])/fabs(a)); }} return m; };

    // ── Gate A: decode-as-prefill(Sq=S, base=0) == forward ──
    G4DecodeState st = alloc_g4_decode_state(c,maxlen);
    float *d_dec; TCK(cudaMalloc(&d_dec,(size_t)S*V*4));
    gemma4_moe_decode(c,w,st,d_ids,S,d_cf,d_sf,d_cs,d_ss,d_dec,0,uq,0);   // commit=0, last_only=0(전 행)
    std::vector<float> dec((size_t)S*V); TCK(cudaMemcpy(dec.data(),d_dec,dec.size()*4,cudaMemcpyDeviceToHost));
    int badA=0; double relA=0,absA=0,relsigA=0;
    for(int p=0;p<S;p++){ if(amax(dec,p)!=amax(fwd,p))badA++; relA=fmax(relA,relmax(dec,fwd,p)); absA=fmax(absA,abserr(dec,fwd,p)); relsigA=fmax(relsigA,relsig(dec,fwd,p)); }
    printf("[Gate A] decode(prefill-mode Sq=%d)==forward: argmax 불일치 %d/%d | max-abs-err=%.3e | rel(|fwd|>1)=%.2e | rel(all,near-zero불안정)=%.2e\n",S,badA,S,absA,relsigA,relA);

    // ── Gate B: 연속성 — decode step-by-step(Sq=1×S, KV 이어받기) 각 위치 == forward 같은 위치 argmax ──
    reset_g4_decode_state(st);
    int badB=0; std::vector<float> step(V);
    for(int i=0;i<S;i++){
        float *d_step; TCK(cudaMalloc(&d_step,(size_t)V*4));
        gemma4_moe_decode(c,w,st,d_ids+i,1,d_cf,d_sf,d_cs,d_ss,d_step,1,uq,1);   // Sq=1, commit=1, last_only=1
        TCK(cudaMemcpy(step.data(),d_step,V*4,cudaMemcpyDeviceToHost)); cudaFree(d_step);
        int a=0; float bv=step[0]; for(int j=1;j<V;j++) if(step[j]>bv){bv=step[j];a=j;}
        if(a!=amax(fwd,i)) badB++;
    }
    printf("[Gate B] decode step-by-step(KV 이어받기) 각 위치==forward: argmax 불일치 %d/%d\n",badB,S);

    bool ok = (badA==0 && badB==0);
    printf("\n[게이트] Gate A(decode==forward)=%s | Gate B(연속성)=%s | 스킴 내(q4)\n", badA==0?"PASS":"FAIL", badB==0?"PASS":"FAIL");
    printf("[test_gemma4_moe_decode] %s (26B decode graph == forward[HF-골든 PASS] 엔진 정합 + KV 연속성 — decode==HF 전이적)\n", ok?"PASS":"FAIL");
    free_g4_decode_state(st); free_weights_gemma4(w); free_config_gemma4(c);
    return ok?0:1;
}
