// M5 Phase 0 — Gemma4 plain 서빙 daemon (C++ 계층: token-ID in/out만, HTTP/토크나이저 없음).
//   2계층 아키텍처(docs/m5-api-design.md): Python front(serve_api.py)이 HTTP·HF 토큰화·SSE·recipe verify를 담당하고
//   이 daemon은 영속 프로세스로 gemma4를 1회 로드해 stdin의 token-ID 요청을 받아 greedy decode → stdout에 token-ID 스트림.
//   ★정확성 불변: gemma4_decode(검증필, test_gemma4_gen과 동일 경로)를 그대로 재사용 — daemon은 ID를 통과·argmax·루프만.
//     따라서 daemon 경유 greedy 출력 == 직접 gemma4_decode greedy 출력 bit-identical(프로토콜이 계산 안 바꿈).
//   Phase 0 스코프: gemma4 plain 1모델·greedy만(샘플링 없음)·dummy rope(실-rope 정확성은 별도 트랙 — bit-id는 같은 rope라 불변).
//   프로토콜(라인 기반): stdin "<max_tokens> <id0> <id1> ... <idP-1>" → stdout 생성 토큰 "<id>\n"(토큰마다 flush) + "###END\n".
//   사용법: serve_daemon_gemma4 <weights_dir> [uq=2] [maxlen=384]   (uq/config는 front이 recipe에서 읽어 전달)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../kernels/rope.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"[daemon] CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

int main(int argc, char** argv){
    std::string WD = argc>1 ? argv[1] : "artifacts/gemma4-31b/weights";
    int uq = argc>2 ? atoi(argv[2]) : 2;
    int maxlen = argc>3 ? atoi(argv[3]) : 384;

    ModelConfig c = load_config_gemma4((WD+"/manifest.json").c_str());
    const int V = c.vocab_size;
    const int hd_s = c.head_dim, hd_f = c.global_head_dim ? c.global_head_dim : c.head_dim;

    // dummy rope(타이밍/골격 — 값은 bit-id 대조에 불변, 같은 rope 공유). 실-rope 정확성은 다음 트랙.
    float *d_cf,*d_sf,*d_cs,*d_ss;
    CK(cudaMalloc(&d_cf,(size_t)maxlen*hd_f*4)); CK(cudaMalloc(&d_sf,(size_t)maxlen*hd_f*4));
    CK(cudaMalloc(&d_cs,(size_t)maxlen*hd_s*4)); CK(cudaMalloc(&d_ss,(size_t)maxlen*hd_s*4));
    launch_gen_rope(d_cf,d_sf,maxlen,hd_f,1e4f); launch_gen_rope(d_cs,d_ss,maxlen,hd_s,1e4f); CK(cudaDeviceSynchronize());

    G4Weights w = load_weights_gemma4(WD.c_str(), c, uq!=0?1:0, uq==0?1:0);
    G4DecodeState st = alloc_g4_decode_state(c, maxlen);
    int *d_tok; CK(cudaMalloc(&d_tok,(size_t)maxlen*4));
    float *d_logit; CK(cudaMalloc(&d_logit,(size_t)V*4));
    std::vector<float> hlog(V);

    // argmax(host) — greedy. V=262144 host 루프(토큰당 1회, 저렴).
    auto argmax = [&]()->int{ CK(cudaMemcpy(hlog.data(),d_logit,(size_t)V*4,cudaMemcpyDeviceToHost));
        int b=0; float bv=hlog[0]; for(int j=1;j<V;j++) if(hlog[j]>bv){bv=hlog[j];b=j;} return b; };

    fprintf(stderr,"[daemon] ready: gemma4 uq=%d maxlen=%d V=%d (plain greedy, ID-in/out)\n", uq,maxlen,V);
    fflush(stderr);

    std::string line;
    while(std::getline(std::cin, line)){
        std::istringstream is(line);
        long mt; if(!(is>>mt)) continue;                    // 첫 정수 = max_tokens
        std::vector<int> prompt; int id; while(is>>id) prompt.push_back(id);
        int P = (int)prompt.size();
        if(P==0){ printf("###END\n"); fflush(stdout); continue; }
        if(P >= maxlen){ P = maxlen-1; prompt.resize(P); }   // KV 캡 가드(안전 불변식)
        int budget = maxlen - P;                              // 생성 가능 상한(KV 선형 캡)
        if(mt > budget) mt = budget; if(mt < 0) mt = 0;

        reset_g4_decode_state(st);                            // n_past=0(KV 재사용, 요청 독립)
        // 프리필: Sq=P, commit=1(n_past=P), last_only=1(마지막 위치 logit) → 첫 생성 토큰 argmax
        CK(cudaMemcpy(d_tok, prompt.data(), (size_t)P*4, cudaMemcpyHostToDevice));
        gemma4_decode(c,w,st,d_tok,P,d_cf,d_sf,d_cs,d_ss,d_logit,/*commit*/1,uq,/*last_only*/1);
        int next = argmax();
        long gen = 0;
        if(mt>0){ printf("%d\n", next); fflush(stdout); gen++; }  // 스트리밍(토큰마다 flush)
        // decode: Sq=1, 이전 생성 토큰 입력, commit=1(n_past++), last_only=1
        while(gen < mt && st.n_past < maxlen-1){
            CK(cudaMemcpy(d_tok, &next, 4, cudaMemcpyHostToDevice));
            gemma4_decode_step(c,w,st,d_tok,1,d_cf,d_sf,d_cs,d_ss,d_logit,1,uq,1);
            next = argmax();
            printf("%d\n", next); fflush(stdout); gen++;
        }
        printf("###END\n"); fflush(stdout);                   // 요청 종료 마커
    }

    cudaFree(d_tok); cudaFree(d_logit); cudaFree(d_cf); cudaFree(d_sf); cudaFree(d_cs); cudaFree(d_ss);
    free_g4_decode_state(st); free_weights_gemma4(w); free_config_gemma4(c);
    return 0;
}
