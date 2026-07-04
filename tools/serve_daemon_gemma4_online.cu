// M5 온라인 연속 배칭 daemon — Gemma4 plain(greedy). 모델-중립 루프(serving_online.h) + Gemma4OnlineBackend.
//   ★헌법: 배치 루프(reader 스레드·slot·argmax·태그드 I/O)는 serving_online.h에 모델-중립으로 추출됨.
//   이 파일 = Gemma4OnlineBackend(gemma4 primitive 배선) + setup + run_online_daemon 호출. qwen daemon과 루프 공유.
//   dummy rope(gemma4 서빙 bit-id 게이트는 계산 일관성 — API==단독). 사용법: serve_daemon_gemma4_online <w> [uq=2] [maxlen=384] [Bmax=8]
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_gemma4.h"
#include "../runtime/weights_gemma4.h"
#include "../runtime/decode_state_gemma4.h"
#include "../runtime/serving_online.h"
#include "../kernels/rope.cuh"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"[online] CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

struct Gemma4OnlineBackend : OnlineBackend {
    ModelConfig c; G4Weights w;
    float *d_cf=nullptr,*d_sf=nullptr,*d_cs=nullptr,*d_ss=nullptr,*d_logit=nullptr;
    int *d_tok=nullptr,*d_pos=nullptr,*d_pf=nullptr;
    int V=0, maxlen_=0, uq=2;
    int vocab() const override { return V; }
    int max_len() const override { return maxlen_; }
    void* alloc_state(int Bmax) override { return new G4BatchState(alloc_g4_batch_state(c,Bmax,maxlen_)); }
    void  free_state(void* st) override { G4BatchState* s=(G4BatchState*)st; free_g4_batch_state(*s); delete s; }
    void  prefill_ids(void* st, int b, const int* h_ids, int P) override {
        G4BatchState* s=(G4BatchState*)st;
        CK(cudaMemcpy(d_pf, h_ids, (size_t)P*4, cudaMemcpyHostToDevice));
        gemma4_prefill_batch_slot(c,w,*s,b,d_pf,P,d_cf,d_sf,d_cs,d_ss,d_logit,uq);  // 슬롯 b KV[0..P-1], d_logit[0:V]=마지막
    }
    void  decode_step(void* st, const int* h_toks, const int* h_pos, int Bmax) override {
        G4BatchState* s=(G4BatchState*)st;
        CK(cudaMemcpy(d_tok, h_toks, Bmax*4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos, h_pos, Bmax*4, cudaMemcpyHostToDevice));
        std::vector<int> hp(h_pos, h_pos+Bmax);
        gemma4_decode_batch(c,w,*s,d_tok,d_cf,d_sf,d_cs,d_ss,d_logit,0,uq,d_pos,hp.data());  // ragged(슬롯 독립)
    }
    void  logits_to_host(float* h_dst, int rows) override { CK(cudaMemcpy(h_dst, d_logit, (size_t)rows*V*4, cudaMemcpyDeviceToHost)); }
};

int main(int argc,char**argv){
    std::string WD = argc>1?argv[1]:"artifacts/gemma4-31b/weights";
    int uq = argc>2?atoi(argv[2]):2, maxlen = argc>3?atoi(argv[3]):384, Bmax = argc>4?atoi(argv[4]):8;
    Gemma4OnlineBackend be; be.uq=uq; be.maxlen_=maxlen;
    be.c = load_config_gemma4((WD+"/manifest.json").c_str());
    be.V = be.c.vocab_size;
    const int hd_s=be.c.head_dim, hd_f=be.c.global_head_dim?be.c.global_head_dim:be.c.head_dim;
    CK(cudaMalloc(&be.d_cf,(size_t)maxlen*hd_f*4)); CK(cudaMalloc(&be.d_sf,(size_t)maxlen*hd_f*4));
    CK(cudaMalloc(&be.d_cs,(size_t)maxlen*hd_s*4)); CK(cudaMalloc(&be.d_ss,(size_t)maxlen*hd_s*4));
    launch_gen_rope(be.d_cf,be.d_sf,maxlen,hd_f,1e4f); launch_gen_rope(be.d_cs,be.d_ss,maxlen,hd_s,1e4f); CK(cudaDeviceSynchronize());
    be.w = load_weights_gemma4(WD.c_str(), be.c, uq!=0?1:0, uq==0?1:0);
    CK(cudaMalloc(&be.d_tok,Bmax*4)); CK(cudaMalloc(&be.d_pos,Bmax*4)); CK(cudaMalloc(&be.d_pf,(size_t)maxlen*4));
    CK(cudaMalloc(&be.d_logit,(size_t)Bmax*be.V*4));
    fprintf(stderr,"[online] ready: gemma4 uq=%d maxlen=%d Bmax=%d V=%d (연속 배칭 greedy, 태그드 ID-in/out)\n", uq,maxlen,Bmax,be.V); fflush(stderr);

    run_online_daemon(be, Bmax);      // 모델-중립 루프(serving_online.h)

    cudaFree(be.d_tok);cudaFree(be.d_pos);cudaFree(be.d_pf);cudaFree(be.d_logit);
    cudaFree(be.d_cf);cudaFree(be.d_sf);cudaFree(be.d_cs);cudaFree(be.d_ss);
    free_weights_gemma4(be.w); free_config_gemma4(be.c);
    return 0;
}
