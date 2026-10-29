// M5 온라인 연속 배칭 daemon — Qwen3.5 계열(27b·35b-a3b MoE, plain greedy). 모델-중립 루프(serving_online.h) + QwenOnlineBackend.
//   ★헌법: 배치 루프는 serving_online.h(모델-중립) 공유 — gemma4 daemon과 동일 루프. 이 파일 = qwen primitive 배선 + setup.
//   35b-a3b는 MoE fused 서빙 이미 물림(qwen35_decode_batch). 27b plain도 같은 primitive(가중치/config만 다름).
//   ★qwen은 GDN state 보유 → reset_slot(슬롯 재사용 시 GDN clear) override 필수(gemma4는 no-op).
//   rope: 실제 golden 테이블(ext=pos정확 / long=128타일 근사) — bench_serve.cu와 동일. bit-id 게이트는 계산 일관성(API==단독).
//   사용법: serve_daemon_qwen_online <weights_dir> <golden_dir(rope)> [uq=2] [maxlen=384] [Bmax=8]
#include "runtime/knobs.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"
#include "../models/qwen35_batch.h"
#include "../runtime/serving_online.h"

#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"[online] CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

struct QwenOnlineBackend : OnlineBackend {
    ModelConfig c; QwenWeights w;
    float *d_cos=nullptr,*d_sin=nullptr,*d_logit=nullptr; int *d_tok=nullptr,*d_pf=nullptr;
    int V=0, maxlen_=0, uq=2;
    int vocab() const override { return V; }
    int max_len() const override { return maxlen_; }
    void* alloc_state(int Bmax) override {
        BatchDecodeState* st=new BatchDecodeState(alloc_batch_decode_state(c,Bmax,maxlen_));
        st->B=Bmax; reset_batch_decode_state(c,*st); return st;
    }
    void  free_state(void* st) override { BatchDecodeState* s=(BatchDecodeState*)st; free_batch_decode_state(*s); delete s; }
    void  reset_slot(void* st, int b) override { reset_batch_slot(c, *(BatchDecodeState*)st, b); }   // GDN state clear(누수 방지)
    void  prefill_ids(void* st, int b, const int* h_ids, int P) override {
        BatchDecodeState* s=(BatchDecodeState*)st;
        CK(cudaMemcpy(d_pf, h_ids, (size_t)P*4, cudaMemcpyHostToDevice));
        prefill_batch_slot(c,w,*s,b,d_pf,P,d_cos,d_sin,d_logit,uq,512);   // 슬롯 b KV[0..P-1], d_logit[0:V]=마지막
    }
    void  decode_step(void* st, const int* h_toks, const int* h_pos, int Bmax) override {
        BatchDecodeState* s=(BatchDecodeState*)st; s->B=Bmax;
        CK(cudaMemcpy(d_tok, h_toks, Bmax*4, cudaMemcpyHostToDevice));
        std::vector<int> hp(h_pos, h_pos+Bmax);
        qwen35_decode_batch(c,w,*s,d_tok,d_cos,d_sin,d_logit,uq,hp.data());   // ragged(슬롯별 위치 host)
    }
    void  logits_to_host(float* h_dst, int rows) override { CK(cudaMemcpy(h_dst, d_logit, (size_t)rows*V*4, cudaMemcpyDeviceToHost)); }
};

// golden 바이너리 float 읽기
static std::vector<float> rf(const std::string& p, size_t n){
    FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"[online] rope 파일 없음: %s\n",p.c_str());exit(1);}
    std::vector<float> v(n); if(fread(v.data(),4,n,f)!=n){fprintf(stderr,"[online] rope 읽기 실패\n");exit(1);} fclose(f); return v;
}

int main(int argc,char**argv){
    knobs_init("serve_daemon_qwen_online");
    std::string WD = argc>1?argv[1]:"artifacts/qwen35-35b-a3b/weights";
    std::string GD = (argc>2?argv[2]:"artifacts/qwen35-35b-a3b/golden") + std::string("/");
    int uq = argc>3?atoi(argv[3]):2, maxlen = argc>4?atoi(argv[4]):384, Bmax = argc>5?atoi(argv[5]):8;
    QwenOnlineBackend be; be.uq=uq; be.maxlen_=maxlen;
    be.c = load_config_qwen((WD+"/manifest.json").c_str());
    be.V = be.c.vocab_size; int rot = be.c.rotary_dim;

    // rope: ext 테이블(pos>=128 정확) 있으면 maxlen행, 없으면 long 128타일(pos>128 근사) — bench_serve.cu와 동일.
    std::vector<float> cl, sl;
    { FILE*e=fopen((GD+"rope_cos_ext.bin").c_str(),"rb");
      if(e){ fseek(e,0,SEEK_END); long er=ftell(e)/4/rot; fclose(e);
             int use=(er>=maxlen)?maxlen:(int)er; cl=rf(GD+"rope_cos_ext.bin",(size_t)use*rot); sl=rf(GD+"rope_sin_ext.bin",(size_t)use*rot);
             if(use<maxlen){ cl.resize((size_t)maxlen*rot); sl.resize((size_t)maxlen*rot);
                 for(int i=use;i<maxlen;i++)for(int j=0;j<rot;j++){cl[(size_t)i*rot+j]=cl[(size_t)(use-1)*rot+j];sl[(size_t)i*rot+j]=sl[(size_t)(use-1)*rot+j];} }
             fprintf(stderr,"[online] rope: ext %d행(pos>=128 정확)\n",use); }
      else { auto c0=rf(GD+"rope_cos_long.bin",(size_t)128*rot), s0=rf(GD+"rope_sin_long.bin",(size_t)128*rot);
             cl.resize((size_t)maxlen*rot); sl.resize((size_t)maxlen*rot);
             for(int i=0;i<maxlen;i++)for(int j=0;j<rot;j++){cl[(size_t)i*rot+j]=c0[(size_t)(i%128)*rot+j];sl[(size_t)i*rot+j]=s0[(size_t)(i%128)*rot+j];}
             fprintf(stderr,"[online] rope: long 128타일(ext 없음 — pos>128 근사)\n"); } }
    CK(cudaMalloc(&be.d_cos,cl.size()*4)); CK(cudaMemcpy(be.d_cos,cl.data(),cl.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&be.d_sin,sl.size()*4)); CK(cudaMemcpy(be.d_sin,sl.data(),sl.size()*4,cudaMemcpyHostToDevice));

    be.w = load_weights_qwen(WD.c_str(), be.c, 1, uq==0?1:0);
    CK(cudaMalloc(&be.d_tok,Bmax*4)); CK(cudaMalloc(&be.d_pf,(size_t)maxlen*4));
    CK(cudaMalloc(&be.d_logit,(size_t)Bmax*be.V*4));
    fprintf(stderr,"[online] ready: qwen uq=%d maxlen=%d Bmax=%d V=%d rot=%d (연속 배칭 greedy, 태그드 ID-in/out)\n", uq,maxlen,Bmax,be.V,rot); fflush(stderr);

    run_online_daemon(be, Bmax, parse_stop_ids_env());      // 모델-중립 루프(serving_online.h) + EOS 종료(STOP_IDS env)

    cudaFree(be.d_tok);cudaFree(be.d_pf);cudaFree(be.d_logit);cudaFree(be.d_cos);cudaFree(be.d_sin);
    free_weights_qwen(be.w); free_config_qwen(be.c);
    return 0;
}
