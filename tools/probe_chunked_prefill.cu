// chunked prefill de-risk: P토큰 프롬프트를 청크 크기별(1=현행/64/128/256/512=full) prefill → 시간 + mism.
//   기존 qwen35_decode(Sq>1 = full-MMQ 병렬 prefill 커널) 재사용. 청크 크기 무관 last-pos logit 동일해야(causal).
//   판정: ① 청크화 win(1-tok 대비 몇 배 빠른가) ② mism=0(fp32 bit-identical / W4A8 argmax=토큰 동일).
#include "runtime/knobs.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "../runtime/config.h"
#include "../runtime/manifest_qwen.h"
#include "../runtime/weights_qwen.h"
#include "../runtime/decode_state.h"  // qwen35_decode 선언 포함
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
using clk=cudaEvent_t;
static std::string WD,GD;
static std::vector<int> ri(const std::string&p){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);fseek(f,0,SEEK_END);long b=ftell(f);fseek(f,0,SEEK_SET);std::vector<int>v(b/4);if(fread(v.data(),4,v.size(),f)!=v.size())exit(1);fclose(f);return v;}
static std::vector<float> rf(const std::string&p,size_t n){FILE*f=fopen((GD+p).c_str(),"rb");if(!f)exit(1);std::vector<float>v(n);if(fread(v.data(),4,n,f)!=n)exit(1);fclose(f);return v;}
static float* up(const std::vector<float>&h){float*d;CK(cudaMalloc(&d,h.size()*4));CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
    WD=argc>1?argv[1]:"artifacts/qwen35-0.8b/weights"; GD=(argc>2?argv[2]:"artifacts/qwen35-0.8b/mtp_golden")+std::string("/");
    int uq=argc>3?atoi(argv[3]):2; int P=argc>4?atoi(argv[4]):512;
    ModelConfig c=load_config_qwen((WD+"/manifest.json").c_str());
    const int V=c.vocab_size, rot=c.rotary_dim; int maxlen=P+8;
    std::vector<int> ids=ri("token_ids.bin");
    std::vector<int> prompt(P); for(int i=0;i<P;i++) prompt[i]=ids[i%ids.size()];  // 타일링(timing만 중요)
    std::vector<float> cl=rf("rope_cos_long.bin",(size_t)128*rot), sl=rf("rope_sin_long.bin",(size_t)128*rot);
    // rope 테이블은 128행 → P>128이면 부족. long rope 재생성 대신 P<=128 rope는 반복(timing 목적). 실제 위치 rope는
    //   position별인데 여기선 커널 시간·mism 목적이라 사용가능 행 내로 클램프(정확 위치 rope는 엔진이 처리, 여기선 상대 비교).
    // 안전: rope 행 부족 방지 위해 cos/sin을 P행으로 확장(반복). mism은 청크 크기 간 상대 비교라 rope 동일하면 유효.
    if((int)cl.size() < (size_t)P*rot){ std::vector<float> cl2((size_t)P*rot),sl2((size_t)P*rot);
        for(int i=0;i<P;i++)for(int j=0;j<rot;j++){cl2[(size_t)i*rot+j]=cl[(size_t)(i%128)*rot+j];sl2[(size_t)i*rot+j]=sl[(size_t)(i%128)*rot+j];}
        cl=cl2; sl=sl2; }
    float*d_cos=up(cl),*d_sin=up(sl);
    QwenWeights w=load_weights_qwen(WD.c_str(),c,1,uq==0?1:0);
    int*d_tok; CK(cudaMalloc(&d_tok,(size_t)P*4)); CK(cudaMemcpy(d_tok,prompt.data(),(size_t)P*4,cudaMemcpyHostToDevice));
    float*d_logit; CK(cudaMalloc(&d_logit,(size_t)P*V*4));
    DecodeState st=alloc_decode_state(c,maxlen);
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    printf("[probe_chunked_prefill] uq=%d(%s) P=%d\n",uq,uq==0?"fp32":"W4A8",P);

    // 청크 크기별 prefill: last-pos(P-1) logit 캡처 + 시간(reset→청크 루프)
    auto run=[&](int C, std::vector<float>&lastlog)->double{
        int reps=5; float best=1e30f;
        for(int r=0;r<=reps;r++){
            reset_decode_state(c,st);
            CK(cudaEventRecord(e0));
            for(int off=0; off<P; off+=C){ int sq=std::min(C,P-off);
                qwen35_decode(c,w,st,d_tok+off,sq,d_cos,d_sin,d_logit,nullptr,1,nullptr,uq); }
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); if(r>0&&ms<best)best=ms;  // r=0 warmup
        }
        // 마지막 청크의 마지막 행(pos P-1) logit
        int lastC=((P-1)/C)*C; int lastsq=std::min(C,P-lastC); (void)lastsq;
        // d_logit엔 마지막 호출 결과([lastsq,V])만 남음 → 그 마지막 행이 pos P-1
        int rows=std::min(C,P-lastC); std::vector<float> all((size_t)rows*V);
        CK(cudaMemcpy(all.data(),d_logit,(size_t)rows*V*4,cudaMemcpyDeviceToHost));
        lastlog.assign(all.begin()+(size_t)(rows-1)*V, all.end());
        return best;
    };
    std::vector<int> chunks = knob_raw("FULLONLY")? std::vector<int>{512} : std::vector<int>{1,64,128,256,512};
    std::vector<std::vector<float>> logs(chunks.size()); std::vector<double> times(chunks.size());
    for(size_t i=0;i<chunks.size();i++){ if(chunks[i]>P)continue; times[i]=run(chunks[i],logs[i]); }
    // mism: chunk=1(현행) 기준 argmax + max-abs
    auto amax=[&](const std::vector<float>&l){int bi=0;float bv=l[0];for(int j=1;j<V;j++)if(l[j]>bv){bv=l[j];bi=j;}return bi;};
    int ref=amax(logs[0]); double t1=times[0];
    printf("  %-8s %10s %8s %8s %s\n","chunk","prefill_ms","speedup","argmax","max|Δ| vs chunk=1");
    for(size_t i=0;i<chunks.size();i++){ if(chunks[i]>P||logs[i].empty())continue;
        int am=amax(logs[i]); double mx=0; for(int j=0;j<V;j++){double d=fabs(logs[i][j]-logs[0][j]);if(d>mx)mx=d;}
        printf("  %-8d %10.3f %7.1f× %8d %s (max|Δ|=%.3e)\n",chunks[i],times[i],t1/times[i],am,
               am==ref?"동일":"★다름", mx);
    }
    printf("  → 청크화 win = chunk=1 대비 full(512) %.1f×; mism: argmax %s (fp32 bit-identical / W4A8 borderline은 argmax 기준)\n",
           logs.back().empty()?0:t1/times.back(), "위 표 참조");
    free_decode_state(st); free_weights_qwen(w); free_config_qwen(c); return 0;
}
