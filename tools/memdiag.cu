// 안전 진단: forward/큰 커널 실행 없이 메모리 상태만. 크래시 원인 규명용.
//   cudaMemGetInfo(로드 전/후) + context별 est_workspace/est_kv + 가드 판정. decode_state(KV)까지만 할당(안전).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include "../runtime/engine.h"
#include "../runtime/decode_state.h"

static size_t est_workspace_bytes(const ModelConfig &c, int Sq){
    size_t H=c.hidden_size, FFN=c.intermediate_size, HD=c.head_dim;
    size_t qdim2=(size_t)c.num_q_heads*HD*(c.attn_output_gate?2:1), qdim=(size_t)c.num_q_heads*HD, kvdim=(size_t)c.num_kv_heads*HD;
    size_t convd=cfg_gdn_conv_dim(&c), vdim=cfg_gdn_value_dim(&c), kdim=cfg_gdn_key_dim(&c), nvh=c.gdn_num_v_heads, hv=c.gdn_head_v_dim;
    size_t perTok = 8*H + qdim2 + 4*qdim + 3*kvdim + 3*convd + 2*kdim + 4*vdim + 4*nvh;
    if (c.num_experts>0) perTok += (size_t)c.num_experts + 3*(size_t)c.shared_expert_intermediate_size + H;
    else                 perTok += 3*FFN;
    size_t snap = (size_t)nvh*hv*hv;
    return ((size_t)Sq*perTok + snap) * 4;
}
static size_t est_kv_bytes(const ModelConfig &c, int len){
    int nfull=0; for(int i=0;i<c.num_layers;i++) if(c.layer_is_full?c.layer_is_full[i]:1) nfull++;
    return (size_t)nfull*(size_t)len*c.num_kv_heads*c.head_dim*2*4;
}
static void mem(const char*tag){ size_t f=0,t=0; cudaMemGetInfo(&f,&t);
    printf("  [mem] %-18s free=%.1fGB total=%.1fGB used=%.1fGB\n",tag,f/1e9,t/1e9,(t-f)/1e9); }

int main(int argc,char**argv){
    std::string WD=argc>1?argv[1]:"artifacts/qwen35-27b/weights";
    mem("로드 전");
    Engine e=engine_load(WD.c_str(),1,0);
    printf("# model H=%d L=%d V=%d experts=%d FFN=%d\n",e.c.hidden_size,e.c.num_layers,e.c.vocab_size,e.c.num_experts,e.c.intermediate_size);
    mem("로드 후(가중치)");
    int ctxs[]={128,2048,8192,32768};
    printf("# context별 예산 (forward 실행 안 함):\n");
    for(int C: ctxs){
        size_t ws=est_workspace_bytes(e.c,C), kv=est_kv_bytes(e.c,C+512+1), need=ws+kv;
        size_t f=0,t=0; cudaMemGetInfo(&f,&t);
        printf("  C=%-6d ws=%.1fGB kv=%.1fGB need=%.1fGB | 가드(0.75×free=%.1fGB): %s\n",
               C, ws/1e9, kv/1e9, need/1e9, 0.75*f/1e9, need>(size_t)(0.75*(double)f)?"SKIP":"통과");
    }
    // KV만 할당(안전, 32K KV ~4.4GB) — workspace/forward는 안 함
    printf("# 32K KV 할당 테스트(안전, forward 없음):\n");
    DecodeState st=alloc_decode_state(e.c, 32768+512+1);
    mem("32K KV 할당 후");
    free_decode_state(st);
    engine_free(e);
    return 0;
}
