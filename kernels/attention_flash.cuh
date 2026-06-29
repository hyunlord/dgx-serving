#ifndef DGX_ATTENTION_FLASH_CUH
#define DGX_ATTENTION_FLASH_CUH
#include <float.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>

// ── attention 커널 landscape (왜 여러 개인가 — mm() 복붙이 조용히 갈라진 교훈, 명시 기록) ──────────
//   1) launch_attn (attention.cuh)            — 레퍼런스 forward(qwen35.cu/smollm3.cu). golden 생성용.
//        S≤HD 제약, 작은 S만(벽 무관). **의도적 독립**: 엔진 FA와 다른 커널 = 교차검증 ground truth.
//   2) launch_flash_attn (이 파일)            — 엔진 prefill+decode(qwen35_decode.cu). FA.1a. 12K 벽 없음.
//        attn_decode를 **교체**. full-attn 16층.
//   3) launch_attn_decode_batch (attention.cuh) — batch decode(qwen35_batch.cu). **아직 12K 벽 있음**
//        (full scores in shared). batch 긴-context는 나중 FA 교체 예정(TODO).
//   4) launch_attn_decode (attention.cuh)     — 옛 엔진 경로. 이제 test_flash_attn의 벽-안 대조 레퍼런스로만.
//   ⚠ FA 개선(FA.1b/FA.2) 시 1)·3)은 자동 안 받음 → 갈라질 잠재. golden 틀어지면 여기부터 의심.

// 챕터7 FA.1a: tiled FlashAttention (online softmax). full-attn 16층용 (GQA causal).
//   현 attn_decode의 두 문제 해결: ① full scores[attendlen]를 shared에 → 48KB 벽(context>12K 깨짐).
//   ② softmax를 단일 스레드 직렬. → FA는 KV를 BC씩 타일 순회 + running(m,l,acc) → shared 상수(BC 고정)
//   + 타일별 softmax. shared = q[HD]+Ks[BC][HD]+Vs[BC][HD]+s[BC] = BC 고정(32K/256K든 동일) → 벽 없음.
//
//   매핑: block=(query i, head h), blockDim=HD. thread d = 출력 dim d 소유(acc[d] running).
//   GQA: kv head = h/(NQ/NKV) — 각 query head가 올바른 KV head attend (데인 함정, 명시).
//   causal: query 절대위치 qpos=base+i, attend [0..qpos] (attendlen=qpos+1). 루프 bound가 causality 보장.
//   BC = 타일 크기(템플릿). 정확성 핵심 = online softmax running 갱신(m 바뀔 때 exp(m_old-m_new) rescale).
template<int HD, int BC>
static __global__ void flash_attn_kernel(const float *Q, const float *Kc, const float *Vc, float *O,
                                         int Sq, int NQ, int NKV, int base) {
    int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int GRP = NQ / NKV, kv = h / GRP;
    int qpos = base + i, attendlen = qpos + 1;
    float scale = 1.0f / sqrtf((float)HD);

    __shared__ float q[HD];
    __shared__ float Ks[BC][HD];
    __shared__ float Vs[BC][HD];
    __shared__ float s[BC];

    q[d] = Q[((size_t)i * NQ + h) * HD + d];
    __syncthreads();

    float m = -FLT_MAX, l = 0.f, acc = 0.f;   // running: m=max, l=denom, acc=출력 dim d
    for (int j0 = 0; j0 < attendlen; j0 += BC) {
        int Bcur = (attendlen - j0 < BC) ? (attendlen - j0) : BC;
        // KV 타일 적재 (thread d = 각 행의 dim d, coalesced)
        for (int r = 0; r < Bcur; r++) {
            Ks[r][d] = Kc[((size_t)(j0 + r) * NKV + kv) * HD + d];
            Vs[r][d] = Vc[((size_t)(j0 + r) * NKV + kv) * HD + d];
        }
        __syncthreads();
        // scores: thread d(<Bcur)가 key j=d 의 full dot (q·Ks[d])
        if (d < Bcur) {
            float dot = 0.f;
            for (int t = 0; t < HD; t++) dot += q[t] * Ks[d][t];
            s[d] = dot * scale;
        }
        __syncthreads();
        // online softmax: 타일 max → running rescale (모든 스레드 중복 계산, Bcur<=BC 작음)
        float tile_m = -FLT_MAX;
        for (int j = 0; j < Bcur; j++) tile_m = fmaxf(tile_m, s[j]);
        float m_new = fmaxf(m, tile_m);
        float alpha = expf(m - m_new);             // 이전 누적 보정 (첫 타일 m=-FLT_MAX → 0)
        float tile_l = 0.f, acc_tile = 0.f;
        for (int j = 0; j < Bcur; j++) {
            float p = expf(s[j] - m_new);
            tile_l += p;
            acc_tile += p * Vs[j][d];              // thread d = 출력 dim d
        }
        l = l * alpha + tile_l;
        acc = acc * alpha + acc_tile;
        m = m_new;
        __syncthreads();                           // 다음 타일이 Ks/Vs 덮기 전
    }
    O[((size_t)i * NQ + h) * HD + d] = acc / l;
}

// 런처: HD는 컴파일타임(head_dim 고정 — 헌법 허용 상수). BC=32 → shared ~33KB 상수(벽 없음).
template<int HD, int BC>
static inline void launch_flash_attn_t(const float *Q, const float *Kc, const float *Vc, float *O,
                                       int Sq, int NQ, int NKV, int base, cudaStream_t st = 0) {
    dim3 grid(Sq, NQ);
    flash_attn_kernel<HD, BC><<<grid, HD, 0, st>>>(Q, Kc, Vc, O, Sq, NQ, NKV, base);
}

// 엔진용 기본 진입. head_dim별 인스턴스(BC는 shared<48KB 유지): HD256→BC16(33KB), HD128→BC32(33KB).
//   ⚠ 미지원 HD는 abort(조용한 return 금지 — 그게 0.8B/27B head_dim=256을 놓친 버그였음).
template<int HD, int BC>   // FA.2 런처 forward 선언 (정의는 파일 하단, 기본인자는 여기만)
static inline void launch_flash_attn2_t(const float *Q, const float *Kc, const float *Vc, float *O,
                                        int Sq, int NQ, int NKV, int base, cudaStream_t st = 0);
static inline void launch_flash_attn(const float *Q, const float *Kc, const float *Vc, float *O,
                                     int Sq, int NQ, int NKV, int HD, int base, cudaStream_t st = 0) {
    // 측정 결과 분기: prefill(Sq>1)=FA.2(bf16 tensor-core, 2-3×) / decode(Sq=1)=FA.1(software, 쿼리1블록 효율).
    //   FA.2는 BR=16 타일이라 Sq=1서 15/16 낭비 → decode 퇴보. decode 특화(FA.2c)는 추후. 미지원 HD는 abort.
    if (Sq > 1) {
        if      (HD == 256) launch_flash_attn2_t<256, 16>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st);
        else if (HD == 128) launch_flash_attn2_t<128, 16>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st);
        else goto unsup;
    } else {
        if      (HD == 256) launch_flash_attn_t<256, 16>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st);   // BC=16 → shared 33KB(48KB내)
        else if (HD == 128) launch_flash_attn_t<128, 32>(Q, Kc, Vc, O, Sq, NQ, NKV, base, st);
        else goto unsup;
    }
    return;
unsup:
    fprintf(stderr, "[flash_attn] 미지원 head_dim=%d (128/256만). 조용한 오답 대신 중단합니다.\n", HD);
    abort();
}

// ── FA.2: bf16 tensor-core FlashAttention (m16n8k16) ─────────────────────────────────────────
//   FA.1a(software) 교체용. S=Q·Kᵀ / O=P·V를 bf16 MMA, online softmax는 shared-왕복(warp-분산 회피, 정확성먼저).
//   fp32 누적 전부(S/acc/running m,l) — vLLM FP8 FA 긴-context 붕괴(누적 부정확) 회피. CTA=1 warp=(쿼리타일 BR, head h).
//   BR=16(MMA m-tile), BC=16(k-step 1). HD 템플릿. GQA kv=h/(NQ/NKV). causal는 행별 qpos 마스크.
__device__ __forceinline__ unsigned fa_pkbf(const __nv_bfloat16 &x, const __nv_bfloat16 &y){
    return (unsigned)(*(const unsigned short*)&x) | ((unsigned)(*(const unsigned short*)&y) << 16);
}
template<int HD, int BR, int BC>
static __global__ void flash_attn2_kernel(const float *Q, const float *Kc, const float *Vc, float *O,
                                          int Sq, int NQ, int NKV, int base) {
    int q0 = blockIdx.x * BR, h = blockIdx.y;
    int GRP = NQ / NKV, kv = h / GRP;
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    float scale = 1.0f / sqrtf((float)HD);
    __shared__ __nv_bfloat16 sQ[BR][HD], sK[BC][HD], sV[BC][HD], sP[BR][BC];
    __shared__ float sS[BR][BC], sAcc[BR][HD], sm[BR], sl[BR];

    for (int i = lane; i < BR*HD; i += 32) { int r=i/HD,c=i%HD,gq=q0+r;
        sQ[r][c] = (gq<Sq) ? __float2bfloat16(Q[((size_t)gq*NQ+h)*HD+c]) : (__nv_bfloat16)0; sAcc[r][c]=0.f; }
    for (int i = lane; i < BR; i += 32) { sm[i]=-FLT_MAX; sl[i]=0.f; }
    __syncwarp();

    int qend = (q0+BR < Sq) ? q0+BR : Sq;   // 타일 내 실제 query 끝 (부분타일 KV OOB 방지)
    int maxlen = base + qend;                // 최대 attendlen (행별 마스크로 causal). ≤ base+Sq = KV 길이
    for (int j0 = 0; j0 < maxlen; j0 += BC) {
        for (int i = lane; i < BC*HD; i += 32) { int r=i/HD,c=i%HD,gk=j0+r;
            bool v = gk < maxlen;
            sK[r][c] = v ? __float2bfloat16(Kc[((size_t)gk*NKV+kv)*HD+c]) : (__nv_bfloat16)0;
            sV[r][c] = v ? __float2bfloat16(Vc[((size_t)gk*NKV+kv)*HD+c]) : (__nv_bfloat16)0; }
        __syncwarp();
        // S = Q·Kᵀ (bf16 MMA, FA.2.0 레이아웃). n-타일 BC/8, k-스텝 HD/16.
        #pragma unroll
        for (int ns=0; ns<BC/8; ns++) {
            float c0=0,c1=0,c2=0,c3=0;
            for (int ks=0; ks<HD; ks+=16) {
                unsigned a0=*(unsigned*)&sQ[g][ks+2*t], a1=*(unsigned*)&sQ[g+8][ks+2*t];
                unsigned a2=*(unsigned*)&sQ[g][ks+2*t+8], a3=*(unsigned*)&sQ[g+8][ks+2*t+8];
                unsigned b0=*(unsigned*)&sK[ns*8+g][ks+2*t], b1=*(unsigned*)&sK[ns*8+g][ks+2*t+8];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                  :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
            sS[g][ns*8+2*t]=c0*scale; sS[g][ns*8+2*t+1]=c1*scale; sS[g+8][ns*8+2*t]=c2*scale; sS[g+8][ns*8+2*t+1]=c3*scale;
        }
        __syncwarp();
        // online softmax (shared-왕복, 행별. 모든 fp32). lane<BR이 행 담당.
        if (lane < BR) { int r=lane, gq=q0+r, qpos=base+gq;
            if (gq < Sq) {
                float tm=-FLT_MAX;
                for (int j=0;j<BC;j++){ int gk=j0+j; float s=(gk<=qpos)?sS[r][j]:-FLT_MAX; sS[r][j]=s; tm=fmaxf(tm,s); }
                float mnew=fmaxf(sm[r],tm), alpha=expf(sm[r]-mnew), tl=0.f;
                for (int j=0;j<BC;j++){ float p=expf(sS[r][j]-mnew); sP[r][j]=__float2bfloat16(p); tl+=p; }
                sl[r]=sl[r]*alpha+tl; sm[r]=mnew;
                for (int c=0;c<HD;c++) sAcc[r][c]*=alpha;
            } else for (int j=0;j<BC;j++) sP[r][j]=(__nv_bfloat16)0;
        }
        __syncwarp();
        // O += P·V (bf16 MMA, FA.2.1 레이아웃). BC=16=k-step 1. n-타일 HD/8.
        #pragma unroll
        for (int n0=0;n0<HD;n0+=8) {
            unsigned a0=*(unsigned*)&sP[g][2*t], a1=*(unsigned*)&sP[g+8][2*t];
            unsigned a2=*(unsigned*)&sP[g][2*t+8], a3=*(unsigned*)&sP[g+8][2*t+8];
            unsigned b0=fa_pkbf(sV[2*t][n0+g], sV[2*t+1][n0+g]);
            unsigned b1=fa_pkbf(sV[2*t+8][n0+g], sV[2*t+9][n0+g]);
            float c0=0,c1=0,c2=0,c3=0;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
              :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            sAcc[g][n0+2*t]+=c0; sAcc[g][n0+2*t+1]+=c1; sAcc[g+8][n0+2*t]+=c2; sAcc[g+8][n0+2*t+1]+=c3;
        }
        __syncwarp();
    }
    if (lane < BR) { int r=lane, gq=q0+r;
        if (gq < Sq) { float li=1.0f/sl[r]; for (int c=0;c<HD;c++) O[((size_t)gq*NQ+h)*HD+c]=sAcc[r][c]*li; } }
}
template<int HD, int BC>
static inline void launch_flash_attn2_t(const float *Q, const float *Kc, const float *Vc, float *O,
                                        int Sq, int NQ, int NKV, int base, cudaStream_t st) {
    dim3 grid((Sq + 15) / 16, NQ);    // BR=16
    flash_attn2_kernel<HD, 16, BC><<<grid, 32, 0, st>>>(Q, Kc, Vc, O, Sq, NQ, NKV, base);
}

#endif // DGX_ATTENTION_FLASH_CUH
