#pragma once
#include <cstdint>
#include "qscale.cuh"
#include "../runtime/config.h"   // QBLK
#include "gemv_w4a8.cuh"         // S=1 fallback

// 챕터7 Phase B: tensor-core int8 MMA W4A8 GEMM. Y[M,N] = (X int8 · (W int4-8) int8) ×(xs·sw).
//   X int8[M,K] + xs[M,K/32], Wq int4 packed[N,K/2] + sw[N,K/32]. MMA m16n8k32.s8.s8.s32.
//   B.0/B.1 검증된 fragment 레이아웃 + 블록 스케일. K-tile=QBLK=32=MMA k 정렬.

// CTA 타일 BM×BN, 8 warp(2×4) 재사용. MMA_PAD=4(stride 36)=스칼라 적재 bank conflict 회피(측정 채택).
//   ※ ldmatrix 경로는 16B 정렬(stride 16배수) 필요했고 측정 평탄이라 기각 → 스칼라+register-blocking 유지.
#define MMA_BM 64
#define MMA_BN 64
#define MMA_PAD 4
// ── Phase C: cp.async 2-stage 더블버퍼 ─────────────────────────────────────
//   ncu 진단: B.2b는 load→sync→MMA 직렬화로 warp stall 59.9%=shared 적재 대기(scoreboard).
//   처방: 다음 K-타일을 cp.async로 prefetch하며 현재 타일 MMA → mem 42%↔compute 35% 겹침.
//   X int8 = 순수 cp.async. W는 packed int4 그대로 cp.async(sWp) → fragment build 때 unpack((q-8)).
#define W_PAD 4
// 2 byte(4 nibble: k,k+1,k+2,k+3) → (q-8) int8 4개를 int 레지스터로 (little-endian 바이트순).
__device__ __forceinline__ int gm_unpack4(const uint8_t *p) {
    unsigned b = *(const unsigned short *)p;
    int q0=(int)(b&0xF)-8, q1=(int)((b>>4)&0xF)-8, q2=(int)((b>>8)&0xF)-8, q3=(int)((b>>12)&0xF)-8;
    return (q0&0xFF)|((q1&0xFF)<<8)|((q2&0xFF)<<16)|((q3&0xFF)<<24);
}
// 값 버전: 정렬된 4-byte read 후 하위 16bit(4 nibble=int4 4개)를 (q-8) int8로. bank conflict 회피용
//   (2-byte sub-word read는 같은 bank word를 분할 접근 → conflict. 4B read + broadcast로 제거).
__device__ __forceinline__ int gm_unpack4v(unsigned h) {
    int q0=(int)(h&0xF)-8, q1=(int)((h>>4)&0xF)-8, q2=(int)((h>>8)&0xF)-8, q3=(int)((h>>12)&0xF)-8;
    return (q0&0xFF)|((q1&0xFF)<<8)|((q2&0xFF)<<16)|((q3&0xFF)<<24);
}
__device__ __forceinline__ void gm_cpa4(void *smem, const void *gmem, bool pred) {
    unsigned s = __cvta_generic_to_shared(smem); int ss = pred ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(ss));
}

// 챕터7 C.2: BM 템플릿화(weight 재read = M/BM회 → BM↑면 weight DRAM 트래픽↓, memory-bound 천장 돌파).
//   8 warp 고정. wm∈{0,1} 2 m-그룹이 각 BM/2행을 (BM/32)개 m16 청크로. ns/wn(BN=64)은 불변.
//   MMA·스케일 산수는 BM 무관 동일 → 정확성 보존(BM=64는 기존과 bitwise 동일).
template<int BM, typename ST>
static __global__ void gemm_mma_w4a8_pipe_kernel(const int8_t *X, const float *xs, const uint8_t *Wq,
                                                 const ST *sw, float *Y, int M, int N, int K) {
    constexpr int MS = BM/32;          // wm 그룹당 m16 청크 수 (BM=64→2, 128→4, 256→8)
    constexpr int MHALF = BM/2;        // wm 그룹당 행 수
    __shared__ int8_t  sX[2][BM][QBLK + MMA_PAD];       // [buf][m][k] X int8
    __shared__ uint8_t sWp[2][MMA_BN][QBLK/2 + W_PAD];  // [buf][n][k/2] W packed int4
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int wm = warp >> 2, wn = warp & 3;
    int m0 = blockIdx.y * BM, n0 = blockIdx.x * MMA_BN, nblk = K / QBLK, K2 = K / 2;
    float acc[MS][2][4];
    #pragma unroll
    for (int a=0;a<MS;a++) for (int b=0;b<2;b++) for (int c=0;c<4;c++) acc[a][b][c]=0.f;

    // 한 K-블록(QBLK)을 buf에 cp.async 적재. X: BM×32B word(4B). W: 64×16B=256 word.
    auto load_blk = [&](int buf, int blk) {
        int k0 = blk * QBLK;
        #pragma unroll
        for (int i = tid; i < BM*QBLK/4; i += 256) {
            int m = i / (QBLK/4), kw = (i % (QBLK/4)) * 4, gm = m0 + m;
            gm_cpa4(&sX[buf][m][kw], &X[(size_t)gm*K + k0 + kw], gm < M);
        }
        for (int i = tid; i < MMA_BN*(QBLK/2)/4; i += 256) {
            int n = i / (QBLK/2/4), kw = (i % (QBLK/2/4)) * 4, gn = n0 + n;
            gm_cpa4(&sWp[buf][n][kw], &Wq[(size_t)gn*K2 + k0/2 + kw], gn < N);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    load_blk(0, 0);
    for (int blk = 0; blk < nblk; blk++) {
        asm volatile("cp.async.wait_group 0;\n");        // 현재(blk) 적재 완료 대기 (이 시점 pending=현재 1개)
        __syncthreads();
        if (blk+1 < nblk) load_blk((blk+1)&1, blk+1);   // 다음 타일 prefetch — 아래 MMA와 겹침(정식 순서)
        int cur = blk & 1;
        // C.2 진단(ncu): L1TEX shared-load scoreboard stall이 limiter(64.6%). A(ms 의존)·B(ns 의존) 프래그먼트를
        //   inner loop서 중복 read(4× where 2×)하던 것을 사전 레지스터 적재 → shared-read/unpack 절반, 선발행으로 latency 은닉.
        int af[MS][4], bf[2][2];
        // A·B 프래그먼트 사전적재(register-blocking, 측정 채택 2.46→2.40ms). ldmatrix-A는 측정 평탄(기각, 메모리 기록)
        //   — L1/TEX 벽은 A-load 명령이 아니라 대역폭(+B unpack bank conflict 11M). de-risk test_ldmatrix는 보존.
        #pragma unroll
        for (int ms=0; ms<MS; ms++) { int mb=wm*MHALF+ms*16;
            af[ms][0]=*(int*)&sX[cur][mb+g][t*4];    af[ms][1]=*(int*)&sX[cur][mb+g+8][t*4];
            af[ms][2]=*(int*)&sX[cur][mb+g][16+t*4]; af[ms][3]=*(int*)&sX[cur][mb+g+8][16+t*4];
        }
        // B 언팩: 2-byte read → 정렬 4-byte read(+shift). 같은 word를 인접 thread가 broadcast → bank conflict↓(11.1M 타깃).
        //   값 동일(t짝수=하위16, t홀수=상위16 = 기존 &[t*2] 2-byte와 같은 nibble). 행 stride(QBLK/2+W_PAD=20)·(t>>1)*4 다 4B정렬.
        #pragma unroll
        for (int ns=0; ns<2; ns++) { int nb=wn*16+ns*8;
            unsigned w0=*(const unsigned*)&sWp[cur][nb+g][(t>>1)*4], w1=*(const unsigned*)&sWp[cur][nb+g][8+(t>>1)*4];
            bf[ns][0]=gm_unpack4v((t&1)?(w0>>16):w0); bf[ns][1]=gm_unpack4v((t&1)?(w1>>16):w1);
        }
        #pragma unroll
        for (int ms=0; ms<MS; ms++) { int mb=wm*MHALF+ms*16;
            #pragma unroll
            for (int ns=0; ns<2; ns++) { int nb=wn*16+ns*8;
                int c0=0,c1=0,c2=0,c3=0;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                  :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(af[ms][0]),"r"(af[ms][1]),"r"(af[ms][2]),"r"(af[ms][3]),"r"(bf[ns][0]),"r"(bf[ns][1]));
                int rm0=m0+mb+g, rm1=m0+mb+g+8, cn0=n0+nb+t*2, cn1=n0+nb+t*2+1;
                float xm0=(rm0<M)?xs[(size_t)rm0*nblk+blk]:0.f, xm1=(rm1<M)?xs[(size_t)rm1*nblk+blk]:0.f;
                float sn0=(cn0<N)?sval(sw[(size_t)cn0*nblk+blk]):0.f, sn1=(cn1<N)?sval(sw[(size_t)cn1*nblk+blk]):0.f;
                acc[ms][ns][0]+=c0*xm0*sn0; acc[ms][ns][1]+=c1*xm0*sn1; acc[ms][ns][2]+=c2*xm1*sn0; acc[ms][ns][3]+=c3*xm1*sn1;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int ms=0; ms<MS; ms++) { int mb=wm*MHALF+ms*16;
        #pragma unroll
        for (int ns=0; ns<2; ns++) { int nb=wn*16+ns*8;
            int rm0=m0+mb+g, rm1=m0+mb+g+8, cn0=n0+nb+t*2, cn1=n0+nb+t*2+1;
            if (rm0<M){ if(cn0<N)Y[(size_t)rm0*N+cn0]=acc[ms][ns][0]; if(cn1<N)Y[(size_t)rm0*N+cn1]=acc[ms][ns][1]; }
            if (rm1<M){ if(cn0<N)Y[(size_t)rm1*N+cn0]=acc[ms][ns][2]; if(cn1<N)Y[(size_t)rm1*N+cn1]=acc[ms][ns][3]; }
        }
    }
}

// BM 스윕용 진입점(ncu/test). BM은 32의 배수(BM/2가 16배수 정렬).
template<int BM>
static inline void launch_gemm_mma_w4a8_bm(const int8_t *X, const float *xs, const uint8_t *Wq,
                                           const void *sw, float *Y, int M, int N, int K, int scale_dtype = 0, cudaStream_t st = 0) {
    dim3 grid((N + MMA_BN - 1) / MMA_BN, (M + BM - 1) / BM);
    if (scale_dtype == 1) gemm_mma_w4a8_pipe_kernel<BM,__half><<<grid, 256, 0, st>>>(X, xs, Wq, (const __half*)sw, Y, M, N, K);
    else                  gemm_mma_w4a8_pipe_kernel<BM,float ><<<grid, 256, 0, st>>>(X, xs, Wq, (const float*)sw, Y, M, N, K);
}

// 엔진 경로: BM=64 고정(기존과 bitwise 동일, de-risk). 측정 우월 확인+합의 후에만 BM 교체.
static inline void launch_gemm_mma_w4a8(const int8_t *X, const float *xs, const uint8_t *Wq,
                                        const void *sw, float *Y, int M, int N, int K, int scale_dtype = 0, cudaStream_t st = 0) {
    launch_gemm_mma_w4a8_bm<MMA_BM>(X, xs, Wq, sw, Y, M, N, K, scale_dtype, st);
}
