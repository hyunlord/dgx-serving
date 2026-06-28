#pragma once
#include <cstdint>
#include "../runtime/config.h"   // QBLK
#include "gemv_w4a8.cuh"         // S=1 fallback

// 챕터7 Phase B: tensor-core int8 MMA W4A8 GEMM. Y[M,N] = (X int8 · (W int4-8) int8) ×(xs·sw).
//   X int8[M,K] + xs[M,K/32], Wq int4 packed[N,K/2] + sw[N,K/32]. MMA m16n8k32.s8.s8.s32.
//   B.0/B.1 검증된 fragment 레이아웃 + 블록 스케일. K-tile=QBLK=32=MMA k 정렬.

// CTA 타일 BM×BN, 8 warp(2×4) 재사용. pad는 bank conflict 회피.
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
__device__ __forceinline__ void gm_cpa4(void *smem, const void *gmem, bool pred) {
    unsigned s = __cvta_generic_to_shared(smem); int ss = pred ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s), "l"(gmem), "r"(ss));
}

static __global__ void gemm_mma_w4a8_pipe_kernel(const int8_t *X, const float *xs, const uint8_t *Wq,
                                                 const float *sw, float *Y, int M, int N, int K) {
    __shared__ int8_t  sX[2][MMA_BM][QBLK + MMA_PAD];   // [buf][m][k] X int8
    __shared__ uint8_t sWp[2][MMA_BN][QBLK/2 + W_PAD];  // [buf][n][k/2] W packed int4
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int wm = warp >> 2, wn = warp & 3;
    int m0 = blockIdx.y * MMA_BM, n0 = blockIdx.x * MMA_BN, nblk = K / QBLK, K2 = K / 2;
    float acc[2][2][4];
    #pragma unroll
    for (int a=0;a<2;a++) for (int b=0;b<2;b++) for (int c=0;c<4;c++) acc[a][b][c]=0.f;

    // 한 K-블록(QBLK)을 buf에 cp.async 적재. X: 64×32B=512 word(4B), 256스레드 2개씩. W: 64×16B=256 word, 1개씩.
    auto load_blk = [&](int buf, int blk) {
        int k0 = blk * QBLK;
        #pragma unroll
        for (int i = tid; i < MMA_BM*QBLK/4; i += 256) {
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
        #pragma unroll
        for (int ms=0; ms<2; ms++) { int mb=wm*32+ms*16;
            #pragma unroll
            for (int ns=0; ns<2; ns++) { int nb=wn*16+ns*8;
                int a0=*(int*)&sX[cur][mb+g][t*4], a1=*(int*)&sX[cur][mb+g+8][t*4];
                int a2=*(int*)&sX[cur][mb+g][16+t*4], a3=*(int*)&sX[cur][mb+g+8][16+t*4];
                int b0=gm_unpack4(&sWp[cur][nb+g][t*2]), b1=gm_unpack4(&sWp[cur][nb+g][8+t*2]);
                int c0=0,c1=0,c2=0,c3=0;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                  :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
                int rm0=m0+mb+g, rm1=m0+mb+g+8, cn0=n0+nb+t*2, cn1=n0+nb+t*2+1;
                float xm0=(rm0<M)?xs[(size_t)rm0*nblk+blk]:0.f, xm1=(rm1<M)?xs[(size_t)rm1*nblk+blk]:0.f;
                float sn0=(cn0<N)?sw[(size_t)cn0*nblk+blk]:0.f, sn1=(cn1<N)?sw[(size_t)cn1*nblk+blk]:0.f;
                acc[ms][ns][0]+=c0*xm0*sn0; acc[ms][ns][1]+=c1*xm0*sn1; acc[ms][ns][2]+=c2*xm1*sn0; acc[ms][ns][3]+=c3*xm1*sn1;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int ms=0; ms<2; ms++) { int mb=wm*32+ms*16;
        #pragma unroll
        for (int ns=0; ns<2; ns++) { int nb=wn*16+ns*8;
            int rm0=m0+mb+g, rm1=m0+mb+g+8, cn0=n0+nb+t*2, cn1=n0+nb+t*2+1;
            if (rm0<M){ if(cn0<N)Y[(size_t)rm0*N+cn0]=acc[ms][ns][0]; if(cn1<N)Y[(size_t)rm0*N+cn1]=acc[ms][ns][1]; }
            if (rm1<M){ if(cn0<N)Y[(size_t)rm1*N+cn0]=acc[ms][ns][2]; if(cn1<N)Y[(size_t)rm1*N+cn1]=acc[ms][ns][3]; }
        }
    }
}

static inline void launch_gemm_mma_w4a8(const int8_t *X, const float *xs, const uint8_t *Wq,
                                        const float *sw, float *Y, int M, int N, int K, cudaStream_t st = 0) {
    dim3 grid((N + MMA_BN - 1) / MMA_BN, (M + MMA_BM - 1) / MMA_BM);
    gemm_mma_w4a8_pipe_kernel<<<grid, 256, 0, st>>>(X, xs, Wq, sw, Y, M, N, K);    // Phase C cp.async 2-stage
}
