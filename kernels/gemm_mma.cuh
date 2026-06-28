#pragma once
#include <cstdint>
#include "../runtime/config.h"   // QBLK
#include "gemv_w4a8.cuh"         // S=1 fallback

// 챕터7 Phase B: tensor-core int8 MMA W4A8 GEMM. Y[M,N] = (X int8 · (W int4-8) int8) ×(xs·sw).
//   X int8[M,K] + xs[M,K/32], Wq int4 packed[N,K/2] + sw[N,K/32]. MMA m16n8k32.s8.s8.s32.
//   B.0/B.1 검증된 fragment 레이아웃 + 블록 스케일. K-tile=QBLK=32=MMA k 정렬.
//   B.2a: warp-per-tile(글로벌 로드) — 한 warp이 16×8 출력 타일, K-블록 순회. (shared 타일링은 B.2b.)

__device__ __forceinline__ int gm_pk4(const int8_t *b){
    return (b[0]&0xFF)|((b[1]&0xFF)<<8)|((b[2]&0xFF)<<16)|((b[3]&0xFF)<<24);
}

// 1 warp = 출력 [16×8] 타일 (m0..m0+15, n0..n0+7). M/N bounds 마스크.
static __global__ void gemm_mma_w4a8_kernel(const int8_t *X, const float *xs, const uint8_t *Wq,
                                            const float *sw, float *Y, int M, int N, int K) {
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int m0 = blockIdx.y * 16, n0 = blockIdx.x * 8;
    int nblk = K / QBLK;
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    // C 매핑: c0(row g, col t*2) c1(g, t*2+1) c2(g+8, t*2) c3(g+8, t*2+1)
    int mr0 = m0 + g, mr1 = m0 + g + 8;          // A row → M
    int nc0 = n0 + t * 2, nc1 = n0 + t * 2 + 1;   // C col → N
    for (int blk = 0; blk < nblk; blk++) {
        int k0 = blk * QBLK; int8_t buf[4]; int a0,a1,a2,a3,b0,b1,c0=0,c1=0,c2=0,c3=0;
        // A = X int8 (mr0/mr1 행, K t*4.. / 16+t*4..). bounds: 행 OOB → 0
        #define LDX(dst, mr, koff) { if((mr)<M){ for(int j=0;j<4;j++) buf[j]=X[(size_t)(mr)*K + k0+(koff)+t*4+j]; } else { for(int j=0;j<4;j++) buf[j]=0; } dst=gm_pk4(buf); }
        LDX(a0, mr0, 0); LDX(a1, mr1, 0); LDX(a2, mr0, 16); LDX(a3, mr1, 16);
        // B = W int4→(wq-8). col(N)=n0+g, K t*4.. . bounds: n OOB → 0
        int wn = n0 + g;
        #define LDW(dst, koff) { if(wn<N){ for(int j=0;j<4;j++){ int kk=k0+(koff)+t*4+j; uint8_t by=Wq[(size_t)wn*(K/2)+(kk>>1)]; int q=(kk&1)?(by>>4):(by&0xF); buf[j]=(int8_t)(q-8);} } else { for(int j=0;j<4;j++)buf[j]=0; } dst=gm_pk4(buf); }
        LDW(b0, 0); LDW(b1, 16);
        #undef LDX
        #undef LDW
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        // 블록 스케일: xs[m,blk]·sw[n,blk]. (bounds OOB 행/열은 acc에 안 씀)
        float xm0 = (mr0<M)?xs[(size_t)mr0*nblk+blk]:0.f, xm1 = (mr1<M)?xs[(size_t)mr1*nblk+blk]:0.f;
        float wn0 = (nc0<N)?sw[(size_t)nc0*nblk+blk]:0.f, wn1 = (nc1<N)?sw[(size_t)nc1*nblk+blk]:0.f;
        acc0 += c0 * xm0 * wn0; acc1 += c1 * xm0 * wn1;
        acc2 += c2 * xm1 * wn0; acc3 += c3 * xm1 * wn1;
    }
    if (mr0 < M) { if (nc0 < N) Y[(size_t)mr0*N+nc0]=acc0; if (nc1 < N) Y[(size_t)mr0*N+nc1]=acc1; }
    if (mr1 < M) { if (nc0 < N) Y[(size_t)mr1*N+nc0]=acc2; if (nc1 < N) Y[(size_t)mr1*N+nc1]=acc3; }
}

// B.2b: shared K-tiling — CTA[BM×BN] 타일, X·W를 shared(M/N-major)에 1회 적재 → 8 warp 재사용.
//   sX[m][k]/sW[n][k] M/N-major → MMA fragment = 연속 4 int8 = 1 int read(packing 불필요).
#define MMA_BM 64
#define MMA_BN 64
#define MMA_PAD 4
static __global__ void gemm_mma_w4a8_tiled_kernel(const int8_t *X, const float *xs, const uint8_t *Wq,
                                                  const float *sw, float *Y, int M, int N, int K) {
    __shared__ int8_t sX[MMA_BM][QBLK + MMA_PAD];   // [m][k]
    __shared__ int8_t sW[MMA_BN][QBLK + MMA_PAD];   // [n][k] (unpacked (wq-8))
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int wm = warp >> 2, wn = warp & 3;              // 2×4 warp 그리드
    int m0 = blockIdx.y * MMA_BM, n0 = blockIdx.x * MMA_BN, nblk = K / QBLK;
    float acc[2][2][4];
    #pragma unroll
    for (int a=0;a<2;a++) for (int b=0;b<2;b++) for (int c=0;c<4;c++) acc[a][b][c]=0.f;

    for (int blk = 0; blk < nblk; blk++) {
        int k0 = blk * QBLK;
        for (int i = tid; i < MMA_BM*QBLK; i += 256) { int m=i/QBLK,k=i%QBLK,gm=m0+m; sX[m][k]=(gm<M)?X[(size_t)gm*K+k0+k]:0; }
        for (int i = tid; i < MMA_BN*QBLK; i += 256) { int n=i/QBLK,k=i%QBLK,gn=n0+n;
            if (gn<N){ int gk=k0+k; uint8_t by=Wq[(size_t)gn*(K/2)+(gk>>1)]; int q=(gk&1)?(by>>4):(by&0xF); sW[n][k]=(int8_t)(q-8); } else sW[n][k]=0; }
        __syncthreads();
        #pragma unroll
        for (int ms=0; ms<2; ms++) { int mb=wm*32+ms*16;
            #pragma unroll
            for (int ns=0; ns<2; ns++) { int nb=wn*16+ns*8;
                int a0=*(int*)&sX[mb+g][t*4], a1=*(int*)&sX[mb+g+8][t*4], a2=*(int*)&sX[mb+g][16+t*4], a3=*(int*)&sX[mb+g+8][16+t*4];
                int b0=*(int*)&sW[nb+g][t*4], b1=*(int*)&sW[nb+g][16+t*4];
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

static inline void launch_gemm_mma_w4a8_tiled(const int8_t *X, const float *xs, const uint8_t *Wq,
                                              const float *sw, float *Y, int M, int N, int K, cudaStream_t st = 0) {
    dim3 grid((N + MMA_BN - 1) / MMA_BN, (M + MMA_BM - 1) / MMA_BM);
    gemm_mma_w4a8_tiled_kernel<<<grid, 256, 0, st>>>(X, xs, Wq, sw, Y, M, N, K);   // B.2b shared-tiled (Phase B 기준)
}

// ── Phase C: cp.async 2-stage 더블버퍼 ─────────────────────────────────────
//   ncu 진단: B.2b는 load→sync→MMA 직렬화로 warp stall 59.9%=shared 적재 대기(scoreboard).
//   처방: 다음 K-타일을 cp.async로 prefetch하며 현재 타일 MMA → mem 42%↔compute 35% 겹침.
//   X int8 = 순수 cp.async. W는 packed int4 그대로 cp.async(sWp) → fragment build 때 unpack((q-8)).
#define W_PAD 4
// 2 byte(4 nibble: k,k+1,k+2,k+3) → (q-8) int8 4개를 int 레지스터로 (little-endian, gm_pk4 호환).
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
        if (blk+1 < nblk) load_blk((blk+1)&1, blk+1);   // 다음 타일 prefetch (겹침)
        asm volatile("cp.async.wait_group 1;\n");        // ≤1 pending → 현재(blk) 적재 완료
        __syncthreads();
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
