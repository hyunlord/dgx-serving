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

static inline void launch_gemm_mma_w4a8(const int8_t *X, const float *xs, const uint8_t *Wq,
                                        const float *sw, float *Y, int M, int N, int K, cudaStream_t st = 0) {
    dim3 grid((N + 7) / 8, (M + 15) / 16);
    gemm_mma_w4a8_kernel<<<grid, 32, 0, st>>>(X, xs, Wq, sw, Y, M, N, K);
}
