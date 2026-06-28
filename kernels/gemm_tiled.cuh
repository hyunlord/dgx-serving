#pragma once
#include <cstdint>
#include "../runtime/config.h"   // QBLK
#include "gemv_q4.cuh"           // S=1 fallback (launch_gemv_q4)

// 챕터7 Phase A: software tiled q4 GEMM (tensor-core 없이). Y[M,N] = X[M,K] · dequant(W[N,K])ᵀ.
//   X[M,K] fp32(M=Sq), W[N,K] q4 packed(row-major, 2 int4/byte), scale[N, K/QBLK].
//   ★ K-slice 타일(Ktile=QBLK=32): X·W 둘 다 shared K-major([K][M]/[K][N]) — full-row staging 금지(데인 지점).
//   ★ bank conflict 0: K-major shared + strided ownership(row=ty+im*16, col=tx+in*16) →
//     warp(tx 0..15)이 연속 col 읽음 = stride-1, 충돌 없음. sX 읽기는 ty만 달라 broadcast.
//   한 CTA = 출력타일 [TM×TN], 256스레드(16×16) 각 4×4 마이크로타일(strided, 레지스터 누적).

#define GT_TM 64
#define GT_TN 64
#define GT_TK 32        // = QBLK (스케일 슬라이스당 1값)
#define GT_PAD 4

static __global__ void gemm_tiled_q4_kernel(const float *X, const uint8_t *Wq, const float *sc,
                                            float *Y, int M, int N, int K) {
    __shared__ float sX[GT_TK][GT_TM + GT_PAD];   // K-major: sX[k][m]
    __shared__ float sW[GT_TK][GT_TN + GT_PAD];   // K-major: sW[k][n] (dequant)
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;       // 16×16
    const int m0 = blockIdx.y * GT_TM, n0 = blockIdx.x * GT_TN;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    const int KB = K / QBLK;

    for (int k0 = 0; k0 < K; k0 += GT_TK) {
        // X 적재 → sX[c][r] (K-major). 글로벌 read coalesce(연속 c=k fastest) 우선 — 측정상 이게 더 빠름
        //   (글로벌 coalescing > shared-write 충돌; m-fastest는 util 79→38% 역효과).
        #pragma unroll
        for (int i = tid; i < GT_TM * GT_TK; i += 256) {
            int r = i / GT_TK, c = i % GT_TK, gm = m0 + r;
            sX[c][r] = (gm < M) ? X[(size_t)gm * K + k0 + c] : 0.f;
        }
        // W 적재+dequant → sW[c][r] (K-major).
        const int slice = k0 / QBLK;
        #pragma unroll
        for (int i = tid; i < GT_TN * GT_TK; i += 256) {
            int r = i / GT_TK, c = i % GT_TK, gn = n0 + r;
            if (gn < N) {
                const uint8_t *wrow = Wq + (size_t)gn * (K / 2);
                int gk = k0 + c;
                uint8_t by = wrow[gk >> 1];
                int q = (gk & 1) ? (by >> 4) : (by & 0x0F);
                sW[c][r] = ((float)q - 8.0f) * sc[(size_t)gn * KB + slice];
            } else sW[c][r] = 0.f;
        }
        __syncthreads();
        // 누적: strided ownership — row=ty+im*16, col=tx+in*16 (충돌 없는 shared 읽기)
        #pragma unroll
        for (int kk = 0; kk < GT_TK; kk++) {
            float xr[4], wr[4];
            #pragma unroll
            for (int im = 0; im < 4; im++) xr[im] = sX[kk][ty + im * 16];   // ty만 달라 broadcast
            #pragma unroll
            for (int in = 0; in < 4; in++) wr[in] = sW[kk][tx + in * 16];   // tx 연속 → stride-1
            #pragma unroll
            for (int im = 0; im < 4; im++)
                #pragma unroll
                for (int in = 0; in < 4; in++) acc[im][in] += xr[im] * wr[in];
        }
        __syncthreads();
    }
    // Y[m0+row, n0+col] 기록 (strided)
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int gm = m0 + ty + im * 16; if (gm >= M) continue;
        #pragma unroll
        for (int in = 0; in < 4; in++) {
            int gn = n0 + tx + in * 16; if (gn < N) Y[(size_t)gm * N + gn] = acc[im][in];
        }
    }
}

static inline void launch_gemm_tiled_q4(const float *X, const uint8_t *Wq, const float *sc,
                                        float *Y, int M, int N, int K, cudaStream_t st = 0) {
    if (M == 1) { launch_gemv_q4(X, Wq, sc, Y, K, N, 1, st); return; }
    dim3 block(256);
    dim3 grid((N + GT_TN - 1) / GT_TN, (M + GT_TM - 1) / GT_TM);
    gemm_tiled_q4_kernel<<<grid, block, 0, st>>>(X, Wq, sc, Y, M, N, K);
}
