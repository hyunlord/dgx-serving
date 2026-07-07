#ifndef DGX_ROPE_GEMMA4_H
#define DGX_ROPE_GEMMA4_H
// Gemma4 daemon용 rope 테이블 host 생성(절대 규약 — HF config·골든 역산으로 확정, 2026-07 재진단):
//   full층  = proportional rope: theta=rope_theta_full(1e6), ★partial_rotary_factor=0.25
//             → 회전쌍 = 0.25×hd/2 = 64쌍만 회전(inv_i = theta^(-2i/hd)), 나머지 쌍은 항등(cos=1·sin=0).
//   sliding = 표준 rope: theta=rope_theta_sliding(1e4), 전체 hd/2쌍 회전.
//   레이아웃 [pos, hd]: 앞 hd/2 = freqs, 뒤 hd/2 = 동일 반복(HF cat(freqs,freqs) — kernels/rope.cuh 규약).
// ★버그 전례 2건(이 파일이 그 대체): daemon launch_gen_rope 하드코딩이 ①full theta 1e4(b1300f5에서 1e6로)
//   ②partial 0.25 누락(전체 회전) — ②가 긴 생성(수백 tok+) 문자-붕괴의 근본 원인(dim 64+ 오회전 누적).
//   spec==plain 상대 게이트는 공유 오류를 못 잡음 — 절대 게이트 = tests/test_gemma4_rope_table.py
//   (이 수식의 python 미러 vs HF 캡처 골든, max|Δ| < 4e-3 = bf16 캡처 반올림 이내. 12B·31B 대조).
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include "manifest.h"
#include "manifest_gemma4.h"   // mf_num_opt

// manifest 권위로 full/sliding 테이블 생성 → 기할당 device 버퍼(d_* = [max_len, hd])에 업로드.
//   manifest_path: <weights>/manifest.json. 반환 0=성공(-1=manifest 없음 → 호출자 abort 권장).
inline int gemma4_gen_rope_tables(const char* manifest_path, int max_len,
                                  int hd_f, float* d_cos_f, float* d_sin_f,
                                  int hd_s, float* d_cos_s, float* d_sin_s){
    char* mf = mf_read_file(manifest_path);
    if(!mf) return -1;
    const float th_f = mf_num_opt(mf, "rope_theta_full", 1e6);
    const float th_s = mf_num_opt(mf, "rope_theta_sliding", 1e4);
    // 회전 원소 수 = manifest rotary_dim_*(extract가 HF partial_rotary_factor×hd로 기록 — 12B/31B/26B
    //   full=128[=64쌍]·sliding=hd 전체). 회전쌍 = rotary_dim/2. 필드 없으면(구 manifest) hd 전체 폴백+경고.
    int rd_f = (int)mf_num_opt(mf, "rotary_dim_full", 0);
    int rd_s = (int)mf_num_opt(mf, "rotary_dim_sliding", 0);
    free(mf);
    if(rd_f<=0){ rd_f=hd_f; fprintf(stderr,"[rope] ★경고: rotary_dim_full 없음 — 전체 회전 폴백(proportional 모델이면 오답: 재변환 필요)\n"); }
    if(rd_s<=0) rd_s=hd_s;
    fprintf(stderr, "[rope] manifest 권위: full theta=%g 회전 %d/%d쌍 · sliding theta=%g 회전 %d/%d쌍\n",
            th_f, rd_f/2, hd_f/2, th_s, rd_s/2, hd_s/2);

    auto fill=[&](int hd, float theta, int rot, float* d_cos, float* d_sin){
        const int half = hd/2;
        std::vector<float> hc((size_t)max_len*hd, 1.f), hs((size_t)max_len*hd, 0.f);
        for(int p=0;p<max_len;p++){
            for(int i=0;i<rot;i++){
                const double ang = (double)p * std::pow((double)theta, -2.0*i/hd);
                const float c=(float)std::cos(ang), s=(float)std::sin(ang);
                hc[(size_t)p*hd+i]=c;      hs[(size_t)p*hd+i]=s;
                hc[(size_t)p*hd+half+i]=c; hs[(size_t)p*hd+half+i]=s;   // cat(freqs,freqs)
            }
        }
        cudaMemcpy(d_cos, hc.data(), hc.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin, hs.data(), hs.size()*4, cudaMemcpyHostToDevice);
    };
    fill(hd_f, th_f, rd_f/2, d_cos_f, d_sin_f);
    fill(hd_s, th_s, rd_s/2, d_cos_s, d_sin_s);
    return 0;
}

#endif // DGX_ROPE_GEMMA4_H
