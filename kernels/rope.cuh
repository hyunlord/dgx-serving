#ifndef DGX_ROPE_CUH
#define DGX_ROPE_CUH

// RoPE (rotate-half, Llama/HF 방식). 모델 독립.
//   T: [S, NH, HD] in-place. cos/sin: [S, HD] (앞 HD/2 = 뒤 HD/2, HF cat(freqs,freqs)).
//   pair (i, i+half): t1'=t1*c - t2*sn ; t2'=t2*c + t1*sn, c=cos[s*HD+i], sn=sin[s*HD+i], i<half.
__global__ void rope_kernel(float *T, const float *cos, const float *sin,
                            int S, int NH, int HD) {
    int half = HD >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * NH * half;
    if (idx >= total) return;
    int i = idx % half;
    int h = (idx / half) % NH;
    int s = idx / (half * NH);
    float *base = T + ((size_t)s * NH + h) * HD;
    float t1 = base[i], t2 = base[i + half];
    float c = cos[(size_t)s * HD + i], sn = sin[(size_t)s * HD + i];
    base[i]        = t1 * c - t2 * sn;
    base[i + half] = t2 * c + t1 * sn;
}

static inline void launch_rope(float *T, const float *cos, const float *sin,
                               int S, int NH, int HD, cudaStream_t st = 0) {
    int block = 256;
    int total = S * NH * (HD >> 1);
    int grid = (total + block - 1) / block;
    rope_kernel<<<grid, block, 0, st>>>(T, cos, sin, S, NH, HD);
}

#endif // DGX_ROPE_CUH
