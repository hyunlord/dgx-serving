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

// Partial RoPE (Qwen3.5): rotate_half 를 앞 rot_dim 차원에만 적용, [rot_dim, HD) 는 통과.
//   T: [S, NH, HD] in-place. cos/sin: [S, rot_dim] (HF apply_rotary_pos_emb, rotary_dim=cos.shape[-1]).
//   pair (i, i+rot_dim/2), i<rot_dim/2: t1'=t1*c - t2*sn ; t2'=t2*c + t1*sn, c=cos[s*rot_dim+i].
//   rot_dim==HD 이면 1단계 full rope 와 동일.
__global__ void rope_partial_kernel(float *T, const float *cos, const float *sin,
                                    int S, int NH, int HD, int rot_dim) {
    int half = rot_dim >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * NH * half;
    if (idx >= total) return;
    int i = idx % half;
    int h = (idx / half) % NH;
    int s = idx / (half * NH);
    float *base = T + ((size_t)s * NH + h) * HD;
    float t1 = base[i], t2 = base[i + half];
    float c = cos[(size_t)s * rot_dim + i], sn = sin[(size_t)s * rot_dim + i];
    base[i]        = t1 * c - t2 * sn;
    base[i + half] = t2 * c + t1 * sn;
}

static inline void launch_rope_partial(float *T, const float *cos, const float *sin,
                                       int S, int NH, int HD, int rot_dim, cudaStream_t st = 0) {
    int block = 256;
    int total = S * NH * (rot_dim >> 1);
    int grid = (total + block - 1) / block;
    rope_partial_kernel<<<grid, block, 0, st>>>(T, cos, sin, S, NH, HD, rot_dim);
}

#endif // DGX_ROPE_CUH
