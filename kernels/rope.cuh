#ifndef DGX_ROPE_CUH
#define DGX_ROPE_CUH

// rope cos/sin 테이블 엔진 내부 생성 (config theta/rotary_dim만 — 골든 rope 의존 제거, 임의 길이/32K).
//   HF default rope: inv_freq[j]=theta^(-2j/rot) (j<rot/2), emb=cat(freqs,freqs) → cos/sin[p,rot] 앞=뒤 절반.
//   모델 독립. 검증: 생성 vs 골든 rope_cos(짧은 S) 오차0.
static __global__ void gen_rope_kernel(float *cosT, float *sinT, int rot, float theta) {
    int p = blockIdx.x;                 // position
    int j = threadIdx.x;                // freq index < rot/2
    int half = rot >> 1;
    if (j >= half) return;
    float inv = powf(theta, -2.0f * j / (float)rot);   // theta^(-2j/rot)
    float a = (float)p * inv;
    float c = cosf(a), s = sinf(a);
    size_t base = (size_t)p * rot;
    cosT[base + j] = c; cosT[base + half + j] = c;     // 앞=뒤 (HF cat(freqs,freqs))
    sinT[base + j] = s; sinT[base + half + j] = s;
}
static inline void launch_gen_rope(float *cosT, float *sinT, int max_len, int rot, float theta, cudaStream_t st = 0) {
    gen_rope_kernel<<<max_len, (rot + 1) >> 1, 0, st>>>(cosT, sinT, rot, theta);   // rot/2 <= 1024 가정
}

// RoPE (rotate-half, Llama/HF 방식). 모델 독립.
//   T: [S, NH, HD] in-place. cos/sin: [S, HD] (앞 HD/2 = 뒤 HD/2, HF cat(freqs,freqs)).
//   pair (i, i+half): t1'=t1*c - t2*sn ; t2'=t2*c + t1*sn, c=cos[s*HD+i], sn=sin[s*HD+i], i<half.
static __global__ void rope_kernel(float *T, const float *cos, const float *sin,
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
static __global__ void rope_partial_kernel(float *T, const float *cos, const float *sin,
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

// batch decode rope (챕터3): B 독립 시퀀스 모두 같은 position(same-length) → cos/sin은 그 position 행 1개 공유.
//   cos/sin 포인터 = d_cos + base*rot (호출측 offset). s(=행=시퀀스)는 cos 인덱싱에 안 들어감.
static __global__ void rope_partial_batch_kernel(float *T, const float *cos, const float *sin,
                                    int B, int NH, int HD, int rot_dim) {
    int half = rot_dim >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NH * half;
    if (idx >= total) return;
    int i = idx % half;
    int h = (idx / half) % NH;
    int s = idx / (half * NH);
    float *base = T + ((size_t)s * NH + h) * HD;
    float t1 = base[i], t2 = base[i + half];
    float c = cos[i], sn = sin[i];                  // 모든 시퀀스 같은 position
    base[i]        = t1 * c - t2 * sn;
    base[i + half] = t2 * c + t1 * sn;
}
static inline void launch_rope_partial_batch(float *T, const float *cos, const float *sin,
                                             int B, int NH, int HD, int rot_dim, cudaStream_t st = 0) {
    int total = B * NH * (rot_dim >> 1); int block = 256; int grid = (total + block - 1) / block;
    rope_partial_batch_kernel<<<grid, block, 0, st>>>(T, cos, sin, B, NH, HD, rot_dim);
}

#endif // DGX_ROPE_CUH
