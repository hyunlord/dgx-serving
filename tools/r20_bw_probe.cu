// R20 A-1: maximum read bandwidth on GB10 with GEMV-like access patterns (pure reads, sums kept live).
// Buffer: 1 GiB (>> L2), read as rows of RB bytes (RB = 1920 = one 3840-element int4 row, or 7680 = K 15360).
// Variants:
//   flat16_b<N>   grid-stride uint4 (16 B) loads, N blocks x 256 threads
//   row4          one warp per row, lane reads 4 B at lane*4 + 128 i (the current GEMV pattern), 8 rows per block
//   row4_pf2      row4 + prefetch.global.L2 two iterations ahead
//   row16_w8      eight warps per row, 16 B per lane (llama MMVQ S=1 pattern), one row per block
//   row16_w1      one warp per row, 16 B per lane
// Each variant: 10 s warm-up (first variant only), then one >= 1 s interval of back-to-back launches.
//   R20_BW variant= rb= GBps= launches= t0_ms= t1_ms=
// Usage: r20_bw_probe [row bytes list, default 1920,7680]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count(); }

__global__ void flat16(const uint4 *p, size_t n, unsigned *sink) {
    unsigned a = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = __ldg(p + i); a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) *sink = a;
}
template <int PF>
__global__ void row4(const uint8_t *p, int rows, int rb, unsigned *sink) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const uint8_t *r = p + (size_t)warp * rb; unsigned a = 0;
    for (int o = lane * 4; o < rb; o += 128) {
        if (PF) { const int q = o + PF * 128; if (q < rb) asm volatile("prefetch.global.L2 [%0];" :: "l"(r + q)); }
        a ^= __ldg(reinterpret_cast<const unsigned *>(r + o));
    }
    if (a == 0x12345678u) *sink = a;
}
template <int NW>
__global__ void row16(const uint8_t *p, int rows, int rb, unsigned *sink) {
    const int tpr = NW * 32;                                   // threads per row
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / tpr, t = (blockIdx.x * blockDim.x + threadIdx.x) % tpr;
    if (row >= rows) return;
    const uint8_t *r = p + (size_t)row * rb; unsigned a = 0;
    for (int o = t * 16; o < rb; o += tpr * 16) { uint4 v = __ldg(reinterpret_cast<const uint4 *>(r + o)); a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) *sink = a;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    std::vector<int> rbs = {1920, 7680};
    if (argc > 1) { rbs.clear(); std::string s(argv[1]); size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) rbs.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
    const size_t bytes = 1ull << 30;
    uint8_t *buf; unsigned *sink; CK(cudaMalloc(&buf, bytes)); CK(cudaMalloc(&sink, 4)); CK(cudaMemset(buf, 0x5a, bytes));
    int sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    auto run = [&](const char *name, int rb, auto launch, double target_ms) {
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        long n = 4; float ms = 0; double t0 = 0, t1 = 0;
        for (;;) {
            CK(cudaDeviceSynchronize()); t0 = now_ms(); CK(cudaEventRecord(e0));
            for (long i = 0; i < n; i++) launch();
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); t1 = now_ms(); CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms >= target_ms) break;
            n = (long)(n * target_ms * 1.1 / (ms > 0.01 ? ms : 0.01)) + 1;
        }
        CK(cudaGetLastError());
        if (name) printf("R20_BW variant=%s rb=%d GBps=%.2f launches=%ld t0_ms=%.1f t1_ms=%.1f\n", name, rb, (double)bytes * n / (ms / 1e3) / 1e9, n, t0, t1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    };
    const size_t n16 = bytes / 16;
    { const double end = now_ms() + 10000; while (now_ms() < end) run(nullptr, 0, [&] { flat16<<<sms * 8, 256>>>((const uint4 *)buf, n16, sink); }, 500); }
    for (int k : {1, 2, 4, 8, 16, 32}) {
        std::string nm = "flat16_b" + std::to_string(sms * k);
        run(nm.c_str(), 0, [&] { flat16<<<sms * k, 256>>>((const uint4 *)buf, n16, sink); }, 1000);
    }
    run("flat16_full", 0, [&] { flat16<<<(unsigned)((n16 + 255) / 256), 256>>>((const uint4 *)buf, n16, sink); }, 1000);
    for (int rb : rbs) {
        const int rows = (int)(bytes / rb);
        run("row4", rb, [&] { row4<0><<<(rows + 7) / 8, 256>>>(buf, rows, rb, sink); }, 1000);
        run("row4_pf2", rb, [&] { row4<2><<<(rows + 7) / 8, 256>>>(buf, rows, rb, sink); }, 1000);
        run("row16_w8", rb, [&] { row16<8><<<rows, 256>>>(buf, rows, rb, sink); }, 1000);
        run("row16_w4", rb, [&] { row16<4><<<(rows + 1) / 2, 256>>>(buf, rows, rb, sink); }, 1000);
        run("row16_w1", rb, [&] { row16<1><<<(rows + 7) / 8, 256>>>(buf, rows, rb, sink); }, 1000);
    }
    return 0;
}
