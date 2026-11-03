// Reuse the exact R20 pure-read kernel; select a single known-byte launch.
#define main r20_unused_main
#include "r20_bw_probe.cu"
#undef main
int main() {
    constexpr size_t bytes = size_t(1) << 30;
    uint4* input = nullptr;
    unsigned* sink = nullptr;
    CK(cudaMalloc(&input, bytes));
    CK(cudaMalloc(&sink, sizeof(unsigned)));
    CK(cudaMemset(input, 0x5a, bytes));
    CK(cudaMemset(sink, 0, sizeof(unsigned)));
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    flat16<<<sms * 8, 256>>>(input, bytes / sizeof(uint4), sink);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    printf("CALIBRATION read_bytes=%zu launches=1 kernel=flat16 blocks=%d threads=256\n", bytes, sms * 8);
    CK(cudaFree(input));
    CK(cudaFree(sink));
    return 0;
}
