#include <cstdio>
#include <cuda_runtime.h>

int main() {
    size_t free_bytes = 0, total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaMemGetInfo: %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("{\"free_bytes\":%zu,\"total_bytes\":%zu}\n", free_bytes, total_bytes);
    return 0;
}
