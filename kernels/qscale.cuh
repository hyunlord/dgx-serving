#pragma once
#include <cuda_fp16.h>

// weight-scale 타입(float 또는 __half)을 float로 읽기. 포인터 산술은 ST로 자연스럽게.
__device__ __forceinline__ float sval(float  x) { return x; }
__device__ __forceinline__ float sval(__half x) { return __half2float(x); }
