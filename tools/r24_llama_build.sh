#!/usr/bin/env bash
# Compile only the R24 harness; the original llama build is read-only input.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
llama=${1:?llama source directory}
out=${2:?output executable under artifacts/r24}
repo=$(cd "$(dirname "$0")/.." && pwd)
[ "$(git -C "$llama" rev-parse HEAD)" = df9beac7153166550571c70cc667f270b745da49 ]
mkdir -p "$(dirname "$out")"
nvcc -O3 -std=c++17 -arch=sm_121a -use_fast_math -extended-lambda \
  -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 \
  -I"$llama/ggml/include" -I"$llama/ggml/src" -I"$llama/ggml/src/ggml-cuda" \
  "$repo/tools/r24_llama_isolated.cu" -L"$llama/build/bin" \
  -Xlinker -rpath -Xlinker "$llama/build/bin" \
  -lggml-cuda -lggml-base -lggml -lcublas -lcublasLt -o "$out"
sha256sum "$out" "$llama/build/bin/libggml-cuda.so.0.27.0"
