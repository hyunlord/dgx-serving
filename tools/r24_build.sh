#!/usr/bin/env bash
# Reproduce diagnostic binaries only; no engine or llama source changes.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?fresh output directory under artifacts/r24}
mkdir -p "$out"
for name in r24_grid r24_bw_calibrate r24_pure_read_bw r24_llama_isolated build.inputs.sha256; do
  [ ! -e "$out/$name" ] || { echo "Refuse overwrite: $out/$name" >&2; exit 2; }
done
sha256sum "$repo"/tools/r24_grid.cu "$repo"/tools/r24_grid_kernel.cuh \
  "$repo"/tools/r24_bw_calibrate.cu "$repo"/tools/r20_bw_probe.cu \
  "$repo"/kernels/gemm_mmq.cuh > "$out/build.inputs.sha256"
nvcc -O2 -std=c++17 -lineinfo -arch=sm_121 -Xptxas=-v -I "$repo" \
  "$repo/tools/r24_grid.cu" -o "$out/r24_grid"
nvcc -O2 -std=c++17 -lineinfo -arch=sm_121 \
  "$repo/tools/r24_bw_calibrate.cu" -o "$out/r24_bw_calibrate"
nvcc -O2 -std=c++17 -lineinfo -arch=sm_121 \
  "$repo/tools/r20_bw_probe.cu" -o "$out/r24_pure_read_bw"
bash "$repo/tools/r24_llama_build.sh" "$HOME/repos/llama.cpp-2026-10" "$out/r24_llama_isolated"
sha256sum "$out"/r24_grid "$out"/r24_bw_calibrate "$out"/r24_pure_read_bw "$out"/r24_llama_isolated > "$out/build.outputs.sha256"
