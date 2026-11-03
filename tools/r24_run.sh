#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
ROOT=$HOME/github/dgx-serving
OUT=$ROOT/artifacts/r24/run-20261103T120344Z
MODE=${1:?calibrate or grid}
mkdir "$OUT/lock"
finish() { local rc=$?; printf '%s\n' "$rc" > "$OUT/$MODE.exit"; rmdir "$OUT/lock"; }
trap finish EXIT
printf '%s\n' "$$" > "$OUT/$MODE.pid"
exec > "$OUT/$MODE.log" 2>&1
cd "$ROOT"
guard() {
    local name=$1
    shift
    local remain=$((1793718224-$(date -u +%s)))
    [ "$remain" -gt 0 ] || { echo DEADLINE; exit 124; }
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    [ -z "$apps" ] || { echo "EXTERNAL_GPU_PROCESS $apps"; exit 87; }
    python3 tools/probe_run_guard.py --name "$name" --log-dir "$OUT" \
      --expected-bytes 34359738368 --cuda-meminfo artifacts/r06-2026-10/cuda_meminfo \
      --evict-weights "$ROOT/artifacts/gemma4-12b-qat-ident-fp16/weights" \
      --evict-weights "$HOME/research" --evict-weights "$HOME/models" \
      --evict-weights "$ROOT/artifacts/r22" --evict-weights "$ROOT/artifacts/r23" \
      --evict-weights "$OUT" -- bash "$HOME/github/dgx-serving-r24diag/tools/r24_deadline.sh" "$@"
}
L2=lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_aperture_device_op_read_lookup_miss.sum,lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum,lts__t_sectors_aperture_peer_op_read_lookup_miss.sum,lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum
if [ "$MODE" = calibrate ]; then
    nvidia-smi > "$OUT/step0.nvidia.txt"
    docker ps > "$OUT/step0.docker.txt"
    guard device "$OUT/r24_grid" device
    guard calibrate ncu --clock-control none --pipeline-boost-state dynamic --cache-control all \
      --kernel-name-base function --kernel-name flat16 --launch-count 1 --metrics "$L2" \
      --export "$OUT/calibrate" "$OUT/r24_bw_calibrate"
    ncu --import "$OUT/calibrate.ncu-rep" --page raw --csv > "$OUT/calibrate.csv"
elif [ "$MODE" = grid ]; then
    guard grid_check "$OUT/r24_grid" check
    guard grid_bench "$OUT/r24_grid" bench
    guard main_sweep "$OUT/r24_grid" sweep
    guard llama_sweep "$OUT/r24_llama_isolated"
elif [ "$MODE" = post ]; then
    repo=$HOME/github/dgx-serving-r24diag
    [ -e "$repo/artifacts" ] || ln -s "$ROOT/artifacts" "$repo/artifacts"
    bash "$repo/tools/r24_deadline.sh" nvcc -O2 -std=c++17 -lineinfo -arch=sm_121 "$repo/tools/r20_bw_probe.cu" \
      -o "$OUT/r24_pure_read_bw" > "$OUT/build_pure_read.log" 2>&1
    sha256sum "$repo/tools/r20_bw_probe.cu" "$OUT/r24_pure_read_bw" > "$OUT/pure_read.sha256"
    bash "$repo/tools/r24_deadline.sh" make -C "$repo" -j4 all tools/knobs_dump > "$OUT/build_tests.log" 2>&1
    guard pure_read_bw "$OUT/r24_pure_read_bw" 1920,7680
    guard make_test env -i HOME="$HOME" PATH="$PATH" LANG=C.UTF-8 make -C "$repo" test
else
    echo "Unknown mode: $MODE"; exit 2
fi
