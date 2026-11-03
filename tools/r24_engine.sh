#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
ROOT=$HOME/github/dgx-serving
OUT=$ROOT/artifacts/r24/run-20261103T120344Z
BASE=$HOME/github/dgx-serving-r22base
BUILD=$ROOT/artifacts/r22/run-20261102-1439/llama-nvtx-build
LL=$HOME/repos/llama.cpp-2026-10
W=$ROOT/artifacts/gemma4-12b-qat-ident-fp16/weights
GG=$HOME/models/qat-gguf/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf
MODE=${1:?trace or ncu or grid_ncu}
mkdir "$OUT/lock"
finish() { local rc=$?; printf '%s\n' "$rc" > "$OUT/$MODE.exit"; rmdir "$OUT/lock"; }
trap finish EXIT
printf '%s\n' "$$" > "$OUT/$MODE.pid"
exec > "$OUT/$MODE.log" 2>&1
cd "$ROOT"
[ "$(git -C "$BASE" rev-parse HEAD)" = 4f65227bb534d70475db8de469816f18b2c17100 ]
git -C "$BASE" diff --quiet
sha256sum "$BASE/tools/bench_gemma4" > "$OUT/$MODE.engine.sha256"
sha256sum -c "$BUILD/reference.before.sha256"
sha256sum -c "$BUILD/instrumented.sha256"
guard() {
    local name=$1
    shift
    local remain=$((1793718224-$(date -u +%s))) apps
    [ "$remain" -gt 0 ] || exit 124
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    [ -z "$apps" ] || { echo "EXTERNAL_GPU_PROCESS $apps"; exit 87; }
    python3 tools/probe_run_guard.py --name "$name" --log-dir "$OUT" \
      --expected-bytes 34359738368 --cuda-meminfo artifacts/r06-2026-10/cuda_meminfo \
      --evict-weights "$W" --evict-weights "$HOME/research" --evict-weights "$HOME/models" \
      --evict-weights "$ROOT/artifacts/r22" --evict-weights "$ROOT/artifacts/r23" \
      --evict-weights "$OUT" -- bash "$HOME/github/dgx-serving-r24diag/tools/r24_deadline.sh" \
      python3 tools/warm_then_run.py --warm "$ROOT/artifacts/rebaseline-2026-10-24/fp4_v1_s2" \
      --line-log "$OUT/warm_$name.jsonl" -- "$@"
}
CLEAN=(env -i HOME="$HOME" PATH="$PATH" LANG=C.UTF-8)
LCLEAN=("${CLEAN[@]}" LD_LIBRARY_PATH="$BUILD/bin:$LL/build/bin" GGML_CUDA_DISABLE_GRAPHS=1)
L2=lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_aperture_device_op_read_lookup_miss.sum,lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum,lts__t_sectors_aperture_peer_op_read_lookup_miss.sum,lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum
EXTRA=$(paste -sd, "$HOME/github/dgx-serving-r24diag/tools/r24_metrics.txt")
NCU=(ncu --clock-control none --pipeline-boost-state dynamic --cache-control all --set full --import-source yes --import-sass yes --launch-count 1 --metrics "$L2,$EXTRA")
if [ "$MODE" = trace ]; then
    for p in 1024 2048; do
        name=main_P$p
        guard "$name" "${CLEAN[@]}" BENCH_G4_SINGLE_UQ=2 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --output "$OUT/$name" "$BASE/tools/bench_gemma4" "$W" 2 "$p" 1 1
        nsys export --type sqlite --output "$OUT/$name.sqlite" "$OUT/$name.nsys-rep"
        nsys stats --report cuda_gpu_trace --format csv --output "$OUT/$name" "$OUT/$name.nsys-rep"
        name=llama_P$p
        guard "$name" "${LCLEAN[@]}" nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --output "$OUT/$name" "$BUILD/bin/llama-bench" -m "$GG" -p "$p" -n 0 -fa 1 -r 1 -b "$p" -ub "$p"
        nsys export --type sqlite --output "$OUT/$name.sqlite" "$OUT/$name.nsys-rep"
    done
elif [ "$MODE" = ncu ] || [ "$MODE" = ncu_resume ]; then
    # These skips must be revalidated against this run's trace before launch.
    [ -f "$OUT/selection_verified.txt" ]
    for p in 1024 2048; do
        for shape in gate down o; do
            case "$shape" in gate) skip=332; weight=ffn_gate;; down) skip=334; weight=ffn_down;; o) skip=331; weight=attn_output;; esac
            name=ncu_main_${shape}_M$p
            if [ ! -f "$OUT/$name.csv" ]; then
            guard "$name" "${CLEAN[@]}" BENCH_G4_SINGLE_UQ=2 "${NCU[@]}" --kernel-name-base function --kernel-name gemm_mmq_kernel --launch-skip "$skip" --export "$OUT/$name" "$BASE/tools/bench_gemma4" "$W" 2 "$p" 1 1
            ncu --import "$OUT/$name.ncu-rep" --page raw --csv > "$OUT/$name.csv"
            fi
            name=ncu_llama_${shape}_M${p}_r2
            lskip=$(awk -F, -v p="$p" -v shape="$shape" '$1==p && $2==shape {print $3}' "$OUT/llama_selection.csv")
            [ -n "$lskip" ]
            guard "$name" "${LCLEAN[@]}" "${NCU[@]}" --kernel-name-base function --kernel-name mul_mat_q --launch-skip "$lskip" --export "$OUT/$name" "$BUILD/bin/llama-bench" -m "$GG" -p "$p" -n 0 -fa 1 -r 1 -b "$p" -ub "$p"
            ncu --import "$OUT/$name.ncu-rep" --page raw --csv > "$OUT/$name.csv"
        done
    done
elif [ "$MODE" = grid_ncu ]; then
    for p in 1024 2048; do
        for shape in gate down o; do
            case "$shape" in gate) n=15360; k=3840;; down) n=3840; k=15360;; o) n=3840; k=4096;; esac
            for variant in main swap; do
                name=grid_${variant}_${shape}_M$p
                guard "$name" "${NCU[@]}" --kernel-name 'regex:.*gemm_mmq_kernel.*|.*r24_grid_swap_kernel.*' --export "$OUT/$name" "$OUT/r24_grid" profile "$variant" "$p" "$n" "$k"
                ncu --import "$OUT/$name.ncu-rep" --page raw --csv > "$OUT/$name.csv"
            done
        done
    done
else
    exit 2
fi
