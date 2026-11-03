#!/usr/bin/env bash
# CPU preparation and guarded diagnostic rerun; originals remain untouched.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
ROOT=$HOME/github/dgx-serving
REPO=$HOME/github/dgx-serving-r24diag
PARENT=$ROOT/artifacts/r24/run-20261103T120344Z
DEADLINE=1793718224
mkdir "$PARENT/cooled.claim"
RUN_ID=cooled-$(date -u +%Y%m%dT%H%M%SZ)-$$
OUT=$PARENT/$RUN_ID
mkdir "$OUT"
printf '%s\n' "$OUT" > "$PARENT/cooled.claim/output"
printf '%s\n' "$$" > "$OUT/cooled.pid"
LOCKED=0
finish() {
    local rc=$?
    printf '%s\n' "$rc" > "$OUT/cooled.exit"
    if [ "$LOCKED" = 1 ]; then rmdir "$PARENT/lock"; fi
}
trap finish EXIT
exec > "$OUT/cooled.log" 2>&1
remaining() {
    [ "$(date -u +%s)" -lt "$DEADLINE" ] || { echo R24_DEADLINE_REACHED; exit 124; }
}
while [ ! -f "$PARENT/post.exit" ]; do remaining; sleep 5; done
[ "$(cat "$PARENT/post.exit")" = 0 ] || { echo POST_FAILED; exit 86; }
while [ -d "$PARENT/lock" ]; do remaining; sleep 1; done
mkdir "$PARENT/lock"
LOCKED=1
remaining
cd "$ROOT"
# Same compiler flags as the original r24_grid, with a fresh output path.
sha256sum "$REPO/tools/r24_grid.cu" "$REPO/tools/r24_grid_kernel.cuh" \
    "$REPO/kernels/gemm_mmq.cuh" "$PARENT/r24_llama_isolated" > "$OUT/build.inputs.sha256"
bash "$REPO/tools/r24_deadline.sh" nvcc -O2 -std=c++17 -lineinfo -arch=sm_121 \
    -Xptxas=-v -I "$REPO" "$REPO/tools/r24_grid.cu" -o "$OUT/r24_grid" \
    > "$OUT/build_grid.log" 2>&1
sha256sum "$OUT/r24_grid" > "$OUT/build.outputs.sha256"
idle() {
    local name=$1 apps now util temp sm zero_since=0
    printf 'epoch,utilization_gpu,temperature_gpu,clocks_sm\n' > "$OUT/idle_$name.csv"
    while true; do
        remaining
        apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [ -z "$apps" ] || { echo "EXTERNAL_GPU_PROCESS $apps"; exit 87; }
        IFS=, read -r util temp sm < <(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,clocks.sm --format=csv,noheader,nounits)
        util=${util//[[:space:]]/}
        [[ "$util" =~ ^[0-9]+$ ]] || { echo INVALID_GPU_UTILIZATION; exit 89; }
        now=$(date -u +%s)
        printf '%s,%s,%s,%s\n' "$now" "$util" "$temp" "$sm" >> "$OUT/idle_$name.csv"
        if [ "$util" = 0 ]; then
            if [ "$zero_since" = 0 ]; then zero_since=$now; fi
            if [ $((now-zero_since)) -ge 120 ]; then break; fi
        else
            zero_since=0
        fi
        sleep 1
    done
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks_throttle_reasons.active \
        --format=csv > "$OUT/start_$name.csv"
}
guard() {
    local name=$1
    shift
    remaining
    python3 tools/probe_run_guard.py --name "$name" --log-dir "$OUT" \
      --expected-bytes 34359738368 --cuda-meminfo artifacts/r06-2026-10/cuda_meminfo \
      --evict-weights "$ROOT/artifacts/gemma4-12b-qat-ident-fp16/weights" \
      --evict-weights "$HOME/research" --evict-weights "$HOME/models" \
      --evict-weights "$ROOT/artifacts/r22" --evict-weights "$ROOT/artifacts/r23" \
      --evict-weights "$PARENT" -- bash "$REPO/tools/r24_deadline.sh" "$@"
}
case_index=0
for m in 128 256 512 1024 2048; do
    for shape in gate down o; do
        case "$shape" in
            gate) n=15360; k=3840;;
            down) n=3840; k=15360;;
            o) n=3840; k=4096;;
        esac
        engines=(main llama)
        if [ $((case_index%2)) = 1 ]; then engines=(llama main); fi
        for engine in "${engines[@]}"; do
            name=${engine}_${shape}_M$m
            idle "$name"
            if [ "$engine" = main ]; then
                guard "$name" "$OUT/r24_grid" sweep "$m" "$n" "$k"
            else
                guard "$name" "$PARENT/r24_llama_isolated" "$m" "$n" "$k"
            fi
        done
        case_index=$((case_index+1))
    done
done
echo COOLED_SWEEP_COMPLETE
