#!/usr/bin/env bash
# CPU-only report export. Usage: bash tools/r24_source_export.sh REPORT OUTDIR [KNOWN_READ_BYTES]
set -euo pipefail
set -o noclobber
if (( $# < 2 || $# > 3 )); then
    echo 'Usage: r24_source_export.sh REPORT OUTDIR [KNOWN_READ_BYTES]' >&2
    exit 2
fi
report=$1
outdir=$2
scriptdir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stem=$(basename -- "$report" .ncu-rep)
mkdir -p -- "$outdir"
raw="$outdir/${stem}_raw.csv"
source_csv="$outdir/${stem}_source.csv"
summary="$outdir/${stem}_summary.json"
for output in "$raw" "$source_csv" "$summary"; do
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite $output" >&2
        exit 2
    fi
done
# GB20B supported all-sample reasons; selected is retained in CSV but excluded
# from stalled-sample totals. not_selected is a waiting state and stays included.
reasons=(barrier branch_resolving dispatch_stall drain lg_throttle long_scoreboard
    math_pipe_throttle membar mio_throttle misc no_instructions not_selected
    selected short_scoreboard sleeping tex_throttle wait)
metrics=inst_executed,memory_l1_wavefronts_shared,memory_l1_wavefronts_shared_ideal
for reason in "${reasons[@]}"; do
    metrics+=",smsp__pcsamp_warps_issue_stalled_${reason}"
done
ncu --import "$report" --csv --page raw --print-units base > "$raw"
ncu --import "$report" --csv --page source --print-source sass \
    --print-units base --metrics "$metrics" > "$source_csv"
args=("$raw" --source "$source_csv")
if (( $# == 3 )); then
    args+=(--calibration-bytes "$3")
fi
python3 "$scriptdir/r24_ncu_summary.py" "${args[@]}" > "$summary"
echo "$summary"
