#!/usr/bin/env bash
set -euo pipefail
OUT=$HOME/github/dgx-serving/artifacts/r24/run-20261103T120344Z
TOOLS=$HOME/github/dgx-serving-r24diag/tools
printf '%s\n' "$$" > "$OUT/sequence.pid"
trap 'printf "%s\n" "$?" > "$OUT/sequence.exit"' EXIT
while [ ! -f "$OUT/ncu_resume.exit" ]; do
    [ "$(date -u +%s)" -lt 1793718224 ] || exit 124
    sleep 5
done
[ "$(cat "$OUT/ncu_resume.exit")" = 0 ]
bash "$TOOLS/r24_run.sh" grid
bash "$TOOLS/r24_engine.sh" grid_ncu
