#!/usr/bin/env bash
set -euo pipefail
OUT=$HOME/github/dgx-serving/artifacts/r24/run-20261103T120344Z
printf '%s\n' "$$" > "$OUT/post_wait.pid"
trap 'printf "%s\n" "$?" > "$OUT/post_wait.exit"' EXIT
while [ ! -f "$OUT/sequence.exit" ]; do
    [ "$(date -u +%s)" -lt 1793718224 ] || exit 124
    sleep 5
done
[ "$(cat "$OUT/sequence.exit")" = 0 ]
bash "$HOME/github/dgx-serving-r24diag/tools/r24_run.sh" post
