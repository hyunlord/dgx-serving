#!/usr/bin/env bash
set -euo pipefail
remaining=$((1793718224-$(date -u +%s)))
if [ "$remaining" -le 0 ]; then
    echo R24_DEADLINE_REACHED >&2
    exit 124
fi
exec timeout "$remaining" "$@"
