#!/bin/sh
# Knob registry lint (run by `make test`): every env read goes through runtime/knobs.h, every
# knob_raw() name is registered, and the generated JSON/Markdown exports match the registry.
set -e
cd "$(dirname "$0")/.."
bad=$(git grep -n 'getenv(' -- '*.cu' '*.cuh' '*.h' | grep -v '^runtime/knobs.h:' || true)
if [ -n "$bad" ]; then echo "knobs_lint: direct getenv outside runtime/knobs.h:"; echo "$bad"; exit 1; fi
names=$(git grep -h -o 'knob_raw("[A-Za-z0-9_]*")' -- '*.cu' '*.cuh' '*.h' | sed 's/knob_raw("\(.*\)")/\1/' | sort -u)
json=$(tools/knobs_dump --json)
for n in $names; do
    echo "$json" | grep -q "\"name\": \"$n\"" || { echo "knobs_lint: knob_raw(\"$n\") is not registered"; exit 1; }
done
echo "$json" | cmp -s - tools/autotune/profiles/knobs.json || { echo "knobs_lint: tools/autotune/profiles/knobs.json is stale (tools/knobs_dump --json)"; exit 1; }
tools/knobs_dump --md | cmp -s - docs/knobs-table.md || { echo "knobs_lint: docs/knobs-table.md is stale (tools/knobs_dump --md)"; exit 1; }
echo "knobs_lint: OK ($(echo "$names" | wc -l | tr -d ' ') names read via knob_raw)"
