# Knob registry (v1)

Every environment-driven setting of the engine is declared once in `runtime/knobs.h` (`g_knobs`).
Engine, tool and test code read values only through `knob_raw(name)`; `tools/knobs_lint.sh` (run by
`make test`) fails on any direct `getenv` outside the registry, on a `knob_raw` name that is not
registered, and on stale exports.

The full generated table is [`docs/knobs-table.md`](knobs-table.md) (`tools/knobs_dump --md`); the
machine-readable form is `tools/autotune/profiles/knobs.json` (`tools/knobs_dump --json`).

## Resolution and startup output

- Priority: explicit env > recipe file > default. The recipe file is named by `DGX_KNOBS_FILE` and holds
  `NAME=VALUE` lines (`#` comments, surrounding spaces ignored). It uses the same names and values as
  env, so a recipe's `settings.backend.fixed_env` / `backend_env` map (`tools/autotune/recipe.py`)
  translates line for line; a YAML parser in the C++ engine is not needed for that.
- `knob_raw` returns `nullptr` when neither env nor the recipe file sets the knob, exactly like the
  `getenv` call it replaced; the call site still applies its own default, and the registry default
  column documents that value. Moving a call site onto the registry therefore changes nothing for
  accepted values.
- On first use the registry validates every set knob: an enum must be one of its choices, an int or
  float must parse completely and fall inside `min..max`, lists must parse element by element. An
  invalid value stops the process with exit code 2 and the allowed range, for example
  `knobs: invalid G4_KV_DTYPE=fp8 (env); allowed: fp32|f16|bf16`. A recipe line with an unknown name
  is also an error.
- Env names under a registry prefix (`G4_`, `BENCH_G4_`, `GEMV_`, `MMQ`, `DRAFT_`, `MTP_`, `MOE_`,
  `DGX_`) that are not registered produce a warning with the closest registered name
  (`unknown knob env G4_ATTN_CHUNKS (did you mean G4_ATTN_CHUNK?)`). They are not errors, because
  old scripts still export names such as `GEMV_D1` from unmerged branches.
- `knobs_init(tool)` prints the effective configuration once on stderr as one line,
  `KNOBS {"tool":...,"recipe":...,"knobs":{"G4_KV_DTYPE":{"value":"f16","source":"default","class":"numeric"},...}}`.
  Every bench, daemon and ppl `main` calls it. Diagnostic knobs appear only when set.

Behavior change from v1 (intended): values the old parsers silently clamped or ignored are now
rejected, e.g. `G4_MMQ_MIN_S=8` (was clamped to 16), `G4_ATTN_SPLIT=2` (was treated as on),
`REP_WINDOW=-1` (was clamped to 0).

## Classes

| class | meaning | search target? |
|---|---|---|
| tunable | output bits unchanged; only speed changes. The `gates` column names the invariant that makes this true, and that gate must still be re-run after a change. | yes |
| numeric | output values change (summation order, storage precision, a different kernel path) | yes, only behind the quality gate below |
| fixed | functional serving setting (stop ids, repetition window, recipe file) | no |
| diagnostic | test/bench harness and debugging switches | no |

"Claimed bitwise" in the gates column means the bit identity is stated in the code comments or an
earlier report but is not re-verified by a standing test; treat such a knob as numeric until a gate
confirms it on the target model.

## Invariants a knob can touch

| invariant | meaning | gate |
|---|---|---|
| spec==greedy | speculative output equals plain greedy for every K | `tests/test_gemma4_spec_e2e` (9 combinations, mism=0), Graph ON and OFF |
| batch==single | each row of a batched step equals the single-sequence step | `tests/test_gemma4_batch_iso`, `tests/test_gemma4_batch` |
| fastpath==batch | daemon single-request fast path equals the batch path | `tools/r14_path_equiv` |
| graph_on==off | Graph v2 (overlap / node update) equals eager | `tools/r13_graph_gate`, `G4_CUDA_GRAPH_STATS` re-instantiation count 0 |
| golden | 12B / 31B / MoE golden tests | `make test`, `tests/test_gemma4` |
| prefill_slot | chunked prefill slot equals c1tok | `tests/test_gemma4_prefill_slot` |

## Speculative decoding knobs

`G4_SPEC_K` (draft length, 1..8) and `G4_SPEC_DRAFT_UQ` (assistant drafter precision, 2 = W4A8,
0 = fp32) are tunable: speculative output equals plain greedy output for every K and drafter
(spec==greedy, `tests/test_gemma4_spec_e2e`, `tests/test_gemma4_spec_serving.py`), so they need no
quality gate, only the spec==greedy gate and a speed measurement. They override the spec daemon's
positional arguments (`serve_api.py --spec-k`); priority is env > recipe file > argument > default (3,
W4A8). When an env or recipe value differs from an explicit argument, the daemon logs which value won.

**Tie rule for tunable defaults.** When two values of a tunable knob measure within noise of each other,
the default is the cheaper one. `G4_SPEC_K` defaults to 3: in R17 serving (12 prompts, W4A8 drafter,
3 alternating rounds) K=3 and K=4 measured +69.2% and +69.0% over plain, a tie, and K=3 wastes fewer
rejected verify rows and less verify memory.

## Searching numeric knobs

A trial that changes a numeric knob is not comparable on speed alone. Before a configuration that
differs in a numeric knob can be adopted (recipe, default, or autotune winner):

1. Golden tests for the affected model (`make test`).
2. The invariants in the knob's `gates` column, re-run with the candidate value.
3. Quality: default-settings chat PPL (`tools/ppl_gemma4 --decode`, docs/ppl-tool-spec.md) must stay
   at or below 29.40 and below 29.6253 for Gemma 4 12B, and within +1.0% of current main. The
   measured noise floor for an order-only change is +0.085% (decode-attn-2026-10 §3). KL against the
   main logits (`--kld-ref`) is recorded alongside.

An autotune harness may screen numeric values for speed, but must run steps 1 to 3 on the final
candidate and discard it on failure, the same "ISO only on the best" pattern as the existing
harness (docs/autotune-harness.md). Tunable knobs need only their listed invariant.

## Adding a knob

1. Add a row to `g_knobs` (name, type, default, range, class, scope, gates, description).
2. Read it with `knob_raw("NAME")` at the call site.
3. Regenerate the exports: `tools/knobs_dump --json > tools/autotune/profiles/knobs.json` and
   `tools/knobs_dump --md > docs/knobs-table.md`.
4. `make test` (lint + `tests/test_knobs.py`).
