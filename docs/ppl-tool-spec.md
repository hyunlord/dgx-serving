# Dense Gemma teacher-forced PPL tool

`tools/ppl_gemma4.cu` is a standalone measurement tool. It reuses `gemma4_decode`, the manifest loader, q4 weight loader, and serving RoPE generator. It does not modify graph/kernel behavior, generate HF references, tokenize text, or insert special tokens.

## Build and invocation

Build separately from GPU measurements:

```sh
export PATH=/usr/local/cuda/bin:$PATH
nvcc -O2 -std=c++17 -arch=sm_120 -o tools/ppl_gemma4 \
  tools/ppl_gemma4.cu models/gemma4_decode.cu runtime/weights_gemma4.cu
```

Run with the existing external memory/process/warm/clock guard:

```sh
env G4_MMQ_PREFILL=0 MMQ256=0 tools/ppl_gemma4 \
  --weights artifacts/gemma4-12b/weights \
  --tokens PREPARED_WINDOWS.bin --window 1024 --nwin 32
```

`G4_MMQ_PREFILL` must explicitly be `0` (SB) or `1` (MMQ); `MMQ256=0` is required. Both scale variants must use the same path. MMQ is requested only under the existing graph dispatch conditions; `path` in the output records the requested route, not an independent kernel trace. Configuration dimensions and scale dtype come from the manifest; dense Gemma models only. MoE input is rejected.

The tool accepts unique named options, rejects unknown options and malformed integers, and requires exactly `window*nwin` int32 tokens with vocabulary-valid IDs. Supported windows are 2..1024. The prepared file uses little-endian signed int32 on the target Linux/aarch64 platform. No padding, truncation, implicit BOS, or chat template insertion occurs.

## Token and loss accounting

For the registered corpus, **1024 includes the prepared BOS**, leaving 1023 text targets in each window. Thirty-two windows therefore contain 32768 input IDs and score **32736** next-token targets. Tokenization/preparation must save the BOS policy, source and token hashes independently; the C++ executable does not infer BOS ID from a model name.

Every window resets `n_past` to zero and overwrites its valid KV range. It calls the existing graph with `use_q4=2`, `amort_mm=1`, and `last_only=0`. Logit row `t` predicts input token `t+1`; the final row is finite-checked but unscored. No target or KV context crosses a window boundary.

The graph already applies manifest-defined final logit softcapping. The tool must not apply softcap again. For each scored row it computes, in double precision:

```
a = max(logits)
NLL = (a - logits[target]) + log(sum(exp(logits[j] - a)))
```

It accumulates double NLL sums and counts. Aggregate PPL is `exp(total_NLL / total_targets)`, not an average of window perplexities. `per_window_nll` contains the mean NLL of each independent window. JSON double output uses 17 significant digits, sufficient for binary64 round-trip. Nonfinite logits/NLL or aggregate overflow fail rather than produce a partial success JSON.

The final stdout line is one raw JSON object with exactly:

```json
{"model":"gemma4-12b","weights":"...","path":"SB","scale_dtype":0,"nwin":32,"ntok":32736,"mean_nll":0,"ppl":1,"per_window_nll":[]}
```

The values above illustrate the schema only and are **not measured values**. `model` is the weight directory's parent folder label; dimensions are manifest-driven. Loader diagnostics may appear before the final line. Tool progress, validation and memory diagnostics go to stderr. On failure, there is no final result JSON. External wrappers must also require exit status zero.

## Replacement tool-verification gates

These gates replace the unavailable same-input eight-position WikiText HF reference; no new HF reference is generated.

1. **Existing five-token golden:** pass the original `golden/token_ids.bin`, `--window 5 --nwin 1 --golden-dir GOLDEN`, with MMQ OFF. The tool verifies that the supplied token IDs exactly equal the golden IDs, reads exact-size existing HF RoPE arrays for these five positions only, and compares all five full-depth logit argmax values against `q4_logits.bin`. It prints each pair and requires 5/5, including the final unscored input row. The PPL denominator is four targets for this validation run. No existing reference file is written. If W4A8 argmax differs, distinguish the same-input existing-harness baseline from a tool discrepancy before proceeding; this tool does not relax the criterion.
2. **Independent NumPy NLL:** run one prepared WikiText window with `--window 1024 --nwin 1 --dump-dir NEW_DIRECTORY`. The parent directory must exist and the destination must not. The tool writes three float32 vocabulary rows at positions **0, 511, 1022**, each with a JSON sidecar containing position, next-token target ID, V, and tool-computed double NLL. Independently recompute each row with NumPy float64 stable log-softmax and require relative error `<1e-6`. Artifacts belong under ignored raw-log storage; these dumps are not HF goldens.
3. **Reset:** use the same prepared windows A and B. Compare B's `per_window_nll[1]` from `[A,B]` against `per_window_nll[0]` from standalone `[B]` in separate executions. Parse JSON to binary64 and compare its bits, not rounded display text.
4. **Repeatability:** execute identical input/settings twice and compare every `per_window_nll` binary64 bit pattern. Proceed to the registered PPL experiment only after all four gates pass.

`--golden-dir` and `--dump-dir` are mutually exclusive. Golden mode only accepts 5×1; dump mode only accepts 1024×1. Files are created exclusively, preserving prior evidence.

`tools/ppl_gemma4 --self-test` exercises the actual CPU NLL routine on uniform logits, extreme logits, translation invariance and standard first-index argmax ties. This numeric smoke test does not initialize a GPU and is supplementary, not a substitute for the four runtime gates.

## Memory and measurement conditions

For V=262144 and window=1024, device logits occupy 1073741824 bytes. At most128 rows are copied into the 134217728-byte host chunk; values are converted to double one at a time during accumulation, avoiding a full host double logits array.

Before loading weights, the tool records a conservative memory estimate: all immediate weight-directory `.bin` file sizes (following symlinks), the largest loading buffer, manifest-specific linear KV for every layer, graph activation and quantization workspace, full logits, host chunk, RoPE storage, and 2GiB runtime reserve. It rejects estimates above 70% of current `/proc/meminfo` MemAvailable. This internal guard supplements the required external process/telemetry guard; it is not permission to overlap CPU preparation, builds, conversion, or another GPU workload. Resident allocations are already reflected in MemAvailable.

Each window prints `PPL_INTERVAL window=<index> start=<epoch> end=<epoch> scope=gemma4_decode_prefill` on stderr. The start is sampled after `cudaDeviceSynchronize`, immediately before the graph call; the end is sampled after the graph and final synchronization. Select only actual100ms clock samples within these bounds for the prefill median. This host envelope includes graph allocations, launches and synchronization, but excludes model loading, token upload and CPU NLL transfer/reduction. It is not an isolated kernel clock. A whole-process clock median must not replace the per-window median. `per_window_nll` remains the **mean** NLL for each window; `PPL_WINDOW` stderr also records the NLL **sum** and target count. PPL is a quality metric, not evidence for an in-engine performance gain.
