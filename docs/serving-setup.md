# Serving setup — weights, RoPE tables, and the API server

> Serving is a **multi-step setup, not a single copy-paste**: model weights are downloaded from
> HuggingFace and converted locally (they are never bundled — each model's license is your
> responsibility), Qwen-family daemons additionally need RoPE tables, and the API front needs the
> model's tokenizer in your local HF cache. The commands below are the exact forms used to produce
> the measured coordinates in [`performance.md`](./performance.md), on a GB10 (DGX Spark) box.

## 0. Prerequisites

- **Hardware**: NVIDIA DGX Spark (GB10, sm_121), 128 GB unified memory. Kernels may build elsewhere,
  but every measured number and memory-budget guard here assumes GB10.
- **Toolkit**: CUDA 13.0 (`nvcc`), C++17, GNU make.
- **Python** (conversion + API front): `pip install transformers safetensors torch`
- **Disk**: converted W4A8 weights are roughly 0.6 GB per B-param (27B ≈ 16 GB, 26B-A4B ≈ 18 GB),
  plus the HuggingFace download cache (bf16 originals are ~2 GB per B-param).
- **Network**: only the conversion step downloads. The server itself runs offline —
  `serve_api.py` sets `HF_HUB_OFFLINE=1` and reads the tokenizer from the local HF cache, which the
  conversion step populates. If you pass a `--tokenizer` id you never downloaded, pre-fetch it once
  with `HF_HUB_OFFLINE=0`.

## 1. Build the serving daemons

```bash
make tools/serve_daemon_gemma4_online tools/serve_daemon_qwen_online tools/serve_daemon_qwen_spec_online
```

## 2. Prepare weights (per model)

The extractor (`tools/extract_weights_qwen35.py`) is config-driven and covers both families —
Gemma MoE router tensors, per-layer-type RoPE, sliding windows etc. are picked up from the model's
`config.json` automatically. `--dtype bfloat16 --q4-only` keeps host RAM inside the 128 GB unified
budget for the big models (a full fp32 load of a 27–31B model is ~110 GB and **will** hit the
memory guard; don't).

### Gemma4-12B — recommended QAT weights

Use `google/gemma-4-12B-it-qat-q4_0-unquantized`, revision
`b6ed86275a6a5735884e208bfed95b445a684ca2` (Apache-2.0). Extract from that pinned
local checkpoint into a new directory; retain the original RTN weights and golden files.
The measured rule is positive absolute maximum / 8.05, signed int4 `[-8, 7]`.

```bash
python tools/extract_weights_qwen35.py \
    --model ~/models/qat/gemma-4-12B-it-qat-q4_0-unquantized \
    --out artifacts/gemma4-12b-qat-ident/weights --dtype bfloat16 --q4-only \
    --quant qat-ident --quant-rule positive-max --quant-c 8.05 --scale-dtype fp32
```

The recommended 12B serving weights use a **simple fp16 scale conversion with q fixed**:

```bash
python tools/convert_scale_fp16.py \
    artifacts/gemma4-12b-qat-ident/weights \
    artifacts/gemma4-12b-qat-ident-fp16/weights
```

Use a new destination. The converter rounds only `_scale.bin` files paired with
`_q4.bin` to nearest fp16, sets manifest `scale_dtype=1`, and symlinks the q4 and
other unchanged files. It does not search scales or reassign integer q. Keep the
source directory: the converted directory depends on those symlinks.
Pass `--weights artifacts/gemma4-12b-qat-ident-fp16/weights` to the Gemma server.
The dedicated HF FP32 reference restored from these converted engine weights belongs in
`artifacts/gemma4-12b-qat-fp16/golden/`; validate it with the unchanged W4A8 test
(`rel < 1.5e-2`, argmax 5/5). R11 verification passed with worst reported rel
1.134e-2 and argmax 5/5, the default 47-entry test suite passed (live-server
degeneration check skipped), and all nine spec workload/K combinations matched greedy.
Do not substitute the fp32-scale QAT or RTN golden.

The fp32-scale alternative remains `artifacts/gemma4-12b-qat-ident/weights`, with
its matching `artifacts/gemma4-12b-qat/golden/` reference. The assistant checkpoint
is unchanged. The autotune profile and recipes still name RTN assets; select the
recommended weight path explicitly until those configurations are updated separately.

With MMQ and flash enabled, the simple conversion measured PPL
29.069840843282211 on the fixed 36-dialogue evaluation and improved decode in all
five alternating pairs (see [performance.md §3f](./performance.md#3f-단순-fp16-스케일-기본-경로-2026-10-28)).
`qat-ident --scale-dtype fp16` is a different, scale-first extraction option: it
rounds each projection scale **before** assigning q. It is not the conversion
command above. Its historical flash-OFF PPL 29.089064374088686 versus fp32
28.904909148350214 failed the then-required <0.5% gate; that result remains recorded.
Embedding/norm remain FP32, and lm_head retains the legacy RTN q assignment.

### Gemma family — RTN alternative, 31B dense and 26B-A4B MoE


```bash
python tools/extract_weights_qwen35.py --model google/gemma-4-12B-it \
    --out artifacts/gemma4-12b/weights --dtype bfloat16 --q4-only
```

Same shape for `google/gemma-4-31B-it` → `artifacts/gemma4-31b/weights` and
`google/gemma-4-26B-A4B-it` → `artifacts/gemma4-26b-a4b/weights` (MoE is auto-detected from
`num_experts` — same daemon serves all three). The Gemma daemon needs no further assets: RoPE
tables are generated on-device at startup.

### Qwen family — 27B / 35B-A3B plain (two steps: weights + RoPE tables)

```bash
python tools/extract_weights_qwen35.py --model Qwen/Qwen3.5-27B \
    --out artifacts/qwen35-27b/weights --dtype bfloat16 --q4-only
mkdir -p artifacts/qwen35-27b/mtp_golden
python tools/gen_rope_long.py artifacts/qwen35-27b 8192   # → mtp_golden/rope_{cos,sin}_ext.bin
```

The Qwen daemon loads `rope_{cos,sin}_ext.bin` from the directory you pass as `--golden`.
`gen_rope_long.py` computes the standard-RoPE tables from the extracted manifest (`rope_theta`,
`rotary_dim`) and, when a forward-captured table is present, cross-checks bit-identity against it
(skipped otherwise — the formula has been verified bit-identical against HF captures for this
family).

### Qwen3.5-27B MTP speculative (`--model spec`) — adds the MTP head

```bash
python tools/dump_mtp_weights.py --model Qwen/Qwen3.5-27B --out artifacts/qwen35-27b/mtp_golden
python tools/gen_rope_long.py artifacts/qwen35-27b 8192
```

The spec daemon reads both the `mtp_*.bin` head weights and the RoPE tables from the same
`--golden` directory.

### Gemma4-12B speculative (`--model gemma4-spec`) — adds the official 0.4B assistant drafter

```bash
python tools/extract_weights_assistant.py --model google/gemma-4-12B-it-assistant \
    --out artifacts/gemma4-12b-assistant/weights
python tools/serve_api.py --model gemma4-spec --tokenizer google/gemma-4-12B-it \
    --weights artifacts/gemma4-12b-qat-ident-fp16/weights \
    --assistant artifacts/gemma4-12b-assistant/weights --port 8000
```

Speculative output is verified token-identical to plain greedy (the drafts are only accepted when
they match the target's own argmax), and measures **~2.1–2.3× faster** end-to-end (client-observed;
drafter runs W4A8 by default — acceptance tuning and verify amortization are the known remaining
levers, see [`gemma-assistant-spec-measured.md`](./gemma-assistant-spec-measured.md)).

## 3. Start the API server

```bash
# Gemma (12B shown; 31B / 26B-A4B identical shape, no --golden needed)
python tools/serve_api.py --model gemma4 --tokenizer google/gemma-4-12B-it \
    --weights artifacts/gemma4-12b-qat-ident-fp16/weights --port 8000

# Qwen plain (27B shown; 35B-A3B identical shape)
python tools/serve_api.py --model qwen --tokenizer Qwen/Qwen3.5-27B \
    --weights artifacts/qwen35-27b/weights --golden artifacts/qwen35-27b/mtp_golden --port 8000

# Qwen 27B MTP speculative
python tools/serve_api.py --model spec --tokenizer Qwen/Qwen3.5-27B \
    --weights artifacts/qwen35-27b/weights --golden artifacts/qwen35-27b/mtp_golden --port 8000
```

- **`--tokenizer` must match the served model** — the chat template and the EOS/stop token ids come
  from it (the default is a Gemma tokenizer; Qwen serving without the flag will tokenize wrongly).
- Wait for `[online] ready` on stderr: big models spend minutes in weight loading on a cold start —
  don't kill the process early.
- `--maxlen` (default 1024) sizes the per-request KV budget (prompt + generation): `max_tokens` is
  silently clamped to `maxlen − prompt − small margin`, so a too-small maxlen shows up as answers
  cut at `finish_reason: length` (the old 384 default truncated at ~350 tokens in practice).
  `--bmax` (default 8) is the continuous-batching width. Long-context serving is a known limit
  (see [`performance.md`](./performance.md) §5).
- Default quantization is W4A8 (`--uq 2`); `--uq 1` = q4. All measured serving coordinates are W4A8.
- Dense Gemma: a lone greedy request decodes through the Graph v2 single-stream session (CUDA
  graph overlap, device-side token feedback); it returns to the batch loop at the next step when
  another request arrives. Outputs are bit-identical to the batch path. `G4_SERVE_FASTPATH=0` keeps
  every request on the batch path; `G4_CUDA_GRAPH=0` runs the single-stream session eagerly.
  Sampling requests (temperature > 0) always use the batch path.
- `stop` (string or up to 4 strings) follows OpenAI semantics: the matched text is not returned and
  `finish_reason` is `stop`. A stop match or a dropped streaming connection cancels the request on
  the GPU (`CANCEL <id>` to the daemon); a dropped non-streaming connection is noticed only when the
  response is written.
- Dense Gemma W4A8 uses MMQ for amortized prefill by default. Set `G4_MMQ_PREFILL=0` to use the
  previous SB path. `G4_MMQ_MIN_S` selects the minimum row count (default 16; smaller values clamp
  to 16; invalid values fall back to 16). Decode and non-amortized calls retain their existing paths.
  `tools/bench_gemma4` now defaults to amortized prefill; `BENCH_G4_AMORT=0` reproduces its prior
  plain GEMV prefill path. This setting does not change the scale dtype selected by the weight manifest.
- Dense Gemma prefill attention uses flash by default. Set `G4_FLASH_PREFILL=0` to restore
  fp32 attention. `G4_FLASH_MIN_S` sets the minimum query count (default 16; values below 16
  clamp to 16; invalid values fall back to 16). KV storage remains fp32; flash converts inputs
  to bf16 for MMA. Single-token decode and the existing verify calls with at most 9 queries
  retain the previous attention path. This setting does not affect 26B MoE.
- Optional: `--recipe recipes/gemma4-12b-gb10.yaml` serves with a tuned config after verifying the
  device fingerprint and memory budget at startup — see [`recipe-schema.md`](./recipe-schema.md).

## 4. Talk to it

Open **`http://localhost:8000/`** in a browser: a zero-dependency chat page with SSE streaming,
per-response tok/s + TTFT metrics, a model badge, a max-tokens control, and truncation indication
(`finish_reason: length`).

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64}'

curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","prompt":"The capital of France is","max_tokens":16}'
```

For access from another machine on your LAN, add `--host 0.0.0.0`.
**Security**: the server has no authentication — bind `0.0.0.0` only on a trusted network and never
port-forward it to the internet (the server prints the same warning).

Decoding scope, honestly stated:

- **Greedy (default)** — deterministic argmax; this is the path every correctness gate runs on.
  Historical note (honesty): long-generation degeneration (e.g. a `1 1 1 …` loop after ~350
  tokens) was **misdiagnosed twice as a greedy-policy property** — the root cause was the serving
  daemons generating their RoPE tables with the wrong full-attention convention (theta, then the
  missing `partial_rotary_factor 0.25`). Both are fixed and locked by an absolute gate against
  HF-captured tables (`tests/test_gemma4_rope_table.py`); with correct RoPE the same prompts
  generate 800 coherent tokens under plain greedy. Greedy repetition loops remain possible in
  principle (no repetition penalty on the greedy path), but are no longer the expected behavior.
- **Sampling (plain daemons only)** — `temperature` (>0 turns it on), `top_p` (default 0.9),
  `repetition_penalty` (default 1.1), `seed` (set for reproducible output; omitted = random per
  request). Implemented as host-side logic in the shared online loop (logits were already on the
  host; an implicit top-64 preselect bounds host cost at ~0.3 ms/token). Sampling is a **serving
  option**: gates keep running greedy, and a request without `temperature` takes the byte-identical
  greedy path.
  - **Repetition penalty is windowed + deduped** (`REP_WINDOW` env, default 128). Honesty note:
    the original penalty divided each history token's logit *per occurrence over the full history*,
    so common tokens accumulated `repp^N` and were crushed as generation grew — this itself caused
    long-generation sampling degeneration (language drift, repetition loops, junk LaTeX). Measured:
    a 3-workload × rep{1.0,1.1,1.2} × 2-seed grid at 850 tokens went from **3/6 (rep=1.1) and 4/6
    (rep=1.2) degenerate → 0/6** after windowing+dedup, with greedy output byte-identical (the
    penalty path only runs when `temperature>0`). Locked by `tests/test_longgen_degen.py`.
    `REP_WINDOW=0` keeps full history but still deduped (no compounding).
  - **UI presets** (`tools/chat.html`): 안정 `t0.5`, 균형 `t0.7` (default), 창의 `t0.9/top_p0.95`
    — all `rep 1.1` (the measured-clean value; temperature is the diversity knob). Presets are
    reasonable defaults, not an exhaustive temperature sweep.
- **Speculative daemons are greedy-only** (`--model gemma4-spec` / `spec`): drafts are verified
  against the target's argmax, so `temperature>0` returns an explicit 400. Pick per server:
  speed (speculative, greedy) or variety (plain, sampling).
- `n>1` / `logprobs` / `best_of` remain unsupported. Generation stops at `generation_config`
  EOS ids (`finish_reason: stop`) or the token budget (`length`).

Multi-user note: the concurrency win (verify/draft concat batching) is about **several people,
each in their own browser** — one person opening many tabs just splits their own tokens.

## 5. Correctness gates (optional, recommended)

- **Serving-path gates** (Python, run against a built daemon + prepared artifacts):
  `tests/test_serve_api_front.py` (HTTP surface vs a fake daemon — chat/completions spec, stream ==
  non-stream), `tests/test_chat_template.py` (chat-template application), `tests/test_moe26b_online.py`
  and `tests/test_spec_online.py` (concurrent serving == solo serving, bit-identical).
- **HF golden gates** (per model, manual make targets — `tests/test_27b`, `tests/test_gemma4*`, …):
  need reference dumps from `tools/make_reference_*.py`. For 27B+ models use `--dtype bfloat16` —
  fp32 reference loads exceed the 128 GB unified budget.
- The default `make test` suite gates the shared kernels and the small dev models; it expects
  `artifacts/smollm3-3b` and `artifacts/qwen35-0.8b` reference artifacts prepared with the same
  extract/reference tools (see the docstring at the top of each tool).
