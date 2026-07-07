# dgx-serving

**LLM serving engine for NVIDIA DGX Spark (GB10 / sm_121)** — with built-in Bayesian auto-tuning and one-file, reproducible serving recipes.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

Purpose-built for a single machine (GB10, ~273 GB/s unified LPDDR5X, 128 GB unified memory) rather than as a general-purpose engine. The design is a **model-independent CUDA kernel library (`kernels/`) + a thin per-model graph (`models/`)**: adding a model is a new graph, not new kernels.

## Features

- **5 models served** out of the box: Qwen3.5-27B, Qwen3.5-35B-A3B (MoE), Gemma4-12B, Gemma4-31B, Gemma4-26B-A4B (MoE) — all W4A8.
- **OpenAI-compatible API + built-in web chat UI** (`/v1/completions`, `/v1/chat/completions`, `/v1/models`) — SSE streaming, online continuous batching, EOS-aware stop (`finish_reason`), and a zero-dependency browser chat page with per-response tok/s + TTFT metrics.
- **Speculative decoding, output-identical to greedy** — Qwen3.5's built-in MTP head (no separate draft model) and Gemma4-12B's official 0.4B assistant drafter (`--model gemma4-spec`); drafts are accepted only when they match the target's own argmax.
- **Auto-tuning pipeline**: `probe → plan → tune (Bayesian, Optuna TPE) → recipe`, with a correctness (bit-identity) gate on every tuned config.
- **One-file recipes**: a `recipe.yaml` captures a tuned serving policy (regime → config) plus verifiable claims; `verify` re-checks device fingerprint, memory budget, and reproduces the numbers on your machine.
- **Pure CUDA kernels, sm_121-native** — no external kernel dependency (no cuBLAS/CUTLASS in the serving path).
- **Large models on one box**: 27–35B-class models in 128 GB unified memory.

## Quick Start

```bash
git clone https://github.com/hyunlord/dgx-serving && cd dgx-serving
make        # nvcc -arch=sm_120 (GB10 sm_121-compatible); builds kernels + test binaries, no model assets needed
```

**Serving requires a multi-step weight + RoPE setup** — weights are downloaded from HuggingFace and
converted locally (never bundled; model licenses are your responsibility), and Qwen-family daemons
additionally need generated RoPE tables. Follow [`docs/serving-setup.md`](./docs/serving-setup.md), then:

```bash
# after setup (example: Gemma4-12B)
python tools/serve_api.py --model gemma4 --tokenizer google/gemma-4-12B-it \
    --weights artifacts/gemma4-12b/weights --port 8000
# → web chat UI: http://localhost:8000/
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

The correctness suite (`make test`) and the per-model HF golden gates need reference-model
artifacts — also covered in [`docs/serving-setup.md`](./docs/serving-setup.md).

Optional — the auto-tuning pipeline (device fingerprint → tune → shareable recipe), after serving setup:

```bash
python tools/autotune/probe.py   --out tools/autotune/artifacts/gb10-auto.json
python tools/autotune/autotune.py --model tools/autotune/profiles/qwen35-27b.json --phase2b
python tools/autotune/recipe.py  verify recipes/qwen35-27b-gb10.yaml   # reproduce claims locally
```

> Serving with `--model qwen` (27B/35B plain) or `--model spec` (27B MTP speculative) follows the
> same shape; see [`docs/serving-setup.md`](./docs/serving-setup.md) and
> [`docs/recipe-schema.md`](./docs/recipe-schema.md). Weights are never bundled.

## Performance

Measured on GB10 (sm_121). Representative serving throughput (continuous batching, warm):

| Model | Op point | Throughput | tpot95 | Conditions |
|---|---|---|---|---|
| Qwen3.5-27B (MTP spec) | B16 K7W2, λ4 | ~51.7 tok/s | ~0.34 s | nreq60, w4a8+mtp_q4 |
| Qwen3.5-35B-A3B (MoE) | B8 | ~62.7 tok/s | ~0.11 s | nreq40, maxlen256 |
| Gemma4-12B (dense) | B8 (all regimes) | ~96 tok/s | ~0.05 s | nreq48, maxlen384 |
| Gemma4-31B (dense) | B8 (knee) | 36–38 tok/s | ~0.14 s | nreq48, maxlen384 |
| Gemma4-26B-A4B (MoE) | single-stream | 43–46 tok/s | — | HTTP, active-width decode |

Roofline attainment (weight-movement upper bound, single decode): 27B **62.5%**, Gemma4-12B 57.6%, Gemma4-31B 53.3%, 35B MoE 29.8%.

Honest comparison vs `llama.cpp` (same model, same prompts, HTTP, warm, ~q4-class):

**Gemma4-26B-A4B (MoE):**

| Concurrency | dgx-serving | llama.cpp | ratio |
|---|---|---|---|
| 1 | 43–46 tok/s | 52.6 | **0.82–0.87×** |
| 8 (=Bmax) | 80.6 tok/s | 168.3 | **0.48×** |

Single-stream used to be 14.7 tok/s (0.28×): the online loop decoded single requests at full batch
width — fixed by active-width decode (raw kernel speed recovered, 2.5×). The N=1 range spans
full-accounting vs decode-only tok/s. The batch-8 row is unaffected by that fix and is **kernel
maturity** (26B MoE is our newest, un-tuned path; llama.cpp's kernels are years-optimized), not an
engine ceiling — continuous batching itself scales near-linearly.

**Gemma4-12B (dense), plain vs plain, single decode:** 23.3 vs 26.8 tok/s = **0.87×** (near-parity;
llama-bench tg64, speculative off on both sides).

**Gemma4-12B speculative (`--model gemma4-spec`, official 0.4B assistant drafter, W4A8):** serving
measures **50.5 tok/s chat / 46.7 code / 31.9 prose** vs ~22 plain = **1.43–2.30×**, with output
verified token-identical to plain greedy (drafts are only accepted when they match the target's own
argmax). Benchmarked head-to-head on the same GB10 against the MTP-fork `llama-server` under
identical conditions (same prompts, same official assistant, K=4, same client-observed accounting):
llama measures 56.2 / 69.6 / 45.6 tok/s — we are at **0.90× on chat** (the speculative multiplier
itself is ours to win there, 2.30× vs 2.20×), **0.67× on code**, 0.70× on prose. The acceptance gap
is fully diagnosed, including a zero-implementation K-quant-quality injection experiment: even with
32%-more-accurate weights (Q4_K_M values), code acceptance barely moves — **the code wall is 4-bit
quantization itself, not scheme quality** (prose, interestingly, recovers fully). Adaptive draft
length was then measured and closed too: three host-side policies (confidence early-stop,
extension-only, HF-style dynamic K) all lose or tie against fixed K=4 — our drafts are so cheap
(2.2 ms vs ~18 ms per accepted token) that cutting them never pays; llama's adaptive gain assumes
a costlier draft. Its byproduct arithmetic (verify cost is nearly flat in rows) did pay: **verify
concat amortization** — all active slots' verify chunks run as one weight-load-amortized call,
gated bit-identical against the per-slot path — lifts multi-user speculative serving throughput
**+33–41%** (e.g. code 44.9 → 62.5 tok/s aggregate at 2 concurrent requests; single-stream
unchanged) ([`docs/gemma-assistant-spec-measured.md`](./docs/gemma-assistant-spec-measured.md)).

We are still slower at these coordinates and say so. This is **not** a general-purpose
fastest-engine; it targets one machine and a fixed model set. Full coordinates, accounting
definitions (raw / decode-only / full), roofline breakdown, and the GB10 bandwidth landscape:
[`docs/performance.md`](./docs/performance.md).

## Requirements

- **Hardware**: NVIDIA DGX Spark (GB10, sm_121). Kernels may build elsewhere, but tuning assets are GB10-specific.
- **Toolkit**: CUDA 13.0 (`nvcc`), C++17.
- **Python** (front-end/tokenizer & weight conversion): `transformers`, `safetensors`, `torch`.
- **Disk**: space for model weights (converted from HuggingFace; not included).

## Documentation

- [`docs/serving-setup.md`](./docs/serving-setup.md) — per-model weight conversion, RoPE tables, starting the API server, web chat UI.
- [`docs/performance.md`](./docs/performance.md) — measured coordinates, roofline attainment, honest benchmarks, bandwidth landscape, known limits.
- [`docs/m6-backend-boundary.md`](./docs/m6-backend-boundary.md) — architecture, correctness contract, and how to add a new model family.
- [`docs/recipe-schema.md`](./docs/recipe-schema.md) — recipe format and the `verify` / `run` workflow.

## License

[Apache License 2.0](./LICENSE). Model weights are **not** included; each model (Qwen, Gemma, …) is subject to its own license.

*"DGX", "DGX Spark", and "GB10" are trademarks of NVIDIA Corporation. This is an independent open-source project, not affiliated with or endorsed by NVIDIA; the names are used only to identify the target hardware.*
