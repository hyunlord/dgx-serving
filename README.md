# dgx-serving

**LLM serving engine for NVIDIA DGX Spark (GB10 / sm_121)** — with built-in Bayesian auto-tuning and one-file, reproducible serving recipes.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

Purpose-built for a single machine (GB10, ~273 GB/s unified LPDDR5X, 128 GB unified memory) rather than as a general-purpose engine. The design is a **model-independent CUDA kernel library (`kernels/`) + a thin per-model graph (`models/`)**: adding a model is a new graph, not new kernels.

## Features

- **4 models served** out of the box: Qwen3.5-27B, Qwen3.5-35B-A3B (MoE), Gemma4-31B, 26B-A4B (MoE) — all W4A8.
- **OpenAI-compatible API** (`/v1/completions`, `/v1/models`) with SSE streaming and online continuous batching.
- **Built-in MTP speculative decoding** — the Multi-Token-Prediction head ships with the model; no separate draft model to host.
- **Auto-tuning pipeline**: `probe → plan → tune (Bayesian, Optuna TPE) → recipe`, with a correctness (bit-identity) gate on every tuned config.
- **One-file recipes**: a `recipe.yaml` captures a tuned serving policy (regime → config) plus verifiable claims; `verify` re-checks device fingerprint, memory budget, and reproduces the numbers on your machine.
- **Pure CUDA kernels, sm_121-native** — no external kernel dependency (no cuBLAS/CUTLASS in the serving path).
- **Large models on one box**: 27–35B-class models in 128 GB unified memory.

## Quick Start

```bash
# 1. Build (nvcc -arch=sm_120, GB10 sm_121-compatible)
git clone https://github.com/hyunlord/dgx-serving && cd dgx-serving
make                              # builds test binaries; ARCH=/NVCC= overridable

# 2. Prepare weights (not distributed — downloaded from HuggingFace and converted).
#    Model licenses (Qwen/Gemma) apply and are your responsibility.
pip install transformers safetensors torch      # reference/tokenizer deps
python tools/extract_weights_qwen35.py --model Qwen/Qwen3.5-27B --out artifacts/qwen35-27b/weights
python tools/make_reference_qwen35.py  --model Qwen/Qwen3.5-27B --out artifacts/qwen35-27b/golden

# 3. Build the serving daemon and start the API server
make tools/serve_daemon_qwen_online
python tools/serve_api.py --model qwen \
    --weights artifacts/qwen35-27b/weights \
    --golden  artifacts/qwen35-27b/golden \
    --port 8000

# 4. Query it (OpenAI-compatible)
curl http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen","prompt":"The capital of France is","max_tokens":16}'
```

Optional — the auto-tuning pipeline (device fingerprint → tune → shareable recipe):

```bash
python tools/autotune/probe.py   --out tools/autotune/artifacts/gb10-auto.json
python tools/autotune/autotune.py --model tools/autotune/profiles/qwen35-27b.json --phase2b
python tools/autotune/recipe.py  verify recipes/qwen35-27b-gb10.yaml   # reproduce claims locally
```

> Serving with `--model gemma4` (dense) or `--model spec` (27B MTP speculative) follows the same shape;
> see [`docs/recipe-schema.md`](./docs/recipe-schema.md). Weights are never bundled.

## Performance

Measured on GB10 (sm_121). Representative serving throughput (continuous batching, warm):

| Model | Op point | Throughput | tpot95 | Conditions |
|---|---|---|---|---|
| Qwen3.5-27B (spec) | B16 K7W2, λ4 | ~51.7 tok/s | ~0.34 s | nreq60, w4a8+mtp_q4 |
| Gemma4-31B (dense) | B8 (knee) | 36–38 tok/s | ~0.14 s | nreq48, maxlen384 |
| Qwen3.5-35B-A3B (MoE) | B8 | ~62.7 tok/s | ~0.11 s | nreq40, maxlen256 |

Roofline attainment (weight-movement upper bound): 27B dense single **62.5%**, Gemma4 53.3%, 35B MoE 29.8%.

Honest comparison vs `llama.cpp` (same Gemma4-26B-A4B, ~q4, HTTP, warm):

| Concurrency | dgx-serving | llama.cpp | ratio |
|---|---|---|---|
| 1 | 14.7 tok/s | 52.6 | 0.28× |
| 8 (=Bmax) | 80.6 tok/s | 168.3 | 0.48× |

We are slower here. The gap is **kernel maturity** (26B MoE is the newest, un-tuned path; llama.cpp's kernels are years-optimized), not an engine ceiling — continuous batching scales near-linearly (5.5× from N=1→8). This is **not** a general-purpose fastest-engine; it targets one machine and a fixed model set. Full coordinates, roofline breakdown, and the GB10 bandwidth landscape: [`docs/performance.md`](./docs/performance.md).

## Requirements

- **Hardware**: NVIDIA DGX Spark (GB10, sm_121). Kernels may build elsewhere, but tuning assets are GB10-specific.
- **Toolkit**: CUDA 13.0 (`nvcc`), C++17.
- **Python** (front-end/tokenizer & weight conversion): `transformers`, `safetensors`, `torch`.
- **Disk**: space for model weights (converted from HuggingFace; not included).

## Documentation

- [`docs/performance.md`](./docs/performance.md) — measured coordinates, roofline attainment, honest benchmarks, bandwidth landscape, known limits.
- [`docs/m6-backend-boundary.md`](./docs/m6-backend-boundary.md) — architecture, correctness contract, and how to add a new model family.
- [`docs/recipe-schema.md`](./docs/recipe-schema.md) — recipe format and the `verify` / `run` workflow.

## License

[Apache License 2.0](./LICENSE). Model weights are **not** included; each model (Qwen, Gemma, …) is subject to its own license.

*"DGX", "DGX Spark", and "GB10" are trademarks of NVIDIA Corporation. This is an independent open-source project, not affiliated with or endorsed by NVIDIA; the names are used only to identify the target hardware.*
