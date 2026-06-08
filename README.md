# unison-evals

[![CI](https://github.com/Unison-Workspace/Unison-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Unison-Workspace/Unison-evals/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Public benchmark harness for [Unison](https://github.com/Unison-Workspace/unison-brain) and comparable agent / memory systems.

Treats the production agent as a black box: every system implements one adapter (~80 LOC), points at its API/CLI, and is scored on the same datasets with the same metrics under the same constraints.

> **v0.1 — early but real.** Ships 6 datasets (LongMemEval, MuSiQue, FRAMES, BitempoQA, MS MARCO, MemoryAgentBench) across 4 tracks, with adapters for `unison-agent` and `mem0-agent`, raw-model baselines (`anthropic-raw`, `openai-gpt5`, `google-gemini`), and CLI agents (`claude-code`, `codex`, `gemini-cli`). The hosted leaderboard is next.

## Repository structure

Two parallel eval systems live here, intentionally separate:

| Subpackage | What it evaluates | Style |
|---|---|---|
| **`src/unison_evals/memory_evals/`** | Single-question Q&A over a per-question corpus | Track 1/2/3 — retrieval, oracle, E2E |
| **`src/unison_evals/benchmarks/`** | Task-shaped, multi-turn, end-state-scored benchmarks | τ-bench, Letta Context-Bench, etc. |

The two share infrastructure (`cli.py`, `config.py`, `types.py`, `server/`) but have different adapter contracts and scoring shapes. Pick the one that matches the question you're asking:

- *"Does my agent retrieve and reason well over a corpus given a question?"* → `memory_evals/`
- *"Does my agent's interface let it accomplish a multi-turn task correctly?"* → `benchmarks/`

See each subpackage's README for details:
- [`memory_evals/`](./src/unison_evals/memory_evals/__init__.py) — the original style, described below in this README
- [`benchmarks/README.md`](./src/unison_evals/benchmarks/README.md) — task-shaped multi-turn
  - [`benchmarks/tau_bench/README.md`](./src/unison_evals/benchmarks/tau_bench/README.md) — current architectural ablation: Unison's bash+md vs native function-calls on retail CRUD

## Top-level docs reference

| File | What it is |
|---|---|
| `README.md` (this file) | quickstart + repo structure |
| [`METHODOLOGY.md`](./METHODOLOGY.md) | how scores are computed, hardware, datasets |
| [`DEPLOY.md`](./DEPLOY.md) | deployment of the hosted leaderboard |

## What it measures

Three tracks isolate three different failure modes:

| Track | What it tests | Method |
|---|---|---|
| **1. Brain only** | Retrieval ranking quality | Fixed queries → score returned chunks vs gold (no LLM) |
| **2. Agent oracle** | Reasoning given perfect context | Hand the agent gold context → score answer (no retrieval) |
| **3. Agent + brain (E2E)** | What users actually experience | Full pipeline → score answer + faithfulness + cost + latency |

## What you're comparing in Track 3

**Track 3 = "given the same per-question corpus, who handles it best?"**

Every system receives the identical per-question document corpus. What differs is *how* each system consumes it:

| System type | How it handles the corpus |
|---|---|
| `unison-agent` | Ingests docs into a brain via `seedDocs`; retrieves only what's relevant for the question |
| `mem0-agent` | Extracts memories via Mem0; retrieves top-k facts |
| Raw-model adapters (`anthropic-raw`, `openai-gpt5`, `google-gemini`) | Stuffs the entire corpus inline into the prompt (long-context inlining) |
| CLI adapters (`claude-code`, `codex`, `gemini-cli`) | Same as raw-model — no persistent store, full corpus in prompt |

The **headline metric** is `efficiency_ratio = baseline_tokens / system_tokens`, where `baseline_tokens` is the mean input tokens consumed by `anthropic-raw` (the most token-hungry naive approach). A ratio of 8× means Unison answered the same questions using 8× fewer prompt tokens.

Run the comprehensive comparison with `--track all` to get Track 1 + Track 2 + Track 3 in one combined JSON:

```bash
uv run unison-evals run \
  --systems unison-agent,anthropic-raw,mem0-agent \
  --dataset longmemeval \
  --track all \
  --limit 25
```

## Results — LongMemEval (Track 3, agentic end-to-end)

**Methodology.** Split: `longmemeval_s_cleaned` — full ~50-session haystacks **with distractors** (the hard split Zep/Mem0 report on, not the `oracle` split). Track 3 = ingest → **multi-step agent** retrieves + reasons + answers. This is **end-to-end answer accuracy**, *not* retrieval recall@k and *not* single-pass QA — a strictly harder metric. Sampling: category-weighted proportional (`EVAL_STRATIFIED=proportional`), so `n=150` mirrors the full 500-set category mix. Judge: `gemini-3.1-flash-lite` (the `--dev` judge — see caveats).

**`unison-agent`** (configured model via the Unison server's auto-routing; see the Unison server for model details), **~$0.02–0.03 / question**:

| Weighted run (seed) | Overall (n=150) |
|---|---|
| 1234 | 90.0% |
| 5678 | 91.3% |
| 9012 | 86.0% |
| **mean** | **89.1%** |

Per-category (representative weighted run, n=150):

| Category | Accuracy |
|---|---|
| multi-session | 90.0% |
| temporal-reasoning | 90.0% |
| knowledge-update | 91.3% |
| single-session-assistant | 100% |
| single-session-user | 81–100% (high variance, n≈21) |
| single-session-preference | 86–89% |

Reproduce:

```bash
EVAL_STRATIFIED=proportional EVAL_SEED=1234 \
  uv run unison-evals run --dataset longmemeval --systems unison-agent --limit 150 --dev
```

**Caveats — read before citing.**
- **Dev judge.** These numbers use `gemini-3.1-flash-lite`, *not* `gpt-4o`. The publishable, cross-system-comparable number requires the gpt-4o judge (`JUDGE_MODEL=gpt-4o-2024-08-06`); it can move the score in either direction.
- **Variance.** Run-to-run decoding variance is ±2–3pp even at n=150 (the `auto` tier samples non-deterministically). Per-category cells at n≈20–40 swing ±10–18pp — treat the **weighted overall**, not individual cells, as the signal.
- **No benchmark contamination.** Prompts contain only general principles, not question-specific exemplars; a locked `EVAL_SPLIT=dev|holdout` partition guards against overfitting (tune on `dev`, validate on `holdout`).

## Quickstart

```bash
# 1. Clone + install
git clone https://github.com/Unison-Workspace/Unison-evals.git
cd Unison-evals
uv sync                                   # uses .python-version (3.12)

# 2. Configure
cp .env.example .env
$EDITOR .env                              # set UNISON_JWT, ANTHROPIC_API_KEY
# To run the `unison-agent` system, you need a Unison brain server — get the
# open-source server + clients at https://github.com/Unison-Workspace/unison-brain
# Tip: when running against a local Unison server, leave UNISON_JWT blank and set
# UNISON_LOCAL_EVAL_TENANT_ID in the brain server's .env — no JWT needed.

# 3. Run an eval — CLI
uv run unison-evals run \
  --systems unison-agent,claude-code,codex,gemini-cli \
  --dataset longmemeval \
  --track agent-oracle \
  --limit 10

# 3. Run an eval — UI
uv run unison-evals-server &              # FastAPI on :8001
cd web && bun install && bun dev          # Next.js on :3000
open http://localhost:3000/runs/new
```

## Publishable comprehensive run

The canonical way to produce a full cross-dataset leaderboard is the comprehensive runner:

```bash
# Smoke run (limit=3 per combo, cheap):
LIMIT=3 bash scripts/run_comprehensive.sh

# Development run (default limit=20):
bash scripts/run_comprehensive.sh

# Publishable run (limit=50, Opus judge, confirm budget gate):
LIMIT=50 JUDGE=claude-opus-4-5-20250101 CONFIRM=1 bash scripts/run_comprehensive.sh

# Scope to one track or one dataset:
ONLY_TRACK=brain bash scripts/run_comprehensive.sh
ONLY_DATASET=bitempoqa bash scripts/run_comprehensive.sh
```

The script:
- Emits a cost estimate before starting; refuses to proceed if the estimate exceeds $20
  unless `CONFIRM=1` is set.
- Logs `[SKIP]` for every (track, dataset, system) combo whose prereqs (API keys,
  services) are missing. Never silently produces fake numbers.
- Logs `[FAIL]` on any adapter crash or HTTP 500 and continues the rest of the matrix.
- Writes one JSON per combo to `results/comprehensive-<TS>/` and an aggregated
  `summary.json` at the end.

After the run, open `/?tab=cross-dataset` in the web UI for the aggregated leaderboard.
Fill in `reports/template.md` with the headline numbers for a shareable report.

## How the comparison stays honest

- **No agent fork.** `unison-agent` adapter calls the same `/api/rest/agents/eval-turn` endpoint that ships in production. Track 3 is bit-for-bit the production agent. Track 2 uses the same agent with the brain/FS/workspace tools disabled (via the `oracleContext` knob in the eval-turn request).
- **Fixed model + temperature.** Judge model pinned per release (`JUDGE_MODEL` env var). All systems use temperature=0 where possible.
- **Fixed dataset versions.** Datasets are downloaded from HuggingFace at a pinned commit hash and cached locally.
- **All numbers reproducible.** Every run writes a JSON artifact with the exact dataset hash, model versions, timestamps, and per-question scores. Re-running the same config on the same hardware gets within ±2%.
- **Comparators run in their preferred config.** Each adapter is configured per the system's docs. Issues / PRs welcome to fix any disadvantage we created accidentally.

## What this benchmark is *for* (and what it isn't)

**For:** comparing memory/workspace agents on tasks that exercise persistent recall, multi-hop reasoning over knowledge, and bitemporal fact correctness.

**Not for:** comparing coding agents on shell tasks (use [SWE-bench](https://github.com/princeton-nlp/SWE-bench), [Terminal-Bench](https://github.com/laude-institute/terminal-bench)) or computer-use agents on browser tasks (use [WebArena](https://webarena.dev/), [OSWorld](https://os-world.github.io/)). Unison isn't a coding agent — running it on SWE-bench would be a category mistake.

## Architecture

```
┌──────────┐     fetch     ┌──────────────┐    asyncio    ┌──────────────┐
│ Next.js  │ ──────────→   │  FastAPI     │ ──────────→   │  Job worker  │
│  web UI  │ ←───── SSE    │  /api/runs/* │               │  (in-process)│
└──────────┘               └──────┬───────┘               └──────┬───────┘
                                  │                              │
                                  ↓                              ↓
                           ┌──────────────┐               ┌──────────────┐
                           │  SQLite      │               │  Adapters    │
                           │  (results)   │               │  (HTTP/CLI)  │
                           └──────────────┘               └──────────────┘
```

CLI calls the same job worker in-process (no HTTP). Server exists for the UI.

## Adding a new system

One file, ~80 LOC. See [src/unison_evals/adapters/base.py](./src/unison_evals/adapters/base.py) for the contract. PRs welcome.

## License

Apache-2.0. Datasets retain their original licenses (LongMemEval is MIT, FRAMES is Apache 2.0, etc.).
