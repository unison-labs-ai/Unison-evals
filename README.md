# unison-evals

[![CI](https://github.com/unison-labs-ai/Unison-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/unison-labs-ai/Unison-evals/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Public benchmark harness for [Unison](https://github.com/unison-labs-ai/unison-brain) and comparable agent / memory systems.

Treats the production agent as a black box: every system implements one adapter (~80 LOC), points at its API/CLI, and is scored on the same datasets with the same metrics under the same constraints.

> **v0.1 — early but real.** Ships 3 datasets (LongMemEval, MemoryAgentBench, Context-Bench) across 2 tracks (agent-oracle, agent-e2e), with adapters for `unison-agent` and `unison-agent-pipeline`. The hosted leaderboard is next.

## Repository structure

Two parallel eval systems live here, intentionally separate:

| Subpackage | What it evaluates | Style |
|---|---|---|
| **`src/unison_evals/memory_evals/`** | Single-question Q&A over a per-question corpus | Track 2 (oracle) + Track 3 (E2E) |
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

## What it measures

Two tracks isolate two different failure modes:

| Track | What it tests | Method |
|---|---|---|
| **agent-oracle** | Reasoning given perfect context | Hand the agent gold context → score answer (no retrieval) |
| **agent-e2e** | What users actually experience | Agent ingests per-question corpus, retrieves, answers → score answer + cost + latency |

## What you're comparing in agent-e2e

**agent-e2e = "given the same per-question corpus, does the agent retrieve and reason correctly?"**

Every system receives the identical per-question document corpus via `seed_docs`. The agent ingests them, retrieves relevant content, and answers — scored by the LLM judge.

| Adapter | What it does |
|---|---|
| `unison-agent` | Ingests docs into a brain via `seedDocs`; retrieves only what's relevant for the question |
| `unison-agent-pipeline` | Same agent, different ingestion pipeline configuration |

The **headline metric** is `pass_rate` — the fraction of questions the agent answered correctly per the LLM judge.

## Results — LongMemEval (Track 3, agentic end-to-end)

**Methodology.** Split: `longmemeval_s_cleaned` — full ~50-session haystacks **with distractors** (the hard split Zep/Mem0 report on, not the `oracle` split). Track 3 = ingest → **multi-step agent** retrieves + reasons + answers. This is **end-to-end answer accuracy**, *not* retrieval recall@k and *not* single-pass QA — a strictly harder metric. Sampling: category-weighted proportional (`EVAL_STRATIFIED=proportional`), so `n=150` mirrors the full 500-set category mix. Judge: `gemini-3.1-flash-lite` (the `--dev` judge — see caveats).

**`unison-agent`** (configured model via the Unison server's auto-routing; see the Unison server for model details), **~$0.02–0.03 / question**. Pooled across **3 weighted runs** (seeds 1234 / 5678 / 9012 — different question samples), **n = 450 judgments**:

| Metric | Result |
|---|---|
| **Overall** (pooled, n=450) | **89.1%** |
| Per-run spread (n=150 each) | 86.0% · 90.0% · 91.3% |

Per-category (pooled across all 3 runs, so cells are stable):

| Category | Accuracy | n |
|---|---|---|
| single-session-assistant | 96.1% | 51 |
| knowledge-update | 92.8% | 69 |
| single-session-user | 92.1% | 63 |
| multi-session | 89.2% | 120 |
| single-session-preference | 88.9% | 27 |
| temporal-reasoning | 82.5% | 120 |

**Reproducing `unison-agent`.** The harness is open source, but `unison-agent` runs against a Unison brain server that is **authenticated and not publicly hosted** — so you cannot reproduce *its* numbers without access. Request an eval access token by emailing **misha@unisonlabs.ai** (briefly state your use case), then point the harness at the provided server:

```bash
export UNISON_API_URL=...        # provided with your token
export UNISON_EVAL_SECRET=...    # your eval access token
EVAL_STRATIFIED=proportional EVAL_SEED=1234 \
  uv run unison-evals run --dataset longmemeval --systems unison-agent --limit 150 --dev
```

Additional comparator adapters can be added via the adapter interface — see [Adding a new system](#adding-a-new-system) below.

**Caveats — read before citing.**
- **Dev judge.** These numbers use `gemini-3.1-flash-lite`, *not* `gpt-4o`. The publishable, cross-system-comparable number requires the gpt-4o judge (`JUDGE_MODEL=gpt-4o-2024-08-06`); it can move the score in either direction.
- **Variance.** Run-to-run decoding variance is ±2–3pp even at n=150 (the `auto` tier samples non-deterministically) — hence the per-run spread above and the pooling of 3 runs into n=450 for the reported figures.
- **No benchmark contamination.** Prompts contain only general principles, not question-specific exemplars; a locked `EVAL_SPLIT=dev|holdout` partition guards against overfitting (tune on `dev`, validate on `holdout`).

## Quickstart

```bash
# 1. Clone + install
git clone https://github.com/unison-labs-ai/Unison-evals.git
cd Unison-evals
uv sync                                   # uses .python-version (3.12)

# 2. Configure
cp .env.example .env
$EDITOR .env                              # set UNISON_JWT, ANTHROPIC_API_KEY
# To run the `unison-agent` system, you need a Unison brain server — get the
# open-source server + clients at https://github.com/unison-labs-ai/unison-brain
# Tip: when running against a local Unison server, leave UNISON_JWT blank and set
# UNISON_LOCAL_EVAL_TENANT_ID in the brain server's .env — no JWT needed.

# 3. Run an eval — CLI
uv run unison-evals run \
  --systems unison-agent \
  --dataset longmemeval \
  --track agent-oracle \
  --limit 10

# 3. Run an eval — UI
uv run unison-evals-server &              # FastAPI on :8001
cd web && bun install && bun dev          # Next.js on :3000
open http://localhost:3000/runs/new
```

## How the comparison stays honest

- **Same production agent loop.** The `unison-agent` adapter calls the `/api/rest/agents/eval-turn` endpoint, which runs the **same `runAgent` loop that ships in production** — retrieve → reason → answer, including the counting-verification gate (no eval-only forks in the answer path). Track 2 disables the brain/FS/workspace tools via the `oracleContext` request flag. *One honest caveat:* the eval seeds each question's brain **synchronously** (`extractFromDocument → recordFact`), which runs the same extraction logic as production but **skips the asynchronous production ingestion pipeline** (the signal notability gate, reconcile, and compaction). So the *answering* path is production; the *brain-building* path is a faster eval-time shortcut over identical extraction.
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

One file, ~80 LOC. See [src/unison_evals/memory_evals/adapters/base.py](./src/unison_evals/memory_evals/adapters/base.py) for the contract. PRs welcome.

## License

Apache-2.0. Datasets retain their original licenses (LongMemEval is MIT, MemoryAgentBench is MIT).
