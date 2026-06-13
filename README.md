<div align="center">

<img src="https://raw.githubusercontent.com/unison-labs-ai/unison-brain/main/assets/brain.svg" width="140" />

# Unison-evals

**Memory benchmarks you can actually reproduce — full results, nothing cherry-picked.**

Benchmark harness for [Unison brain](https://unisonlabs.ai) and any agent memory system.<br>
Plug in your adapter (~80 LOC), run the same datasets under the same constraints, compare honestly.

[![CI](https://github.com/unison-labs-ai/Unison-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/unison-labs-ai/Unison-evals/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/unison-labs-ai/Unison-evals?style=social)](https://github.com/unison-labs-ai/Unison-evals)

[**Benchmarks**](#benchmarks) • [**Results**](#results) • [**Quickstart**](#quickstart) • [**Architecture**](#architecture) • [**Add a system**](#adding-a-new-system)

</div>

---

## Benchmarks

Two eval tracks isolate two distinct failure modes:

| Track | What it tests | Method |
|---|---|---|
| **agent-oracle** | Reasoning given perfect context | Hand the agent gold context → score answer (no retrieval) |
| **agent-e2e** | What users actually experience | Agent ingests per-question corpus, retrieves, answers → score answer + cost + latency |

Two parallel subpackages, intentionally separate:

| Subpackage | What it evaluates | Style |
|---|---|---|
| **`memory_evals/`** | Single-question Q&A over a per-question corpus | Track 2 (oracle) + Track 3 (E2E) |
| **`benchmarks/`** | Task-shaped, multi-turn, end-state-scored | τ-bench, Letta Context-Bench, etc. |

Supported datasets:

- **LongMemEval** — ICLR 2025, `longmemeval_s_cleaned` hard split with full ~50-session haystacks + distractors
- **LOCOMO** — ACL 2024 ([snap-research/locomo](https://github.com/snap-research/locomo)), categories 1–4 (single-hop, multi-hop, temporal, open-domain)
- **MemoryAgentBench** — task-shaped memory benchmark
- **Context-Bench** — Letta multi-turn context benchmark
- **τ-bench** — architectural ablation: Unison's bash+md vs native function-calls on retail CRUD

## Results

End-to-end **answer accuracy** (LLM-judge `gemini-3.1-flash-lite`): the agent ingests the corpus, retrieves, and answers; the judge grades the final answer against ground truth. Run-to-run variance ≈ ±2–3pp.

### LongMemEval — `longmemeval_s_cleaned` (n=150, proportional, seed 9012)

| System | Answer accuracy |
|---|---|
| **unison-agent** | **91.3%** |

```bash
export UNISON_API_URL=...        # provided with your eval token
export UNISON_EVAL_SECRET=...    # request at misha@unisonlabs.ai
EVAL_STRATIFIED=proportional EVAL_SEED=9012 \
  uv run unison-evals run --dataset longmemeval --systems unison-agent --limit 150 --dev
```

### LOCOMO — `locomo10.json` cats 1–4 (n=128, proportional, seed 1234)

| Category | Answer accuracy |
|---|---|
| **Overall** | **85.9%** |
| open-domain | 93% (n=70) |
| multi-hop | 81% (n=27) |
| single-hop | 78% (n=23) |
| temporal | 62% (n=8) |

<sub>LOCOMO carries ~6.4% documented label errors ([penfieldlabs audit](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer)) that affect any system scored on it.</sub>

```bash
EVAL_STRATIFIED=proportional EVAL_SEED=1234 \
  uv run unison-evals run --dataset locomo --systems unison-agent --limit 128 --dev
```

Every run writes a JSON artifact with the exact dataset hash, model versions, timestamps, and per-question scores. Re-running the same config on the same hardware gets within ±2%.

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
# UNISON_LOCAL_EVAL_WORKSPACE_ID in the brain server's .env — no JWT needed.

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

- **Same production agent loop.** `unison-agent` calls `/api/rest/agents/eval-turn` — the same `runAgent` loop that ships in production. No eval-only forks in the answer path.
- **New brain-context contract.** `unison-brain-context` reflects the post-restructure server: provision → seed via `/v1/eval/seed` → `GET /v1/brain/context` → reader LLM answers. Retrieval and generation fully decoupled.
- **Fixed model + temperature.** Judge model pinned per release (`JUDGE_MODEL` env var). Temperature=0 where possible.
- **Fixed dataset versions.** Downloaded from HuggingFace at a pinned commit hash, cached locally.
- **Comparators run in their preferred config.** PRs welcome to fix any accidental disadvantage.

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

One file, ~80 LOC. See [`src/unison_evals/memory_evals/adapters/base.py`](./src/unison_evals/memory_evals/adapters/base.py) for the contract. PRs welcome.

## License

Apache-2.0. Datasets retain their original licenses (LongMemEval is MIT, MemoryAgentBench is MIT).

---

## Star history

[<img src="https://api.star-history.com/svg?repos=unison-labs-ai/Unison-evals&type=Date" width="600" />](https://star-history.com/#unison-labs-ai/Unison-evals&Date)

If this saves you from publishing cherry-picked numbers, leave a ⭐ — it helps others find it.

---

## Part of the Unison Labs constellation

**One brain, every agent.** Every repo below reads from _and writes to_ the same [Unison brain](https://unisonlabs.ai) — no per-tool memory silos.

| Repo | What it does |
|---|---|
| [unison-brain](https://github.com/unison-labs-ai/unison-brain) | CLI · SDK · MCP server — the core |
| [claude-unison](https://github.com/unison-labs-ai/claude-unison) | Memory for Claude Code |
| [cursor-unison](https://github.com/unison-labs-ai/cursor-unison) | Memory for Cursor |
| [codex-unison](https://github.com/unison-labs-ai/codex-unison) | Memory for OpenAI Codex CLI |
| [opencode-unison](https://github.com/unison-labs-ai/opencode-unison) | Memory for OpenCode |
| [openclaw-unison](https://github.com/unison-labs-ai/openclaw-unison) | Memory for OpenClaw |
| [pipecat-unison](https://github.com/unison-labs-ai/pipecat-unison) | Memory for Pipecat voice agents |
| [python-sdk](https://github.com/unison-labs-ai/python-sdk) | Python SDK for the brain |
| [install-mcp](https://github.com/unison-labs-ai/install-mcp) | One-command MCP installer |
| [code-chunk](https://github.com/unison-labs-ai/code-chunk) | AST-aware code chunking |
| [unison-fs](https://github.com/unison-labs-ai/unison-fs) | Mount the brain as a filesystem |
| [backchannel](https://github.com/unison-labs-ai/backchannel) | Async messaging between agents |
| **[Unison-evals](https://github.com/unison-labs-ai/Unison-evals)** | **Open memory benchmark suite ← you are here** |
