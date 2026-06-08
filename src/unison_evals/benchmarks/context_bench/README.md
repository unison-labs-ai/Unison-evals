# context_bench/

Adapter for [Letta's Context-Bench Filesystem Suite](https://www.letta.com/blog/context-bench) — multi-hop entity navigation across 10 fictional text files. Letta's published thesis is that filesystem-as-memory beats specialized memory stores, which makes this **the closest direct architectural test of Unison's "everything is a .md filesystem" claim** available publicly.

Upstream pinned at `vendor/letta-evals/` (git submodule).

## What it measures

Letta's harness gives the agent two tools — `open_files` (read) and `grep_files` (search) — over 10 plain-text data files about fictional people and their entities (pets, vehicles, addresses, bank accounts, credit cards, employments, insurance policies, internet accounts, medical records). Questions require **joining across multiple files via shared `person_id` keys**:

> *"Among all people who live in the same state as the owner of the pet named 'Dawn', who owns the most vehicles? If there's a tie, who among them is the oldest?"*

Scoring: LLM-as-judge (`gpt-5-mini`, temperature 0) using `vendor/letta-evals/.../rubric.txt`. Outputs are scored 0.0 / 0.5 / 1.0 per the published rubric:
- 1.0 — answer matches ground truth (with equivalence rules for case / number formats / minor phrasing)
- 0.5 — explicit refusal without attempting the task
- 0.0 — wrong answer or missed the key value

## Why this benchmark, in addition to τ-bench

τ-bench retail is rigid CRUD inside a fixed verb registry — the regime where typed function-calls beat bash by 44pp on our paired ablation. Context-Bench is multi-hop entity reasoning where the agent **composes primitives** (cat / grep / sort / cross-reference) the API designer didn't anticipate. If Unison's interface helps anywhere, it should help here.

This is the publishable architectural test:

| Cell | Tool surface | Score |
|---|---|---|
| Letta agent + GPT-5.2-codex | `open_files` + `grep_files` (two purpose-built tools) | **93%** (leaderboard.letta.com, 2026-03-13) |
| Letta agent + Sonnet 4.6 | same | **88%** |
| **Unison + Sonnet 4.5** | **single `bash` over `/private/sources/eval/context-bench/*.md`** | **measure here** |

Same model. Same corpus. Same dataset. Same judge. Same rubric. Only the **interface** differs.

## What we built

```
context_bench/
├── README.md          ← this file
├── __init__.py
├── seed.py            ← preload the 10 corpus files into /private/sources/eval/context-bench/ + a SCHEMA.md
├── target.py          ← UnisonContextBenchTarget — one-shot Q&A via /eval-turn
├── judge.py           ← LLM-judge using Letta's vendored rubric.txt
└── run.py             ← CLI entry; loads dataset, runs target, grades, writes summary
```

Per-row protocol:
1. Wipe + reseed the eval tenant brain (10 .txt files + a generic SCHEMA.md).
2. POST the question to `/api/rest/agents/eval-turn` with `memoryMode: "fresh"` (no Memory-v2 carryover across rows).
3. Capture the agent's final text answer.
4. Grade with Letta's exact rubric + judge model.
5. Persist per-row JSON immediately so a crash doesn't waste prior rows.

Reuses the τ-bench `brain_client.py` for seed / wipe — no duplication.

## How to run

```bash
# Smoke (3 rows, ~5 min, < $0.50)
.venv/bin/python -m unison_evals.benchmarks.context_bench.run -n 3

# Larger paired n
.venv/bin/python -m unison_evals.benchmarks.context_bench.run -n 25

# Full dataset
.venv/bin/python -m unison_evals.benchmarks.context_bench.run --task-ids all

# Specific rows
.venv/bin/python -m unison_evals.benchmarks.context_bench.run --task-ids 0,5,10

# Override agent or judge
.venv/bin/python -m unison_evals.benchmarks.context_bench.run -n 5 --model claude-sonnet-4-6
.venv/bin/python -m unison_evals.benchmarks.context_bench.run -n 5 --judge-model gpt-4o-mini
```

Prerequisites:
- Unison API up at `$UNISON_API_URL` (default `http://localhost:3001`)
- Eval tenant provisioned (see your Unison server `.env`)
- `OPENAI_API_KEY` (judge + reading the dataset)
- `ANTHROPIC_API_KEY` (Sonnet agent inside Unison)
- `git submodule update --init vendor/letta-evals` (corpus + dataset + rubric)

## Where results land

```
results/context-bench/unison/<model>/
├── summary.json       ← aggregated score, paired with Letta's published cell
└── row-<NNN>.json     ← per-row: question, GT, agent answer, judge score, costs
```

## Costs & timing (rough, per row)

| Component | Per row |
|---|---|
| Agent (Sonnet 4.5, Unison) | ~$0.05-0.10 (10-30 bash calls × ~2K tokens) |
| Judge (`gpt-5-mini`) | ~$0.002 |
| Wall time | ~30-90s |

For n=25: ~$2-3, ~30-45 min. For full dataset: depends on dataset size (vendored jsonl has the count).

## Honest reading

If Unison + Sonnet 4.5 **matches or beats** Letta agent + Sonnet 4.5 (74.0%), the architectural claim — *bash over .md is at least as good as purpose-built filesystem tools for entity-navigation tasks* — has direct empirical support on a public benchmark a competitor built.

If Unison **loses by < 5pp**, the interface is competitive but specialized tools edge it out. Still publishable.

If Unison **loses by ≥ 10pp**, the architectural claim doesn't hold even on the task class it should win on, and the next move is to find what specifically is wrong with the adapter (over-broad SCHEMA, missing skill, etc.) before drawing the deeper conclusion.

## Hard constraints (per the project's benchmarks/ rules)

- This adapter never modifies Unison. All glue lives in this directory.
- The SCHEMA.md we seed describes file layout only — no strategic recipes, no bash examples, no "always do X". Same fairness boundary as τ-bench Mode B.
- Letta's rubric, dataset, and corpus are vendored verbatim (`vendor/letta-evals/`, pinned commit).
