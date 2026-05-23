# Contributing to unison-evals

Thanks for helping improve the benchmark. The most valuable contributions are
**new adapters** (so another system can be compared on equal terms) and fixes that
correct any disadvantage we created for a system by accident.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
make lint     # ruff check + format check
make test     # pytest
```

CI runs exactly these, plus a `typecheck` + `lint` on the `web/` leaderboard UI.

## Adding a system (adapter)

Each system implements one small adapter (~80 LOC):

- Memory / single-question QA systems → `src/unison_evals/memory_evals/adapters/`
- Task-shaped, multi-turn benchmarks → `src/unison_evals/benchmarks/<bench>/`

Register it in the adapter registry, point it at your API/CLI, and run:

```bash
uv run unison-evals run --systems <your-system> --dataset longmemeval --track agent-oracle --limit 5
```

See an existing adapter (`anthropic_raw.py`, `mem0.py`) for the contract. Configure
each system per its own docs — if we set yours up sub-optimally, that's a bug; open
a PR.

## Adding a dataset

Implement a loader under `src/unison_evals/memory_evals/datasets/` that returns
`Question` / `BrainQuestion` objects, register it, and ship a small **embedded smoke
sample** so offline and CI runs work without network.

## Submitting results

Every run writes a reproducible JSON artifact (dataset hash, model versions,
per-question scores). To submit numbers, open a PR with the artifact plus the exact
command and config. We publish results in full — wins and losses alike.

## Conventions

- `make lint` and `make test` must pass; CI enforces them.
- Conventional-commit messages (`feat:`, `fix:`, `docs:`).
- Pin judge model + dataset versions for any reported number.
- Be honest about cost and latency; never hand-tune a competitor into a worse config.
