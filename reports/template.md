# unison-evals Comprehensive Run Report

> Fill in the sections below after a `bash scripts/run_comprehensive.sh` run completes.
> Replace all `<...>` placeholders with real numbers from `results/comprehensive-<TS>/summary.json`.

---

## Run metadata

| Field | Value |
|---|---|
| Date | <YYYY-MM-DD> |
| Comprehensive ID | `<TS>` (e.g. `20260510T123456Z`) |
| Limit per combo | `<LIMIT>` questions |
| Judge model | `<JUDGE>` |
| Total combos run | <N> |
| Total combos done | <N_DONE> |
| Total combos failed | <N_FAILED> |
| Total skipped | <N_SKIPPED> |
| Estimated cost | $<COST> |

---

## Headline

> 1–3 sentences summarising the most important finding.

Example: "At limit=20, Unison Brain achieves Recall@10 of X% vs pgvector-naive's Y% on BitempoQA,
while claude-code reaches Z% pass rate on LongMemEval Track 2."

---

## Aggregated cross-dataset leaderboard

> Populate from `results/comprehensive-<TS>/summary.json`.
> `pass rate` = Track 2+3 mean pass rate across all datasets.
> `Recall@10` = Track 1 mean recall across all brain datasets.

| System | Mean pass rate | Mean Recall@10 | Mean $/solved | Mean p50 ms |
|---|---|---|---|---|
| unison-agent | <!-- % --> | n/a | <!-- $ --> | <!-- ms --> |
| claude-code | <!-- % --> | n/a | <!-- $ --> | <!-- ms --> |
| unison-brain | n/a | <!-- % --> | n/a | <!-- ms --> |
| pgvector-naive | n/a | <!-- % --> | n/a | <!-- ms --> |
| mem0 | n/a | <!-- % --> | n/a | <!-- ms --> |
| letta | n/a | <!-- % --> | n/a | <!-- ms --> |
| _(add rows)_ | | | | |

---

## Per-dataset breakdown

> Matrix: rows = systems, columns = datasets. Cells = headline metric (pass rate or Recall@10).
> Write "n/a" where a system was skipped for a dataset.

| System | bitempoqa | longmemeval | memoryagentbench | musique | frames | msmarco |
|---|---|---|---|---|---|---|
| unison-agent | | | | | | n/a |
| claude-code | | | | | | n/a |
| unison-brain | | | | n/a | n/a | |
| pgvector-naive | | | | n/a | n/a | |
| mem0 | | | | n/a | n/a | n/a |
| letta | | | | n/a | n/a | n/a |
| _(add rows)_ | | | | | | |

---

## Per-track summary

### Track 1 — Brain only (Recall@10)

| System | bitempoqa | longmemeval | memoryagentbench | msmarco |
|---|---|---|---|---|
| unison-brain | | | | |
| pgvector-naive | | | | |
| mem0 | | | | |
| letta | | | | |

### Track 2 — Agent oracle (pass rate)

| System | bitempoqa | longmemeval | memoryagentbench | musique | frames |
|---|---|---|---|---|---|
| unison-agent | | | | | |
| claude-code | | | | | |
| mem0-agent | | | | | |
| anthropic-raw | | | | | |

### Track 3 — Agent E2E (pass rate)

| System | bitempoqa | longmemeval | musique | frames |
|---|---|---|---|---|
| unison-agent | | | | |
| claude-code | | | | |
| mem0-agent | | | | |

---

## Cost summary

| System | Total cost ($) | Cost per question ($) | Cost per solved task ($) |
|---|---|---|---|
| unison-agent | | | |
| claude-code | | | |
| _(add rows)_ | | | |

---

## Methodology

See [METHODOLOGY.md](../METHODOLOGY.md) for:
- Per-metric formulas (Recall@10, nDCG@10, MRR, pass rate)
- Dataset hashes + HuggingFace IDs
- Judge prompt and temperature
- Hardware spec
- Reproduction instructions

---

## Raw output

Results directory: `results/comprehensive-<TS>/`

Each combo is a file like `brain-bitempoqa-pgvector-naive.json` containing:
```json
{
  "summary": { ... },
  "results": [ ... ],
  "exported_at": "..."
}
```

Aggregated summary: `results/comprehensive-<TS>/summary.json`
