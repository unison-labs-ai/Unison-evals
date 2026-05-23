<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright 2026 unison-evals contributors -->
<!-- License: Creative Commons Attribution 4.0 International (CC BY 4.0) -->
<!-- https://creativecommons.org/licenses/by/4.0/ -->

# BitempoQA v0

A hand-curated bitemporal question-answering dataset for evaluating memory systems
that must reason over facts with time-bounded validity. v0 ships 100 questions over
a synthetic SaaS/tech-industry corpus of 110 atomic facts.

## What this dataset tests

Most memory systems collapse all facts to "current truth" — whichever value was
written last wins. Bitemporal systems (e.g. Unison's brain, which carries
`valid_from` / `valid_to` / `supersedes` on every fact) can answer both current
and historical questions correctly because they preserve the full fact timeline.

BitempoQA probes four distinct failure modes:

| Question type | Example | Naive system | Bitemporal system |
|---|---|---|---|
| `current_truth` | "Who is Acme's CEO today?" | usually correct | correct |
| `historical_truth` | "Who was Acme's CEO on 2024-06-01?" | wrong (returns current) | correct |
| `predecessor` | "Who came before Bob as Acme's CEO?" | wrong or hallucinated | correct |
| `transition` | "When did Acme change CEO?" | wrong or missing | correct |

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
You may use, distribute, and adapt it for any purpose, including commercial use,
provided you attribute the source.

Attribution: "BitempoQA v0 — unison-evals, 2026"

## Corpus schema — `corpus.jsonl`

Each line is a JSON object representing one atomic fact:

```json
{
  "fact_id": "f001",
  "subject": "Acme Corp",
  "predicate": "ceo",
  "object": "Alice Smith",
  "valid_from": "2024-01-15",
  "valid_to": "2025-03-22",
  "supersedes": null,
  "source_id": "src001"
}
```

| Field | Type | Description |
|---|---|---|
| `fact_id` | string | Unique fact identifier, e.g. `f001` |
| `subject` | string | The entity the fact is about |
| `predicate` | string | The relationship / attribute (e.g. `ceo`, `hq_city`, `funding_series`) |
| `object` | string | The value of the attribute during the validity window |
| `valid_from` | ISO date | When this fact became true (inclusive) |
| `valid_to` | ISO date or null | When this fact ceased to be true (exclusive); null = still current |
| `supersedes` | fact_id or null | The fact_id this fact replaced (null for first assertions) |
| `source_id` | string | Groups related facts about the same entity/event |

Invariant: if `valid_to` is non-null, there must exist a successor fact for the
same (subject, predicate) pair whose `valid_from` equals this `valid_to` and whose
`supersedes` equals this `fact_id`.

## Questions schema — `questions.jsonl`

Each line is a JSON object representing one evaluation question:

```json
{
  "id": "q001",
  "question": "Who is Acme Corp's current CEO?",
  "expected_answer": "Bob Lee",
  "as_of": null,
  "fact_ids": ["f002"],
  "question_type": "current_truth",
  "difficulty": 1
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique question identifier, e.g. `q001` |
| `question` | string | The natural-language question posed to the system |
| `expected_answer` | string | The exact correct answer |
| `as_of` | ISO date or null | The temporal anchor for `historical_truth` questions; null for "now" |
| `fact_ids` | list[string] | Corpus fact_ids needed to answer correctly |
| `question_type` | enum | One of: `current_truth`, `historical_truth`, `predecessor`, `transition` |
| `difficulty` | int 1–3 | 1 = single fact, 2 = as_of filtering, 3 = multi-fact temporal reasoning |

## Methodology

All facts and company names are **synthetic and fictional**. No real-world persons,
addresses, or identifying information appears in the dataset. Companies are named in
the style of generic SaaS/tech startups (e.g. "Velox Systems", "Orbita Analytics").

Facts were hand-authored to ensure:

1. Every fact chain has a clear valid_from/valid_to structure with no gaps or
   ambiguities (a system that implements bitemporal queries correctly gets 100%)
2. Every question has exactly one correct answer derivable from the corpus
3. Supersession chains are consistent: if f002 supersedes f001, then f001.valid_to
   equals f002.valid_from exactly
4. Questions cover ≥20 distinct subjects so the dataset doesn't over-index on one
   company

Questions were authored after facts, not before — ensuring the corpus is the source
of truth and questions are a strict subset of what the corpus can answer.

## Distribution

v0 (this release):
- Corpus: 110 atomic facts across 30 subjects
- Questions: 100 questions (25 per question_type)
- Difficulty: ~38 at level 1, ~37 at level 2, ~25 at level 3

v0.2 (planned):
- Expand to 300 questions
- Add 2-3 hop transition chains
- Publish to HuggingFace dataset hub

## Versioning

v0 is the initial release. Dataset versions are pinned in eval runs via git commit
hash. Breaking schema changes require a version bump.
