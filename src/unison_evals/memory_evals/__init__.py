"""Memory-evals — single-question Q&A evaluation framework.

A fixed dataset of questions, each with a per-question corpus, run through
one adapter at a time, scored by an LLM judge. Two tracks:

  Track 2 (agent-oracle) — agent given gold context; isolates reasoning quality.
  Track 3 (agent-e2e)    — agent ingests per-question corpus and retrieves; full pipeline.

For task-shaped, multi-turn, end-state-scored evaluations (e.g. τ-bench
retail, Context-Bench filesystem), see `unison_evals.benchmarks`.
"""
