"""Memory-evals — single-question Q&A evaluation framework.

The original style of evaluation in this harness: a fixed dataset of
questions, each with a per-question corpus, run through one adapter at
a time, scored by an LLM judge or retrieval metric. Tracks 1/2/3
isolate brain-only retrieval, agent-oracle reasoning, and agent+brain
end-to-end performance.

For task-shaped, multi-turn, end-state-scored evaluations (e.g. τ-bench
retail, Context-Bench filesystem), see `unison_evals.benchmarks`.
"""
