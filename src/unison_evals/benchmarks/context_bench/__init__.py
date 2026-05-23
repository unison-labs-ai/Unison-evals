"""Adapter for Letta's Context-Bench (Filesystem Suite).

Letta's published benchmark for "agentic context engineering" — multi-hop
entity navigation across 10 fictional text files via filesystem tools.
Tests the exact surface Unison's md filesystem is designed for.

Upstream: https://github.com/letta-ai/letta-evals
Suite:    letta-leaderboard/filesystem-agent/
Vendored: vendor/letta-evals/ (git submodule)

The Letta leaderboard cells use a Letta agent with `open_files` /
`grep_files` tools. Sonnet 4.5 on that setup scores 74.0%. Our adapter
runs the same dataset, same rubric, same judge model, against Unison's
single `bash` tool over `/wiki/*.txt` — the architectural comparison.
"""
