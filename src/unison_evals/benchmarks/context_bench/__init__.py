"""Adapter for Letta's Context-Bench (Filesystem Suite).

Letta's published benchmark for "agentic context engineering" — multi-hop
entity navigation across 10 fictional text files via filesystem tools.
Tests the exact surface Unison's md filesystem is designed for.

Upstream: https://github.com/letta-ai/letta-evals
Suite:    letta-leaderboard/filesystem-agent/
Vendored: vendor/letta-evals/ (git submodule)

The Letta leaderboard cells use a Letta agent with `open_files` /
`grep_files` tools (leaderboard.letta.com: Sonnet 4.6 88%, GPT-5.2-codex
93%, as of 2026-03-13). Our adapter runs the same dataset, same rubric,
same judge model, against Unison's single `bash` tool over the corpus
seeded at `/private/sources/eval/context-bench/*.md` — the architectural
comparison.
"""
