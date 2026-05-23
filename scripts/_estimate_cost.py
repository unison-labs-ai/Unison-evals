#!/usr/bin/env python3
"""_estimate_cost.py — print a per-(track,dataset,system,limit,judge) cost estimate.

Called by run_comprehensive.sh to gate runs > $20.

Usage:
    python scripts/_estimate_cost.py --limit 20 --judge claude-haiku-4-5 \\
        --tracks brain agent together --datasets bitempoqa longmemeval \\
        --systems unison-brain pgvector-naive mem0 letta zep \\
                  unison-agent claude-code codex gemini-cli mem0-agent \\
                  anthropic-raw openai-gpt5 google-gemini

Exit code 1 if total > $20 and --check flag is set (used by the shell gate).
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Cost table: cost_per_question_usd[track][system]
# Values are rough per-question estimates (adapter API cost + judge cost).
# Judge cost is separate and added on top for agent tracks.
# ---------------------------------------------------------------------------

BRAIN_COST_PER_Q: dict[str, float] = {
    "unison-brain": 0.001,       # tRPC call, no LLM
    "pgvector-naive": 0.001,     # embeddings only (text-embedding-3-small ~$0.00002/token)
    "mem0": 0.50,                # cloud per-op pricing
    "letta": 0.25,               # cloud API per call
    "zep": 0.50,                 # cloud per-op pricing
}

AGENT_COST_PER_Q: dict[str, float] = {
    "unison-agent": 0.033,
    "claude-code": 0.033,
    "codex": 0.033,
    "gemini-cli": 0.010,
    "mem0-agent": 0.033,
    "anthropic-raw": 0.033,
    "openai-gpt5": 0.050,
    "google-gemini": 0.010,
}

# Together track (E2E) = brain ingest + agent call
TOGETHER_COST_PER_Q: dict[str, float] = {
    s: AGENT_COST_PER_Q.get(s, 0.05) + 0.001
    for s in list(AGENT_COST_PER_Q) + ["unison-agent"]
}

JUDGE_COST_PER_Q: dict[str, float] = {
    "claude-haiku-4-5": 0.0006,
    "claude-haiku-4-0": 0.0006,
    "claude-opus-4-5-20250101": 0.015,
    "claude-opus-4-0": 0.015,
    "claude-sonnet-4-5": 0.003,
}
_DEFAULT_JUDGE_COST = 0.003  # fallback for unknown judge


def cost_for_combo(
    track: str,
    system: str,
    limit: int,
    judge: str,
) -> float:
    """Return estimated USD cost for one (track, dataset, system) combo at `limit` questions."""
    if track == "brain":
        per_q = BRAIN_COST_PER_Q.get(system, 0.10)
        judge_per_q = 0.0  # no judge in brain track
    elif track == "agent":
        per_q = AGENT_COST_PER_Q.get(system, 0.033)
        judge_per_q = JUDGE_COST_PER_Q.get(judge, _DEFAULT_JUDGE_COST)
    else:  # together / E2E
        per_q = TOGETHER_COST_PER_Q.get(system, 0.05)
        judge_per_q = JUDGE_COST_PER_Q.get(judge, _DEFAULT_JUDGE_COST)

    return (per_q + judge_per_q) * limit


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate comprehensive run cost.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--judge", default="claude-haiku-4-5")
    parser.add_argument("--tracks", nargs="+", default=["brain", "agent", "together"])
    parser.add_argument("--datasets", nargs="+", default=[
        "bitempoqa", "longmemeval", "memoryagentbench", "musique", "frames", "msmarco"
    ])
    parser.add_argument("--brain-systems", nargs="+", dest="brain_systems", default=[
        "unison-brain", "pgvector-naive", "mem0", "letta", "zep"
    ])
    parser.add_argument("--agent-systems", nargs="+", dest="agent_systems", default=[
        "unison-agent", "claude-code", "codex", "gemini-cli", "mem0-agent",
        "anthropic-raw", "openai-gpt5", "google-gemini"
    ])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if total > $20 (used by the shell budget gate).",
    )
    parser.add_argument("--threshold", type=float, default=20.0)
    args = parser.parse_args()

    # Datasets per track (not all datasets work with all tracks).
    # brain-track datasets:  bitempoqa, longmemeval, memoryagentbench, msmarco
    # agent-track datasets:  bitempoqa, longmemeval, memoryagentbench, musique, frames
    # together-track datasets: bitempoqa, longmemeval, musique, frames
    brain_datasets = [
        d for d in args.datasets
        if d in {"bitempoqa", "longmemeval", "memoryagentbench", "msmarco", "musique"}
    ]
    agent_datasets = [
        d for d in args.datasets
        if d in {"bitempoqa", "longmemeval", "memoryagentbench", "musique", "frames"}
    ]
    together_datasets = [
        d for d in args.datasets
        if d in {"bitempoqa", "longmemeval", "musique", "frames"}
    ]

    rows: list[tuple[str, str, str, float]] = []

    for track in args.tracks:
        if track == "brain":
            ds_list = brain_datasets
            sys_list = args.brain_systems
        elif track == "agent":
            ds_list = agent_datasets
            sys_list = args.agent_systems
        else:
            ds_list = together_datasets
            sys_list = args.agent_systems

        for dataset in ds_list:
            for system in sys_list:
                c = cost_for_combo(track, system, args.limit, args.judge)
                rows.append((track, dataset, system, c))

    total = sum(r[3] for r in rows)

    # Print table
    print(f"{'Track':<12} {'Dataset':<22} {'System':<18} {'Est. cost':>10}")
    print("-" * 68)
    for track, dataset, system, cost in sorted(rows, key=lambda r: -r[3]):
        print(f"{track:<12} {dataset:<22} {system:<18} ${cost:>9.4f}")
    print("-" * 68)
    print(f"{'TOTAL':<54} ${total:>9.4f}")
    print()
    print(f"Limit={args.limit}  Judge={args.judge}  Threshold=${args.threshold:.2f}")

    if args.check and total > args.threshold:
        print(
            f"\n[BUDGET] Estimated cost ${total:.2f} > ${args.threshold:.2f} threshold.\n"
            "Set CONFIRM=1 to proceed, or reduce LIMIT / scope.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
