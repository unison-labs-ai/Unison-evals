#!/usr/bin/env python3
"""_estimate_cost.py — print a per-(track,dataset,system,limit,judge) cost estimate.

Called by run_comprehensive.sh to gate runs > $20.

Usage:
    python scripts/_estimate_cost.py --limit 20 --judge gpt-4o-2024-08-06 \\
        --tracks agent together --datasets longmemeval memoryagentbench \\
        --agent-systems unison-agent unison-agent-pipeline

Exit code 1 if total > $20 and --check flag is set (used by the shell budget gate).
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Cost table: cost_per_question_usd[track][system]
# Values are rough per-question estimates (adapter API cost + judge cost).
# Judge cost is separate and added on top for agent tracks.
# ---------------------------------------------------------------------------

AGENT_COST_PER_Q: dict[str, float] = {
    "unison-agent": 0.033,
    "unison-agent-pipeline": 0.033,
}

# Together track (E2E) = brain ingest + agent call
TOGETHER_COST_PER_Q: dict[str, float] = {
    s: AGENT_COST_PER_Q.get(s, 0.05) + 0.001
    for s in AGENT_COST_PER_Q
}

JUDGE_COST_PER_Q: dict[str, float] = {
    "claude-haiku-4-5": 0.0006,
    "claude-haiku-4-0": 0.0006,
    "claude-opus-4-5-20250101": 0.015,
    "claude-opus-4-0": 0.015,
    "claude-sonnet-4-5": 0.003,
    "gpt-4o-2024-08-06": 0.005,
}
_DEFAULT_JUDGE_COST = 0.003  # fallback for unknown judge


def cost_for_combo(
    track: str,
    system: str,
    limit: int,
    judge: str,
) -> float:
    """Return estimated USD cost for one (track, dataset, system) combo at `limit` questions."""
    if track == "agent":
        per_q = AGENT_COST_PER_Q.get(system, 0.033)
        judge_per_q = JUDGE_COST_PER_Q.get(judge, _DEFAULT_JUDGE_COST)
    else:  # together / E2E
        per_q = TOGETHER_COST_PER_Q.get(system, 0.05)
        judge_per_q = JUDGE_COST_PER_Q.get(judge, _DEFAULT_JUDGE_COST)

    return (per_q + judge_per_q) * limit


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate comprehensive run cost.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--judge", default="gpt-4o-2024-08-06")
    parser.add_argument("--tracks", nargs="+", default=["agent", "together"])
    parser.add_argument("--datasets", nargs="+", default=[
        "longmemeval", "memoryagentbench"
    ])
    parser.add_argument("--agent-systems", nargs="+", dest="agent_systems", default=[
        "unison-agent", "unison-agent-pipeline"
    ])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if total > threshold (used by the shell budget gate).",
    )
    parser.add_argument("--threshold", type=float, default=20.0)
    args = parser.parse_args()

    rows: list[tuple[str, str, str, float]] = []

    for track in args.tracks:
        ds_list = args.datasets
        sys_list = args.agent_systems

        for dataset in ds_list:
            for system in sys_list:
                c = cost_for_combo(track, system, args.limit, args.judge)
                rows.append((track, dataset, system, c))

    total = sum(r[3] for r in rows)

    # Print table
    print(f"{'Track':<12} {'Dataset':<22} {'System':<22} {'Est. cost':>10}")
    print("-" * 72)
    for track, dataset, system, cost in sorted(rows, key=lambda r: -r[3]):
        print(f"{track:<12} {dataset:<22} {system:<22} ${cost:>9.4f}")
    print("-" * 72)
    print(f"{'TOTAL':<58} ${total:>9.4f}")
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
