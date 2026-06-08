# benchmarks/

Task-shaped multi-turn benchmarks where the agent reads, reasons, and (often) mutates a workspace, then is scored on the **final state**. This is a separate eval style from the `memory_evals/` subpackage, which scores single-question Q&A.

## Why a separate subpackage

Memory-evals (the original style):
- Fixed dataset of `(question, gold_answer)` pairs.
- Each row gets a per-question corpus.
- One adapter call → one answer → LLM-judge scores it.
- Examples: LongMemEval, MemoryAgentBench.

Task-shaped benchmarks (this subpackage):
- Multi-turn dialogue between the agent and a simulated user.
- Agent reads + (often) mutates state.
- Scored on the **final state** of the environment (hash match, rubric judge, etc.), not just the final natural-language answer.
- Examples: τ-bench (retail / airline), Letta Context-Bench, Workspace-Bench.

The two styles share infrastructure (config, types, HTTP clients) but diverge on adapter contract, scoring shape, and what they measure architecturally.

## Layout convention

```
benchmarks/
├── README.md              ← this file
├── __init__.py
├── tau_bench/             ← Sierra/Anthropic retail+airline
│   ├── README.md          ← what tau-bench is + our results
│   ├── smoke.py           ← Mode A entry (native function-calls)
│   ├── run_mode_b.py      ← Mode B entry (Unison bash+md)
│   ├── md_overlay.py
│   ├── action_translator.py
│   ├── mode_b_agent.py
│   ├── brain_client.py
│   └── __init__.py
└── context_bench/         ← Letta filesystem entity navigation
    ├── README.md
    ├── ...
    └── __init__.py
```

Per-benchmark results live at `results/<benchmark-slug>/<mode>/<model>/`. Per-task trajectories (when captured) live under that.

## How to add a new benchmark

1. Vendor its harness as a git submodule under `vendor/<bench-name>/`.
2. `mkdir src/unison_evals/benchmarks/<bench_name>/`.
3. Implement the smallest possible adapter that:
   - Drives the benchmark's harness against Unison's `/api/rest/agents/eval-turn` (or whichever Unison API surface fits).
   - **Never modifies Unison itself.** All glue lives here.
   - Reads brain state via Postgres if needed (the existing `tau_bench/brain_client.py` pattern).
4. Add a `README.md` describing: what it tests, what's measured, how to run, current results, costs.
5. Run a smoke against the smallest subset (3-5 tasks) before any full matrix.

## Hard constraints (for every benchmark adapter)

- **Don't change Unison code.** All benchmark-specific logic lives here.
- **Don't change the upstream benchmark.** Pin its commit hash in the submodule.
- **Eval-side instrumentation only.** Trajectories, dispatched actions, costs — read from Postgres or harness logs after-the-fact, never via Unison code patches.
- **Same model both sides of an ablation.** When comparing interfaces (Mode A vs Mode B), the LLM under both interfaces must be identical; the only variable is the tool surface.
