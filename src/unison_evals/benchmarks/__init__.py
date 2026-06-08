"""Task-shaped, multi-turn benchmarks scored on final environment state.

Distinct from `unison_evals.memory_evals/` (single-question Q&A
evaluations). A benchmark here typically:

  - Hands the agent a task description from a simulated user
  - Lets the agent take multiple turns to read + mutate a workspace
  - Scores the final state against ground truth (exact hash match,
    rubric judge, or similar)

Active subpackages:

  tau_bench/      — Sierra + Anthropic's customer-service benchmark.
                    Tests model reasoning within a fixed tool registry
                    (retail CRUD). Our headline numbers there are an
                    architectural ablation: same model, Mode A native
                    function-calls vs Mode B Unison bash+md.

  context_bench/  — Letta's filesystem multi-hop entity navigation
                    benchmark. Tests agentic context engineering over
                    a private prose corpus, the exact surface Unison's
                    md filesystem is designed for.

Each subpackage owns its README with: what the benchmark is, what's
been built, what's measured, current results, how to run.
"""
