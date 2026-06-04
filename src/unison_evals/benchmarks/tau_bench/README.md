# tau_bench/

Adapter for [Sierra + Anthropic's τ-bench](https://github.com/sierra-research/tau-bench) (arxiv 2406.12045). Pinned upstream commit: `59a200c` (`vendor/tau-bench/`).

## What τ-bench measures

Multi-turn customer-service tasks (retail + airline domains) where the agent:
1. Reads a simulated customer's request (drip-fed by a gpt-4o user-simulator).
2. Reads + mutates a small relational DB (orders / users / products).
3. Is scored 1.0 iff the final `env.data` hash matches ground-truth **and** every expected output string appears in some `RESPOND` message.

It's a **model-reasoning benchmark inside a fixed tool registry** — given the right verbs (`find_user_id_by_name_zip`, `get_order_details`, `exchange_delivered_order_items`, …), can the LLM (a) pick the right sequence, (b) extract correct arguments from a vague customer, (c) respect policy guards.

## What we built

Two modes that run the **same model on the same task set with the same user-simulator** — only the tool surface differs:

| Mode | Tool surface | Purpose |
|---|---|---|
| **A** (`smoke.py`) | τ-bench's native typed function-call tools | Control cell |
| **B** (`run_mode_b.py`) | Unison's single `bash` tool over a virtual `/private/taubench/<table>/<id>.md` filesystem | Treatment cell — tests Unison's "everything is a .md filesystem" thesis |

Mode B's mechanism:
1. Seed `env.data` into Unison's brain as `/private/taubench/orders/<id>.md`, `/private/taubench/users/<id>.md`, `/private/taubench/products/<id>.md` (JSON-in-fenced-block codec; see `md_overlay.py`).
2. Agent navigates + mutates via `bash` (cat / grep / sed / heredoc).
3. After each agent turn, snapshot `/private/taubench/` and diff vs the previous snapshot → translate to τ-bench `Action` calls via `action_translator.py` → run through `env.step()` so policy guards still fire.

The translator is the load-bearing piece: it must dispatch every brain mutation through `env.step()` (not direct `env.data` mutation), so policy enforcement remains identical to Mode A.

## What we measured

Headline ablation (Claude Sonnet 4.5 agent, gpt-4o user-sim, retail tasks 0–24, single trial):

| Mode | Score | Mean reward |
|---|---|---|
| **A — Sonnet + native function-calls** | **21/25** | **0.840** |
| **B — Sonnet + Unison bash+md** | **10/25** | **0.400** |
| Δ (B − A) | -11 tasks | **-44.0 pp** |

Paired McNemar contingency (same tasks, same model):

```
                  B pass    B fail
A pass               9         12     ← A wins 12 tasks
A fail               1          3     ← B wins  1 task
```

McNemar χ²(1, continuity-corrected) ≈ **7.69, p ≈ 0.02** — not chance.

## Honest interpretation

**On rigid CRUD with a constrained verb set (τ-bench retail), typed function-calls beat bash+md by 44pp on the same model.** This is the architecture under test losing on this task class — the freedom bash provides becomes wandering when the optimal procedure is short and well-defined.

**This does NOT mean** bash+md is universally worse. τ-bench measures model reasoning within a fixed tool registry; it doesn't test what Unison's interface is *for* (knowledge navigation, multi-hop retrieval, composing primitives the API designer didn't anticipate). That hypothesis is the target of `benchmarks/context_bench/`.

We keep τ-bench in the harness as the **negative-result evidence** that delineates where bash+md helps vs hurts.

## How to run

Same model both sides (the interface ablation):

```bash
# Mode A (native function-calls) — control
.venv/bin/python -m unison_evals.benchmarks.tau_bench.smoke -n 25

# Mode B (Unison bash+md) — treatment
.venv/bin/python -m unison_evals.benchmarks.tau_bench.run_mode_b -n 25
```

Useful flags either runner accepts:

| Flag | Effect |
|---|---|
| `-n 25` | Run tasks 0..24 |
| `--task-ids 0,5,10,15` | Explicit task IDs |
| `--task-ids all` | Full retail 115 tasks |
| `--model claude-sonnet-4-6` | Override agent model (provider auto-derived for claude-* / gpt-* / gemini-*) |
| `--provider anthropic` | Force litellm provider |

Mode B also needs Unison running locally with the eval tenant configured (see `~/IdeaProjects/Unison/.env`'s `UNISON_LOCAL_EVAL_TENANT_ID` / `UNISON_LOCAL_EVAL_USER_ID`).

## Where results land

```
results/tau-bench/
├── smoke/<model>/                       ← Mode A
│   ├── summary.json
│   └── <upstream-checkpoint>.json
└── smoke-mode-b/<model>/                ← Mode B
    ├── summary.json
    └── trajectories/
        └── task-<NNN>.json              ← every bash command + output Sonnet ran
```

## Cost / time (n=25 paired)

- Mode A: ~$0.70, ~30 min wall.
- Mode B: ~$1.60, ~2-3 hours wall (Unison turns are slower per-step).
- Total: ~$2.30, ~3-4h if parallel.

## Files

| File | Purpose |
|---|---|
| `smoke.py` | Mode A entry — wraps τ-bench's stock `tool_calling_agent` |
| `run_mode_b.py` | Mode B entry — drives `UnisonModeBAgent` over the task set |
| `mode_b_agent.py` | `UnisonModeBAgent(tau_bench.Agent)` — multi-turn loop driving Unison via `/api/rest/agents/eval-turn` |
| `md_overlay.py` | `env.data` ↔ `/private/taubench/<table>/<id>.md` codec (JSON-in-fenced-block, round-trip-safe for position-sensitive lists) |
| `action_translator.py` | Brain-state diff → `tau_bench.Action` dispatcher (multiset diff over `items[]`; handles cancel / exchange / return / modify variants) |
| `brain_client.py` | Postgres-direct seed / snapshot / wipe / trajectory dump against the eval tenant |
