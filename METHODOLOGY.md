# unison-evals — Methodology

> Version: v1.0+ canonical reference  
> Last updated: 2026-05-10

---

## 1. Overview

unison-evals is a public benchmark harness that measures memory/workspace agent
and brain-retrieval systems on tasks that require persistent recall, multi-hop
reasoning over stored knowledge, and bitemporal fact correctness. Every system
implements a single adapter (~80 LOC), points at its production API or CLI, and
is scored against the same datasets with identical metrics under identical
constraints. The harness produces JSON artifacts with every run — dataset hash,
model versions, per-question scores, latency measurements — so any result can be
reproduced or independently challenged.

**What this benchmark measures:** retrieval ranking quality (Track 1), reasoning
quality given perfect context (Track 2), and end-to-end quality including
retrieval (Track 3). **What it does not measure:** coding agent skill (see
SWE-bench, Terminal-Bench), browser-use agent skill (see WebArena, OSWorld), or
any task unrelated to memory and knowledge management. Running a coding agent on
this benchmark would be a category mistake; running Unison on SWE-bench would be
a different category mistake. This benchmark is for teams building or buying
memory/workspace agents who want a principled comparison.

**Intended audience:** systems researchers, product teams, and technically
sophisticated evaluators who want reproducible numbers, not a press-release
score.

---

## 2. Two Tracks

### Track 2 — Agent oracle (reasoning quality only)

**What it isolates:** the agent's reasoning and generation quality, completely
decoupled from retrieval. The gold context (oracle_context) is handed to the
agent directly — the brain/FS/workspace tools are disabled server-side for
Unison, and the oracle context is injected into the prompt for Claude Code and
Mem0-agent. If an agent fails in oracle mode, the failure is in reasoning, not
retrieval.

**Runner lifecycle** (`src/unison_evals/memory_evals/runners/agent_oracle.py`):

1. `adapter.setup()` — establish connection / confirm binary present
2. For each (question, system) pair in bounded-concurrency asyncio:
   a. `adapter.answer(q.question, oracle_context=q.oracle_context)`
   b. If the adapter returns an error, score is 0.0 (no judge call)
   c. Otherwise: `LLMJudge.judge(q.question, q.expected_answer, answer.answer)`
   d. Record `passed = score >= pass_threshold`, cost, latency
3. `adapter.teardown()`
4. Aggregate: pass rate, cost per solved task, p50/p95/p99 latency per system

**Why isolation matters:** oracle mode is the upper-bound for what the system
could achieve with perfect retrieval. If oracle pass rate is low, no retrieval
improvement can fix it — the reasoning gap is the bottleneck.

### Track 3 — Agent end-to-end (what users actually experience)

**What it isolates:** the full pipeline — ingest, retrieve, reason, generate —
under identical conditions for every system.

**Runner lifecycle** (`src/unison_evals/memory_evals/runners/agent_e2e.py`):

1. `adapter.setup()` for each system (adapters whose `setup()` raises are skipped)
2. For each (question, system) pair: call `adapter.answer(q.query, seed_docs=q.corpus)` — the adapter ingests the per-question corpus, retrieves, and answers
3. Score answer with the LLM judge
4. Report: pass rate + cost + latency

**Status:** Track 3 is fully runnable for `unison-agent` and `unison-agent-pipeline`.

## 3. Metrics — Exact Formulas

Track 2 and Track 3 metrics are implemented in
`src/unison_evals/memory_evals/runners/agent_oracle.py` and
`src/unison_evals/memory_evals/runners/agent_e2e.py` (`_build_summary`).

### Pass rate

```
pass_rate = n_passed / n_questions

where n_passed = count(questions where judge_score >= pass_threshold)
```

Default `pass_threshold = 1.0` (strict — only a judge score of 1.0 counts).
`pass_threshold = 0.5` is acceptable for partial-credit datasets where partial
answers are meaningfully better than wrong answers.

### Cost per solved task

```
cost_per_solved_usd = total_cost_usd / n_passed
```

`None` when `n_passed == 0` (avoids division by zero and infinite numbers in the
leaderboard). `total_cost_usd` includes both adapter cost (LLM calls the system
made) and judge cost. Adapter cost for Unison is returned by the server; for
Claude Code it is estimated from token counts using Anthropic pricing at the
time of the run.

Implementation: `SystemSummary.cost_per_solved_usd` in
`src/unison_evals/memory_evals/runners/agent_oracle.py` (`_build_summary`).

### Latency

Wall-clock time from the start of the adapter call to the return of the answer.
Measured in milliseconds inside each adapter using `time.perf_counter()`.

**Never report only the mean.** The harness reports:
- `p50_latency_ms` — median (50th percentile)
- `p95_latency_ms` — 95th percentile (tail latency budget)
- `avg_latency_ms` — arithmetic mean (included for completeness; do not use for
  SLA discussions)

p99 is not computed in Track 1/2 summaries in v1.0 because sample sizes
(50–500) make p99 statistically unreliable. p99 is available from the raw
per-question result JSON for large runs.

Implementation: `_percentile()` helper in both runner files (linear
interpolation, not nearest-rank).

---

## 4. LLM Judge

### Configuration

| Parameter | Default | Override |
|---|---|---|
| Model | `claude-opus-4-5-20250101` | `JUDGE_MODEL` env var |
| Temperature | 0 | not configurable |
| Max tokens | 300 | not configurable |
| Pass threshold | 1.0 | `pass_threshold` arg to `LLMJudge()` |

Implementation: `src/unison_evals/metrics/llm_judge.py`.

Cheaper CI alternative: `claude-haiku-4-5` costs ~0.0005 USD/question vs.
~0.005 USD/question for Opus — 10x cheaper, slightly noisier (see Known
Limitations below). Published numbers always use Opus.

### Exact judge prompt

The following is the verbatim `JUDGE_PROMPT` constant from
`src/unison_evals/metrics/llm_judge.py`, lines 42–66:

```
You are evaluating whether an AI agent's answer correctly answers a question, given the expected answer.

Score:
- 1.0 (correct): captures the key information from the expected answer; minor wording differences and extra correct context are fine
- 0.5 (partial): some overlap with expected answer but missing key info, or includes material errors mixed with correct content
- 0.0 (incorrect): wrong, missing, contradictory, or non-responsive

Be strict but fair. A different correct phrasing of the same fact is fully correct (1.0). A correct fact with extra fabricated detail is partial (0.5). A confidently-stated wrong fact is incorrect (0.0).

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

AGENT'S ANSWER:
{actual}

Respond with ONLY a JSON object on a single line, no markdown:
{"score": 0.0|0.5|1.0, "confidence": 0.0-1.0, "reasoning": "one short sentence"}
```

The harness parses the JSON response with a tolerant parser that strips markdown
fences and finds the first `{...}` block if the model adds surrounding text.

### Scoring semantics

- `score = 1.0` — correct: the key information is present; wording differences
  and additional correct context do not penalise
- `score = 0.5` — partial: some overlap with the expected answer, but missing
  key information or contains material errors mixed with correct content
- `score = 0.0` — incorrect: wrong, missing, contradictory, or non-responsive

An empty-string answer skips the judge call and is scored 0.0 immediately.

### Known limitation: judge non-determinism

Even at temperature=0, LLM judges exhibit ~1–2% per-question noise across
independent runs on the same inputs. This is documented across the LLM-judge
literature (see Zheng et al., "Judging LLM-as-a-Judge with MT-Bench", NeurIPS
2023). Mitigations: (a) run 3 independent full-dataset passes and report median
± IQR; (b) for borderline cases (score oscillating 0.5/1.0), examine the
`confidence` field and the `reasoning` string in the per-question JSON artifact.

---

## 5. Datasets

### LongMemEval

| Field | Value |
|---|---|
| Paper | "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" |
| arXiv | 2410.10813 |
| Venue | ICLR 2025 |
| HuggingFace | `xiaowu0162/longmemeval` |
| License | MIT |
| Size | 500 questions across 5 memory ability types |
| Splits used | `longmemeval_oracle` (default), `longmemeval_s` |

LongMemEval presents agents with a haystack of up to 53 conversation sessions
and asks questions spanning five memory abilities: single-session user facts,
multi-session synthesis, temporal reasoning, knowledge update (a fact changed
over time), and abstention (the answer was never stated). The `longmemeval_oracle`
split strips distractor sessions to give Track 2 a cleaner reasoning signal; the
`longmemeval_s` split is the smallest subset for fast iteration.

Loader specifics (`src/unison_evals/memory_evals/datasets/longmemeval.py`): The haystack
sessions are flattened into a single `oracle_context` string with dated
`## Session N — YYYY-MM-DD` headers. For Track 3, each session is ingested as a
separate document. If the HuggingFace download fails, the loader falls back to a
3-question embedded smoke set so CI runs offline without surprise.

```bibtex
@inproceedings{wu2025longmemeval,
  title     = {LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author    = {Wu, Di and He, Hongwei and Liu, Wenhao and Han, Weijian and Ma, Jiacheng and Yu, Dong},
  booktitle = {ICLR 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2410.10813}
}
```

---

### MemoryAgentBench

| Field | Value |
|---|---|
| Paper | "MemoryAgentBench: A Comprehensive Benchmark for Memory-Augmented LLM Agents" |
| arXiv | 2507.05257 |
| Venue | ICLR 2026 |
| HuggingFace | `ai-hyz/MemoryAgentBench` |
| License | MIT |
| Size | 146 trajectory rows → ~10K Q/A pairs across 4 splits |
| Splits | `Accurate_Retrieval` (22 rows), `Test_Time_Learning` (6), `Long_Range_Understanding` (110), `Conflict_Resolution` (8) |

MemoryAgentBench uses an "inject once, query multiple times" design: one
multi-turn trajectory context yields 60–100 Q/A pairs. The four memory ability
splits test: (1) looking up specific facts from a trajectory, (2) incorporating
new skills or facts encountered mid-session, (3) synthesising across many turns
in a long context, and (4) resolving contradictory information over time. The
Conflict Resolution split is the most discriminating — SOTA scores approximately
6% on it as of 2026-05. Systems with bitemporal models (Unison) have a
structural advantage there because they can track which version of a fact was
most recently asserted.

Loader specifics (`src/unison_evals/memory_evals/datasets/memoryagentbench.py`): Each HF row
is exploded into one `Question` per Q/A pair. The full trajectory is formatted as
`oracle_context` under a `## Trajectory` header. The `memory_ability` field is
preserved in question metadata for per-ability breakdowns. Note: the canonical
HF id `ai-hyz/MemoryAgentBench` was verified on 2026-05-10 against the paper's
GitHub repository; if the authors publish a different canonical id, the loader's
`HF_DATASET` constant should be updated and results re-pinned.

```bibtex
@inproceedings{he2026memoryagentbench,
  title     = {MemoryAgentBench: A Comprehensive Benchmark for Memory-Augmented LLM Agents},
  author    = {He, Yizhu and others},
  booktitle = {ICLR 2026},
  year      = {2026},
  url       = {https://arxiv.org/abs/2507.05257}
}
```

---

### Context-Bench

Context-Bench is the Letta leaderboard's filesystem entity navigation benchmark.
The agent is given a structured filesystem of entity pages and must navigate,
read, and synthesise them to answer multi-step questions. Scored on the final
natural-language answer using the Letta rubric judge.

Loader specifics (`src/unison_evals/benchmarks/context_bench/`): The benchmark
runs via the dedicated `run_context_bench()` function (not the memory-evals runner);
it creates fresh per-run sessions and judges with the Letta-parity judge
(`gpt-5-mini` for publishable runs). Always invoked when `--dataset context-bench`
is passed to the CLI.

---

## 6. Adapters

### Agent adapters (Track 2 / Track 3)

**unison-agent** (`src/unison_evals/memory_evals/adapters/unison_agent.py`): Posts to
`/api/rest/agents/eval-turn` — the production endpoint, bit-for-bit. Track 3
calls with no `oracleContext`, so the full brain + FS + workspace toolchain runs.
Track 2 passes `oracleContext` to the endpoint, which disables brain/FS/workspace
tools server-side so the agent reasons from the injected context only. This means
Track 2 and Track 3 use the same binary; the `oracleContext` flag is a supported
production knob, not a test-mode bypass. Requires `UNISON_EVAL_SECRET` or
`UNISON_JWT` and the Unison server running.

**unison-agent-pipeline** (variant of `unison-agent`): Same endpoint, different
ingestion pipeline configuration. Used for A/B comparisons of brain pipeline
variants.

---

## 7. Reproduction

### Environment

- **Python:** 3.12, managed via uv (see `.python-version`)
- **Dependencies:** `uv sync` (installs from `uv.lock`, pinned)
- **Hardware:** any modern Mac/Linux with 16 GB RAM handles smoke sets and
  limit-20 runs. Full LongMemEval (500 q) needs 32 GB if the HuggingFace cache
  is being built; subsequent runs use the cache and fit in 16 GB.
- **Network:** stable enough for HuggingFace dataset download (~few hundred MB,
  cached under `~/.cache/unison-evals/hf` after first run)

### Required API keys

| Key | Required for |
|---|---|
| `ANTHROPIC_API_KEY` | LLM judge (all Track 2 / Track 3 runs) |
| `UNISON_JWT` | `unison-agent` + `unison-agent-pipeline` adapters |

Copy `.env.example` to `.env` and fill in the keys you need for the systems
you're benchmarking.

### Step-by-step reproduction

```bash
# 1. Clone
git clone https://github.com/Unison-Workspace/Unison-evals.git
cd Unison-evals

# 2. Install
uv sync                          # creates .venv, installs all deps

# 3. Configure
cp .env.example .env
$EDITOR .env                     # set ANTHROPIC_API_KEY at minimum

# 4. Smoke test — offline, no API keys needed beyond ANTHROPIC_API_KEY
uv run unison-evals run \
  --systems unison-agent \
  --dataset longmemeval \
  --track agent-oracle \
  --limit 3

# 5. Full agent-oracle run — LongMemEval (requires UNISON_JWT + ANTHROPIC_API_KEY)
uv run unison-evals run \
  --systems unison-agent \
  --dataset longmemeval \
  --track agent-oracle \
  --limit 500

# 6. agent-e2e run — LongMemEval (requires UNISON_JWT + ANTHROPIC_API_KEY)
uv run unison-evals run \
  --systems unison-agent \
  --dataset longmemeval \
  --track agent-e2e \
  --limit 50

# 7. Run all tests
uv run pytest

# 8. Lint check
uv run ruff check src tests
```

### Makefile targets

```bash
make smoke       # limit=3, unison-agent, longmemeval, no judge (fast CI check)
make test        # pytest
make lint        # ruff check
make typecheck   # pyright
```

### Verifying a published number

Every run writes a JSON artifact to `results/`. The artifact includes:
- `dataset_name`, `hf_dataset_id`, `hf_commit_hash` (when available)
- `judge_model`, `pass_threshold`
- `systems` with their adapter names
- `started_at`, `finished_at` (UTC)
- Per-question results: `question_id`, `answer`, `judge_score`, `judge_reasoning`,
  `latency_ms`, `cost_usd`

To verify a published number: re-run with the same config on the same dataset
version and compare the JSON artifacts. Differences beyond ±2% (the documented
LLM-judge noise band) indicate a genuine discrepancy worth investigating.

---

## 8. Variance and Stability

### Sample size

**Never publish numbers from fewer than 50 questions.** The CI smoke set uses
10 questions (fast pass/fail gate — not publishable). A 10-question subset at
40% pass rate has a 95% CI of roughly ±25 percentage points (exact depends on
distribution). At 50 questions the 95% CI tightens to ±14 pp; at 200, ±7 pp;
at 500 (full LongMemEval), ±4 pp.

| Sample size | Approx. 95% CI width at 50% pass rate |
|---|---|
| 10 | ±31 pp — do not publish |
| 50 | ±14 pp — minimum for directional claims |
| 100 | ±10 pp |
| 200 | ±7 pp |
| 500 | ±4 pp — sufficient for leaderboard |

### LLM judge noise

Even at temperature=0, independent judge runs on identical inputs differ on
approximately 1–2% of questions. This affects any metric derived from judge
scores (pass rate, cost per solved task). Mitigation: run 3 independent full
passes and report median and IQR. For the published leaderboard, runs are
averaged when multiple results are available.

### Network latency variance

Latency measurements for hosted adapters (Unison API) depend on the network
between the evaluation machine and the service's data center. Published latency
numbers are only comparable when measured from the same network as the publish.
If re-running from a different ISP or geography, report the latency numbers
separately and do not directly compare them to published numbers from another
location.

### Recommendation for publishers

- Run each configuration 3 times
- Report median pass rate and IQR across runs
- Report p50/p95 latency (not mean) per adapter, from the same network
- Pin the judge model version (`JUDGE_MODEL`) and record it in the artifact
- Pin the dataset commit hash (available in the artifact when HF provides it)

---

## 9. Known Limitations

1. **LLM judge is the cost bottleneck.** Opus costs ~$0.005/question;
   a 500-question LongMemEval run costs ~$2.50 in judge calls. Haiku is 10x
   cheaper (~$0.0005/question) but introduces slightly more noise on close calls
   (score oscillating between 0.5 and 1.0). Published numbers always use Opus.
   Haiku is acceptable for CI smoke and rapid iteration.

2. **No human evaluation baseline.** The judge is LLM-only. For domains where
   LLM judges are known to diverge from human judgement (highly technical content,
   domain-specific conventions), human spot-checks are recommended.

3. **Unison adapter runs production code.** The `unison-agent` adapter calls the
   same `/api/rest/agents/eval-turn` endpoint as a real user session. This means
   results reflect the production agent's current capabilities. If the Unison
   codebase is updated between runs, scores may change. The artifact records the
   Unison server version when available.

---

## 10. Citation

If you use unison-evals in research or a product comparison, please cite:

```bibtex
@software{unison2026evals,
  title     = {unison-evals: A Benchmark Harness for Memory and Workspace Agents},
  author    = {{Unison}},
  year      = {2026},
  url       = {https://github.com/Unison-Workspace/Unison-evals},
  license   = {Apache-2.0},
  version   = {0.1.0}
}
```

Dataset citations are listed in §5 under each dataset.

When comparing numbers across systems, cite the artifact JSON (not the paper)
so readers can verify the exact run configuration.
