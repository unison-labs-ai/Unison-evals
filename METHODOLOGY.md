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

## 2. Four Tracks

### Track 1 — Brain retrieval (retrieval quality only)

**What it isolates:** the retrieval pipeline's ranking ability, completely
decoupled from the agent's reasoning or generation. No LLM inference runs in
Track 1. The scorer compares returned document paths against gold paths from the
dataset.

**Runner lifecycle** (`src/unison_evals/runners/brain_retrieval.py`):

1. `adapter.setup()` — establish connection / load config
2. For each question in the dataset:
   a. `adapter.reset()` — clear the corpus (TRUNCATE for pgvector, UUID rotation
      for Mem0, agent deletion for Letta)
   b. `adapter.ingest(q.corpus)` — load the per-question document set
   c. `adapter.search(q.query, k=10)` — retrieve top-10 chunks
   d. Score: compute Recall@10, nDCG@10, MRR, Hit@1 against `q.gold_doc_paths`
3. `adapter.teardown()` — close connections
4. Aggregate: mean metrics across all questions, p50/p95 latency

**Why isolation matters:** retrieval quality explains approximately 60% of
end-to-end QA quality variance (arxiv 2405.07437 — "Large Language Models as
Probabilistic Retrieval Models"). A system that looks good E2E because its
generator is stronger can hide a weak retrieval layer. Track 1 makes retrieval
quality independently auditable.

### Track 2 — Agent oracle (reasoning quality only)

**What it isolates:** the agent's reasoning and generation quality, completely
decoupled from retrieval. The gold context (oracle_context) is handed to the
agent directly — the brain/FS/workspace tools are disabled server-side for
Unison, and the oracle context is injected into the prompt for Claude Code and
Mem0-agent. If an agent fails in oracle mode, the failure is in reasoning, not
retrieval.

**Runner lifecycle** (`src/unison_evals/runners/agent_oracle.py`):

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

**Runner lifecycle** (`src/unison_evals/runners/agent_e2e.py`, planned):

1. `brain_adapter.setup()` + `agent_adapter.setup()`
2. For each question: ingest corpus, call agent (no oracle context), score answer
3. Report: pass rate + faithfulness + cost + latency

**Status:** Track 3 is not yet runnable for `unison-brain`. The Unison API does
not expose a bulk-ingest endpoint suitable for an eval harness; the eval-side
ingest contract is documented in the Unison ingest API (not yet public). Track 3 is available
for `pgvector-naive` + `mem0` + `letta` against datasets that include a
per-question corpus (BitempoQA, MuSiQue).

---

### Track 4 — Scale (retrieval at realistic corpus sizes)

**What it tests:** retrieval quality at 1M-10M document scale — the size where
index choices, recency boosts, and optional reranking actually matter.

**Lifecycle** (`src/unison_evals/runners/scale_retrieval.py`):

1. (One-time setup, not part of the runner) Bulk-load a fixed reference corpus
   into the brain via `scripts/load_corpus_<dataset>.sh`. This is paid once;
   embedding cost is O(corpus size).
2. `adapter.setup()` — establish connection / load config.
3. Emit `corpus_announced` event so the UI can display the corpus label.
4. For each query in the question set, **no reset, no ingest**:
   a. `adapter.search(q.query, k=10)` — retrieve top-10 chunks from the
      pre-loaded corpus.
   b. Score: Recall@10, nDCG@10, MRR, Hit@1 against `q.gold_doc_paths`.
5. `adapter.teardown()` — close connections.
6. Aggregate: mean metrics per system, p50/p95/p99 latency. p99 is included
   (unlike Tracks 1-3) because sample sizes at scale are large enough to make
   it statistically reliable.

**Why it matters:** Tracks 1-3 use small per-question corpora (10-100 docs),
making ranking almost trivial. Track 4 stresses the brain at the corpus sizes
where index structure, approximate nearest-neighbour parameters, and retrieval
fusion strategies have measurable impact.

**Reference corpora:**

| Corpus | Passages | Dev queries | One-time cost |
|---|---|---|---|
| MS MARCO v1.1 (100k subset) | 100K | 6,980 | ~$0.40 |
| MS MARCO v1.1 (1M subset) | 1M | 6,980 | ~$4 |
| MS MARCO v1.1 (full) | 8.8M | 6,980 | ~$35 |

BEIR subsets and Cohere multilingual benchmark (10M+ vectors) are planned for
v0.3. Add new corpora by implementing `load_scale_questions()` on the dataset
class and a corresponding `scripts/load_corpus_<name>.sh` loader.

**Runner signature:**

```python
runner = ScaleRetrievalRunner(
    systems={"pgvector-naive": adapter},
    corpus_label="msmarco-passages-v1-100k",
)
async for event in runner.run(questions, dataset_name="msmarco"):
    ...
summary = await runner.run_to_completion(questions)
```

**CLI:**

```bash
# One-time corpus load (100k subset, ~$0.40, ~10 min):
SYSTEM=pgvector-naive SUBSET=100k bash scripts/load_corpus_msmarco.sh

# Run Track 4 against the pre-loaded corpus:
uv run unison-evals run \
  --systems pgvector-naive \
  --dataset msmarco \
  --track scale \
  --corpus msmarco-passages-v1-100k \
  --limit 500
```

## 3. Metrics — Exact Formulas

All retrieval metric implementations are in
`src/unison_evals/metrics/retrieval.py`. Track 2 metrics are in
`src/unison_evals/runners/agent_oracle.py` (`_build_summary`).

**Conventions in all formulas:**
- `retrieved` — ordered list of doc paths returned by the system (index 0 = rank 1)
- `gold` — set of doc paths considered relevant (ground truth)
- `k` — cut-off depth

### Recall@k

```
recall@k = |retrieved[:k] ∩ gold| / |gold|
```

Fraction of gold documents that appear in the top-k results. Maximum 1.0 when
all gold docs are in the top-k; 0.0 when none are.

Edge cases:
- `retrieved` empty → 0.0
- `gold` empty → 0.0

Implementation: `recall_at_k()`, lines 19–29.

### Hit@k

```
hit@k = 1.0 if (retrieved[:k] ∩ gold) else 0.0
```

Binary: did the system return at least one gold document in the top-k? Useful as
a "found it" signal when gold sets are large and full recall is impractical at
small k.

Edge cases: same as Recall@k.

Implementation: `hit_at_k()`, lines 32–39.

### Precision@k

```
precision@k = |retrieved[:k] ∩ gold| / k
```

Fraction of the top-k results that are gold. Penalises returning many irrelevant
documents even if all gold docs are present. Unlike Recall@k, Precision@k
decreases as k grows if the system pads with non-gold docs.

Edge cases: `retrieved` empty or `gold` empty → 0.0. Denominator is always k
(not the actual number returned), so a system returning fewer than k results is
penalised.

Implementation: `precision_at_k()`, lines 59–70.

### MRR — Mean Reciprocal Rank

```
RR(q)  = 1 / rank_of_first_gold_in_retrieved,  or 0 if no gold appears
MRR    = mean( RR(q) )  over all questions q
```

Rewards putting the first relevant document near the top of the list. A system
that always puts the gold doc at rank 1 scores 1.0; at rank 2, 0.5; at rank 10,
0.1.

Edge cases: `retrieved` empty, `gold` empty, or no gold doc in `retrieved` → 0.0
for that question.

Implementation: `mrr()`, lines 42–56 (per-query RR; caller averages across
questions).

### nDCG@k — Normalised Discounted Cumulative Gain

Binary relevance (1 if in gold, else 0).

```
DCG@k  = Σ_{i=1}^{k}  rel_i / log2(i + 1)

IDCG@k = Σ_{i=1}^{min(|gold|, k)}  1 / log2(i + 1)   (perfect ranking)

nDCG@k = DCG@k / IDCG@k
```

Logarithmic discount penalises relevant documents found at lower ranks. IDCG@k
is the score of a hypothetical perfect ranker that puts all gold docs first.
nDCG@k is always in [0, 1].

Edge cases:
- `retrieved` empty → 0.0
- `gold` empty → 0.0 (IDCG = 0 → return 0.0 to avoid division by zero)
- `|gold| > k` → IDCG capped at k (unrecoverable gold docs don't penalise the
  system for failing to surface them beyond cut-off)

Implementation: `ndcg_at_k()`, lines 73–97.

### Pass rate (Track 2)

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
`src/unison_evals/runners/agent_oracle.py`, lines 246–253.

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

Loader specifics (`src/unison_evals/datasets/longmemeval.py`): The haystack
sessions are flattened into a single `oracle_context` string with dated
`## Session N — YYYY-MM-DD` headers. Track 3 will ingest each session as a
separate document (v0.2). If the HuggingFace download fails, the loader falls
back to a 3-question embedded smoke set so CI runs offline without surprise.

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

Loader specifics (`src/unison_evals/datasets/memoryagentbench.py`): Each HF row
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

### FRAMES

| Field | Value |
|---|---|
| Paper | "FRAMES: Factuality, Retrieval, And Reasoning MEasurement Set" |
| arXiv | 2409.12941 |
| Venue | NAACL 2025 |
| HuggingFace | `google/frames-benchmark` |
| License | Apache 2.0 |
| Size | 824 questions, test split |

FRAMES (DeepMind / Harvard / Meta) tests end-to-end RAG factuality + retrieval +
reasoning with questions that require synthesising information across multiple
Wikipedia articles. SOTA without retrieval is ~40%; with multi-step retrieval
pipelines, ~66%. Each row includes gold Wikipedia URLs (`wiki_links`) that serve
as gold doc paths for Track 1.

Loader specifics (`src/unison_evals/datasets/frames.py`): `oracle_context` is
set to `None` for all FRAMES questions. FRAMES was designed to test retrieval
pipelines, and the ~40% SOTA on raw LLM knowledge without retrieval is the
natural Track 2 baseline. Supplying oracle context from Wikipedia URLs would
require fetching and parsing live pages, which introduces non-determinism and
network dependencies — that is Track 3 territory, scheduled for v0.2. For Track
2 runs the adapter receives only the question and must answer from parametric
knowledge. Track 1 is also unavailable in v1.0 (no corpus loading); Track 1
will be wired in v0.2 when the Wikipedia article fetcher ships.

```bibtex
@inproceedings{krishna2025frames,
  title     = {FRAMES: Factuality, Retrieval, And Reasoning MEasurement Set},
  author    = {Krishna, Kalpesh and others},
  booktitle = {NAACL 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2409.12941}
}
```

---

### MuSiQue

| Field | Value |
|---|---|
| Paper | "MuSiQue: Multihop Questions via Single-hop Question Composition" |
| arXiv | 2108.00573 |
| Venue | EMNLP 2022 |
| HuggingFace | `dgslibisey/MuSiQue` |
| License | CC BY 4.0 |
| Size | ~25,000 questions (train split default), 2–4 hop chains |

MuSiQue is designed to be substantially harder than HotpotQA by requiring
genuine multi-hop chains that cannot be short-circuited via single-hop shortcuts.
Each question ships with exactly 20 paragraphs — a private per-question corpus —
of which 2–5 are gold (`is_supporting=True`). The benchmark evaluates both the
final answer string and which paragraphs the system identified as evidence.

Loader specifics (`src/unison_evals/datasets/musique.py`): All 20 paragraphs are
concatenated with `### [N] Title` headers into `oracle_context` for Track 2
(given perfect retrieval, can the agent reason across the chain?). Track 1 is
immediately available: treat each paragraph as a Document, ingest all 20, score
the brain's ability to surface the gold ones. The `gold_paragraph_indexes` field
in question metadata identifies which paragraph indices are gold, enabling exact
Track 1 comparison.

```bibtex
@inproceedings{trivedi2022musique,
  title     = {MuSiQue: Multihop Questions via Single-hop Question Composition},
  author    = {Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  booktitle = {EMNLP 2022},
  year      = {2022},
  url       = {https://arxiv.org/abs/2108.00573}
}
```

---

### BitempoQA

| Field | Value |
|---|---|
| Origin | unison-evals (this repository), v0 |
| HuggingFace | not yet published (v0.2 target) |
| License | CC BY 4.0 |
| Size | 110 facts across 30 fictional SaaS/tech subjects; 100 questions (25 per type) |
| Question types | `current_truth`, `historical_truth`, `predecessor`, `transition` |

BitempoQA is the benchmark's own contribution. It probes bitemporal correctness:
the ability to distinguish between "what is X now," "what was X on date D," "who
came before Y as X's attribute," and "when did X change." The corpus is synthetic
(fictional SaaS companies with realistic metadata changes) so there is no
Wikipedia knowledge contamination. Each fact carries `valid_from`, `valid_to`,
and `supersedes` fields.

Loader specifics (`src/unison_evals/datasets/bitempoqa.py`): The corpus is
read from `data/bitempoqa/corpus.jsonl` and questions from
`data/bitempoqa/questions.jsonl`, both shipped in the repository. Oracle context
is assembled per question from the relevant fact IDs plus an `as_of` hint for
`historical_truth` questions. BitempoQA is the only dataset where Track 1, 2, and
3 are all immediately runnable without a remote HF download.

**Limitation:** 100 questions is too small for stable aggregate scores. BitempoQA
v0.2 (scheduled for v0.5 milestone) expands to 300 questions and will be
published on HuggingFace with a versioned commit hash.

```bibtex
@dataset{unison2026bitempoqa,
  title     = {BitempoQA: A Bitemporal Question-Answering Benchmark for Memory Agents},
  author    = {{Unison}},
  year      = {2026},
  license   = {CC BY 4.0},
  url       = {https://github.com/Unison-Workspace/Unison-evals}
}
```

---

## 6. Adapters

### Brain adapters (Track 1)

**pgvector-naive** (`src/unison_evals/adapters/pgvector_naive.py`): Pure cosine
similarity over OpenAI `text-embedding-3-small` (1536-dim) in a single Postgres
table. HNSW index with default parameters. No reranking, no hybrid search, no
chunking — every document is its own chunk. This is the "what you build in an
afternoon" baseline: its score is the floor that any serious system must beat.
Preferred config: Postgres 17 with pgvector, `OPENAI_API_KEY` for embeddings,
`PGVECTOR_DSN` pointing at the instance. Batch size: 96 docs/embed call,
256 inserts/transaction.

**unison-brain** (`src/unison_evals/adapters/unison_brain.py`): Unison's
production brain — Postgres-native hybrid retrieval (pgvector dense + tsvector
BM25, RRF fusion, optional cross-encoder rerank via Voyage/Cohere/local bge),
kind boosts (`wiki_page > note > raw`), recency decay, and importance
multipliers from importance signals. Called via the Unison server's retrieval API
(auth via the eval secret/JWT documented in the adapter).
Ingest is not implemented in v1.0 (see Known Limitations); Track 1 requires
pre-seeding via the Unison server's ingest endpoint.

**mem0** (`src/unison_evals/adapters/mem0.py`, `Mem0BrainAdapter`): Mem0 cloud
managed memory (https://mem0.ai). Hybrid dense + sparse retrieval proprietary to
Mem0. Per-question isolation achieved via UUID rotation in `reset()` — no
`delete_all()` roundtrip needed, which avoids the ~5 s API call that would
dominate latency. Requires `MEM0_API_KEY`. SDK: `mem0ai >= 0.1.0`.

**letta** (`src/unison_evals/adapters/letta.py`): Letta (formerly MemGPT) archival
memory — embedding-based semantic search over structured long-term memory passages.
Each question gets a fresh ephemeral agent (`reset()` = delete old + create new).
Embedding model defaults to `openai/text-embedding-3-small` (Letta cloud default
as of 2026). The LLM model attached to the agent is never invoked in Track 1 —
only the archival memory passages API is exercised. Requires `LETTA_API_KEY`. SDK:
`letta-client >= 1.10.3`.

### Agent adapters (Track 2 / Track 3)

**unison-agent** (`src/unison_evals/adapters/unison_agent.py`): Posts to
`/api/rest/agents/eval-turn` — the production endpoint, bit-for-bit. Track 3
calls with no `oracleContext`, so the full brain + FS + workspace toolchain runs.
Track 2 passes `oracleContext` to the endpoint, which disables brain/FS/workspace
tools server-side so the agent reasons from the injected context only. This means
Track 2 and Track 3 use the same binary; the `oracleContext` flag is a supported
production knob, not a test-mode bypass. Requires `UNISON_EVAL_SECRET` or
`UNISON_JWT` and the Unison server running.

**claude-code** (`src/unison_evals/adapters/claude_code.py`): Subprocess to
`claude --print`. Claude Code has no persistent memory, so Track 2 and Track 3
behave identically — oracle context is always injected into the prompt. This is
the honest comparison: Claude Code is a capable general-purpose agent without a
brain, so persistent-memory benchmarks should expose that gap. Cost is estimated
from token counts using Anthropic pricing for `claude-sonnet-4-5` (3.00 USD/M
input, 15.00 USD/M output as of 2026-05). Requires the `claude` binary on PATH
and a valid Claude Code authentication.

**mem0-agent** (`src/unison_evals/adapters/mem0.py`, `Mem0AgentAdapter`): Mem0
retrieval combined with Anthropic `claude-sonnet-4-5` generation. In oracle mode,
the oracle context is added to Mem0 memory for the current user and then retrieved
before answering; in E2E mode, the adapter relies on whatever is in the Mem0 store
from prior ingest. Requires both `MEM0_API_KEY` and `ANTHROPIC_API_KEY`.

---

## 7. Reproduction

### Environment

- **Python:** 3.12, managed via uv (see `.python-version`)
- **Dependencies:** `uv sync` (installs from `uv.lock`, pinned)
- **Postgres:** 17 with pgvector extension — required only for `pgvector-naive`
  Track 1. Quick setup: `docker run -d --name pgvec -p 5433:5432 -e POSTGRES_PASSWORD=evals pgvector/pgvector:pg17`
- **Hardware:** any modern Mac/Linux with 16 GB RAM handles smoke sets and small
  datasets (BitempoQA, limit-20 runs). Full LongMemEval (500 q) or FRAMES (824 q)
  need 32 GB if HuggingFace caches are being built; subsequent runs use the cache
  and fit in 16 GB.
- **Network:** stable enough for HuggingFace dataset download (~few hundred MB,
  cached under `~/.cache/unison-evals/hf` after first run)

### Required API keys

| Key | Required for |
|---|---|
| `ANTHROPIC_API_KEY` | LLM judge (all Track 2 runs); `mem0-agent` generation |
| `OPENAI_API_KEY` | `pgvector-naive` embeddings |
| `MEM0_API_KEY` | `mem0` brain adapter + `mem0-agent` |
| `LETTA_API_KEY` | `letta` brain adapter |
| `UNISON_JWT` | `unison-agent` + `unison-brain` adapters |

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
  --systems claude-code \
  --dataset longmemeval \
  --track agent-oracle \
  --limit 3

# 5. Full Track 2 run — LongMemEval, two systems (requires UNISON_JWT + ANTHROPIC_API_KEY)
uv run unison-evals run \
  --systems unison-agent,claude-code \
  --dataset longmemeval \
  --track agent-oracle \
  --limit 500

# 6. Track 1 run — BitempoQA, pgvector-naive (requires OPENAI_API_KEY + PGVECTOR_DSN)
uv run unison-evals run \
  --systems pgvector-naive \
  --dataset bitempoqa \
  --track brain-retrieval

# 7. Track 1 — MemoryAgentBench, pgvector-naive + mem0 (requires MEM0_API_KEY)
uv run unison-evals run \
  --systems pgvector-naive,mem0 \
  --dataset memoryagentbench \
  --track brain-retrieval \
  --limit 50

# 8. Run all tests (should show >=216 passing)
uv run pytest

# 9. Lint check
uv run ruff check src tests
```

### Makefile targets

```bash
make smoke       # limit=10, claude-code, longmemeval, agent-oracle (fast CI check)
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

Latency measurements for hosted adapters (Mem0 cloud, Letta cloud, Unison API)
depend on the network between the evaluation machine and the service's data
center. Published latency numbers are only comparable when measured from the
same network as the publish. If re-running from a different ISP or geography,
report the latency numbers separately and do not directly compare them to
published numbers from another location.

### Recommendation for publishers

- Run each configuration 3 times
- Report median pass rate and IQR across runs
- Report p50/p95 latency (not mean) per adapter, from the same network
- Pin the judge model version (`JUDGE_MODEL`) and record it in the artifact
- Pin the dataset commit hash (available in the artifact when HF provides it)

---

## 9. Known Limitations

1. **Track 3 unavailable for unison-brain.** The Unison API does not yet expose a
   bulk-ingest endpoint suitable for an eval harness. The `unison-brain` adapter
   raises `NotImplementedError` on `ingest()`. Track 3 requires the eval-side
   ingest endpoint documented in the Unison ingest API (not yet public). All other adapters
   (pgvector-naive, mem0, letta) support Track 3 for datasets with per-question
   corpora (BitempoQA, MuSiQue).

2. **BitempoQA is small (100 questions).** Per the sample-size table in §8, 100
   questions yields ±10 pp at the 95% confidence level. BitempoQA v0.2 expands to
   300 questions and will be published to HuggingFace with a pinned commit hash.
   Until then, treat BitempoQA numbers as directional, not definitive.

3. **FRAMES Track 1 unavailable.** The Wikipedia corpus loader that fetches and
   caches article bodies is not yet implemented. FRAMES questions have
   `oracle_context=None`, making them usable only in Track 2 (parametric
   knowledge baseline) in v1.0. Track 1 and full Track 3 for FRAMES are
   scheduled for v0.2.

4. **LLM judge is the cost bottleneck for Track 2.** Opus costs ~$0.005/question;
   a 500-question LongMemEval run across two systems costs ~$5 in judge calls
   alone. Haiku is 10x cheaper (~$0.0005/question) but introduces slightly more
   noise on close calls (score oscillating between 0.5 and 1.0). Published
   numbers always use Opus. Haiku is acceptable for CI smoke and rapid iteration.

5. **No human evaluation baseline.** The judge is LLM-only. For domains where
   LLM judges are known to diverge from human judgement (highly technical content,
   domain-specific conventions), human spot-checks are recommended. We plan to add
   a human evaluation track in v1.0+ for a sampled subset of LongMemEval
   conflict-resolution and knowledge-update questions.

6. **Unison adapter runs production code.** The `unison-agent` adapter calls the
   same `/api/rest/agents/eval-turn` endpoint as a real user session. This means
   results reflect the production agent's current capabilities. If the Unison
   codebase is updated between runs, scores may change. The artifact records the
   Unison server version when available.

7. **mem0 Conflict Resolution scoring.** The Mem0 adapter uses UUID rotation
   for per-question isolation instead of `delete_all()`. This is faster and
   produces the same isolation semantics for most datasets; however, if Mem0's
   fact-extraction layer merges or deduplicates across sessions for the same
   user, UUID rotation may not fully isolate. The tradeoff is documented in the
   adapter's docstring.

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

Dataset citations are listed in §5 under each dataset. If you are specifically
using BitempoQA, cite it separately as the BitempoQA entry in §5.

When comparing numbers across systems, cite the artifact JSON (not the paper)
so readers can verify the exact run configuration.
