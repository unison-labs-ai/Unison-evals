# Benchmark spec — Unified Markdown-Filesystem State for LLM Agents

Working title: **MarkdownBench** (or **UnifiedStateBench / USB**) — final
name TBD.

Status: spec / draft v2 (2026-05-11). Replaces v1 after deep research on
Karpathy's autoeval and the broader benchmark-construction design space.
No code yet — this document gates implementation.

The current `unison-evals` harness is a **mock**. Do not assume any of its
architectural choices carry forward — this spec re-decides each one from
first principles.

---

## 1. Thesis (sharpened)

> Mapping *all* agent state — **tool calls, tool outputs, memory,
> intermediate reasoning, agent identity** — into a single unified markdown
> filesystem produces measurably better task performance than fragmenting
> that state across native APIs, vector stores, and in-context scratchpads.

The variable under test is **unification of representation**.
*Not* markdown-ness alone (controlled by JSON-format variant).
*Not* filesystem-ness alone (controlled by fragmented-md variant).
*Not* "memory layer" (controlled by classic vector-DB baseline).

This thesis is broader than:
- **Letta's** "filesystem as memory" claim (filesystem.com/blog/letta-filesystem)
- **BrainBench's** static-corpus filesystem-navigation claim
- **MemGPT's** tiered-memory paging claim

## 2. Existing-work positioning (with concrete deltas)

| Existing benchmark | What it measures | What it doesn't measure that we will |
|---|---|---|
| MemGPT (2310.08560) | tiered memory in context-exhaustion tasks | unification of *non-memory* state (tool logs, reasoning trace) |
| Letta filesystem (LoCoMo) | filesystem retrieval on dialogue QA | multi-step agent state; A/B against fragmented baseline |
| BrainBench (dhasson04) | navigation strategy on static md FS | per-task corpora; agent uses its OWN state representation, not the test FS |
| LongMemEval (2410.10813) | conversational memory abilities | tool-state representation; agentic workflow |
| TAU-bench (2406.12045) | tool-agent reliability via `pass^k` | memory architecture; state representation |
| MemoryAgentBench (2507.05257) | 4 cognitive memory competencies | filesystem unification of tool/reasoning state |
| AGENTS.md (2602.11988) | counter-evidence: md context can HURT SWE-bench | unification effect at constant token budget |
| Structured Context (2602.05447) | file-context +2.7pp on SQL (frontier only) | controlled for model tier (we will too) |

**Empirical gap:** no benchmark today isolates unified vs. fragmented state
representation as the only variable.

## 3. Karpathy autoeval — primary-source finding

**Karpathy has not published a system specifically called "autoeval."** The
term appears only in community writing extrapolating from his
`autoresearch` project. Verified via exhaustive search of github.com/karpathy,
karpathy.bearblog.dev, karpathy.ai, and x.com/karpathy through May 2026.

The closest Karpathy artifact is `hn-time-capsule` (Dec 2025) — an automated
LLM-judge pipeline that grades decade-old HN comments against historical
outcomes. This is the canonical pattern we should mirror.

### 3a. What `hn-time-capsule` actually is

**Repo:** github.com/karpathy/hn-time-capsule
**Blog post:** karpathy.bearblog.dev/auto-grade-hn/ (2025-12-10)
**Scale:** 930 queries, $58 total cost, ~1h wall-clock, single GPT-5.1 judge

**Architecture (verbatim from `pipeline.py`):**

```
fetch → prompt → analyze → parse → render
```

Each stage caches output to disk. Parallel analysis with
`ThreadPoolExecutor(max_workers=5)`. Each test item is a directory:
`data/<date>/<item_id>/` containing `meta.json`, `article.txt`,
`comments.json`, `prompt.md`, `response.md`, `grades.json`.

**The rubric (verbatim prompt fragment):**

```
1. Give a brief summary of the article and the discussion thread.
2. What ended up happening to this topic? (research the topic briefly...)
3. Give out awards for "Most prescient" and "Most wrong" comments...
4. Mention any other fun or notable aspects of the article or discussion.
5. Give out grades to specific people for their comments, considering
   what happened.
6. At the end, give a final score (from 0-10) for how interesting this
   article and its retrospect analysis was.

As for the format of Section 5, use the header "Final grades" and follow
it with simply an unordered list of people and their grades in the format
of "name: grade (optional comment)". Here is an example:

Final grades
- speckx: A+ (excellent predictions on ...)
- tosh:   A  (correctly predicted this or that ...)
- bgwalter: D
- fsflover: F (completely wrong on ...)

Please follow the format exactly because I will be parsing it
programmatically.
```

**Grade parser (verbatim from `pipeline.py`):**

```python
def grade_to_numeric(grade: str) -> float:
    base = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
    value = base.get(grade[0].upper(), 0.0)
    if len(grade) > 1:
        if grade[1] == '+':
            value += 0.3
        elif grade[1] in '-−':
            value -= 0.3
    return value
```

### 3b. What we should take from Karpathy verbatim

1. **Rubric inside the prompt, not in a separate file.** Each task ships its
   own prompt that doubles as its grading spec. No DSL, no YAML rubric layer.
2. **Letter grades + optional rationale.** Familiar to humans, easy for the
   model to produce consistently. Don't invent a new ordinal scale.
3. **Programmatic parsing of structured judge output.** "Please follow the
   format exactly because I will be parsing it programmatically" — Karpathy
   tells the judge to emit machine-parseable output. The parser is ~15 lines
   of Python.
4. **A separate 0–10 interestingness score** for task-level quality
   filtering. Decouples "did the agent do well?" from "is this a good test?"
5. **Directory-per-task on disk**, not JSONL. Simpler diffs, easier inspection,
   git-friendly. Each task is just a folder you can `ls`.
6. **Cache every stage.** `fetch`, `prompt`, `analyze`, `parse`, `render` all
   cache. Re-runs are cheap.
7. **Single judge, not a pool.** Karpathy uses one judge (GPT-5.1 Thinking).
   Multi-judge ensembles are not in his pattern.

### 3c. What we should NOT take from Karpathy

1. **Single-judge limitation.** Karpathy's `hn-time-capsule` is a personal
   project; he doesn't claim methodological rigor. For a publishable
   architecture benchmark, single-judge is a methodological smell — judge
   self-preference inflation (Claude judging Claude scores ~10pp higher than
   GPT-4o judging the same Claude outputs, per "Justice or Prejudice?"
   2410.02736). For Stage 1 we use multiple judges and report inter-judge
   agreement.
2. **Pure LLM-grading.** Karpathy grades prose against historical outcomes —
   inherently LLM-dependent. Our benchmark has filesystem state we can verify
   programmatically. We should grade *outcomes* with code, not with LLMs.
   LLMs are only needed when the answer is genuinely free-form.

### 3d. Karpathy's broader eval views (relevant primary sources)

- **"Evaluation crisis" tweet** (2025-03-02, x.com/karpathy/status/1896266683301659068):
  *"MMLU was a good and useful for a few years but that's long over.
  SWE-Bench Verified (real, practical, verified problems) I really like
  and is great but itself too narrow."* Vibe checks suffer from
  confirmation bias and low sample size.
- **"Verifiability" post** (2025-11-17, karpathy.bearblog.dev/verifiability/):
  Verifiability has three preconditions — *resettable*, *efficient*,
  *rewardable*. "The more a task/job is verifiable, the more amenable it
  is to automation." This is the core property our benchmark must have.
- **2025 LLM Year in Review** (karpathy.bearblog.dev/year-in-review-2025/):
  *"benchmarks are almost by construction verifiable environments and
  are therefore immediately susceptible to RLVR and weaker forms of it
  via synthetic data generation… training on the test set is a new art
  form."* Our defense: held-out seeds + procedural generation +
  behavioral (not knowledge) tasks.

## 4. The benchmark — architectural choices

### 4a. Test-case construction: **procedural simulator + held-out seeds**

Not static datasets. Not pure templates. **Both:**

1. **Procedural simulator (the environment).** A live mock filesystem the
   agent navigates with `read_file`, `write_file`, `list_directory`,
   `move_file` tools. Final state is checkable programmatically. This is
   the right shape per Karpathy's verifiability criteria (resettable,
   efficient, rewardable) and matches TAU-bench's pattern.
2. **Parameterized task templates.** ~5 task families (multi-session
   investigation, decision-tree-with-rollback, long-horizon planning,
   negation, adversarial distraction). Each template parameterizes the
   filesystem layout, the goal, and the expected end state. Instances
   generated from a **held-out random seed** that's not in the public repo.

**Why this beats static datasets:** the specific test instances don't enter
training data (only the templates do). Contamination becomes much harder
to gain leverage from.

**Why this beats pure templates:** the simulator gives us real interaction
testing — we measure trajectory, not just final answer.

### 4b. Grading: **layered, programmatic-first**

Three layers, applied in order:

1. **Programmatic outcome verification (primary, ~80% of grading).** After
   the agent finishes, serialize the directory tree (`tree -J` or
   `find . -type f -exec sha256sum {} \;`) and diff against expected state.
   No ambiguity, no judge cost, fully reproducible. Same model as SWE-bench
   (test-suite passes) and TAU-bench (DB state diff).
2. **Trajectory grading (secondary).** Log every tool call. Measure: step
   count, tool-call type distribution, file re-read rate, error recovery
   rate. Step-efficiency = optimal-steps / actual-steps. Re-read rate is
   the most diagnostic signal for our thesis — unified state representation
   should reduce repeated reads.
3. **LLM-as-judge (fallback, ~20% of grading).** Only for tasks with
   genuinely free-form answers (e.g. "summarize the project's status from
   these files"). Use the Karpathy rubric pattern: rubric in the prompt,
   letter grades + 0–10 score, programmatic parsing of structured output.

**Multi-judge pool.** Three judges from different model families:
Claude Opus 4.5, GPT-5, Gemini 2.5 Pro. Report inter-judge agreement
(Krippendorff's α). α<0.6 = our rubric needs work; α≥0.8 = ship it.

### 4c. Statistical methodology: **bootstrap CIs + paired tests + effect size**

| Metric | Test | When |
|---|---|---|
| Pass rate | Bootstrap 95% CI (1000-resample, seeded) | every headline number |
| Pass-rate delta (A vs C) | **McNemar's test** | paired binary outcomes |
| Continuous score delta | Paired t-test (or Wilcoxon if non-normal) | latency, cost, step count |
| Effect size | **Cohen's d** | always reported alongside p |
| Reliability across k runs | **pass^k** (k=3 minimum) | TAU-bench convention |
| Coverage across k runs | **pass@k** | for completeness |
| Judge agreement | **Krippendorff's α** | inter-judge validation |
| Ranking correlation vs other benchmarks | Spearman ρ | external validity |

**Pre-registration is mandatory.** Hypothesis, primary metric, statistical
test, and minimum-n sample size committed to OSF before competitors are
run. Eliminates HARKing (Hypothesizing After Results are Known).

### 4d. Framework: **Inspect AI** (UK AISI, inspect.ai-safety-institute.org.uk)

Out of the entire landscape — OpenAI Evals, lm-eval-harness, lighteval,
DeepEval, RAGAS, promptfoo, LangSmith, TruLens, Braintrust, W&B Weave,
Phoenix — Inspect AI is the only framework designed explicitly for
**multi-step agent evaluation with sandboxed environments**. Built by UK
AISI specifically for the agent-eval use case. Provides:

- Python task definitions (not YAML — composable, testable)
- Per-task Docker sandboxes (clean state every run)
- First-class tool support
- Trajectory logging built in
- Multi-judge LLM-as-judge integration
- Multi-model support out of the box

Other frameworks fail the test:
- **OpenAI Evals** — single-turn, OpenAI-centric, no agent state
- **lm-eval-harness / lighteval** — base LM benchmarks, no tool use
- **DeepEval / RAGAS / TruLens** — RAG-focused, wrong domain
- **promptfoo** — regression testing, not formal benchmarking
- **LangSmith / Braintrust / W&B Weave** — production observability layers,
  not benchmark harnesses. Usable on top of Inspect for the dashboard.

**Layer it like this:**
1. Inspect AI for the eval engine + Docker sandbox + trajectory logs
2. Braintrust or W&B Weave for the hosted dashboard (Stage 2 leaderboard)
3. Custom Python scorer (~100 lines) for the filesystem-state diff
4. Custom Karpathy-style judge prompts for the LLM-graded subset

We will **vendor a fork of Inspect AI** if needed — they're moving fast and
breaking changes are possible. Pin a specific commit.

### 4e. Heterogeneous-agent comparison: **two-stage**

This is the methodological land mine. Mixing Claude Code + Codex + Cursor +
Letta + Mem0 + Unison conflates *architecture* with *base model* with
*agent identity*. Solution:

**Stage 1 — The architecture proof.** Single-agent harness, single base
model (Claude Sonnet 4.5), conditions A/B/C/D/E (see §5). This is the
controlled ablation that proves the thesis. ~80 tasks. The paper.

**Stage 2 — The ecological-validity leaderboard.** Run heterogeneous agents
(Claude Code, Codex, Cursor, Letta, Mem0, Unison) against the same task
set. Report stratified by capability tier, with cost / latency / pass^k
axes. Multi-axis Pareto, not single-number ranking. This is the brand.

**Critical:** publish Stage 1 first. The methodology paper is what gets
cited. Stage 2 without Stage 1 is just a vendor leaderboard.

### 4f. Contamination strategy: **template + held-out seeds**

- 5–10 parameterized templates published openly.
- Instances generated at eval time from a **never-published random seed**.
- Quarterly seed rotation (LiveBench pattern).
- Behavioral-not-knowledge framing — testing whether the agent can *do*
  filesystem manipulation, not whether it has memorized facts. ICML 2025
  work shows behavioral benchmarks are much more contamination-resistant.

If we ever publicly host a leaderboard, run inference server-side against
held-out instances — only the templates are public, never the answers.
This is the SWE-bench Verified model.

## 5. The five-condition ablation (the core experiment)

All conditions: same base model, same prompt scaffolding, same tasks,
same token budget. Only the state-representation mechanism changes.

| Condition | What lives in unified md-FS | Tests |
|---|---|---|
| **A — Full** | tool calls + outputs + memory + reasoning + identity | the thesis |
| **B — Memory-only** | only LT memory in md-FS; tool outputs ephemeral; ReAct scratchpad | does the unification matter beyond memory? |
| **C — Baseline** | ReAct + vector memory; tool outputs ephemeral; no FS | classical SOTA |
| **D — Format control** | same as A but **JSON** instead of markdown | is it markdown specifically? |
| **E — Unity control** | multiple separate md files (memory.md / tools.md / reasoning.md) | is it unification, beyond just markdown? |

**A vs C** is the headline.
**D** kills the "just because markdown is hot in 2026" critique.
**E** kills the "just splitting things into files" critique.

If A wins A-vs-C significantly **and** A wins A-vs-D significantly **and**
A wins A-vs-E significantly — then the unification claim is defensible.
Any one of those falling means we revise the thesis.

## 6. Task families (~80 tasks total, 16 each, 5 difficulty tiers)

Effect only appears on tasks that need:
- Cross-step state reference (look up tool output > 3 steps back)
- Cross-session state continuity
- State-conditional reasoning
- Verifiable end states

1. **Multi-session investigation.** 20 sub-tasks across 4 simulated
   sessions; final task references session-1 resolution. Tests cross-session
   state retention.
2. **Decision-tree with rollback.** Strategy change mid-task; must refer
   back to original goal. Tests unified-state coherence under change.
3. **Long-horizon planning with intermediate state.** Gather 10 facts,
   synthesize using subset. Tests whether tool outputs are correctly
   retrievable as state.
4. **Negation / contradiction handling.** Contradictory info; must resolve
   via correct provenance. Tests audit-trail value of unified representation.
5. **Adversarial distraction.** Corpus with noise that should NOT enter
   state. Tests whether the architecture appropriately discriminates.

Five difficulty tiers (`L1`–`L5`) per family. Report by tier — never aggregate
to a single number (Simpson's paradox is real, per-stratum reversals are
common in heterogeneous benchmarks).

## 7. Common failure modes we will explicitly defend against

1. **Cherry-picked task selection.** Pre-register templates on OSF before
   running competitors.
2. **Base-model dominance.** Single base model across all conditions. The
   2602.05447 finding (21pp model-tier gap vs. 2.7pp architecture gap) is
   the warning.
3. **Prompt-engineering asymmetry.** Same prompt scaffolding across
   conditions. Only state-representation mechanism differs.
4. **Cost-blind reporting.** Token cost and $/task reported on every metric.
   Accuracy-at-budget curves, not just peak accuracy. Engages with the
   AGENTS.md counter-evidence (2602.11988) head-on.
5. **Judge bias.** Multi-judge pool (Claude + GPT + Gemini), randomized
   answer-order in pairwise comparisons, inter-judge agreement (Krippendorff
   α) reported. Mitigates "Justice or Prejudice?" (2410.02736) biases.
6. **Single-number aggregation hiding per-stratum reversals.** Report by
   difficulty tier, by task family, by condition. Never aggregate to one
   number in the headline.
7. **No reliability metric.** Always report `pass^k`, not just `pass@1`.
8. **Contamination.** Template-only public, instances private with held-out
   seeds.
9. **Lack of pre-registration.** OSF pre-registration of hypothesis +
   primary metric + statistical test + sample-size calculation. Done before
   any competitor agent is run.
10. **Insufficient n.** Power analysis up front. McNemar at n=80 with d=0.3
    has ~80% power; that's our floor.

## 8. Two-stage publication path

**Stage 1 — Controlled ablation paper.**
- Single agent harness, fixed base model, conditions A–E
- ~80 tasks × 5 difficulty tiers
- Full statistical reporting: bootstrap 95% CIs + McNemar's + paired t +
  Cohen's d
- Multi-judge pool with α reporting
- Pre-registered, with the registered protocol committed to the repo
- 8–12 pages, arxiv (cs.AI)
- **This is what gets cited.**

**Stage 2 — Public leaderboard.**
- Same task set; heterogeneous agents (Claude Code, Codex, Cursor, Letta,
  Mem0, Unison, future submissions)
- Reported by capability stratum + cost + reliability axes
- Multi-axis Pareto plots
- Hosted at `unison.ai/markdown-bench` or `markdownbench.org`
- HuggingFace dataset hub for the template definitions (templates public,
  instances held out)
- HuggingFace Spaces for the leaderboard render
- Quarterly seed rotation + version freezes (v1.0, v1.1, …)
- **This is what drives adoption.**

Don't conflate Stage 1 and Stage 2. The methodology claim and the brand
claim ride different vehicles.

## 9. Open product questions

| Question | Default unless changed |
|---|---|
| Base model for the ablation | Claude Sonnet 4.5 (multi-model would be ideal but doubles cost) |
| Repo location | New standalone repo `markdown-bench` — don't fold into unison-evals |
| License (templates + code) | CC-BY-4.0 for templates, MIT for harness code |
| Public or invite-only Stage 2 leaderboard | Public — gates adoption |
| Stage 1 paper authors | TBD |
| Stage 1 timeline | 4–6 weeks from start of implementation |
| Pre-registration platform | OSF (osf.io) |
| Inspect AI vendor strategy | Vendor a pinned-commit fork until v1.0 stable |

## 10. Out of scope (v1)

- Multi-modal tasks (images, audio in agent state)
- Real-time interactive eval (user can intervene)
- Self-improvement loops / Reflexion-style multi-trial within a single task
- Cross-language tasks (English only at launch)
- Production-trace replay (privacy + consent issues)

## 11. Definition-of-done for the spec

We move from spec → implementation when **all** are true:
- [ ] §9 product questions resolved
- [ ] Inspect AI pinned to a specific commit
- [ ] One template prototyped end-to-end (Karpathy-style: directory layout +
      prompt + parser + scorer) to validate the contract
- [ ] OSF pre-registration drafted (not yet submitted)
- [ ] Author list finalized
- [ ] Budget for ~5 full eval runs across 5 conditions × 80 tasks ×
      multi-judge pool (estimated ~$500–$1500 in API costs)

---

## References

**Primary Karpathy sources:**
- hn-time-capsule (the canonical pattern): github.com/karpathy/hn-time-capsule
- Auto-grade HN blog post: karpathy.bearblog.dev/auto-grade-hn/ (2025-12-10)
- autoresearch (deterministic-metric autonomous loop, NOT LLM eval): github.com/karpathy/autoresearch
- nanochat (benchmark-based eval, NOT LLM-as-judge): github.com/karpathy/nanochat
- LLM Council (multi-judge experiment, "vibe coded"): github.com/karpathy/llm-council
- Verifiability post: karpathy.bearblog.dev/verifiability/ (2025-11-17)
- 2025 LLM Year in Review: karpathy.bearblog.dev/year-in-review-2025/ (2025-12-19)
- Evaluation crisis tweet: x.com/karpathy/status/1896266683301659068 (2025-03-02)

**Methodology references:**
- MemGPT — arxiv:2310.08560
- LongMemEval — arxiv:2410.10813
- LoCoMo — arxiv:2402.17753
- MemoryAgentBench — arxiv:2507.05257
- TAU-bench — arxiv:2406.12045
- GAIA — arxiv:2311.12983
- SWE-bench — arxiv:2310.06770
- AGENTS.md harm — arxiv:2602.11988
- Structured Context Engineering — arxiv:2602.05447
- LiveBench — arxiv:2406.19314
- G-Eval — arxiv:2303.16634
- RAGAS — arxiv:2309.15217
- Justice or Prejudice (judge bias taxonomy) — arxiv:2410.02736
- Min-K% Prob (contamination detection) — arxiv:2310.16789
- Anthropic statistical evals — anthropic.com/research/statistical-approach-to-model-evals

**Frameworks:**
- Inspect AI — inspect.ai-safety-institute.org.uk
- OpenAI Evals — github.com/openai/evals
- lm-eval-harness — github.com/EleutherAI/lm-evaluation-harness
- DeepEval — deepeval.com
- RAGAS — docs.ragas.io
- promptfoo — promptfoo.dev
- LangSmith — smith.langchain.com
- Braintrust — braintrust.dev
- W&B Weave — wandb.ai/site/weave
