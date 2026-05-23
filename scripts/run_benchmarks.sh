#!/usr/bin/env bash
# run_benchmarks.sh — run the v1.0 reference benchmark suite.
#
# Reads .env for credentials (if present). Skips any benchmark whose required
# env vars or services are missing, with a clear "SKIP: <reason>" log line —
# never silently produces fake numbers.
#
# Usage:
#   bash scripts/run_benchmarks.sh                # default limit=20
#   LIMIT=50 bash scripts/run_benchmarks.sh       # publishable run
#   LIMIT=3  bash scripts/run_benchmarks.sh       # smoke (cheap)
#
# Output:
#   results/v1.0-<dataset>-<adapter>-<timestamp>.json per benchmark run
#
# Prerequisites:
#   uv sync --all-extras   (done once per environment)
#   ANTHROPIC_API_KEY      needed for Track 2 LLM judge + claude-code adapter
#   UNISON_JWT             needed for unison-agent adapter
#   MEM0_API_KEY           needed for mem0 and mem0-agent adapters
#   LETTA_API_KEY          needed for letta adapter
#   OPENAI_API_KEY         needed for pgvector-naive embeddings
#   PGVECTOR_DSN           needed for pgvector-naive (local Postgres with pgvector)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LIMIT="${LIMIT:-20}"
JUDGE="${JUDGE:-claude-haiku-4-5}"   # cheap default; use claude-opus-4-5-20250101 for v1.0 publish
TS=$(date -u +%Y%m%dT%H%M%SZ)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$REPO_ROOT/results"

# Load .env if it exists (sourcing gives us env vars without exporting manually).
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -o allexport
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +o allexport
fi

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

log_info()  { echo "[INFO]  $*"; }
log_skip()  { echo "[SKIP]  $*"; }
log_run()   { echo "[RUN]   $*"; }
log_done()  { echo "[DONE]  $*"; }
log_error() { echo "[ERROR] $*" >&2; }

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

# uv must be available.
if ! command -v uv &>/dev/null; then
  log_error "uv not found. Install from https://docs.astral.sh/uv/ and re-run."
  exit 1
fi

# Verify the harness is installed.
if ! uv run unison-evals --help &>/dev/null; then
  log_info "Running uv sync --all-extras to install harness..."
  uv sync --all-extras
fi

# ---------------------------------------------------------------------------
# Track 2 — Agent Oracle benchmarks
# (agent is given gold context; tests reasoning quality)
# ---------------------------------------------------------------------------

run_track2_longmemeval() {
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    log_skip "longmemeval (Track 2) — needs ANTHROPIC_API_KEY"
    return
  fi

  systems="claude-code"
  [[ -n "${UNISON_JWT:-}" ]] && systems="$systems,unison-agent"
  [[ -n "${MEM0_API_KEY:-}" ]] && systems="$systems,mem0-agent"

  out="$REPO_ROOT/results/v1.0-longmemeval-track2-$TS.json"
  log_run "longmemeval track=agent-oracle systems=$systems limit=$LIMIT judge=$JUDGE"
  uv run unison-evals run \
    --track agent-oracle \
    --dataset longmemeval \
    --systems "$systems" \
    --limit "$LIMIT" \
    --judge "$JUDGE" \
    --output "$out"
  log_done "longmemeval → $out"
}

run_track2_musique() {
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    log_skip "musique (Track 2) — needs ANTHROPIC_API_KEY"
    return
  fi

  systems="claude-code"
  [[ -n "${UNISON_JWT:-}" ]] && systems="$systems,unison-agent"
  [[ -n "${MEM0_API_KEY:-}" ]] && systems="$systems,mem0-agent"

  out="$REPO_ROOT/results/v1.0-musique-track2-$TS.json"
  log_run "musique track=agent-oracle systems=$systems limit=$LIMIT judge=$JUDGE"
  uv run unison-evals run \
    --track agent-oracle \
    --dataset musique \
    --systems "$systems" \
    --limit "$LIMIT" \
    --judge "$JUDGE" \
    --output "$out"
  log_done "musique → $out"
}

# ---------------------------------------------------------------------------
# Track 1 — Brain Only benchmarks
# (fixed corpus ingest → retrieval; no LLM, pure ranking quality)
# ---------------------------------------------------------------------------

run_track1_bitempoqa() {
  local systems=""

  # pgvector-naive needs OPENAI_API_KEY (embeddings) + PGVECTOR_DSN (Postgres).
  if [[ -n "${OPENAI_API_KEY:-}" && -n "${PGVECTOR_DSN:-}" ]]; then
    systems="${systems:+$systems,}pgvector-naive"
  else
    _pgvec_missing=()
    [[ -z "${OPENAI_API_KEY:-}" ]] && _pgvec_missing+=("OPENAI_API_KEY")
    [[ -z "${PGVECTOR_DSN:-}" ]] && _pgvec_missing+=("PGVECTOR_DSN (local Postgres+pgvector)")
    log_skip "bitempoqa pgvector-naive — needs ${_pgvec_missing[*]}"
  fi

  if [[ -n "${MEM0_API_KEY:-}" ]]; then
    systems="${systems:+$systems,}mem0"
  else
    log_skip "bitempoqa mem0 — needs MEM0_API_KEY"
  fi

  if [[ -n "${LETTA_API_KEY:-}" ]]; then
    systems="${systems:+$systems,}letta"
  else
    log_skip "bitempoqa letta — needs LETTA_API_KEY"
  fi

  if [[ -z "$systems" ]]; then
    log_skip "bitempoqa (Track 1) — no brain adapters available (needs OPENAI_API_KEY+PGVECTOR_DSN, MEM0_API_KEY, or LETTA_API_KEY)"
    return
  fi

  out="$REPO_ROOT/results/v1.0-bitempoqa-track1-$TS.json"
  log_run "bitempoqa track=brain-only systems=$systems limit=$LIMIT"
  uv run unison-evals run \
    --track brain-only \
    --dataset bitempoqa \
    --systems "$systems" \
    --limit "$LIMIT" \
    --output "$out"
  log_done "bitempoqa → $out"
}

run_track1_memoryagentbench() {
  local systems=""

  if [[ -n "${OPENAI_API_KEY:-}" && -n "${PGVECTOR_DSN:-}" ]]; then
    systems="${systems:+$systems,}pgvector-naive"
  else
    log_skip "memoryagentbench pgvector-naive — needs OPENAI_API_KEY + PGVECTOR_DSN"
  fi

  if [[ -n "${MEM0_API_KEY:-}" ]]; then
    systems="${systems:+$systems,}mem0"
  else
    log_skip "memoryagentbench mem0 — needs MEM0_API_KEY"
  fi

  if [[ -n "${LETTA_API_KEY:-}" ]]; then
    systems="${systems:+$systems,}letta"
  else
    log_skip "memoryagentbench letta — needs LETTA_API_KEY"
  fi

  if [[ -z "$systems" ]]; then
    log_skip "memoryagentbench (Track 1) — no brain adapters available"
    return
  fi

  out="$REPO_ROOT/results/v1.0-memoryagentbench-track1-$TS.json"
  log_run "memoryagentbench track=brain-only systems=$systems limit=$LIMIT"
  uv run unison-evals run \
    --track brain-only \
    --dataset memoryagentbench \
    --systems "$systems" \
    --limit "$LIMIT" \
    --output "$out"
  log_done "memoryagentbench → $out"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log_info "=== unison-evals v1.0 benchmark suite ==="
log_info "LIMIT=$LIMIT  JUDGE=$JUDGE  TS=$TS"
log_info "Results will be written to $REPO_ROOT/results/"
echo

# Track 2 — Agent Oracle
log_info "--- Track 2: Agent Oracle ---"
run_track2_longmemeval
run_track2_musique
echo

# Track 1 — Brain Only
log_info "--- Track 1: Brain Only ---"
run_track1_bitempoqa
run_track1_memoryagentbench
echo

log_info "=== Suite complete. Check $REPO_ROOT/results/ for output files. ==="
log_info "Tip: LIMIT=50 JUDGE=claude-opus-4-5-20250101 bash scripts/run_benchmarks.sh for publishable numbers."
