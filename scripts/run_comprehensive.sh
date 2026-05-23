#!/usr/bin/env bash
# run_comprehensive.sh — the full eval matrix.
#
# Tracks:   brain (Track 1), agent (Track 2), together (Track 3 / agent-e2e)
# Datasets: bitempoqa, longmemeval, memoryagentbench, musique, frames, msmarco
# Systems:
#   brain  = unison-brain, pgvector-naive, mem0, letta, zep
#   agent  = unison-agent, claude-code, codex, gemini-cli, mem0-agent,
#             anthropic-raw, openai-gpt5, google-gemini
#   together = same as agent (each also drives a brain ingest per question)
#
# Usage:
#   bash scripts/run_comprehensive.sh                               # default LIMIT=20
#   LIMIT=50 bash scripts/run_comprehensive.sh                      # publishable
#   LIMIT=3 JUDGE=claude-haiku-4-5 bash scripts/run_comprehensive.sh   # smoke
#   ONLY_TRACK=brain bash scripts/run_comprehensive.sh              # one track only
#   ONLY_DATASET=bitempoqa bash scripts/run_comprehensive.sh        # one dataset only
#   CONFIRM=1 bash scripts/run_comprehensive.sh                     # skip $20 budget gate
#
# Output: results/comprehensive-<TS>/
#   One JSON per successful (track, dataset, system) combo.
#   summary.json aggregating all runs.
#
# Skip/Fail behaviour:
#   [SKIP] — prereqs missing (env var, service). Logged with reason. Never silently faked.
#   [FAIL] — run command returned non-zero. Logged with exit code. Suite continues.

set -euo pipefail

LIMIT="${LIMIT:-20}"
JUDGE="${JUDGE:-claude-haiku-4-5}"
ONLY_TRACK="${ONLY_TRACK:-}"     # brain | agent | together | (empty = all)
ONLY_DATASET="${ONLY_DATASET:-}"
CONFIRM="${CONFIRM:-0}"
TS=$(date -u +%Y%m%dT%H%M%SZ)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/results/comprehensive-$TS"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/run.log"
SUMMARY_FILE="$OUT_DIR/summary.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log() { echo "$*" | tee -a "$LOG_FILE"; }
log_info()  { _log "[INFO]  $*"; }
log_skip()  { _log "[SKIP]  $*"; }
log_run()   { _log "[RUN]   $*"; }
log_done()  { _log "[DONE]  $*"; }
log_fail()  { _log "[FAIL]  $*"; }
log_error() { _log "[ERROR] $*"; }

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -o allexport
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +o allexport
fi

# ---------------------------------------------------------------------------
# Prereqs
# ---------------------------------------------------------------------------

if ! command -v uv &>/dev/null; then
  log_error "uv not found. Install from https://docs.astral.sh/uv/ and re-run."
  exit 1
fi

if ! uv run unison-evals --help &>/dev/null 2>&1; then
  log_info "Running uv sync --all-extras to install harness..."
  uv sync --all-extras
fi

# ---------------------------------------------------------------------------
# Budget estimate + gate
# ---------------------------------------------------------------------------

log_info "=== unison-evals comprehensive run === TS=$TS"
log_info "LIMIT=$LIMIT  JUDGE=$JUDGE  ONLY_TRACK=${ONLY_TRACK:-all}  ONLY_DATASET=${ONLY_DATASET:-all}"
log_info "Output → $OUT_DIR"
echo

log_info "--- Cost estimate ---"

ESTIMATE_ARGS=(
  "--limit" "$LIMIT"
  "--judge" "$JUDGE"
)

[[ -n "$ONLY_TRACK" ]] && ESTIMATE_ARGS+=("--tracks" "$ONLY_TRACK")
[[ -n "$ONLY_DATASET" ]] && ESTIMATE_ARGS+=("--datasets" "$ONLY_DATASET")

# Run cost estimate (do not abort on non-zero since --check sets exit 1)
if uv run python "$SCRIPT_DIR/_estimate_cost.py" "${ESTIMATE_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
  :
else
  ESTIMATE_EXIT=${PIPESTATUS[0]}
  if [[ "$ESTIMATE_EXIT" -ne 0 ]]; then
    if [[ "$CONFIRM" == "1" ]]; then
      log_info "Budget gate exceeded but CONFIRM=1 — proceeding."
    else
      log_error "Budget gate: estimated cost exceeds \$20. Set CONFIRM=1 to override."
      exit 1
    fi
  fi
fi

echo

# ---------------------------------------------------------------------------
# Tracking arrays for summary.json
# ---------------------------------------------------------------------------

declare -a COMBO_RESULTS=()   # each entry: "track|dataset|system|status|file"

# ---------------------------------------------------------------------------
# Core run helper
# ---------------------------------------------------------------------------

# run_one <track_cli> <dataset> <system> <out_file> [extra_args...]
run_one() {
  local track_cli="$1"
  local dataset="$2"
  local system="$3"
  local out_file="$4"
  shift 4
  local extra_args=("$@")

  local cmd=(
    uv run unison-evals run
    --track "$track_cli"
    --dataset "$dataset"
    --systems "$system"
    --limit "$LIMIT"
    --judge "$JUDGE"
    --output "$out_file"
    "${extra_args[@]}"
  )

  log_run "track=$track_cli dataset=$dataset system=$system → $out_file"
  if "${cmd[@]}" >> "$LOG_FILE" 2>&1; then
    log_done "track=$track_cli dataset=$dataset system=$system"
    COMBO_RESULTS+=("$track_cli|$dataset|$system|done|$out_file")
  else
    local rc=$?
    log_fail "track=$track_cli dataset=$dataset system=$system exit=$rc"
    COMBO_RESULTS+=("$track_cli|$dataset|$system|failed|")
  fi
}

# ---------------------------------------------------------------------------
# Prereq check helpers (return 0 if prereqs met, 1 if not)
# ---------------------------------------------------------------------------

need_vars() {
  # Usage: need_vars VAR1 VAR2 ...  (stores missing list in $MISSING_VARS)
  MISSING_VARS=()
  for v in "$@"; do
    [[ -z "${!v:-}" ]] && MISSING_VARS+=("$v")
  done
  [[ ${#MISSING_VARS[@]} -eq 0 ]]
}

# ---------------------------------------------------------------------------
# Track filter helper
# ---------------------------------------------------------------------------

skip_track() {
  # Returns 0 (true) if we should skip this track
  [[ -n "$ONLY_TRACK" && "$ONLY_TRACK" != "$1" ]]
}

skip_dataset() {
  [[ -n "$ONLY_DATASET" && "$ONLY_DATASET" != "$1" ]]
}

# ---------------------------------------------------------------------------
# Track 1 — Brain Only
# ---------------------------------------------------------------------------

run_brain_track() {
  local dataset="$1"

  skip_track "brain" && return
  skip_dataset "$dataset" && return

  log_info "--- Track 1 (brain): $dataset ---"

  # unison-brain — needs UNISON_JWT or UNISON_LOCAL_EVAL_TENANT_ID
  if need_vars UNISON_JWT || need_vars UNISON_LOCAL_EVAL_TENANT_ID; then
    run_one "brain-only" "$dataset" "unison-brain" \
      "$OUT_DIR/brain-${dataset}-unison-brain.json"
  else
    log_skip "track=brain dataset=$dataset system=unison-brain reason=${MISSING_VARS[*]} missing"
  fi

  # pgvector-naive — needs OPENAI_API_KEY + PGVECTOR_DSN
  if need_vars OPENAI_API_KEY PGVECTOR_DSN; then
    run_one "brain-only" "$dataset" "pgvector-naive" \
      "$OUT_DIR/brain-${dataset}-pgvector-naive.json"
  else
    log_skip "track=brain dataset=$dataset system=pgvector-naive reason=${MISSING_VARS[*]} missing"
  fi

  # mem0
  if need_vars MEM0_API_KEY; then
    run_one "brain-only" "$dataset" "mem0" \
      "$OUT_DIR/brain-${dataset}-mem0.json"
  else
    log_skip "track=brain dataset=$dataset system=mem0 reason=MEM0_API_KEY missing"
  fi

  # letta
  if need_vars LETTA_API_KEY; then
    run_one "brain-only" "$dataset" "letta" \
      "$OUT_DIR/brain-${dataset}-letta.json"
  else
    log_skip "track=brain dataset=$dataset system=letta reason=LETTA_API_KEY missing"
  fi

  # zep
  if need_vars ZEP_API_KEY; then
    run_one "brain-only" "$dataset" "zep" \
      "$OUT_DIR/brain-${dataset}-zep.json"
  else
    log_skip "track=brain dataset=$dataset system=zep reason=ZEP_API_KEY missing"
  fi
}

# ---------------------------------------------------------------------------
# Track 2 — Agent Oracle
# ---------------------------------------------------------------------------

run_agent_track() {
  local dataset="$1"

  skip_track "agent" && return
  skip_dataset "$dataset" && return

  log_info "--- Track 2 (agent): $dataset ---"

  # All agent systems need at minimum ANTHROPIC_API_KEY for the judge.
  if ! need_vars ANTHROPIC_API_KEY; then
    log_skip "track=agent dataset=$dataset reason=ANTHROPIC_API_KEY missing (judge required)"
    return
  fi

  # unison-agent
  if need_vars UNISON_JWT || need_vars UNISON_LOCAL_EVAL_TENANT_ID; then
    run_one "agent-oracle" "$dataset" "unison-agent" \
      "$OUT_DIR/agent-${dataset}-unison-agent.json"
  else
    log_skip "track=agent dataset=$dataset system=unison-agent reason=UNISON_JWT or UNISON_LOCAL_EVAL_TENANT_ID missing"
  fi

  # claude-code — needs ANTHROPIC_API_KEY (already checked above)
  run_one "agent-oracle" "$dataset" "claude-code" \
    "$OUT_DIR/agent-${dataset}-claude-code.json"

  # codex
  if need_vars OPENAI_API_KEY; then
    run_one "agent-oracle" "$dataset" "codex" \
      "$OUT_DIR/agent-${dataset}-codex.json"
  else
    log_skip "track=agent dataset=$dataset system=codex reason=OPENAI_API_KEY missing"
  fi

  # gemini-cli
  if need_vars GEMINI_API_KEY || need_vars GOOGLE_API_KEY; then
    run_one "agent-oracle" "$dataset" "gemini-cli" \
      "$OUT_DIR/agent-${dataset}-gemini-cli.json"
  else
    log_skip "track=agent dataset=$dataset system=gemini-cli reason=GEMINI_API_KEY or GOOGLE_API_KEY missing"
  fi

  # mem0-agent
  if need_vars MEM0_API_KEY; then
    run_one "agent-oracle" "$dataset" "mem0-agent" \
      "$OUT_DIR/agent-${dataset}-mem0-agent.json"
  else
    log_skip "track=agent dataset=$dataset system=mem0-agent reason=MEM0_API_KEY missing"
  fi

  # anthropic-raw — needs ANTHROPIC_API_KEY (already checked)
  run_one "agent-oracle" "$dataset" "anthropic-raw" \
    "$OUT_DIR/agent-${dataset}-anthropic-raw.json"

  # openai-gpt5
  if need_vars OPENAI_API_KEY; then
    run_one "agent-oracle" "$dataset" "openai-gpt5" \
      "$OUT_DIR/agent-${dataset}-openai-gpt5.json"
  else
    log_skip "track=agent dataset=$dataset system=openai-gpt5 reason=OPENAI_API_KEY missing"
  fi

  # google-gemini
  if need_vars GEMINI_API_KEY || need_vars GOOGLE_API_KEY; then
    run_one "agent-oracle" "$dataset" "google-gemini" \
      "$OUT_DIR/agent-${dataset}-google-gemini.json"
  else
    log_skip "track=agent dataset=$dataset system=google-gemini reason=GEMINI_API_KEY or GOOGLE_API_KEY missing"
  fi
}

# ---------------------------------------------------------------------------
# Track 3 — E2E (agent + brain)
# ---------------------------------------------------------------------------

run_together_track() {
  local dataset="$1"

  skip_track "together" && return
  skip_dataset "$dataset" && return

  log_info "--- Track 3 (together/E2E): $dataset ---"

  # All E2E runs need ANTHROPIC_API_KEY for judge.
  if ! need_vars ANTHROPIC_API_KEY; then
    log_skip "track=together dataset=$dataset reason=ANTHROPIC_API_KEY missing (judge required)"
    return
  fi

  # unison-agent E2E
  if need_vars UNISON_JWT || need_vars UNISON_LOCAL_EVAL_TENANT_ID; then
    run_one "agent-e2e" "$dataset" "unison-agent" \
      "$OUT_DIR/together-${dataset}-unison-agent.json"
  else
    log_skip "track=together dataset=$dataset system=unison-agent reason=UNISON_JWT or UNISON_LOCAL_EVAL_TENANT_ID missing"
  fi

  # claude-code E2E
  run_one "agent-e2e" "$dataset" "claude-code" \
    "$OUT_DIR/together-${dataset}-claude-code.json"

  # mem0-agent E2E
  if need_vars MEM0_API_KEY; then
    run_one "agent-e2e" "$dataset" "mem0-agent" \
      "$OUT_DIR/together-${dataset}-mem0-agent.json"
  else
    log_skip "track=together dataset=$dataset system=mem0-agent reason=MEM0_API_KEY missing"
  fi

  # anthropic-raw E2E
  run_one "agent-e2e" "$dataset" "anthropic-raw" \
    "$OUT_DIR/together-${dataset}-anthropic-raw.json"

  # openai-gpt5 E2E
  if need_vars OPENAI_API_KEY; then
    run_one "agent-e2e" "$dataset" "openai-gpt5" \
      "$OUT_DIR/together-${dataset}-openai-gpt5.json"
  else
    log_skip "track=together dataset=$dataset system=openai-gpt5 reason=OPENAI_API_KEY missing"
  fi

  # google-gemini E2E
  if need_vars GEMINI_API_KEY || need_vars GOOGLE_API_KEY; then
    run_one "agent-e2e" "$dataset" "google-gemini" \
      "$OUT_DIR/together-${dataset}-google-gemini.json"
  else
    log_skip "track=together dataset=$dataset system=google-gemini reason=GEMINI_API_KEY or GOOGLE_API_KEY missing"
  fi
}

# ---------------------------------------------------------------------------
# Main matrix
# ---------------------------------------------------------------------------

# Brain-only datasets
BRAIN_DATASETS=(bitempoqa longmemeval memoryagentbench musique msmarco)

# Agent-oracle datasets (all that have oracle_context support)
AGENT_DATASETS=(bitempoqa longmemeval memoryagentbench musique frames)

# E2E datasets (need load_brain_questions support)
TOGETHER_DATASETS=(bitempoqa longmemeval musique frames)

log_info "--- Track 1: Brain Only ---"
for ds in "${BRAIN_DATASETS[@]}"; do
  run_brain_track "$ds"
done
echo

log_info "--- Track 2: Agent Oracle ---"
for ds in "${AGENT_DATASETS[@]}"; do
  run_agent_track "$ds"
done
echo

log_info "--- Track 3: Agent E2E ---"
for ds in "${TOGETHER_DATASETS[@]}"; do
  run_together_track "$ds"
done
echo

# ---------------------------------------------------------------------------
# Write summary.json
# ---------------------------------------------------------------------------

log_info "--- Writing $SUMMARY_FILE ---"

python3 - <<PYEOF
import json, os, pathlib, datetime

out_dir = pathlib.Path("$OUT_DIR")
results = []

combo_lines = """${COMBO_RESULTS[*]}""".strip()
combos = [c for c in combo_lines.split(" ") if c] if combo_lines else []

for combo in combos:
    parts = combo.split("|")
    if len(parts) < 5:
        continue
    track, dataset, system, status, filepath = parts
    entry = {
        "track": track,
        "dataset": dataset,
        "system": system,
        "status": status,
        "file": filepath if filepath else None,
        "comprehensive_id": "$TS",
    }
    # Embed headline metrics if file exists
    if filepath and pathlib.Path(filepath).exists():
        try:
            data = json.loads(pathlib.Path(filepath).read_text())
            summary = data.get("summary", {})
            summaries = summary.get("summaries", [])
            sys_summary = next((s for s in summaries if s.get("system") == system), None)
            if sys_summary:
                entry["pass_rate"] = sys_summary.get("pass_rate")
                entry["recall_at_10"] = sys_summary.get("mean_recall_at_10")
                entry["cost_per_solved_usd"] = sys_summary.get("cost_per_solved_usd")
                entry["p50_latency_ms"] = sys_summary.get("p50_latency_ms")
                entry["n_questions"] = sys_summary.get("n_questions")
        except Exception as e:
            entry["parse_error"] = str(e)
    results.append(entry)

payload = {
    "comprehensive_id": "$TS",
    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "limit": $LIMIT,
    "judge": "$JUDGE",
    "n_combos": len(results),
    "n_done": sum(1 for r in results if r["status"] == "done"),
    "n_failed": sum(1 for r in results if r["status"] == "failed"),
    "combos": results,
}
pathlib.Path("$SUMMARY_FILE").write_text(json.dumps(payload, indent=2))
print(f"Written {len(results)} combo entries to $SUMMARY_FILE")
PYEOF

echo
log_info "=== Comprehensive run complete ==="
log_info "Output:  $OUT_DIR"
log_info "Log:     $LOG_FILE"
log_info "Summary: $SUMMARY_FILE"
log_info ""
log_info "Done combos:  $(grep -c '\[DONE\]' "$LOG_FILE" || echo 0)"
log_info "Skipped:      $(grep -c '\[SKIP\]' "$LOG_FILE" || echo 0)"
log_info "Failed:       $(grep -c '\[FAIL\]' "$LOG_FILE" || echo 0)"
log_info ""
log_info "Tip: open the web UI and check /?tab=leaderboard for the aggregated cross-dataset view."
log_info "Tip: publishable run → LIMIT=50 JUDGE=claude-opus-4-5-20250101 CONFIRM=1 bash scripts/run_comprehensive.sh"
