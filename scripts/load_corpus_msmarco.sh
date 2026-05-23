#!/usr/bin/env bash
# load_corpus_msmarco.sh — one-time bulk-load MS MARCO passages into a brain.
#
# Usage:
#   SYSTEM=pgvector-naive bash scripts/load_corpus_msmarco.sh
#   SYSTEM=unison-brain UNISON_TENANT_ID=<uuid> bash scripts/load_corpus_msmarco.sh
#   SUBSET=1M SYSTEM=pgvector-naive bash scripts/load_corpus_msmarco.sh
#
# Environment variables:
#   SYSTEM           Brain adapter name. Required. Use `unison-evals systems` to list.
#   SUBSET           Passage subset: 100k (default), 1M, full (all 8.8M).
#   UNISON_TENANT_ID Unison tenant UUID (only needed for unison-brain, currently unsupported).
#
# Cost (one-time, OpenAI text-embedding-3-small @ $0.02/1M tokens):
#   100k passages  ≈ $0.40    ~10 min
#   1M   passages  ≈ $4.00    ~60 min
#   8.8M passages  ≈ $35      ~6 hours
#
# Disk (vector storage in Postgres):
#   100k passages  ≈ 1 GB
#   1M   passages  ≈ 10 GB
#   8.8M passages  ≈ 80 GB
#
# Critical: `pgvector-naive` writes documents with kind="raw" (no Unison
# extract pipeline), so embedding cost is the only per-doc cost.
# For `unison-brain`: the adapter does not yet support ingest() — the script
# will exit with an error and instructions to use `brain-cli import` instead.
#
# After loading, verify a sample query:
#   uv run unison-evals run \
#     --systems pgvector-naive \
#     --dataset msmarco \
#     --track scale \
#     --corpus msmarco-passages-v1-100k \
#     --limit 20

set -euo pipefail

SYSTEM="${SYSTEM:-pgvector-naive}"
SUBSET="${SUBSET:-100k}"
UNISON_TENANT_ID="${UNISON_TENANT_ID:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== MS MARCO corpus loader ==="
echo "  System : ${SYSTEM}"
echo "  Subset : ${SUBSET}"
echo "  Repo   : ${REPO_ROOT}"
echo ""

# Validate environment.
if [[ -z "${SYSTEM}" ]]; then
    echo "ERROR: SYSTEM env var is required." >&2
    exit 1
fi

# Run the Python helper inside the project's venv via uv.
cd "${REPO_ROOT}"
uv run python scripts/_load_corpus.py \
    --system "${SYSTEM}" \
    --subset "${SUBSET}" \
    ${UNISON_TENANT_ID:+--unison-tenant-id "${UNISON_TENANT_ID}"}

echo ""
echo "=== Load complete ==="
echo "Run a quick smoke test with:"
echo ""
echo "  uv run unison-evals run \\"
echo "    --systems ${SYSTEM} \\"
echo "    --dataset msmarco \\"
echo "    --track scale \\"
echo "    --corpus msmarco-passages-v1-${SUBSET} \\"
echo "    --limit 20"
