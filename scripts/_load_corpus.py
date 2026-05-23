"""Corpus loader helper — streams passages from HuggingFace and bulk-ingests
them into a BrainAdapter.

Called by load_corpus_msmarco.sh. Not meant to be imported; run directly:

    python scripts/_load_corpus.py \
        --system pgvector-naive \
        --subset 100k \
        [--unison-tenant-id <uuid>]

Exit codes: 0 = success, 1 = error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the project src is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unison_evals.memory_evals.adapters import BRAIN_REGISTRY, get_brain_adapter  # noqa: E402
from unison_evals.types import Document  # noqa: E402

# Embedding-batch size for bulk ingest. Smaller than the eval batch (96) to
# keep memory usage bounded during a 6-hour 8.8M-passage run.
INGEST_BATCH_SIZE = 256

# Subset → passage count limits.
SUBSET_LIMITS: dict[str, int | None] = {
    "100k": 100_000,
    "1M": 1_000_000,
    "full": None,  # all 8.8M
}

HF_DATASET = "microsoft/ms_marco"
HF_CONFIG = "v1.1"
HF_SPLIT = "train"  # training split has all 8.8M passages in passage_id/text form


def _passage_to_doc(passage_id: str, passage_text: str) -> Document:
    """Convert a raw MS MARCO passage to a Document.

    Path scheme: /msmarco/passages/<passage_id>.md

    This MUST match the scheme used in MsMarcoDataset._row_to_scale_question()
    so that gold_doc_paths align with ingested doc_paths.
    """
    return Document(
        path=f"/msmarco/passages/{passage_id}.md",
        body=passage_text,
        metadata={"passage_id": passage_id, "source": "ms_marco_v1.1"},
    )


async def _load(system: str, subset: str, unison_tenant_id: str | None) -> None:
    limit = SUBSET_LIMITS.get(subset)
    if subset not in SUBSET_LIMITS:
        print(
            f"Unknown subset '{subset}'. Choose from: {', '.join(SUBSET_LIMITS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if system not in BRAIN_REGISTRY:
        print(
            f"Unknown brain system '{system}'. Available: {', '.join(sorted(BRAIN_REGISTRY))}",
            file=sys.stderr,
        )
        sys.exit(1)

    if system == "unison-brain":
        print(
            "WARNING: unison-brain adapter does not support ingest() in v1.0.\n"
            "Use `brain-cli import` or wait for the Unison seedDocs endpoint.\n"
            "Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    adapter = get_brain_adapter(system)
    await adapter.setup()

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print("`datasets` library not installed. Run `uv sync`.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading MS MARCO {HF_CONFIG} passages from HuggingFace …")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT, streaming=True)

    seen_ids: set[str] = set()
    batch: list[Document] = []
    total_ingested = 0

    for row in ds:
        if limit is not None and total_ingested >= limit:
            break

        passages = row.get("passages", {})
        if isinstance(passages, dict):
            ids = passages.get("passage_id", [])
            texts = passages.get("passage_text", [])
        else:
            ids = [p.get("passage_id", "") for p in passages]
            texts = [p.get("passage_text", "") for p in passages]

        for pid, text in zip(ids, texts):
            pid = str(pid)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            if not text:
                continue
            batch.append(_passage_to_doc(pid, text))

            if len(batch) >= INGEST_BATCH_SIZE:
                await adapter.ingest(batch)
                total_ingested += len(batch)
                batch = []
                if total_ingested % 10_000 == 0:
                    print(f"  … {total_ingested:,} passages ingested")

                if limit is not None and total_ingested >= limit:
                    break

    if batch:
        await adapter.ingest(batch)
        total_ingested += len(batch)

    await adapter.teardown()
    print(f"Done. Ingested {total_ingested:,} passages into {system}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-load MS MARCO passages into a brain adapter.")
    parser.add_argument("--system", required=True, help="Brain adapter name (e.g. pgvector-naive)")
    parser.add_argument(
        "--subset",
        default="100k",
        choices=list(SUBSET_LIMITS),
        help="Passage subset to load. Default: 100k (~$0.40 embedding cost, ~10 min).",
    )
    parser.add_argument(
        "--unison-tenant-id",
        default=None,
        help="Unison tenant ID (only needed for unison-brain, currently unsupported).",
    )
    args = parser.parse_args()
    asyncio.run(_load(args.system, args.subset, args.unison_tenant_id))


if __name__ == "__main__":
    main()
