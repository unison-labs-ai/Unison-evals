"""Seed Letta's filesystem-agent corpus into a fresh Unison /wiki/ tenant.

The 10 fictional text files (people / pets / vehicles / addresses /
bank_accounts / credit_cards / employments / insurance_policies /
internet_accounts / medical_records) are vendored under
`vendor/letta-evals/letta-leaderboard/filesystem-agent/files/`. We
preserve the original filenames so the agent sees the same surface a
Letta agent would.
"""

from __future__ import annotations

from pathlib import Path

from ..tau_bench.brain_client import BrainPage, seed_pages_sync, wipe_tenant_sync

_REPO_ROOT = Path(__file__).resolve().parents[4]
CORPUS_DIR = (
    _REPO_ROOT / "vendor" / "letta-evals" / "letta-leaderboard" / "filesystem-agent" / "files"
)

# The 10 corpus files. Order is irrelevant for storage but explicit so we
# fail loudly if one goes missing in a future submodule update.
CORPUS_FILES = (
    "addresses.txt",
    "bank_accounts.txt",
    "credit_cards.txt",
    "employments.txt",
    "insurance_policies.txt",
    "internet_accounts.txt",
    "medical_records.txt",
    "people.txt",
    "pets.txt",
    "vehicles.txt",
)


SCHEMA_MD = """# Workspace — Context-Bench filesystem corpus

Ten plain-text files under /wiki/ describe a fictional set of people and
their entities. Files are flat text (line-oriented or pipe-delimited);
the actual format is documented inside each file.

## Layout

- `/wiki/people.txt`             — person records (id, name, age, …)
- `/wiki/addresses.txt`          — address records, linked to people
- `/wiki/pets.txt`                — pets owned by people
- `/wiki/vehicles.txt`            — vehicles owned by people
- `/wiki/employments.txt`         — jobs and employers
- `/wiki/bank_accounts.txt`       — bank accounts linked to people
- `/wiki/credit_cards.txt`        — credit cards linked to people
- `/wiki/insurance_policies.txt`  — insurance policies linked to people
- `/wiki/internet_accounts.txt`   — online accounts linked to people
- `/wiki/medical_records.txt`     — medical records linked to people

All cross-references between files use a `person_id` field (or a similar
key documented inside the file). Questions typically require joining
across two or more files.
"""


def _build_pages() -> list[BrainPage]:
    pages: list[BrainPage] = [
        BrainPage(path="/wiki/SCHEMA.md", body_md=SCHEMA_MD, kind="wiki_page"),
    ]
    missing: list[str] = []
    for name in CORPUS_FILES:
        src = CORPUS_DIR / name
        if not src.exists():
            missing.append(str(src))
            continue
        pages.append(
            BrainPage(
                path=f"/wiki/{name}",
                body_md=src.read_text(),
                kind="wiki_page",
            )
        )
    if missing:
        raise FileNotFoundError(
            "Letta corpus files not vendored. Run "
            "`git submodule update --init vendor/letta-evals`. Missing: " + ", ".join(missing)
        )
    return pages


def fresh_tenant(tenant_id: str, user_id: str) -> tuple[int, int]:
    """Wipe the tenant + seed the Context-Bench corpus from scratch.

    Returns (wiped_doc_count, seeded_page_count)."""
    counts = wipe_tenant_sync(tenant_id)
    wiped = counts.get("cortex_documents", 0)
    pages = _build_pages()
    seeded = seed_pages_sync(tenant_id, user_id, pages)
    return wiped, seeded
