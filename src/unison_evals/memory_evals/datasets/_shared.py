"""Shared dataset helpers — centralized so every benchmark loader reuses the
SAME sampling/split logic and conversation rendering instead of copy-pasting it.

Operates on normalized rows: each loader maps its upstream schema to dicts that
carry at least ``question_id`` and ``question_type`` (the keys the sampler keys
on), then routes through these helpers. Keeps EVAL_CATEGORY / EVAL_SPLIT /
EVAL_STRATIFIED behaviour identical across datasets.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


def maybe_stratify(rows: Any, limit: int | None) -> list:
    """Deterministic sampling shared by every dataset.

    - ``EVAL_CATEGORY=<type[,type...]>`` keeps only those ``question_type``s
      (isolate one weak category for low-noise before/after measurement).
    - ``EVAL_SPLIT=dev|holdout`` is a stable 50/50 partition by a hash of
      ``question_id`` — tune on dev, validate on holdout (never inspect holdout
      failures). A change that lifts dev but not holdout is overfit.
    - ``EVAL_STRATIFIED=proportional`` samples each category at its real share so
      a small dev sample mirrors the full-set category mix; any other truthy
      value = equal round-robin coverage. No-op without a limit.
    """
    rows = list(rows)
    cats = os.environ.get("EVAL_CATEGORY")
    if cats:
        wanted = {c.strip() for c in cats.split(",") if c.strip()}
        rows = [r for r in rows if str(r.get("question_type") or "?") in wanted]
    split = os.environ.get("EVAL_SPLIT")
    if split in ("dev", "holdout"):
        want_dev = split == "dev"
        rows = [
            r
            for r in rows
            if (int(hashlib.md5(str(r.get("question_id")).encode()).hexdigest(), 16) % 2 == 0)
            == want_dev
        ]
    mode = os.environ.get("EVAL_STRATIFIED")
    if not mode or limit is None:
        return rows
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(str(r.get("question_type") or "?"), []).append(r)
    keys = list(groups.keys())

    if mode == "proportional":
        import random

        total = sum(len(groups[k]) for k in keys)
        rng = random.Random(int(os.environ.get("EVAL_SEED", "1234")))
        raw = {k: limit * len(groups[k]) / total for k in keys}
        base = {k: int(raw[k]) for k in keys}
        remainder = limit - sum(base.values())
        for k in sorted(keys, key=lambda k: raw[k] - base[k], reverse=True)[:remainder]:
            base[k] += 1
        picked: list = []
        for k in keys:
            pool = list(groups[k])
            rng.shuffle(pool)
            picked.extend(pool[: base[k]])
        rng.shuffle(picked)
        return picked

    ordered: list = []
    while any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                ordered.append(groups[k].pop(0))
    return ordered


def with_question_date(date: str | None, question: str) -> str:
    """Prepend the conversation's reference date as the agent's "now" — required
    for temporal questions ("how long ago / how many weeks since")."""
    if date:
        return f"Today's date is {date}.\n\n{question}"
    return question


def _turn_role(turn: dict[str, Any]) -> str:
    return str(turn.get("role") or turn.get("speaker") or "?")


def _turn_text(turn: dict[str, Any]) -> str:
    return str(turn.get("content") or turn.get("text") or "")


def format_session(session: Any, date: str | None, idx: int) -> str:
    """Render one conversation session as a markdown document body. Handles both
    {role, content} (LongMemEval) and {speaker, text} (LOCOMO) turn shapes."""
    header = f"## Session {idx + 1}"
    if date:
        header += f" — {date}"
    parts: list[str] = [header]
    if not isinstance(session, list):
        parts.append(str(session))
        return "\n".join(parts)
    for turn in session:
        if isinstance(turn, dict):
            parts.append(f"{_turn_role(turn).upper()}: {_turn_text(turn)}")
        else:
            parts.append(str(turn))
    return "\n".join(parts)


def format_oracle(sessions: list[Any], dates: list[str | None]) -> str:
    """Flatten all sessions into one oracle context string (Track 2)."""
    if not sessions:
        return ""
    parts: list[str] = []
    for i, session in enumerate(sessions):
        parts.append(format_session(session, dates[i] if i < len(dates) else None, i))
        parts.append("")
    return "\n".join(parts)


def stable_id(prefix: str, payload: Any) -> str:
    """Deterministic id when upstream has none — hash the payload."""
    seed = json.dumps(payload, sort_keys=True, default=str)
    return f"{prefix}-" + hashlib.sha256(seed.encode()).hexdigest()[:12]
