"""Postgres-direct brain ops for Phase-2 τ-bench smoke.

Mode B's translator needs to read brain state after every agent turn and
wipe state between tasks. Unison has no `/api/rest/brain/*` endpoints yet
(v2 ask), so we hit Postgres directly. Local-only — the dedicated eval
tenant is the single namespace.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import asyncpg

DEFAULT_DSN = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"


@dataclass(frozen=True)
class BrainPage:
    path: str
    body_md: str
    kind: str


def _dsn() -> str:
    return os.environ.get("UNISON_DB_URL") or DEFAULT_DSN


async def _connect() -> asyncpg.Connection:
    return await asyncpg.connect(_dsn())


# Tenant-scoped tables Memory-v2 + the agent runtime populate during a turn.
# All must wipe between tasks for iid scoring — leaving any of them lets task
# N see facts/messages/sessions/jobs accumulated by tasks 1..N-1.
_TENANT_SCOPED_RESET_TABLES = (
    # Memory-v2 signal pipeline
    "cortex_signals",
    "cortex_facts",
    "cortex_trajectories",
    "cortex_conflicts",
    # Agent runtime state
    "agent_messages",
    "agent_events",
    "agent_input_requests",
    "agent_sessions",
    "cortex_runs",
    "cortex_run_steps",
    "cortex_escalations",
    # Background-job queue (kill any pending extract.turn etc.)
    "cortex_jobs",
    # Entity graph (re-built from /wiki/ on next turn)
    "cortex_aliases",
    "cortex_entities",
    "cortex_links",
    "cortex_tags",
    "cortex_sync_state",
    # Brain content — keep last so FK cascades have already cleared
    "cortex_documents",
)


async def wipe_tenant(tenant_id: str) -> dict[str, int]:
    """Hard-delete every tenant-scoped row across Memory-v2 + agent tables.
    Returns {table: rows_deleted}. Used between tasks to reset to iid state."""
    conn = await _connect()
    try:
        counts: dict[str, int] = {}
        for table in _TENANT_SCOPED_RESET_TABLES:
            result = await conn.execute(
                f"DELETE FROM {table} WHERE tenant_id = $1",
                tenant_id,
            )
            counts[table] = int(result.split()[-1]) if result else 0
        return counts
    finally:
        await conn.close()


# Back-compat wrapper — old callers expected a single int.
async def wipe_wiki(tenant_id: str) -> int:
    counts = await wipe_tenant(tenant_id)
    return counts.get("cortex_documents", 0)


async def seed_pages(
    tenant_id: str,
    user_id: str,
    pages: list[BrainPage],
) -> int:
    """Bulk-INSERT wiki pages. Skips the embedding column entirely
    (semantic search isn't used by the agent's bash workflow)."""
    conn = await _connect()
    try:
        await conn.executemany(
            """
            INSERT INTO cortex_documents
                (tenant_id, kind, path, body_md, actor_kind, actor_id, created_by, owner_user_id)
            VALUES ($1::uuid, $2, $3, $4, 'human', $5::text, $6::uuid, $7::uuid)
            ON CONFLICT (tenant_id, path) WHERE parent_id IS NULL AND deleted_at IS NULL
            DO UPDATE SET body_md = EXCLUDED.body_md, updated_at = now()
            """,
            [(tenant_id, p.kind, p.path, p.body_md, user_id, user_id, user_id) for p in pages],
        )
        return len(pages)
    finally:
        await conn.close()


async def snapshot_wiki(tenant_id: str, prefix: str = "/wiki/") -> dict[str, str]:
    """Return {path: body_md} for every live wiki page under prefix."""
    conn = await _connect()
    try:
        rows = await conn.fetch(
            """
            SELECT path, body_md FROM cortex_documents
            WHERE tenant_id = $1 AND path LIKE $2 AND deleted_at IS NULL
            ORDER BY path
            """,
            tenant_id,
            f"{prefix}%",
        )
        return {r["path"]: r["body_md"] for r in rows}
    finally:
        await conn.close()


# Sync wrappers for non-async callers (the τ-bench Agent loop is sync).
def wipe_wiki_sync(tenant_id: str) -> int:
    return asyncio.run(wipe_wiki(tenant_id))


def wipe_tenant_sync(tenant_id: str) -> dict[str, int]:
    return asyncio.run(wipe_tenant(tenant_id))


# ── Trajectory dump ──────────────────────────────────────────────────────


async def dump_trajectory(tenant_id: str, session_id: str) -> list[dict]:
    """Reconstruct the agent's full per-turn trajectory from agent_messages.

    Each message row has `content` as a JSONB array of content parts
    (text / tool_use / tool_result). We return a flat list of structured
    entries ordered by created_at — readable enough to inspect what the
    agent actually ran without modifying Unison.

    Output shape per row:
      {
        "ts": iso8601,
        "role": "user" | "assistant" | "tool" | "system",
        "text": str            # concatenated text blocks
        "tool_uses": [{"name": str, "input": dict, "id": str}, ...]
        "tool_results": [{"tool_use_id": str, "content": str}, ...]
      }
    """
    conn = await _connect()
    try:
        rows = await conn.fetch(
            """
            SELECT created_at, role, content
            FROM agent_messages
            WHERE tenant_id = $1 AND session_id = $2
            ORDER BY created_at, id
            """,
            tenant_id,
            session_id,
        )
    finally:
        await conn.close()

    import json as _json

    out: list[dict] = []
    for row in rows:
        content = row["content"]
        if isinstance(content, str):
            try:
                content = _json.loads(content)
            except Exception:
                content = [{"type": "text", "text": content}]
        # role=tool rows often store content as a single object, not a list.
        # Normalise to a list of parts so the loop below handles both.
        if isinstance(content, dict):
            parts = [content]
        elif isinstance(content, list):
            parts = content
        else:
            parts = []

        text_parts: list[str] = []
        tool_uses: list[dict] = []
        tool_results: list[dict] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            t = part.get("type") or ""
            # Unison stores content-part types as kebab-case (`tool-call`,
            # `tool-result`); we also accept Anthropic-SDK underscore form
            # for forward-compat.
            if t == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            elif t == "reasoning" and isinstance(part.get("text"), str):
                text_parts.append(f"[reasoning] {part['text']}")
            elif t in ("tool-call", "tool_use"):
                tool_uses.append(
                    {
                        "id": part.get("toolCallId") or part.get("id", ""),
                        "name": part.get("toolName") or part.get("name", ""),
                        "input": part.get("input", {}),
                    }
                )
            elif t in ("tool-result", "tool_result") or (
                # role=tool rows store the result directly without a type field
                str(row["role"]) == "tool" and isinstance(part.get("result"), dict)
            ):
                result = part.get("result")
                if isinstance(result, dict):
                    stdout = result.get("stdout", "") or ""
                    stderr = result.get("stderr", "") or ""
                    truncated = result.get("truncated", False)
                    body = stdout
                    if stderr:
                        body = f"{body}\n[stderr] {stderr}" if body else f"[stderr] {stderr}"
                    if truncated:
                        body = f"{body}\n[truncated]"
                    tr_content: str | list = body
                else:
                    raw = part.get("content", "")
                    if isinstance(raw, list):
                        tr_content = "".join(
                            p.get("text", "")
                            if isinstance(p, dict) and p.get("type") == "text"
                            else ""
                            for p in raw
                        )
                    else:
                        tr_content = str(raw)
                tool_results.append(
                    {
                        "tool_use_id": part.get("toolCallId") or part.get("tool_use_id", ""),
                        "content": tr_content,
                    }
                )
        out.append(
            {
                "ts": row["created_at"].isoformat() if row["created_at"] else None,
                "role": str(row["role"]),
                "text": "\n".join(text_parts),
                "tool_uses": tool_uses,
                "tool_results": tool_results,
            }
        )
    return out


def dump_trajectory_sync(tenant_id: str, session_id: str) -> list[dict]:
    return asyncio.run(dump_trajectory(tenant_id, session_id))


def seed_pages_sync(tenant_id: str, user_id: str, pages: list[BrainPage]) -> int:
    return asyncio.run(seed_pages(tenant_id, user_id, pages))


def snapshot_wiki_sync(tenant_id: str, prefix: str = "/wiki/") -> dict[str, str]:
    return asyncio.run(snapshot_wiki(tenant_id, prefix))
