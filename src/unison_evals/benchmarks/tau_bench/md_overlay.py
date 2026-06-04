"""env.data ↔ /private/taubench/ filesystem codec for τ-bench retail.

τ-bench env.data is a dict-of-dict:
    { "orders":   {order_id: OrderRecord},
      "users":    {user_id:  UserRecord},
      "products": {product_id: ProductRecord} }

We serialize each record as `/private/taubench/<table>/<id>.md` with the entire record
embedded as a fenced JSON block (NOT YAML frontmatter — retail records have
position-sensitive lists and deep-nested objects that YAML round-trips
unreliably; JSON is unambiguous).

`#` is stripped from order IDs to make valid path slugs:
    "#W2611340" → "/private/taubench/orders/W2611340.md"

Two extra documents seed alongside the records:
    /private/taubench/policy.md   — τ-bench's wiki text (return rules etc.)
    /private/taubench/SCHEMA.md   — tells the Unison agent what's where
"""

from __future__ import annotations

import json
from typing import Any

from .brain_client import BrainPage

# ── Path helpers ─────────────────────────────────────────────────────────


def order_path(order_id: str) -> str:
    return f"/private/taubench/orders/{order_id.lstrip('#')}.md"


def user_path(user_id: str) -> str:
    return f"/private/taubench/users/{user_id}.md"


def product_path(product_id: str) -> str:
    return f"/private/taubench/products/{product_id}.md"


def order_id_from_path(path: str) -> str:
    # /private/taubench/orders/W2611340.md → "#W2611340"
    return "#" + path.rsplit("/", 1)[-1].removesuffix(".md")


def user_id_from_path(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".md")


def product_id_from_path(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".md")


# ── Serialization ────────────────────────────────────────────────────────


def _to_md(record: dict[str, Any]) -> str:
    """Embed a JSON record inside a fenced markdown block.

    The agent reads via bash; cat returns the markdown source verbatim.
    Lossless: position-sensitive lists, nested objects, and numeric types
    all round-trip through json.loads(json.dumps(x))."""
    return "```json\n" + json.dumps(record, indent=2, sort_keys=False) + "\n```\n"


def _from_md(body_md: str) -> dict[str, Any]:
    """Parse the JSON fenced block back to a dict. Tolerates surrounding
    text — extracts the first ```json ... ``` block."""
    fence = "```json"
    start = body_md.find(fence)
    if start == -1:
        raise ValueError(f"no json fence in body: {body_md[:200]!r}")
    start += len(fence)
    end = body_md.find("```", start)
    if end == -1:
        raise ValueError("unclosed json fence")
    return json.loads(body_md[start:end])


# ── Public seeding API ───────────────────────────────────────────────────


SCHEMA_MD = """# Workspace schema — retail customer-service brain

The brain holds the live retail database. Mutations you write to these
files are persisted and become the source of truth for the store.

## Layout

- `/private/taubench/orders/<id>.md`   — one file per order
- `/private/taubench/users/<id>.md`    — one file per customer profile
- `/private/taubench/products/<id>.md` — product catalogue, each with `variants[]`

Each file contains a fenced ` ```json ... ``` ` block holding the canonical
record. The JSON block IS the data; surrounding markdown is ignored. To
mutate state, overwrite the file with the updated JSON.

## File naming

- Order ids in JSON include the `#` prefix (`order_id: "#W2378156"`); the
  filename strips it: `/private/taubench/orders/W2378156.md`.
- User ids and product ids match the filename exactly.

## Record shapes

- `users/<id>.md` — `name`, `address`, `email`, `payment_methods`, and
  `orders[]`. The `orders[]` array is the authoritative list of every
  order belonging to that customer.
- `orders/<id>.md` — `order_id`, `user_id`, `address`, `items[]` (each
  with `item_id`, `product_id`, `price`, `options{}`), `fulfillments[]`,
  `status`, `payment_history[]`.
- `products/<id>.md` — `name`, `product_id`, `variants{<item_id>: {
  options{}, available, price }}`.

## Mutations

The system observes file edits and dispatches them through the live
ordering system; policy guards still fire and will reject invalid
transitions. Canonical field shapes for each business action:

- **Cancel** a `pending` order: set `status` to `"cancelled"`.
- **Modify address** on a `pending` order: edit `address.*`.
- **Modify items** on a `pending` order: replace one or more
  `items[].item_id` entries with new variants.
- **Exchange items** on a `delivered` order: replace one or more
  `items[].item_id` entries with new variants AND append a
  `payment_history` entry with `{transaction_type, amount, payment_method_id}`.
- **Return items** from a `delivered` order: set `status` to
  `"return requested"`, add `return_items: [<item_id>, ...]` and
  `return_payment_method_id: "<id>"` to the order record.

The fenced ` ```json ` block must remain valid JSON after every edit.
"""


def build_seed_pages(env_data: dict[str, Any]) -> list[BrainPage]:
    """Materialize env.data + SCHEMA into a flat list of BrainPages ready
    for bulk INSERT.

    Policy is NOT seeded as a brain file — it's injected as part of the
    first user message in mode_b_agent (matching Mode A's system-prompt
    placement, so the policy gets the same authoritative framing both
    modes give it)."""
    pages: list[BrainPage] = [
        BrainPage(path="/private/taubench/SCHEMA.md", body_md=SCHEMA_MD, kind="raw"),
    ]
    for order_id, rec in env_data.get("orders", {}).items():
        pages.append(BrainPage(path=order_path(order_id), body_md=_to_md(rec), kind="raw"))
    for user_id, rec in env_data.get("users", {}).items():
        pages.append(BrainPage(path=user_path(user_id), body_md=_to_md(rec), kind="raw"))
    for product_id, rec in env_data.get("products", {}).items():
        pages.append(
            BrainPage(path=product_path(product_id), body_md=_to_md(rec), kind="raw")
        )
    return pages


def parse_snapshot(
    snapshot: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Re-hydrate a brain snapshot back into env.data shape.

    Returns {table: {id: record}} — only includes orders/users/products,
    ignores SCHEMA.md / policy.md / anything outside the three table dirs.
    """
    orders, users, products = {}, {}, {}
    for path, body in snapshot.items():
        try:
            rec = _from_md(body)
        except ValueError:
            continue
        if path.startswith("/private/taubench/orders/"):
            orders[order_id_from_path(path)] = rec
        elif path.startswith("/private/taubench/users/"):
            users[user_id_from_path(path)] = rec
        elif path.startswith("/private/taubench/products/"):
            products[product_id_from_path(path)] = rec
    return {"orders": orders, "users": users, "products": products}
