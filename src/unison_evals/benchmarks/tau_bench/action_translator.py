"""Brain-state diff → tau_bench Action dispatcher (retail).

After every Unison turn we snapshot the wiki and compare against the
previous snapshot. Each detected change maps to one τ-bench Action that
we run through `env.step()` so policy guards (e.g. "no exchange on
returned orders") still fire.

Coverage targeted at smoke tasks 0/1/2 (exchange + return). Unmapped
diffs are logged but do not raise — reward will reflect the gap.

The translator is intentionally one-way (brain → tau_bench). The agent
writes the brain freely; we read what it did and dispatch. We do NOT
back-port env.step() rejections into the brain — for v1 that's the
deliberate v1.5 limitation (see UNISON_BENCHMARKS.md V2 list).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tau_bench.types import Action


@dataclass
class TranslatedAction:
    action: Action
    reason: str  # human-readable why this action was emitted
    source_path: str  # wiki path that triggered it


def translate(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> tuple[list[TranslatedAction], list[str]]:
    """Compute Actions to dispatch given before/after env.data-shaped dicts.

    Returns (actions, unmapped_diffs)."""
    actions: list[TranslatedAction] = []
    unmapped: list[str] = []

    actions.extend(_translate_orders(before.get("orders", {}), after.get("orders", {}), unmapped))
    actions.extend(_translate_users(before.get("users", {}), after.get("users", {}), unmapped))
    return actions, unmapped


# ── Orders ───────────────────────────────────────────────────────────────


def _translate_orders(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    unmapped: list[str],
) -> list[TranslatedAction]:
    out: list[TranslatedAction] = []
    for order_id, after_rec in after.items():
        before_rec = before.get(order_id)
        if before_rec is None:
            unmapped.append(f"orders/{order_id}: created — not a valid agent action")
            continue
        if before_rec == after_rec:
            continue

        path = f"/private/taubench/orders/{order_id.lstrip('#')}.md"
        status_before = before_rec.get("status")
        status_after = after_rec.get("status")

        # Address change on pending order
        if before_rec.get("address") != after_rec.get("address"):
            if status_before == "pending":
                addr = after_rec["address"]
                out.append(
                    TranslatedAction(
                        Action(
                            name="modify_pending_order_address",
                            kwargs={
                                "order_id": order_id,
                                "address1": addr.get("address1", ""),
                                "address2": addr.get("address2", ""),
                                "city": addr.get("city", ""),
                                "state": addr.get("state", ""),
                                "country": addr.get("country", ""),
                                "zip": addr.get("zip", ""),
                            },
                        ),
                        reason="address modified on pending order",
                        source_path=path,
                    )
                )
            else:
                unmapped.append(
                    f"orders/{order_id}: address changed but status={status_before} not pending"
                )

        # Status transitions
        if status_before != status_after:
            if status_before == "pending" and status_after == "cancelled":
                out.append(
                    TranslatedAction(
                        Action(
                            name="cancel_pending_order",
                            kwargs={
                                "order_id": order_id,
                                "reason": "no longer needed",
                            },
                        ),
                        reason="status: pending → cancelled",
                        source_path=path,
                    )
                )
            elif status_before == "delivered" and status_after in ("return requested", "returned"):
                ret = _emit_return(order_id, before_rec, after_rec, path)
                if ret:
                    out.append(ret)
                else:
                    unmapped.append(
                        f"orders/{order_id}: return requested but couldn't infer item_ids/payment_method"
                    )
            else:
                unmapped.append(
                    f"orders/{order_id}: unhandled status transition {status_before} → {status_after}"
                )

        # Items array changed (exchange or modify)
        if before_rec.get("items") != after_rec.get("items"):
            if status_before == "delivered":
                ex = _emit_exchange(order_id, before_rec, after_rec, path)
                if ex:
                    out.append(ex)
                else:
                    unmapped.append(
                        f"orders/{order_id}: items changed on delivered order but couldn't infer swap"
                    )
            elif status_before == "pending":
                ex = _emit_modify_items(order_id, before_rec, after_rec, path)
                if ex:
                    out.append(ex)
            else:
                unmapped.append(f"orders/{order_id}: items changed with status={status_before}")
    return out


def _emit_return(order_id: str, before: dict, after: dict, path: str) -> TranslatedAction | None:
    # τ-bench's return_delivered_order_items writes two canonical fields onto
    # the order: `return_items` (list[item_id]) and `return_payment_method_id`.
    # See vendor/tau-bench/.../tools/return_delivered_order_items.py:36-38.
    # Partial returns are legal, so we trust `return_items` verbatim instead
    # of inferring item_ids from payment_history.
    return_items = after.get("return_items")
    payment_method_id = after.get("return_payment_method_id")
    if not return_items or not payment_method_id:
        return None
    return TranslatedAction(
        Action(
            name="return_delivered_order_items",
            kwargs={
                "order_id": order_id,
                "item_ids": list(return_items),
                "payment_method_id": payment_method_id,
            },
        ),
        reason="status: delivered → return requested (return_items + return_payment_method_id present)",
        source_path=path,
    )


def _diff_item_ids(
    before_items: list[dict], after_items: list[dict]
) -> tuple[list[str], list[str]] | None:
    """Multiset diff over items[].item_id, preserving duplicate multiplicity.

    Returns (removed_ids, added_ids) in the order each ID appears in `before`
    / `after`. Robust to:
      - reordering (positional shuffles emit no action)
      - duplicates (7/1000 retail orders have repeated item_ids — partial
        replacement of a duplicated id must be detected, not silently
        dropped by set-membership)

    Returns None if the change isn't a clean swap (added != removed count
    after multiset diff)."""
    from collections import Counter

    before_ids = [it.get("item_id") for it in before_items if it.get("item_id")]
    after_ids = [it.get("item_id") for it in after_items if it.get("item_id")]

    # Walk before/after; each occurrence that doesn't have a matching
    # counterpart in the other side is recorded. This treats item_ids as
    # a multiset, so a partial swap of duplicated items is correctly
    # identified as one removed + one added.
    remaining_after: Counter = Counter(after_ids)
    removed: list[str] = []
    for iid in before_ids:
        if remaining_after[iid] > 0:
            remaining_after[iid] -= 1
        else:
            removed.append(iid)

    remaining_before: Counter = Counter(before_ids)
    added: list[str] = []
    for iid in after_ids:
        if remaining_before[iid] > 0:
            remaining_before[iid] -= 1
        else:
            added.append(iid)

    if not removed and not added:
        return None
    if len(removed) != len(added):
        return None
    return removed, added


def _emit_exchange(order_id: str, before: dict, after: dict, path: str) -> TranslatedAction | None:
    """Detect items[].item_id swaps for exchange_delivered_order_items."""
    diff = _diff_item_ids(before.get("items", []), after.get("items", []))
    if diff is None:
        return None
    swapped_old, swapped_new = diff
    # Payment method: take the new payment_history entry's payment_method_id
    ph_after = after.get("payment_history", [])
    ph_before = before.get("payment_history", [])
    new_entries = ph_after[len(ph_before) :]
    payment_method_id = (
        new_entries[0].get("payment_method_id")
        if new_entries
        else ph_before[-1].get("payment_method_id")
        if ph_before
        else None
    )
    return TranslatedAction(
        Action(
            name="exchange_delivered_order_items",
            kwargs={
                "order_id": order_id,
                "item_ids": swapped_old,
                "new_item_ids": swapped_new,
                "payment_method_id": payment_method_id,
            },
        ),
        reason=f"items swapped on delivered order ({len(swapped_old)} item(s))",
        source_path=path,
    )


def _emit_modify_items(
    order_id: str, before: dict, after: dict, path: str
) -> TranslatedAction | None:
    """Same shape as exchange, but for pending orders."""
    diff = _diff_item_ids(before.get("items", []), after.get("items", []))
    if diff is None:
        return None
    swapped_old, swapped_new = diff
    ph_before = before.get("payment_history", [])
    payment_method_id = ph_before[-1].get("payment_method_id") if ph_before else None
    return TranslatedAction(
        Action(
            name="modify_pending_order_items",
            kwargs={
                "order_id": order_id,
                "item_ids": swapped_old,
                "new_item_ids": swapped_new,
                "payment_method_id": payment_method_id,
            },
        ),
        reason=f"items swapped on pending order ({len(swapped_old)} item(s))",
        source_path=path,
    )


# ── Users ────────────────────────────────────────────────────────────────


def _translate_users(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    unmapped: list[str],
) -> list[TranslatedAction]:
    out: list[TranslatedAction] = []
    for user_id, after_rec in after.items():
        before_rec = before.get(user_id)
        if before_rec is None or before_rec == after_rec:
            continue
        if before_rec.get("address") != after_rec.get("address"):
            addr = after_rec["address"]
            out.append(
                TranslatedAction(
                    Action(
                        name="modify_user_address",
                        kwargs={
                            "user_id": user_id,
                            "address1": addr.get("address1", ""),
                            "address2": addr.get("address2", ""),
                            "city": addr.get("city", ""),
                            "state": addr.get("state", ""),
                            "country": addr.get("country", ""),
                            "zip": addr.get("zip", ""),
                        },
                    ),
                    reason="user address modified",
                    source_path=f"/private/taubench/users/{user_id}.md",
                )
            )
    return out
