"""Task instances and the ground-truth oracle.

Task text is the natural-language customer message given to the agent.
Ground truth is computed from the base database by `solve()` (never
hard-coded), so data and answers can't drift apart.
"""
import re
import sqlite3
from datetime import datetime, timedelta

from rlvr_task.config import (
    CURRENT_DATE,
    DB_PATH,
    DEFAULT_REFUND_RATIO,
    MANUAL_REVIEW_THRESHOLD,
    PHOTO_MAX_HOURS_AFTER_DELIVERY,
    RETURN_ARRIVED_STATUS,
    RETURN_WINDOW_DAYS,
)

TASK_TEMPLATE = (
    "【售后工单】订单号 {order_id}：我现在收到的快递是坏的，照片已经上传了，可以退货吗？"
    "请按平台售后政策处理本工单，并以 JSON 输出最终结果："
    '{{"order_id": str, "decision": "refund|reject|escalate|wait_return", '
    '"refund_amount": number, "refund_id": str|null, '
    '"reject_reason": "outside_window|invalid_evidence|non_returnable"|null}}'
    "（decision 为 reject 时必须给出 reject_reason，否则为 null）"
)

TASKS = {
    oid: {"task_id": f"refund-{oid}", "order_id": oid, "text": TASK_TEMPLATE.format(order_id=oid)}
    for oid in ("O1001", "O1002", "O1003", "O1004", "O1005", "O1006", "O1007", "O1008", "O1009", "O1010")
}

ORDER_IN_TEXT_RE = re.compile(r"O\d{4}")


def find_task(task_text):
    """Map task text (or a bare task_id / order_id) back to a task instance."""
    if not isinstance(task_text, str):
        return None
    for task in TASKS.values():
        if task_text == task["text"] or task_text == task["task_id"]:
            return task
    match = ORDER_IN_TEXT_RE.search(task_text)
    return TASKS.get(match.group(0)) if match else None


def _parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def valid_photo_items(items, attachments, delivered):
    """Item ids of this order that have at least one photo taken within the window."""
    ids = {it["item_id"] for it in items}
    deadline = delivered + timedelta(hours=PHOTO_MAX_HOURS_AFTER_DELIVERY)
    return {a["item_id"] for a in attachments
            if a["item_id"] in ids and delivered <= _parse_dt(a["taken_at"]) <= deadline}


def eligible_items(items, attachments, delivered):
    """Per-item evidence rule shared by the oracle.

    A photo is valid if taken within PHOTO_MAX_HOURS_AFTER_DELIVERY after delivery.
    For each returnable item with valid photos:
        accepted_qty = min(max claimed damaged_qty, number of valid photos, bought qty)
    (every damaged unit needs at least one photo; one unit may have several).
    Returns {item_id: accepted_qty}.
    """
    by_id = {it["item_id"]: it for it in items}
    deadline = delivered + timedelta(hours=PHOTO_MAX_HOURS_AFTER_DELIVERY)
    valid = {}
    for att in attachments:
        if delivered <= _parse_dt(att["taken_at"]) <= deadline and att["item_id"] in by_id:
            valid.setdefault(att["item_id"], []).append(att["damaged_qty"])
    result = {}
    for item_id, claims in valid.items():
        item = by_id[item_id]
        if not item["returnable"]:
            continue
        qty = min(max(claims), len(claims), item["qty"])
        if qty > 0:
            result[item_id] = qty
    return result


def solve(order_id, db_path=DB_PATH):
    """Reference oracle: apply the business rules to the base DB.

    Rule precedence:
      1. outside tier return window (from delivery date)            -> reject
      2. no returnable item with a valid photo (<=24h after delivery) -> reject
         (non_returnable if valid photos exist but only for non-returnable items,
          otherwise invalid_evidence)
      3. refund amount > tier review threshold                         -> escalate
      4. return parcel not at merchant                                 -> wait_return
      5. otherwise                                                     -> refund
    refund_amount = sum(unit_price * accepted_qty * ratio) over eligible items
    for non-reject decisions, 0 for reject.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cust = conn.execute(
            "SELECT c.tier FROM orders o JOIN customers c ON o.customer_id=c.customer_id "
            "WHERE o.order_id=?", (order_id,)).fetchone()
        delivery = conn.execute("SELECT * FROM deliveries WHERE order_id=?", (order_id,)).fetchone()
        atts = [dict(r) for r in conn.execute("SELECT * FROM attachments WHERE order_id=?", (order_id,))]
        items = [dict(r) for r in conn.execute("SELECT * FROM order_items WHERE order_id=?", (order_id,))]
    finally:
        conn.close()

    def answer(decision, amount=0.0, reason="", item_ids=(), reject_reason=None):
        return {"order_id": order_id, "decision": decision, "refund_amount": round(amount, 2),
                "reject_reason": reject_reason, "reason": reason, "item_ids": sorted(item_ids)}

    if cust is None or delivery is None or not atts:
        return answer("reject", reason="missing order/delivery/evidence", reject_reason="invalid_evidence")

    tier = cust["tier"]
    delivered = _parse_dt(delivery["delivered_at"])
    today = datetime.strptime(CURRENT_DATE, "%Y-%m-%d")
    if (today.date() - delivered.date()).days > RETURN_WINDOW_DAYS[tier]:
        return answer("reject", reason="outside return window", reject_reason="outside_window")

    accepted = eligible_items(items, atts, delivered)
    if not accepted:
        # non_returnable: every item with valid photo evidence is non-returnable;
        # invalid_evidence: no item has a valid photo at all.
        if valid_photo_items(items, atts, delivered):
            return answer("reject", reason="damaged items are non-returnable", reject_reason="non_returnable")
        return answer("reject", reason="no valid photo evidence", reject_reason="invalid_evidence")
    price = {it["item_id"]: it["unit_price"] for it in items}
    amount = sum(price[i] * q for i, q in accepted.items()) * DEFAULT_REFUND_RATIO

    if amount > MANUAL_REVIEW_THRESHOLD[tier]:
        return answer("escalate", amount, "amount above tier review threshold", accepted)
    if delivery["return_status"] != RETURN_ARRIVED_STATUS:
        return answer("wait_return", amount, "return parcel not yet at merchant", accepted)
    return answer("refund", amount, "eligible", accepted)
