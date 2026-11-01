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
    '"refund_amount": number, "refund_id": str|null}}'
)

TASKS = {
    oid: {"task_id": f"refund-{oid}", "order_id": oid, "text": TASK_TEMPLATE.format(order_id=oid)}
    for oid in ("O1001", "O1002", "O1003", "O1004", "O1005", "O1006", "O1007", "O1008", "O1009")
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


def solve(order_id, db_path=DB_PATH):
    """Reference oracle: apply the business rules to the base DB.

    Rule precedence:
      1. damaged item not returnable          -> reject
      2. outside tier return window           -> reject  (counted from delivery date)
      3. no photo / photo > 24h after delivery-> reject
      4. amount > tier review threshold       -> escalate
      5. return parcel not at merchant        -> wait_return
      6. otherwise                            -> refund
    refund_amount = unit_price * damaged_qty * ratio for non-reject decisions, 0 for reject.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cust = conn.execute(
            "SELECT c.tier FROM orders o JOIN customers c ON o.customer_id=c.customer_id "
            "WHERE o.order_id=?", (order_id,)).fetchone()
        delivery = conn.execute("SELECT * FROM deliveries WHERE order_id=?", (order_id,)).fetchone()
        att = conn.execute("SELECT * FROM attachments WHERE order_id=? ORDER BY taken_at",
                           (order_id,)).fetchone()
        item = None
        if att is not None:
            item = conn.execute("SELECT * FROM order_items WHERE item_id=?", (att["item_id"],)).fetchone()
    finally:
        conn.close()

    def answer(decision, amount=0.0, reason=""):
        return {"order_id": order_id, "decision": decision,
                "refund_amount": round(amount, 2), "reason": reason}

    if cust is None or delivery is None or att is None or item is None:
        return answer("reject", reason="missing order/delivery/evidence")

    tier = cust["tier"]
    delivered = _parse_dt(delivery["delivered_at"])
    today = datetime.strptime(CURRENT_DATE, "%Y-%m-%d")
    amount = item["unit_price"] * att["damaged_qty"] * DEFAULT_REFUND_RATIO

    if not item["returnable"]:
        return answer("reject", reason="item not returnable")
    if (today.date() - delivered.date()).days > RETURN_WINDOW_DAYS[tier]:
        return answer("reject", reason="outside return window")
    taken = _parse_dt(att["taken_at"])
    if not (delivered <= taken <= delivered + timedelta(hours=PHOTO_MAX_HOURS_AFTER_DELIVERY)):
        return answer("reject", reason="photo not taken within 24h after delivery")
    if amount > MANUAL_REVIEW_THRESHOLD[tier]:
        return answer("escalate", amount, reason="amount above tier review threshold")
    if delivery["return_status"] != RETURN_ARRIVED_STATUS:
        return answer("wait_return", amount, reason="return parcel not yet at merchant")
    return answer("refund", amount, reason="eligible")
