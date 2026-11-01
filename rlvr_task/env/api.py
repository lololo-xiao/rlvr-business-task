"""After-sales refund environment exposed to the agent as Python class methods.

The agent never touches SQLite directly; every capability goes through a
public method of `RefundEnv`. Each method returns a dict:
    {"ok": True, ...data}  or  {"ok": False, "error": "<reason>"}
and never raises, so a bad call cannot crash a training rollout.

Each env instance works on a private in-memory copy of the base database,
so episodes are isolated and fully reproducible.
"""
import re
import sqlite3
from pathlib import Path

from rlvr_task.config import (
    CURRENT_DATE,
    DB_PATH,
    DEFAULT_REFUND_RATIO,
    MANUAL_REVIEW_THRESHOLD,
    PHOTO_MAX_HOURS_AFTER_DELIVERY,
    RETURN_WINDOW_DAYS,
)

ORDER_ID_RE = re.compile(r"^O\d{4}$")
CUSTOMER_ID_RE = re.compile(r"^C\d{2}$")

# Tool descriptions handed to the agent (e.g. as a tool schema in the harness).
TOOL_DESCRIPTIONS = {
    "check_order": "check_order(customer_id) -> 该顾客的历史订单列表（订单号、下单日期、订单总额、状态）。",
    "get_policy": "get_policy(order_id) -> 订单商品明细与售后政策（policy_id、customer_id、商品单价/数量/是否可退、退货规则说明）。退款前必须先获取政策。",
    "check_attachment": "check_attachment(order_id) -> 顾客为该订单上传的附件（对应商品、损坏描述、损坏数量、拍摄时间）。",
    "check_delivery": "check_delivery(order_id) -> 物流信息（承运商、签收时间、退货包裹单号与状态）。",
    "get_customer_tier": "get_customer_tier(customer_id) -> 顾客会员等级（bronze/silver/gold）及联系方式。",
    "need_permission": "need_permission(customer_id, amount) -> 该退款金额对该顾客是否需要人工审核。",
    "get_address": "get_address(order_id) -> 顾客收货地址与商家退货地址。",
    "notify_refund": "notify_refund(order_id, message) -> 通知顾客（如提醒按要求寄回商品）。",
    "dryrun_refund": "dryrun_refund(order_id, item_id, amount, policy_id) -> 退款安全预检（须已调用 get_policy 和 check_attachment；只做安全检查，不判断是否符合退货政策），通过则返回一次性 dryrun_token。",
    "issue_refund": "issue_refund(order_id, dryrun_token) -> 正式发起退款，返回 refund_id。必须使用 dryrun_refund 返回的 token。",
}


def _err(msg):
    return {"ok": False, "error": msg}


class RefundEnv:
    def __init__(self, db_path=DB_PATH):
        db_path = Path(db_path)
        if not db_path.exists():
            from rlvr_task.data.build_db import build
            build(db_path)
        src = sqlite3.connect(db_path)
        self._conn = sqlite3.connect(":memory:")
        src.backup(self._conn)
        src.close()
        self._conn.row_factory = sqlite3.Row
        self._fetched_policies = {}   # order_id -> policy_id fetched this episode
        self._checked_attachments = set()  # order_ids whose attachments were inspected
        self._dryrun_tokens = {}      # token -> (order_id, item_id, amount)
        self._counter = 0

    # ------------------------------------------------------------ helpers
    def _rows(self, sql, params=()):
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _order_exists(self, order_id):
        return bool(self._rows("SELECT 1 FROM orders WHERE order_id=?", (order_id,)))

    @staticmethod
    def _valid_order_id(order_id):
        return isinstance(order_id, str) and bool(ORDER_ID_RE.match(order_id))

    @staticmethod
    def _valid_customer_id(customer_id):
        return isinstance(customer_id, str) and bool(CUSTOMER_ID_RE.match(customer_id))

    def _check_order_arg(self, order_id):
        if not self._valid_order_id(order_id):
            return _err(f"invalid order_id format: {order_id!r} (expected like 'O1001')")
        if not self._order_exists(order_id):
            return _err(f"order not found: {order_id}")
        return None

    def call(self, name, **kwargs):
        """Generic dispatcher for harnesses: env.call('get_policy', order_id='O1001')."""
        if name not in TOOL_DESCRIPTIONS:
            return _err(f"unknown tool: {name!r}")
        try:
            return getattr(self, name)(**kwargs)
        except TypeError as exc:
            return _err(f"bad arguments for {name}: {exc}")
        except Exception as exc:  # last-resort guard: never crash a rollout
            return _err(f"internal error in {name}: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ query APIs
    def check_order(self, customer_id):
        """Redundant API: order id is already given in the task text."""
        if not self._valid_customer_id(customer_id):
            return _err(f"invalid customer_id format: {customer_id!r}")
        orders = self._rows(
            "SELECT order_id, order_date, order_total, status, remark FROM orders "
            "WHERE customer_id=? ORDER BY order_date", (customer_id,))
        return {"ok": True, "customer_id": customer_id, "orders": orders}

    def get_policy(self, order_id):
        bad = self._check_order_arg(order_id)
        if bad:
            return bad
        order = self._rows("SELECT customer_id, order_date FROM orders WHERE order_id=?", (order_id,))[0]
        items = self._rows(
            "SELECT item_id, name, category, unit_price, qty, returnable FROM order_items "
            "WHERE order_id=?", (order_id,))
        for it in items:
            it["returnable"] = bool(it["returnable"])
        policy_id = f"POL-{order_id}"
        self._fetched_policies[order_id] = policy_id
        rules = {
            "return_window": "退货时限自签收日起算，天数按会员等级：" + "，".join(
                f"{k} {v} 天" for k, v in RETURN_WINDOW_DAYS.items()),
            "evidence": f"损坏照片须在签收后 {PHOTO_MAX_HOURS_AFTER_DELIVERY} 小时内拍摄，且对应具体商品。",
            "non_returnable": "returnable=false 的商品（如电脑）不支持退货退款。",
            "refund_amount": f"按损坏商品单价 × 损坏数量 × 退款比例计算，默认比例 {DEFAULT_REFUND_RATIO:.0%}，运费由商家承担；调整比例或加收运费须转人工。",
            "manual_review": "退款金额超过会员等级对应的审核阈值须转人工（可调用 need_permission 查询）。",
            "refund_timing": "退货包裹到达商家后方可退款；未到达时应提醒顾客寄回并等待。",
        }
        return {"ok": True, "policy_id": policy_id, "order_id": order_id,
                "customer_id": order["customer_id"], "order_date": order["order_date"],
                "items": items, "rules": rules, "current_date": CURRENT_DATE}

    def check_attachment(self, order_id):
        bad = self._check_order_arg(order_id)
        if bad:
            return bad
        atts = self._rows(
            "SELECT attachment_id, item_id, file_name, damage_type, damaged_qty, taken_at, uploaded_at "
            "FROM attachments WHERE order_id=?", (order_id,))
        self._checked_attachments.add(order_id)
        return {"ok": True, "order_id": order_id, "attachments": atts}

    def check_delivery(self, order_id):
        bad = self._check_order_arg(order_id)
        if bad:
            return bad
        rows = self._rows("SELECT * FROM deliveries WHERE order_id=?", (order_id,))
        if not rows:
            return _err(f"no delivery record for {order_id}")
        return {"ok": True, **rows[0]}

    def get_customer_tier(self, customer_id):
        if not self._valid_customer_id(customer_id):
            return _err(f"invalid customer_id format: {customer_id!r}")
        rows = self._rows("SELECT customer_id, name, tier, phone, registered_at FROM customers "
                          "WHERE customer_id=?", (customer_id,))
        if not rows:
            return _err(f"customer not found: {customer_id}")
        return {"ok": True, **rows[0]}

    def need_permission(self, customer_id, amount):
        tier_info = self.get_customer_tier(customer_id)
        if not tier_info["ok"]:
            return tier_info
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount < 0:
            return _err(f"amount must be a non-negative number, got {amount!r}")
        threshold = MANUAL_REVIEW_THRESHOLD[tier_info["tier"]]
        return {"ok": True, "customer_id": customer_id, "amount": float(amount),
                "need_manual_review": float(amount) > threshold}

    def get_address(self, order_id):
        """Redundant API."""
        bad = self._check_order_arg(order_id)
        if bad:
            return bad
        return {"ok": True, **self._rows("SELECT * FROM addresses WHERE order_id=?", (order_id,))[0]}

    # ------------------------------------------------------------ action APIs
    def notify_refund(self, order_id, message=""):
        bad = self._check_order_arg(order_id)
        if bad:
            return bad
        if not isinstance(message, str):
            return _err("message must be a string")
        self._conn.execute("INSERT INTO notifications VALUES (?,?)", (order_id, message))
        return {"ok": True, "order_id": order_id, "notified": True}

    def dryrun_refund(self, order_id, item_id, amount, policy_id):
        """Safety pre-check only (policy fetched, attachments checked, item belongs
        to order, amount bound).

        It deliberately does NOT judge business eligibility (window, evidence,
        review threshold) - that reasoning is the agent's job.
        """
        bad = self._check_order_arg(order_id)
        if bad:
            return bad
        if self._fetched_policies.get(order_id) is None:
            return _err("policy not fetched: call get_policy(order_id) before refunding")
        if policy_id != self._fetched_policies[order_id]:
            return _err(f"policy_id mismatch for {order_id}")
        if order_id not in self._checked_attachments:
            return _err("attachments not checked: call check_attachment(order_id) before refunding")
        items = self._rows("SELECT unit_price, qty FROM order_items WHERE order_id=? AND item_id=?",
                           (order_id, item_id))
        if not items:
            return _err(f"item {item_id!r} not in order {order_id}")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            return _err(f"amount must be a positive number, got {amount!r}")
        max_amount = items[0]["unit_price"] * items[0]["qty"]
        if amount > max_amount + 1e-9:
            return _err(f"amount {amount} exceeds paid amount {max_amount} for item {item_id}")
        self._counter += 1
        token = f"DRY-{order_id}-{self._counter:03d}"
        self._dryrun_tokens[token] = (order_id, item_id, float(amount))
        return {"ok": True, "dryrun_token": token, "order_id": order_id,
                "item_id": item_id, "amount": float(amount)}

    def issue_refund(self, order_id, dryrun_token):
        bad = self._check_order_arg(order_id)
        if bad:
            return bad
        info = self._dryrun_tokens.pop(dryrun_token, None) if isinstance(dryrun_token, str) else None
        if info is None:
            return _err("invalid or already-used dryrun_token: call dryrun_refund first")
        if info[0] != order_id:
            return _err("dryrun_token does not belong to this order")
        self._counter += 1
        refund_id = f"RF-{order_id}-{self._counter:03d}"
        self._conn.execute("INSERT INTO refunds VALUES (?,?,?,?,?)",
                           (refund_id, order_id, info[1], info[2], CURRENT_DATE))
        return {"ok": True, "refund_id": refund_id, "order_id": order_id, "amount": info[2]}

    # ------------------------------------------------------------ harness hooks
    def export_state(self):
        """Final environment state, passed to the verifier by the harness."""
        return {
            "refunds": self._rows("SELECT refund_id, order_id, item_id, amount FROM refunds"),
            "notifications": self._rows("SELECT order_id, message FROM notifications"),
        }

    def close(self):
        self._conn.close()
