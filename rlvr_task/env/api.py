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
ITEM_ID_RE = re.compile(r"^I\d{4}[A-Z]$")

# Tool descriptions handed to the agent (e.g. as a tool schema in the harness).
TOOL_DESCRIPTIONS = {
    "check_order": "check_order(customer_id) -> 该顾客的历史订单列表（订单号、下单日期、订单总额、状态）。",
    "get_policy": "get_policy(order_id) -> 订单商品明细与平台售后政策条款全集（policy_id、customer_id、商品单价/数量/是否可退、clauses）。不同条款适用于不同情形。退款前必须先获取政策。",
    "check_attachment": "check_attachment(order_id) -> 顾客为该订单上传的附件（对应商品、损坏描述、损坏数量、拍摄时间）。",
    "check_delivery": "check_delivery(order_id) -> 物流信息（承运商、签收时间、退货包裹单号与状态）。",
    "get_customer_tier": "get_customer_tier(customer_id) -> 顾客会员等级（bronze/silver/gold）及联系方式。",
    "need_permission": "need_permission(customer_id, amount) -> 该退款金额对该顾客是否需要人工审核。",
    "get_address": "get_address(order_id) -> 顾客收货地址与商家退货地址。",
    "get_product_info": "get_product_info(item_id) -> 商品详情（品牌、产地、保修期天数、重量、评分、库存）。",
    "notify_refund": "notify_refund(order_id, message) -> 通知顾客（如提醒按要求寄回商品）。",
    "dryrun_refund": "dryrun_refund(order_id, item_id | [item_id, ...], amount, policy_id) -> 退款安全预检（须已调用 get_policy 和 check_attachment；只做安全检查，不判断是否符合退货政策），通过则返回一次性 dryrun_token。",
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
        self._dryrun_tokens = {}      # unused token -> (order_id, item_ids, amount)
        self._dryrun_log = []         # every PASSED dry-run, exported for the verifier
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
        # The full after-sales policy book. Only some clauses apply to a given
        # ticket; the agent must pick the applicable ones (R2/R4/R5/R6 for a
        # damage claim). R1 and R3 are plausible distractors with different
        # window start dates / refund formulas.
        clauses = [
            {"clause_id": "R1", "title": "七天无理由退货",
             "applies_to": "商品完好、无质量问题，顾客主观不想要",
             "window": "自下单日起 7 天内",
             "refund": "按商品单价退款，扣除往返运费"},
            {"clause_id": "R2", "title": "质量问题 / 运输损坏退货",
             "applies_to": "顾客收到的商品存在损坏或质量问题",
             "window_start": "delivered_at",
             "window_days_by_tier": dict(RETURN_WINDOW_DAYS),
             "evidence": {"photo_taken_within_hours_after_delivery": PHOTO_MAX_HOURS_AFTER_DELIVERY,
                          "min_valid_photos_per_damaged_unit": 1},
             "refund": {"basis": "unit_price × accepted_damaged_qty",
                        "ratio": DEFAULT_REFUND_RATIO, "shipping": "merchant_pays",
                        "ratio_or_shipping_change": "manual_review"}},
            {"clause_id": "R3", "title": "生鲜 / 食品",
             "applies_to": "category=food 的商品",
             "window": "签收后 48 小时内", "refund": "仅退商品款，不需寄回"},
            {"clause_id": "R4", "title": "人工审核",
             "applies_to": "所有退款",
             "rule": "退款金额超过客户等级对应的审核阈值时，不得直接退款，须转人工",
             "tool": "need_permission"},
            {"clause_id": "R5", "title": "不可退商品",
             "applies_to": "returnable=false 的商品",
             "rule": "不支持退货退款，质量问题也不例外"},
            {"clause_id": "R6", "title": "退款时机",
             "applies_to": "需要寄回的退货",
             "rule": "退货包裹状态为 arrived_at_merchant 后才可退款；否则通知顾客寄回并等待"},
        ]
        return {"ok": True, "policy_id": policy_id, "order_id": order_id,
                "customer_id": order["customer_id"], "order_date": order["order_date"],
                "items": items, "clauses": clauses, "current_date": CURRENT_DATE}

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

    def get_product_info(self, item_id):
        """Redundant API. warranty_days is a distractor (warranty != return window)."""
        if not isinstance(item_id, str) or not ITEM_ID_RE.match(item_id):
            return _err(f"invalid item_id format: {item_id!r} (expected like 'I1001A')")
        rows = self._rows(
            "SELECT p.item_id, i.name, i.category, p.brand, p.origin, p.warranty_days, "
            "p.weight_kg, p.rating, p.stock FROM products p JOIN order_items i ON p.item_id=i.item_id "
            "WHERE p.item_id=?", (item_id,))
        if not rows:
            return _err(f"item not found: {item_id}")
        return {"ok": True, **rows[0]}

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
        item_ids = [item_id] if isinstance(item_id, str) else item_id
        if not isinstance(item_ids, (list, tuple)) or not item_ids or \
                not all(isinstance(i, str) for i in item_ids) or len(set(item_ids)) != len(item_ids):
            return _err("item_id must be an item id string or a non-empty list of distinct ids")
        max_amount = 0.0
        for iid in item_ids:
            rows = self._rows("SELECT unit_price, qty FROM order_items WHERE order_id=? AND item_id=?",
                              (order_id, iid))
            if not rows:
                return _err(f"item {iid!r} not in order {order_id}")
            max_amount += rows[0]["unit_price"] * rows[0]["qty"]
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            return _err(f"amount must be a positive number, got {amount!r}")
        if amount > max_amount + 1e-9:
            return _err(f"amount {amount} exceeds paid amount {max_amount} for items {item_ids}")
        self._counter += 1
        token = f"DRY-{order_id}-{self._counter:03d}"
        joined = ",".join(sorted(item_ids))
        self._dryrun_tokens[token] = (order_id, joined, float(amount))
        self._dryrun_log.append({"dryrun_token": token, "order_id": order_id,
                                 "item_ids": joined, "amount": float(amount)})
        return {"ok": True, "dryrun_token": token, "order_id": order_id,
                "item_ids": sorted(item_ids), "amount": float(amount)}

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
        self._conn.execute("INSERT INTO refunds VALUES (?,?,?,?,?,?)",
                           (refund_id, order_id, info[1], info[2], dryrun_token, CURRENT_DATE))
        return {"ok": True, "refund_id": refund_id, "order_id": order_id, "amount": info[2]}

    # ------------------------------------------------------------ harness hooks
    def export_state(self):
        """Final environment state, passed to the verifier by the harness."""
        return {
            "refunds": self._rows("SELECT refund_id, order_id, item_id, amount, dryrun_token FROM refunds"),
            "dryruns": [dict(d) for d in self._dryrun_log],
            "notifications": self._rows("SELECT order_id, message FROM notifications"),
        }

    def close(self):
        self._conn.close()
