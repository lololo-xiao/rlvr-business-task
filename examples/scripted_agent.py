"""A scripted (non-LLM) agent that solves a task using ONLY env APIs.

It follows the typical execution path documented in docs/DESIGN.md and
can inject specific mistakes to produce realistic wrong answers.
"""
from datetime import datetime, timedelta

from rlvr_task.config import (
    DEFAULT_REFUND_RATIO,
    PHOTO_MAX_HOURS_AFTER_DELIVERY,
    RETURN_ARRIVED_STATUS,
    RETURN_WINDOW_DAYS,
)

MISTAKES = (
    "window_from_order_date",  # count return window from order date
    "ignore_tier_window",      # always use the longest window
    "use_order_total",         # use order total as refund amount
    "refund_whole_line",       # refund bought qty instead of accepted damaged qty
    "trust_claimed_qty",       # trust claimed damaged_qty even with fewer photos
    "skip_issue_refund",       # dry-run passes, never issue refund, still claim refund
    "fabricate_refund",        # claim refund with neither dry-run nor refund
    "double_refund",           # refund twice
    "wrong_reject_reason",     # always cite outside_window when rejecting
)


def _dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def run_agent(env, order_id, mistake=None):
    """Returns the final JSON answer (dict). `mistake` is None or one of MISTAKES."""
    # Step 1: policy -> customer_id, items, clauses (damage claim -> R2/R4/R5/R6), policy_id
    policy = env.get_policy(order_id)
    customer_id = policy["customer_id"]
    today = datetime.strptime(policy["current_date"], "%Y-%m-%d")
    items = {i["item_id"]: i for i in policy["items"]}
    # Step 2: attachments -> damaged items, photo times
    attachments = env.check_attachment(order_id)["attachments"]
    # Step 3: delivery -> delivered_at, return parcel status
    delivery = env.check_delivery(order_id)
    delivered = _dt(delivery["delivered_at"])
    # Step 4: tier -> return window
    tier = env.get_customer_tier(customer_id)["tier"]
    window = RETURN_WINDOW_DAYS[tier]
    if mistake == "ignore_tier_window":
        window = max(RETURN_WINDOW_DAYS.values())

    def done(decision, amount=0.0, refund_id=None, reject_reason=None):
        if reject_reason and mistake == "wrong_reject_reason":
            reject_reason = "outside_window"
        return {"order_id": order_id, "decision": decision, "refund_amount": round(amount, 2),
                "refund_id": refund_id, "reject_reason": reject_reason}

    start = datetime.strptime(policy["order_date"], "%Y-%m-%d") if mistake == "window_from_order_date" else delivered
    if (today.date() - start.date()).days > window:
        return done("reject", reject_reason="outside_window")

    # Evidence: valid photos per item (taken within N hours after delivery)
    deadline = delivered + timedelta(hours=PHOTO_MAX_HOURS_AFTER_DELIVERY)
    claims = {}
    for att in attachments:
        if delivered <= _dt(att["taken_at"]) <= deadline:
            claims.setdefault(att["item_id"], []).append(att["damaged_qty"])

    # Cross-API computation: sum(unit_price x accepted qty x ratio) over eligible items
    accepted = {}
    for item_id, qtys in claims.items():
        item = items.get(item_id)
        if item is None or not item["returnable"]:
            continue
        qty = min(max(qtys), len(qtys), item["qty"])
        if mistake == "trust_claimed_qty":
            qty = min(max(qtys), item["qty"])
        if mistake == "refund_whole_line":
            qty = item["qty"]
        accepted[item_id] = qty
    if not accepted:
        # photos are valid but only for non-returnable items -> non_returnable
        return done("reject", reject_reason="non_returnable" if claims else "invalid_evidence")
    amount = sum(items[i]["unit_price"] * q for i, q in accepted.items()) * DEFAULT_REFUND_RATIO
    if mistake == "use_order_total":
        amount = sum(i["unit_price"] * i["qty"] for i in policy["items"])

    # Step 5: review threshold depends on tier and on the computed amount
    if env.need_permission(customer_id, amount)["need_manual_review"]:
        return done("escalate", amount)
    if delivery["return_status"] != RETURN_ARRIVED_STATUS:
        env.notify_refund(order_id, "请按退货地址寄回商品，商家签收后为您退款。")
        return done("wait_return", amount)

    if mistake == "fabricate_refund":
        return done("refund", amount, refund_id="RF-FAKE")
    # Step 6: dry-run then issue refund (environment enforces this order)
    item_ids = sorted(accepted)
    token = env.dryrun_refund(order_id, item_ids, amount, policy["policy_id"])["dryrun_token"]
    if mistake == "skip_issue_refund":
        return done("refund", amount, refund_id="RF-FAKE")
    refund = env.issue_refund(order_id, token)
    if mistake == "double_refund":
        token2 = env.dryrun_refund(order_id, item_ids, amount, policy["policy_id"])["dryrun_token"]
        env.issue_refund(order_id, token2)
    return done("refund", amount, refund["refund_id"])
