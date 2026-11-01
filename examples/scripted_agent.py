"""A scripted (non-LLM) agent that solves a task using ONLY env APIs.

It follows the typical execution path documented in docs/DESIGN.md and
can inject specific mistakes to produce realistic wrong answers.
"""
from datetime import datetime, timedelta


def _dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def run_agent(env, order_id, mistake=None):
    """mistake: None | 'window_from_order_date' | 'ignore_tier_window' |
                'use_order_total' | 'skip_issue_refund' | 'double_refund'
    Returns the final JSON answer (dict)."""
    # Step 1: policy -> customer_id, items, rules, policy_id
    policy = env.get_policy(order_id)
    customer_id = policy["customer_id"]
    today = datetime.strptime(policy["current_date"], "%Y-%m-%d")
    # Step 2: attachments -> which item is damaged, photo time
    att = env.check_attachment(order_id)["attachments"][0]
    item = next(i for i in policy["items"] if i["item_id"] == att["item_id"])
    # Step 3: delivery -> delivered_at, return parcel status
    delivery = env.check_delivery(order_id)
    delivered = _dt(delivery["delivered_at"])
    # Step 4: tier -> return window
    tier = env.get_customer_tier(customer_id)["tier"]
    window = {"bronze": 7, "silver": 15, "gold": 30}[tier]
    if mistake == "ignore_tier_window":
        window = 30

    def done(decision, amount=0.0, refund_id=None):
        return {"order_id": order_id, "decision": decision,
                "refund_amount": round(amount, 2), "refund_id": refund_id}

    if not item["returnable"]:
        return done("reject")
    start = datetime.strptime(policy["order_date"], "%Y-%m-%d") if mistake == "window_from_order_date" else delivered
    if (today.date() - start.date()).days > window:
        return done("reject")
    taken = _dt(att["taken_at"])
    if not (delivered <= taken <= delivered + timedelta(hours=24)):
        return done("reject")

    # Cross-API computation: damaged item price x damaged qty x ratio (1.0)
    amount = item["unit_price"] * att["damaged_qty"] * 1.0
    if mistake == "use_order_total":
        amount = sum(i["unit_price"] * i["qty"] for i in policy["items"])

    # Step 5: review threshold depends on tier and on the computed amount
    if env.need_permission(customer_id, amount)["need_manual_review"]:
        return done("escalate", amount)
    if delivery["return_status"] != "arrived_at_merchant":
        env.notify_refund(order_id, "请按退货地址寄回商品，商家签收后为您退款。")
        return done("wait_return", amount)

    # Step 6: dry-run then issue refund (environment enforces this order)
    token = env.dryrun_refund(order_id, item["item_id"], amount, policy["policy_id"])["dryrun_token"]
    if mistake == "skip_issue_refund":
        return done("refund", amount, refund_id="RF-FAKE")
    refund = env.issue_refund(order_id, token)
    if mistake == "double_refund":
        token2 = env.dryrun_refund(order_id, item["item_id"], amount, policy["policy_id"])["dryrun_token"]
        env.issue_refund(order_id, token2)
    return done("refund", amount, refund["refund_id"])
