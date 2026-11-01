"""RLVR verifier: objective reward in [0, 1] from the agent's final result.

Unified entry point for training harnesses:

    from rlvr_task.verifier import compute_reward
    r = compute_reward(task_text, agent_output, env_state=env.export_state())

It judges only the final result (the answer plus the final environment
state it claims to have produced). It never inspects the call trajectory
and never calls an LLM. It never raises: any bad input yields reward 0.
"""
import json
import math

from rlvr_task.config import (
    AMOUNT_ABS_TOLERANCE,
    AMOUNT_REL_TOLERANCE,
    AMOUNT_ZERO_CREDIT_REL,
    CAP_ACTION_INCONSISTENT,
    CAP_DUPLICATE_REFUND,
    CAP_HARMFUL_REFUND,
    CAP_INVALID_FORMAT,
    CAP_WRONG_DECISION,
    DB_PATH,
    DECISIONS,
    REWARD_WEIGHTS,
)
from rlvr_task.tasks import find_task, solve


def _parse_output(agent_output):
    """Accept a dict or a JSON string (optionally wrapped in ``` fences)."""
    if isinstance(agent_output, dict):
        return agent_output, None
    if not isinstance(agent_output, str):
        return None, f"output must be dict or JSON string, got {type(agent_output).__name__}"
    text = agent_output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None, "no JSON object found"
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return None, "JSON root must be an object"
    return obj, None


def _validate(obj):
    decision = obj.get("decision")
    if decision not in DECISIONS:
        return f"decision must be one of {DECISIONS}, got {decision!r}"
    amount = obj.get("refund_amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
        return f"refund_amount must be a non-negative number, got {amount!r}"
    refund_id = obj.get("refund_id")
    if refund_id is not None and not isinstance(refund_id, str):
        return "refund_id must be a string or null"
    return None


def _amount_score(pred, truth):
    err = abs(pred - truth)
    if err <= max(AMOUNT_ABS_TOLERANCE, AMOUNT_REL_TOLERANCE * truth):
        return 1.0
    if truth == 0:
        return 0.0
    rel = err / truth
    return max(0.0, 1.0 - rel / AMOUNT_ZERO_CREDIT_REL)


def evaluate(task_text, agent_output, env_state=None, db_path=DB_PATH):
    """Detailed evaluation. Returns {"reward": float, "components": ..., "caps": [...], ...}."""
    result = {"reward": 0.0, "components": {}, "caps": [], "error": None, "ground_truth": None}
    try:
        task = find_task(task_text)
        if task is None:
            result["error"] = "unknown task"
            return result
        truth = solve(task["order_id"], db_path)
        result["ground_truth"] = truth

        obj, err = _parse_output(agent_output)
        if err is None:
            err = _validate(obj)
        if err is None and obj.get("order_id") not in (None, task["order_id"]):
            err = f"order_id mismatch: {obj.get('order_id')!r}"
        if err is not None:
            result["error"] = err
            result["caps"].append(("invalid_format", CAP_INVALID_FORMAT))
            result["reward"] = CAP_INVALID_FORMAT
            return result

        decision_ok = obj["decision"] == truth["decision"]
        amount_s = _amount_score(float(obj["refund_amount"]), truth["refund_amount"])

        # ---- execution: does the final world state match the claim and the truth?
        caps = []
        refunds = None
        if isinstance(env_state, dict) and isinstance(env_state.get("refunds"), list):
            refunds = [r for r in env_state["refunds"]
                       if isinstance(r, dict) and r.get("order_id") == task["order_id"]]
        if refunds is not None:
            refunded = len(refunds) > 0
            if len(refunds) > 1:
                caps.append(("duplicate_refund", CAP_DUPLICATE_REFUND))
            if refunded and truth["decision"] != "refund":
                caps.append(("harmful_refund", CAP_HARMFUL_REFUND))
            claims_refund = obj["decision"] == "refund"
            id_matches = refunded and obj.get("refund_id") in {r.get("refund_id") for r in refunds}
            if claims_refund != refunded or (claims_refund and not id_matches):
                caps.append(("action_inconsistent", CAP_ACTION_INCONSISTENT))
            if truth["decision"] == "refund":
                exec_s = 1.0 if (len(refunds) == 1 and id_matches and
                                 _amount_score(float(refunds[0].get("amount", -1)), truth["refund_amount"]) == 1.0) else 0.0
            else:
                exec_s = 1.0 if not refunded else 0.0
        else:
            # No env state supplied: only check internal consistency of the answer.
            has_id = bool(obj.get("refund_id"))
            consistent = has_id == (obj["decision"] == "refund")
            if not consistent:
                caps.append(("action_inconsistent", CAP_ACTION_INCONSISTENT))
            exec_s = 1.0 if consistent and decision_ok else 0.0

        if not decision_ok:
            caps.append(("wrong_decision", CAP_WRONG_DECISION))

        components = {"decision": 1.0 if decision_ok else 0.0, "amount": amount_s, "execution": exec_s}
        reward = sum(REWARD_WEIGHTS[k] * v for k, v in components.items())
        for _, cap in caps:
            reward = min(reward, cap)
        result.update(reward=round(max(0.0, min(1.0, reward)), 4), components=components, caps=caps)
        return result
    except Exception as exc:  # the verifier must never crash the trainer
        result["error"] = f"verifier internal error: {type(exc).__name__}: {exc}"
        result["reward"] = 0.0
        return result


def compute_reward(task_text, agent_output, env_state=None):
    """Unified entry point. Returns a float reward in [0, 1]."""
    return evaluate(task_text, agent_output, env_state)["reward"]
