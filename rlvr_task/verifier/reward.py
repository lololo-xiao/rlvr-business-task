"""RLVR verifier: objective reward in [0, 1] from the agent's final result.

Unified entry point for training harnesses:

    from rlvr_task.verifier import compute_reward
    r = compute_reward(task_text, agent_output, env_state=env.export_state())

It judges only the final result (the answer plus the final environment
state it claims to have produced). It never inspects the call trajectory
and never calls an LLM. It never raises: unparsable output yields 0, any valid answer at least REWARD_FLOOR.
"""
import json
import math

from rlvr_task.config import (
    AMOUNT_ABS_TOLERANCE,
    AMOUNT_REL_TOLERANCE,
    AMOUNT_ZERO_CREDIT_REL,
    CAP_ACTION_INCONSISTENT,
    CAP_DRYRUN_WITHOUT_REFUND,
    CAP_DUPLICATE_REFUND,
    CAP_FABRICATED_REFUND,
    CAP_REFUND_WITHOUT_DRYRUN,
    CAP_HARMFUL_REFUND,
    CAP_WRONG_DECISION,
    CAP_WRONGFUL_ESCALATE,
    CAP_WRONGFUL_REJECT,
    DB_PATH,
    DECISIONS,
    INVALID_FORMAT_REWARD,
    REWARD_FLOOR,
    REJECT_REASONS,
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
    if decision == "reject" and obj.get("reject_reason") not in REJECT_REASONS:
        return f"reject requires reject_reason in {REJECT_REASONS}, got {obj.get('reject_reason')!r}"
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
            result["caps"].append(("invalid_format", INVALID_FORMAT_REWARD))
            result["reward"] = INVALID_FORMAT_REWARD
            return result

        decision_ok = obj["decision"] == truth["decision"] and (
            truth["decision"] != "reject" or obj.get("reject_reason") == truth["reject_reason"])
        amount_s = _amount_score(float(obj["refund_amount"]), truth["refund_amount"])

        # ---- execution: does the final world state match the claim and the truth?
        caps = []
        claims_refund = obj["decision"] == "refund"
        state_ok = isinstance(env_state, dict) and isinstance(env_state.get("refunds"), list)
        if state_ok:
            oid = task["order_id"]
            refunds = [r for r in env_state["refunds"] if isinstance(r, dict) and r.get("order_id") == oid]
            dryruns = [d for d in env_state.get("dryruns") or [] if isinstance(d, dict) and d.get("order_id") == oid]
            passed_tokens = {d.get("dryrun_token") for d in dryruns}
            refunded = len(refunds) > 0

            if len(refunds) > 1:
                caps.append(("duplicate_refund", CAP_DUPLICATE_REFUND))
            if refunded and truth["decision"] != "refund":
                caps.append(("harmful_refund", CAP_HARMFUL_REFUND))
            if any(r.get("dryrun_token") not in passed_tokens for r in refunds):
                caps.append(("refund_without_dryrun", CAP_REFUND_WITHOUT_DRYRUN))
            if claims_refund and not refunded:
                if dryruns:
                    caps.append(("dryrun_passed_but_no_refund", CAP_DRYRUN_WITHOUT_REFUND))
                else:
                    caps.append(("fabricated_refund", CAP_FABRICATED_REFUND))
            id_matches = refunded and obj.get("refund_id") in {r.get("refund_id") for r in refunds}
            if refunded and (not claims_refund or not id_matches):
                caps.append(("action_inconsistent", CAP_ACTION_INCONSISTENT))

            if truth["decision"] == "refund":
                exec_s = 1.0 if (len(refunds) == 1 and id_matches and
                                 _amount_score(float(refunds[0].get("amount", -1)), truth["refund_amount"]) == 1.0) else 0.0
            else:
                exec_s = 1.0 if not refunded else 0.0
        else:
            # No env state supplied: a refund claim cannot be verified, so it is
            # capped (prevents rewarding a hallucinated refund_id).
            # Harnesses must pass env.export_state().
            has_id = bool(obj.get("refund_id"))
            if claims_refund:
                caps.append(("refund_unverifiable_without_env_state", CAP_ACTION_INCONSISTENT))
                exec_s = 0.0
            else:
                if has_id:
                    caps.append(("action_inconsistent", CAP_ACTION_INCONSISTENT))
                exec_s = 1.0 if (not has_id and decision_ok) else 0.0

        if not decision_ok:
            if obj["decision"] == "reject" and truth["decision"] != "reject":
                caps.append(("wrongful_reject", CAP_WRONGFUL_REJECT))
            elif obj["decision"] == "escalate" and truth["decision"] != "escalate":
                caps.append(("wrongful_escalate", CAP_WRONGFUL_ESCALATE))
            else:
                caps.append(("wrong_decision", CAP_WRONG_DECISION))

        components = {"decision": 1.0 if decision_ok else 0.0, "amount": amount_s, "execution": exec_s}
        reward = sum(REWARD_WEIGHTS[k] * v for k, v in components.items())
        for _, cap in caps:
            reward = min(reward, cap)
        reward = max(REWARD_FLOOR, min(1.0, reward))
        result.update(reward=round(reward, 4), components=components, caps=caps)
        return result
    except Exception as exc:  # the verifier must never crash the trainer
        result["error"] = f"verifier internal error: {type(exc).__name__}: {exc}"
        result["reward"] = 0.0
        return result


def compute_reward(task_text, agent_output, env_state=None):
    """Unified entry point. Returns a float reward in [0, 1]."""
    return evaluate(task_text, agent_output, env_state)["reward"]
