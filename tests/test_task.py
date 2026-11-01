import pytest

from examples.scripted_agent import run_agent
from rlvr_task.env import RefundEnv
from rlvr_task.tasks import TASKS, find_task, solve
from rlvr_task.verifier import compute_reward


def _play(order_id, mistake=None):
    env = RefundEnv()
    answer = run_agent(env, order_id, mistake)
    state = env.export_state()
    env.close()
    return answer, state


@pytest.mark.parametrize("order_id", list(TASKS))
def test_api_only_agent_gets_full_reward(order_id):
    answer, state = _play(order_id)
    assert compute_reward(TASKS[order_id]["text"], answer, state) == 1.0


@pytest.mark.parametrize("order_id,mistake", [
    ("O1001", "window_from_order_date"),
    ("O1002", "ignore_tier_window"),
    ("O1005", "use_order_total"),
    ("O1001", "skip_issue_refund"),
    ("O1001", "double_refund"),
])
def test_each_hidden_constraint_changes_reward(order_id, mistake):
    answer, state = _play(order_id, mistake)
    assert compute_reward(TASKS[order_id]["text"], answer, state) <= 0.2


def test_ground_truth_covers_all_branches():
    decisions = {solve(o)["decision"] for o in TASKS}
    assert decisions == {"refund", "reject", "escalate", "wait_return"}


@pytest.mark.parametrize("bad", [None, 42, "", "not json", "{bad", [], {"decision": "refund"},
                                 {"decision": "refund", "refund_amount": float("nan")},
                                 {"decision": "refund", "refund_amount": True}])
def test_invalid_outputs_get_zero_without_raising(bad):
    assert compute_reward(TASKS["O1001"]["text"], bad) == 0.0


def test_unknown_task_gets_zero():
    assert compute_reward("随便一个问题", {"decision": "reject", "refund_amount": 0}) == 0.0
    assert find_task(None) is None


def test_json_string_with_fences_is_accepted():
    out = '```json\n{"order_id": "O1002", "decision": "reject", "refund_amount": 0, "refund_id": null}\n```'
    assert compute_reward(TASKS["O1002"]["text"], out, {"refunds": []}) == 1.0


def test_amount_partial_credit():
    # escalate is correct, amount off by 10% -> amount component 0.5
    out = {"order_id": "O1004", "decision": "escalate", "refund_amount": 612.0, "refund_id": None}
    assert compute_reward(TASKS["O1004"]["text"], out, {"refunds": []}) == pytest.approx(0.5 + 0.15 + 0.2)


def test_env_enforces_policy_and_dryrun_before_refund():
    env = RefundEnv()
    assert not env.dryrun_refund("O1001", "I1001A", 299, "POL-O1001")["ok"]
    assert not env.issue_refund("O1001", "DRY-X")["ok"]
    pol = env.get_policy("O1001")
    assert not env.dryrun_refund("O1001", "I1001A", 9999, pol["policy_id"])["ok"]
    tok = env.dryrun_refund("O1001", "I1001A", 299, pol["policy_id"])["dryrun_token"]
    assert env.issue_refund("O1001", tok)["ok"]
    assert not env.issue_refund("O1001", tok)["ok"]  # token is single-use
    env.close()


def test_episodes_are_isolated():
    _play("O1001")
    assert RefundEnv().export_state()["refunds"] == []


@pytest.mark.parametrize("name,kwargs", [
    ("get_policy", {"order_id": 1001}), ("get_policy", {"order_id": "O9999"}),
    ("get_customer_tier", {"customer_id": "X"}), ("need_permission", {"customer_id": "C01", "amount": "a"}),
    ("nope", {}), ("check_delivery", {}),
])
def test_bad_api_calls_return_errors(name, kwargs):
    res = RefundEnv().call(name, **kwargs)
    assert res["ok"] is False and res["error"]
