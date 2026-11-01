"""Demo of the standard harness calling convention: env APIs + verifier.

Run: python -m examples.run_examples
"""
import json

from examples.scripted_agent import run_agent
from rlvr_task.env import RefundEnv
from rlvr_task.tasks import TASKS
from rlvr_task.verifier import evaluate

SCENARIOS = [
    # (title, order_id, mistake, expected outcome)
    ("正确: 可退款 (签收日起算, 只退损坏商品)", "O1001", None, "correct"),
    ("正确: 超期拒绝 (bronze 7天)", "O1002", None, "correct"),
    ("正确: 照片超24小时拒绝", "O1003", None, "correct"),
    ("正确: 大额转人工 (silver 阈值500)", "O1004", None, "correct"),
    ("正确: 等待退货包裹到达", "O1005", None, "correct"),
    ("正确: 电脑不可退", "O1006", None, "correct"),
    ("正确: gold 14天时限内可退", "O1007", None, "correct"),
    ("正确: bronze 阈值200 -> 转人工", "O1008", None, "correct"),
    ("正确: 多张照片, 只退损坏数量 (2 x 60)", "O1009", None, "correct"),
    ("正确: 多商品多附件, 只退有有效照片的那件", "O1010", None, "correct"),
    ("错误: 从下单日起算时限 -> 误拒", "O1001", "window_from_order_date", "wrong"),
    ("错误: 忽略等级用最长时限 -> 误退款", "O1002", "ignore_tier_window", "wrong"),
    ("错误: 用订单总额判阈值 -> 误转人工", "O1005", "use_order_total", "wrong"),
    ("错误: 按整行数量退款 -> 金额错 (部分分)", "O1009", "refund_whole_line", "wrong"),
    ("错误: 照片数少于声称损坏数仍按声称退", "O1010", "trust_claimed_qty", "wrong"),
    ("错误: dry-run 通过但没真正退款, 却声称已退", "O1001", "skip_issue_refund", "wrong"),
    ("错误: 没有 dry-run 也没退款, 纯编造已退款", "O1001", "fabricate_refund", "wrong"),
    ("错误: 重复退款", "O1001", "double_refund", "wrong"),
]


def run_scenario(title, order_id, mistake):
    task = TASKS[order_id]
    env = RefundEnv()                       # 1. fresh, isolated episode
    answer = run_agent(env, order_id, mistake)  # 2. agent acts through APIs only
    result = evaluate(task["text"], answer, env.export_state())  # 3. reward
    env.close()
    return answer, result


def main():
    print("=" * 78)
    print("RLVR 售后退款任务 —— 环境 API + 验证引擎调用样例")
    print("=" * 78)
    for title, order_id, mistake, _ in SCENARIOS:
        answer, res = run_scenario(title, order_id, mistake)
        print(f"\n[{title}] task={TASKS[order_id]['task_id']}")
        print(f"  agent answer : {json.dumps(answer, ensure_ascii=False)}")
        print(f"  ground truth : decision={res['ground_truth']['decision']} "
              f"amount={res['ground_truth']['refund_amount']} ({res['ground_truth']['reason']})")
        print(f"  components   : {res['components']}  caps={[c[0] for c in res['caps']]}")
        print(f"  REWARD       : {res['reward']}")

    print("\n" + "-" * 78)
    print("非法输入 / 异常调用 (不会中断流程):")
    task = TASKS["O1001"]["text"]
    for label, out in [("非 JSON 文本", "我觉得可以退"), ("非法 decision", {"decision": "maybe", "refund_amount": 1}),
                       ("None", None), ("负金额", {"decision": "reject", "refund_amount": -5})]:
        res = evaluate(task, out)
        print(f"  {label:<14} -> reward={res['reward']}  error={res['error']}")
    env = RefundEnv()
    print("  issue_refund 未 dry-run  ->", env.issue_refund("O1001", "DRY-NOPE"))
    print("  dryrun 未先查政策         ->", env.dryrun_refund("O1001", "I1001A", 299, "POL-O1001"))
    print("  非法订单号               ->", env.check_delivery("1001; DROP TABLE orders"))
    print("  未知工具                 ->", env.call("delete_all_orders"))
    print("  参数错误                 ->", env.call("get_policy", oid="O1001"))
    env.close()


if __name__ == "__main__":
    main()
