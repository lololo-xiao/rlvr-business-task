# RLVR 业务分析任务：电商售后退款

**所选题目：题目一 —— 复杂业务分析 RLVR 任务构建**

一个可直接接入 RL 训练 Harness 的任务单元，包含三部分：
- **API 交互环境**（`RefundEnv`）：Agent 只能通过 Python 类方法查询 SQLite 业务数据、执行退款；
- **任务实例**：10 个"收到的商品坏了，能退吗？"售后工单；
- **验证引擎**：`compute_reward()` 根据最终结果输出 0～1 的客观奖励，不使用大模型。

完整的任务设计、接口定义、典型执行路径和奖励逻辑见 **[docs/DESIGN.md](docs/DESIGN.md)**。

## 目录结构

```
rlvr_task/
  config.py            # 业务规则、奖励权重、阈值（统一配置常量）
  data/build_db.py     # 构建 SQLite 业务数据（含陷阱数据）
  env/api.py           # RefundEnv：10 个 API（含 2 个冗余 API），参数校验 + 错误字典
  tasks.py             # 任务实例 + 标准答案求解器（从数据库计算）
  verifier/reward.py   # compute_reward / evaluate：奖励计算
examples/
  scripted_agent.py    # 只用 API 解题的脚本 Agent（可注入特定错误）
  run_examples.py      # 调用样例：10 个正确场景 + 9 个错误场景 + 非法输入
tests/test_task.py     # pytest（53 个用例）
docs/DESIGN.md         # 设计说明
Dockerfile, requirements.txt
```

## 本地运行

需要 Python 3.8+。运行时只用标准库，测试需要 pytest。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m rlvr_task.data.build_db      # 可选：首次创建 RefundEnv 时会自动建库
python -m examples.run_examples        # 运行调用样例
python -m pytest -q                    # 运行测试
```

## Docker

```bash
docker build -t rlvr-refund .
docker run --rm rlvr-refund            # 自动运行调用样例，输出各场景奖励
```

镜像内置依赖、构建好的业务数据库和执行入口（`CMD python -m examples.run_examples`）。

**挂载本地目录，供训练框架调用：**

```bash
# 1) 把训练脚本挂进容器，直接 import 环境和验证模块（容器内 PYTHONPATH=/app）
docker run --rm -v /path/to/trainer:/trainer rlvr-refund python /trainer/train.py

# 2) 把本仓库代码挂进容器（修改规则或数据后无需重新构建）
docker run --rm -v "$PWD":/app rlvr-refund python -m examples.run_examples
```

## 训练集成调用方式

```python
from rlvr_task.env import RefundEnv, TOOL_DESCRIPTIONS   # 工具说明可直接作为 tool schema
from rlvr_task.tasks import TASKS
from rlvr_task.verifier import compute_reward

task = TASKS["O1001"]
env = RefundEnv()                          # 每个 rollout 一个隔离的环境
# ... Harness 把 task["text"] 交给策略模型，并把模型的工具调用转发到：
obs = env.call("get_policy", order_id="O1001")
# ... 模型输出最终 JSON 后：
reward = compute_reward(task["text"], final_answer, env_state=env.export_state())
```

> Harness **必须**传入 `env_state=env.export_state()`，其中包含退款记录和通过的 dry-run 记录。
> 不传时，验证器无法核实退款是否真实发生，声称 `refund` 的答案最高只能得 0.2，以防奖励幻觉出来的 `refund_id`。

## 复现与验收步骤

1. `docker build -t rlvr-refund . && docker run --rm rlvr-refund`
2. 预期输出：10 个正确场景都是 `REWARD: 1.0`。错误场景的奖励如下：
   | 错误场景 | 奖励 | 原因 |
   | --- | --- | --- |
   | 从下单日起算时限 → 误拒 | 0.1 | 误拒，上限 0.1 |
   | 忽略会员等级 → 误退款 | 0.1 | 资金损失，上限 0.1 |
   | 用订单总额判阈值 → 误转人工 | 0.1 | 误转人工且金额错误，金额错误上限 0.1 |
   | 按整行数量退款（应退 2×60，实退 4×60） | 0.1 | 金额错误（多退钱），上限 0.1 |
   | 照片数少于声称的损坏数，仍按声称数量退款 | 0.1 | 金额错误（多退钱），上限 0.1 |
   | dry-run 通过但没真正退款，却声称已退 | 0.2 | 上限 0.2 |
   | 既没 dry-run 也没退款，纯编造已退款 | 0.1 | 上限 0.1 |
   | 重复退款 | 0.1 | 上限 0.1 |
   | 拒绝原因码写错（应为 invalid_evidence） | 0.2 | 决策错误，上限 0.2 |

   此外还会演示两点：无法解析的输出得 0 分，reject 缺少原因码也算格式错误，这是唯一会得 0 的情况；非法 API 调用返回错误字典，不会中断流程。
3. `docker run --rm rlvr-refund python -m pytest -q`，预期 53 passed

## 调用样例输出（节选）

```
[正确: 可退款 (签收日起算, 只退损坏商品)] task=refund-O1001
  agent answer : {"order_id": "O1001", "decision": "refund", "refund_amount": 299.0, "refund_id": "RF-O1001-002", "reject_reason": null}
  components   : {'decision': 1.0, 'amount': 1.0, 'execution': 1.0}  caps=[]
  REWARD       : 1.0
[错误: 忽略等级用最长时限 -> 误退款] task=refund-O1002
  agent answer : {"order_id": "O1002", "decision": "refund", "refund_amount": 199.0, "refund_id": "RF-O1002-002", "reject_reason": null}
  components   : {'decision': 0.0, 'amount': 0.0, 'execution': 0.0}  caps=['harmful_refund', 'wrong_decision']
  REWARD       : 0.1
```
