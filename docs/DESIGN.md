# 设计说明：电商售后退款 RLVR 任务

## 1. 任务设计思路

场景：电商售后客服 Agent 处理"收到的商品损坏，能否退货"工单。
这类任务对 LLM Agent 有真实难度：答案不在任何一张表里，必须
**从多个 API 拼出事实 → 套用分散在政策中的规则 → 计算金额 → 执行（或不执行）退款动作**，
任何一步出错都会改变最终结果。

### 自然语言输入

```
【售后工单】订单号 O1001：我现在收到的快递是坏的，照片已经上传了，可以退货吗？
请按平台售后政策处理本工单，并以 JSON 输出最终结果：
{"order_id": str, "decision": "refund|reject|escalate|wait_return", "refund_amount": number, "refund_id": str|null}
```

共 9 个任务实例（`rlvr_task/tasks.py: TASKS`），同一模板、不同订单，覆盖全部 4 种决策分支（refund 3 / reject 3 / escalate 2 / wait_return 1），
避免单题被策略"背答案"。

### 最终输出格式

| 字段 | 含义 |
| --- | --- |
| `order_id` | 工单订单号（可省略；若给出必须一致） |
| `decision` | `refund` 已退款 / `reject` 不予退款 / `escalate` 转人工 / `wait_return` 等待退货包裹 |
| `refund_amount` | 应退金额；`reject` 时为 0；`escalate`/`wait_return` 填按政策计算出的应退金额 |
| `refund_id` | `decision=refund` 时必须是 `issue_refund` 返回的真实 id，否则为 `null` |

### 成功判定

奖励 = 1.0 当且仅当：决策正确、金额在容差内、且环境最终状态与决策一致
（应退款则恰好退了一次且 id/金额匹配；不应退款则没有任何退款记录）。

### 业务规则（`rlvr_task/config.py` 统一管理）

| 规则 | 值 |
| --- | --- |
| 退货时限 | **自签收日起算**；bronze 7 天 / silver 15 天 / gold 30 天 |
| 附件证据 | 损坏照片须在**签收后 24 小时内**拍摄，并对应具体商品 |
| 不可退商品 | `returnable=false`（如电脑）一律拒绝 |
| 应退金额 | 损坏商品单价 × 损坏数量 × 比例（默认 100%，运费商家全包）；改比例/加运费须转人工 |
| 人工审核阈值 | 应退金额超过 bronze 500 / silver 500 / gold 1000 须转人工 |
| 退款时机 | 退货包裹到达商家（`arrived_at_merchant`）后才能退款，否则提醒寄回并等待 |
| 规则优先级 | 不可退 → 超时限 → 证据无效（以上均 reject）→ 超阈值 escalate → 包裹未到 wait_return → refund |

### 隐含约束（题面不写，需 Agent 推导）与陷阱数据

| 隐含约束 | 来源 | 陷阱实例 | 忽略后的错误结果 |
| --- | --- | --- | --- |
| 时限从**签收日**而非下单日起算 | `get_policy.rules` + `check_delivery` | O1001：下单 20 天、签收 12 天（silver 15 天） | 误判超期 → reject |
| 时限随**会员等级**变化 | `get_customer_tier` | O1002：bronze，签收 9 天；O1007：gold，签收 20 天 | 套用 30 天 → 误退款；套用 7/15 天 → 误拒 |
| 照片须在签收后 24h 内 | `check_attachment` + `check_delivery` | O1003：签收后 54h 拍摄 | 误退款 |
| 阈值按等级，且按**损坏商品**金额而非订单总额 | `need_permission` + `get_policy.items` | O1004：silver 680 > 500；O1008：bronze 520 > 500；O1005：总额 1139 但咖啡机 899 ≤ 1000 | 误退款 / 误转人工 |
| 只退**损坏数量**，而非整行或整单 | `check_attachment.damaged_qty` | O1009：4 只杯子碎 2 只 → 120 | 退 240 / 270 → 金额错 |
| 包裹未到商家不能退 | `check_delivery.return_status` | O1005：`in_transit` | 提前退款 |
| 不可退品类 | `get_policy.items.returnable` | O1006：笔记本电脑 4999 | 误转人工 |
| 退款必须先查政策、先 dry-run | **由环境强制**（token 机制） | — | 无法拿到 `refund_id` |

## 2. API 接口定义（`rlvr_task/env/api.py: RefundEnv`）

所有方法返回 `{"ok": True, ...}` 或 `{"ok": False, "error": "..."}`，**从不抛异常**；
也可通过 `env.call(name, **kwargs)` 统一分发（未知工具 / 参数错误同样返回错误字典）。

| API | 类型 | 必要性 | 返回要点 |
| --- | --- | --- | --- |
| `get_policy(order_id)` | 查询 | 必要 | `policy_id`、`customer_id`、`order_date`、商品明细（单价/数量/`returnable`）、规则说明、`current_date` |
| `check_attachment(order_id)` | 查询 | 必要 | 附件：`item_id`、`damage_type`、`damaged_qty`、`taken_at` |
| `check_delivery(order_id)` | 查询 | 必要 | `delivered_at`、`return_status`（及承运商、单号等干扰字段） |
| `get_customer_tier(customer_id)` | 查询 | 必要 | `tier`（及姓名、电话等干扰字段） |
| `need_permission(customer_id, amount)` | 查询 | 必要 | `need_manual_review`（需 Agent 先算出正确金额） |
| `dryrun_refund(order_id, item_id, amount, policy_id)` | 动作 | 退款时必要 | 一次性 `dryrun_token`；仅做安全检查（已查政策、商品属于订单、金额不超过实付），**不判断业务资格** |
| `issue_refund(order_id, dryrun_token)` | 动作 | 退款时必要 | `refund_id`，写入退款表 |
| `notify_refund(order_id, message)` | 动作 | wait_return 时建议 | 记录通知 |
| `check_order(customer_id)` | 查询 | **冗余** | 历史订单与 `order_total`（总额是金额陷阱） |
| `get_address(order_id)` | 查询 | **冗余** | 收货/退货地址 |

干扰字段：`phone`、`registered_at`、`remark`、`carrier`、`tracking_no`、`file_name`、`uploaded_at`、`order_total` 等。
注意 `uploaded_at` 与 `taken_at` 的区别：证据规则看拍摄时间而非上传时间。

每个 episode 的 `RefundEnv()` 都会把基础库复制到内存数据库，回合之间互不影响，结果可复现。

## 3. 典型执行路径（以 O1001 为例）

| 步 | 调用 | 目标 | 关键返回 | 依赖 |
| --- | --- | --- | --- | --- |
| 1 | `get_policy("O1001")` | 获取客户、商品、规则 | `customer_id=C01`、`policy_id`、耳机 299 / 手机壳 59 | 工单中的订单号 |
| 2 | `check_attachment("O1001")` | 确认损坏商品与证据 | `item_id=I1001A`、`taken_at=10-20 20:30` | — |
| 3 | `check_delivery("O1001")` | 签收时间、退货包裹状态 | `delivered_at=10-20 10:00`、`arrived_at_merchant` | — |
| 4 | `get_customer_tier("C01")` | 等级 → 时限 | `silver` → 15 天 | 步骤 1 的 `customer_id` |
| — | 计算 | 签收 12 天 ≤ 15 ✓；照片签收后 10.5h ✓；金额 = 299 × 1 × 100% = 299 | | 步骤 1+2+3+4 |
| 5 | `need_permission("C01", 299)` | 是否需人工 | `false`（≤ 500） | 步骤 1 + 计算出的金额 |
| 6 | `dryrun_refund("O1001","I1001A",299,policy_id)` | 安全预检 | `dryrun_token` | 步骤 1、2、计算 |
| 7 | `issue_refund("O1001", token)` | 执行退款 | `refund_id` | 步骤 6 |
| 输出 | `{"decision":"refund","refund_amount":299,"refund_id":"RF-..."}` | | | |

**为什么必须多步：** 时限需要 *签收时间*（delivery）× *等级*（tier）× *规则*（policy）三方关联；
金额需要 *损坏商品*（attachment）× *单价*（policy）；审核需要 *金额*（上一步计算）× *等级*。
任何单个 API 都不给出决策或金额；拒绝类工单也至少需要 3 次查询才能确认拒绝理由。

## 4. 奖励计算逻辑（`rlvr_task/verifier/reward.py`）

统一入口：`compute_reward(task_text, agent_output, env_state=None) -> float`；
`evaluate(...)` 返回带分项、触发的上限和标准答案的详细结果。

1. 由任务文本解析订单号 → `tasks.solve()` 从**基础库**按规则计算标准答案（不硬编码）。
2. 解析输出（dict 或 JSON 字符串，允许 ``` 包裹）并做 schema 校验；失败 → **0**。
3. 分项（权重在 `config.REWARD_WEIGHTS`）：
   - `decision` 0.5：决策是否等于标准答案；
   - `amount` 0.3：误差 ≤ max(0.01, 1%) 满分，相对误差到 20% 线性衰减为 0；
   - `execution` 0.2：环境最终状态是否与标准答案一致（应退：恰好一笔、id 和金额匹配；不应退：无退款）。
4. 硬上限（取最低）：
   | 情况 | 上限 |
   | --- | --- |
   | 输出格式非法 | 0 |
   | 重复退款 | 0 |
   | 不该退却执行了退款（资金损失） | 0 |
   | 声称退款但未真正退款 / 已退款但声称未退 | 0.2 |
   | 决策错误 | 0.2 |

**只看最终结果，不看轨迹：** 验证器从不检查调用顺序、次数。"先查政策、先 dry-run"由环境的 token 机制强制，
违反顺序就拿不到 `refund_id`，错误自然体现在最终结果上。`env_state` 是训练 Harness 在回合结束时调用
`env.export_state()` 得到的最终世界状态，属于"最终结果"的一部分；未提供时只检查答案内部一致性。

验证器不调用任何大模型，且对任何输入都不会抛异常。

## 5. 适用的 RLVR 训练场景

- 多轮工具调用 Agent 的 RL 训练（如 GRPO/PPO + tool-use rollout）：奖励稀疏但客观，可精确复现；
- 训练"读规则 → 推导隐含条件 → 跨源关联 → 计算 → 再行动"的能力，而不是关键词匹配；
- 安全约束学习：错误退款和重复退款被直接打到 0，策略会学会"拿不准就不执行不可逆动作"；
- 部分分（金额、执行）提供比 0/1 更平滑的信号，同时用上限防止"决策错但格式好"刷分。

## 6. 已知局限与可扩展方向

- 实例数较少（9 个），可用同一规则引擎批量生成参数化订单来扩充；
- 附件是结构化记录而非真实图片（为了能客观判定）；
- `notify_refund` 在 wait_return 分支中不计分；
- 假设每个工单只有一个损坏附件；多商品损坏可扩展为按商品求和。
