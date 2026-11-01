# 设计说明：电商售后退款 RLVR 任务

## 1. 任务设计思路

场景：电商售后客服 Agent 处理"收到的商品坏了，能不能退货"工单。
这类任务对 LLM Agent 有真实难度，因为答案不在任何一张表里。Agent 必须按顺序完成以下步骤：
从多个 API 拼出事实；套用分散在政策里的规则；逐件核对附件证据；计算金额；最后执行或不执行退款。
任何一步出错，最终结果都会变。

### 自然语言输入

```
【售后工单】订单号 O1001：我现在收到的快递是坏的，照片已经上传了，可以退货吗？
请按平台售后政策处理本工单，并以 JSON 输出最终结果：
{"order_id": str, "decision": "refund|reject|escalate|wait_return", "refund_amount": number, "refund_id": str|null,
 "reject_reason": "outside_window|invalid_evidence|non_returnable"|null}
（decision 为 reject 时必须给出 reject_reason，否则为 null）
```

共 10 个任务实例（见 `rlvr_task/tasks.py: TASKS`）。它们用同一个模板，只是订单不同，覆盖了全部 4 种决策分支：
refund 4 个、reject 3 个、escalate 2 个、wait_return 1 个。这样可以避免策略把单道题的答案背下来。

### 最终输出格式

| 字段 | 含义 |
| --- | --- |
| `order_id` | 工单订单号。可以省略；如果给出，必须与工单一致 |
| `decision` | `refund` 已退款 / `reject` 不予退款 / `escalate` 转人工 / `wait_return` 等待退货包裹 |
| `refund_amount` | 应退金额。`reject` 时填 0；`escalate` 和 `wait_return` 填按政策算出的应退金额 |
| `refund_id` | `decision=refund` 时必须是 `issue_refund` 返回的真实 id，否则填 `null` |
| `reject_reason` | 仅 `reject` 时必填：`outside_window` 超出时限 / `invalid_evidence` 没有有效照片 / `non_returnable` 有有效照片的商品都不可退。原因码不对按决策错误处理；缺失或非法按格式错误处理 |

### 成功判定

奖励为 1.0，当且仅当以下三点同时满足：
- 决策正确（reject 时原因码也必须正确）；
- 金额在容差范围内；
- 环境最终状态与决策一致：该退款的，恰好退了一次，且 id 和金额都匹配；不该退款的，没有任何退款记录。

### 业务规则（统一在 `rlvr_task/config.py` 管理）

| 规则 | 值 |
| --- | --- |
| 退货时限 | **从签收日起算**：bronze 7 天、silver 10 天、gold 14 天 |
| 附件证据 | 照片必须在**签收后 24 小时内**拍摄才有效。按商品逐件判定：**每件损坏都至少要有一张有效照片**，同一件可以有多张。认可的损坏数量 = min(照片声称的最大损坏数, 有效照片张数, 购买数量) |
| 不可退商品 | `returnable=false` 的商品（如电脑）不计入退款 |
| 应退金额 | 对每件可退商品计算 单价 × 认可损坏数量 × 比例，再求和。比例默认 100%，运费由商家全包；要改比例或加收运费，必须转人工 |
| 人工审核阈值 | 应退金额超过 bronze 200 / silver 500 / gold 1000 时，必须转人工 |
| 退款时机 | 退货包裹到达商家（状态为 `arrived_at_merchant`）后才能退款；否则提醒顾客寄回，并等待 |
| 规则优先级 | 超出时限（reject / `outside_window`）→ 没有任何可退且有有效证据的商品（reject：有有效照片但都不可退为 `non_returnable`，否则为 `invalid_evidence`）→ 超过阈值转 escalate → 包裹未到为 wait_return → refund |

### 隐含约束（题面不写，需要 Agent 自己推导）与陷阱数据

每个陷阱实例都是专门构造的：如果 Agent 忽略对应的隐含约束，就会得出一个**不同的**错误答案。

| 隐含约束 | 信息来源 | 陷阱实例 | 忽略后的错误结果 |
| --- | --- | --- | --- |
| 损坏退货要选对**条款**：适用 R2（质量问题），不适用 R1（七天无理由） | `get_policy.clauses` | 所有实例 | 按 R1 从下单日起算时限 → 误拒；按 R1 扣运费 → 金额错误 |
| 时限从**签收日**起算，不是下单日 | `get_policy.clauses[R2].window_start` + `check_delivery` | O1001：下单已 20 天，签收才 8 天（silver 10 天） | 误判超期 → reject |
| 时限随**会员等级**变化 | `get_customer_tier` | O1002：bronze，签收 9 天；O1007：gold，签收 12 天 | 套用最长时限 → 误退款；套用 7 或 10 天 → 误拒 |
| 照片必须在签收后 24 小时内拍摄 | `check_attachment` + `check_delivery` | O1003：签收后 54 小时才拍；O1010：水壶照片在签收后 30 小时拍 | 误退款，或金额多算 |
| 每件损坏至少要有一张照片 | `check_attachment` | O1010：声称 2 个杯子坏了，但只有 1 张照片 | 退 258，应退 129 |
| 只退**认可的损坏数量**，不是整行或整单 | `check_attachment.damaged_qty` | O1009：4 个杯子坏了 2 个（有 2 张照片） | 退 240 或 270，应退 120 |
| 阈值按等级定，并且看**应退金额**而不是订单总额 | `need_permission` + `get_policy.items` | O1004：silver 680 > 500；O1008：bronze 520 > 200；O1005：订单总额 1139，但咖啡机只有 899 ≤ 1000 | 误退款，或误转人工 |
| 包裹没到商家不能退款 | `check_delivery.return_status` | O1005：状态为 `in_transit` | 提前退款 |
| 退货时限不是**保修期** | 冗余 API `get_product_info.warranty_days` | O1002：牙刷保修 730 天，但 bronze 的退货时限只有 7 天 | 把保修期当时限 → 误退款 |
| 不可退品类 | `get_policy.items.returnable` | O1006：笔记本电脑 4999 | 误转人工 |
| 退款前必须先查政策、查附件、做 dry-run | **由环境强制**（token 机制） | — | 拿不到 `refund_id` |

## 2. API 接口定义（`rlvr_task/env/api.py: RefundEnv`）

所有方法只返回两种结果：`{"ok": True, ...}` 或 `{"ok": False, "error": "..."}`，**任何情况下都不抛异常**。
也可以通过 `env.call(name, **kwargs)` 统一分发调用。遇到未知工具或参数错误，同样返回错误字典。

| API | 类型 | 是否必要 | 返回要点 |
| --- | --- | --- | --- |
| `get_policy(order_id)` | 查询 | 必要 | `policy_id`、`customer_id`、`order_date`、商品明细（单价、数量、`returnable`）、售后政策条款全集 `clauses`、`current_date` |
| `check_attachment(order_id)` | 查询 | 必要 | 每张附件的 `item_id`、`damage_type`、`damaged_qty`、`taken_at` |
| `check_delivery(order_id)` | 查询 | 必要 | `delivered_at`、`return_status`；另有承运商、单号等干扰字段 |
| `get_customer_tier(customer_id)` | 查询 | 必要 | `tier`；另有姓名、电话等干扰字段 |
| `need_permission(customer_id, amount)` | 查询 | 必要 | `need_manual_review`。Agent 必须先算出正确金额，才能传对参数 |
| `dryrun_refund(order_id, item_id 或 [item_id…], amount, policy_id)` | 动作 | 退款时必要 | 返回一次性 `dryrun_token`。前提是已调用过 `get_policy` 和 `check_attachment`。只做安全检查（商品属于该订单、金额不超过实付），**不判断是否符合退货政策** |
| `issue_refund(order_id, dryrun_token)` | 动作 | 退款时必要 | 返回 `refund_id`，并写入退款表（同时记录用了哪个 token） |
| `notify_refund(order_id, message)` | 动作 | wait_return 时建议调用 | 记录一条通知 |
| `check_order(customer_id)` | 查询 | **冗余** | 历史订单及 `order_total`。订单总额是金额陷阱 |
| `get_address(order_id)` | 查询 | **冗余** | 收货地址和退货地址 |
| `get_product_info(item_id)` | 查询 | **冗余** | 品牌、产地、`warranty_days` 保修期、重量、评分、库存。保修期是陷阱：保修期不是退货时限 |

`clauses` 是平台的完整售后政策，共 6 条。损坏退货只适用 R2（质量问题和运输损坏）、R4（人工审核）、R5（不可退商品）、R6（退款时机）。
R1（七天无理由退货，从**下单日**起算、扣运费）和 R3（生鲜 48 小时）是干扰条款，看起来相关，但套用后会算出错误的时限或金额。
R2 只给出结构化参数（例如 `window_start: delivered_at`），不附带解释，Agent 需要自己理解参数含义。

干扰字段包括 `phone`、`registered_at`、`remark`、`carrier`、`tracking_no`、`file_name`、`uploaded_at`、`order_total` 等。
其中证据规则看的是拍摄时间 `taken_at`，不是上传时间 `uploaded_at`。

每个 episode 新建的 `RefundEnv()` 会把基础库复制一份到内存数据库里，所以回合之间互不影响，结果可以复现。
回合结束时，`export_state()` 返回该回合的最终状态，包括退款记录 `refunds`、通过的 dry-run 记录 `dryruns` 和通知记录 `notifications`。

## 3. 典型执行路径（以 O1001 为例）

| 步 | 调用 | 目标 | 关键返回 | 依赖 |
| --- | --- | --- | --- | --- |
| 1 | `get_policy("O1001")` | 获取客户、商品，并选出适用的条款 R2/R4/R5/R6 | `customer_id=C01`、`policy_id`、耳机 299 / 手机壳 59 | 工单里的订单号 |
| 2 | `check_attachment("O1001")` | 确认损坏的商品和证据 | 耳机 1 张照片，`taken_at=10-24 20:30` | — |
| 3 | `check_delivery("O1001")` | 查签收时间和退货包裹状态 | `delivered_at=10-24 10:00`、`arrived_at_merchant` | — |
| 4 | `get_customer_tier("C01")` | 由等级确定时限 | `silver`，时限 10 天 | 步骤 1 的 `customer_id` |
| — | 计算 | 签收 8 天 ≤ 10 天 ✓；照片拍于签收后 10.5 小时 ✓；金额 = 299 × min(1,1,1) × 100% = 299 | | 步骤 1、2、3、4 |
| 5 | `need_permission("C01", 299)` | 判断是否需要人工 | `false`（299 ≤ 500） | 步骤 1 和计算出的金额 |
| 6 | `dryrun_refund("O1001", ["I1001A"], 299, policy_id)` | 安全预检 | `dryrun_token` | 步骤 1、2 和计算结果 |
| 7 | `issue_refund("O1001", token)` | 执行退款 | `refund_id` | 步骤 6 |
| 输出 | `{"decision":"refund","refund_amount":299,"refund_id":"RF-..."}` | | | |

**为什么必须多步调用：** 每个判断都需要多个 API 的数据关联起来。
- 判断时限：需要先从政策中选出适用的条款 R2，再把签收时间（来自 delivery）、会员等级（来自 tier）三方关联起来。
- 计算金额：需要逐件核对附件证据（来自 attachment），再乘以单价（来自 policy）。
- 判断是否审核：需要上一步算出的金额，再结合等级。

任何一个 API 都不会直接给出决策或金额。即使是拒绝类工单，也至少要查询 3 次才能确认拒绝理由。

## 4. 奖励计算逻辑（`rlvr_task/verifier/reward.py`）

统一入口是 `compute_reward(task_text, agent_output, env_state=None) -> float`。
`evaluate(...)` 返回详细结果，包括各分项得分、触发了哪些上限，以及标准答案。

1. 从任务文本中解析出订单号，再由 `tasks.solve()` 在**基础库**上按规则算出标准答案（不是硬编码的）。
2. 解析 Agent 的输出，然后做 schema 校验。输出可以是 dict，也可以是一段文本：文本里允许先写推理再给 JSON，也允许用 ``` 包裹，验证器取其中**最后一个**能解析的 JSON 对象。校验失败直接得 **0**，这是唯一会得 0 的情况。
3. 计算三个分项（权重见 `config.REWARD_WEIGHTS`）：
   - `decision`，权重 0.5：决策是否与标准答案一致；reject 时原因码也必须一致；
   - `amount`，权重 0.3：误差 ≤ max(0.01, 1%) 时满分。超出容差就会触发下面的"金额错误"上限 0.1，所以金额要么满分，要么整体封顶 0.1；
   - `execution`，权重 0.2：环境最终状态是否与标准答案一致。
4. 套用硬上限。多个上限同时触发时，取最低的那个：

   | 情况 | 上限 |
   | --- | --- |
   | 金额错误：声称的金额或实际退款金额超出容差（多退、少退、按订单总额算等） | 0.1 |
   | 重复退款 | 0.1 |
   | 不该退款却执行了退款（造成资金损失） | 0.1 |
   | 声称已退款，但既没做 dry-run 也没有退款（纯编造） | 0.1 |
   | 退款记录找不到对应的成功 dry-run（防御性检查；环境已强制先 dry-run，正常不会触发） | 0.1 |
   | dry-run 通过了，但没有执行 `issue_refund`，却声称已退款 | 0.2 |
   | 实际已退款，但答案声称没退，或 `refund_id` 对不上 | 0.2 |
   | 没有传 env_state 时声称退款（无法核实） | 0.2 |
   | 误拒：标准答案不是 reject，Agent 却拒绝了（直接伤害顾客） | 0.1 |
   | 误转人工：标准答案不是 escalate，Agent 却转了人工（有人兜底，但浪费审核人力） | 0.2 |
   | 其他决策错误（包括 reject 原因码写错） | 0.2 |

5. **保底 0.1**：只要答案能解析、格式合法，最低就是 0.1（见 `REWARD_FLOOR`）。这样造成损害的行为都落在保底分上，永远不会比无害的错误得分更高。

**只看最终结果，不看调用轨迹。** 验证器从不检查调用顺序或次数。"先查政策、先查附件、先 dry-run"这几条由环境的 token 机制强制执行：
不按顺序就拿不到 `refund_id`，错误自然会体现在最终结果上。`env_state` 是训练 Harness 在回合结束时调用 `env.export_state()` 得到的最终状态，
它属于"最终结果"的一部分，**Harness 必须传入**。如果没有传，验证器就无法核实退款是否真的发生，这时声称 `refund` 的答案最高只能得 0.2。

**平凡策略基线**（用来检查是否存在"什么都不做"的局部最优）：对全部 10 个实例始终输出同一个决策、金额填 0，平均奖励分别为：
- 始终 reject（无论固定哪个原因码）：0.21。没有原因码时是 0.40，见下文说明
- 始终 escalate：0.13
- 始终 wait_return：0.13
- 始终 refund：0.10

测试 `test_constant_policy_baseline_is_low` 保证，任何常数策略的平均奖励都低于 0.3。

引入原因码之前，"始终 reject"能拿到 0.40：10 个实例里有 3 个的正确答案就是 reject，猜中就是满分，误拒也只罚到 0.2。
现在做了两处修改：一是 reject 必须给出正确的原因码，靠猜只能命中其中 1 种；二是误拒的上限降到了 0.1。

验证器不调用任何大模型，对任何输入都不会抛异常。

## 5. 适用的 RLVR 训练场景

- **多轮工具调用 Agent 的 RL 训练**（例如 GRPO/PPO 配合 tool-use rollout）：奖励是稀疏的，但客观、可以精确复现。
- **训练推理链路**：让模型学会"读规则 → 推导隐含条件 → 跨数据源关联 → 计算 → 再行动"，而不是靠关键词匹配。
- **安全约束学习**：错误退款、重复退款、编造退款都会被打到 0.1 的保底分。策略会因此学会：拿不准时，不执行不可逆的动作，也不谎报已经执行。
- **偏严格的信号**：金额只要出错就封顶 0.1，因为在售后场景里多退、少退都是资金差错。代价是奖励比较稀疏，适合先做 SFT 或给出较强的初始策略，再进行 RL。

## 6. 已知局限与可扩展方向

- 实例数量较少（10 个）。可以复用同一套规则，批量生成参数化订单来扩充。
- 附件是结构化记录，不是真实图片。这是为了能客观判定。
- `notify_refund` 在 wait_return 分支里不计分。
- 政策目前对所有订单都一样。可以扩展为**按下单日期选择政策版本**（政策在某天改过规则），这是更强的隐含约束。
- 误拒和误转人工的上限目前是固定值。真实业务里，代价可能还和订单金额有关，可以进一步改成随金额变化。
- 因为"金额错误封顶 0.1"，`config` 里金额线性衰减的参数（`AMOUNT_ZERO_CREDIT_REL`）目前不影响最终奖励，只体现在 `evaluate()` 的分项明细里。如果以后想恢复部分分，调高或去掉 `CAP_WRONG_AMOUNT` 即可。
