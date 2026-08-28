# RobustToM natural-CoT GRPO

本目录是 RobustToM 的 GRPO 训练入口。当前主链路不再要求 actor 输出 JSON 或
`belief_chain`，而是输出轻量自然语言过程：

```text
Think 1:
Henry privately saw the flashlight move to blue_canvas_bag.
State: blue_canvas_bag

Think 2:
Thomas did not observe Henry's later private update.
State: ceramic_jar

Answer: ceramic_jar
```

训练时，每个 prompt 采样 16 条 response。本地规则负责检查 Think 数量、每步
`State` 和最终 `Answer`；`deepseek-v4-flash` 只评价每步 reasoning。一个 prompt
及其 16 条 response 被打包为一次 Judge 请求，默认一个训练 step 的 8 个请求并发执行。

旧版 JSON/few-shot 实验仍可通过 `grpo/run_grpo_json_v3.sh` 复现；
`grpo/run_grpo_v3.sh` 现在是 natural-CoT 脚本的兼容别名。

## 1. 运行环境

正式训练要求：

- Linux x86_64；
- NVIDIA CUDA 12.1；
- Python 3.10；
- PyTorch 2.4.0、vLLM 0.6.3；
- 建议使用 A800 或同等级 GPU。

在项目根目录创建环境：

```bash
conda create -n robusttom python=3.10 pip -y
conda activate robusttom
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.0.post2 --no-build-isolation
python -m pip install -e . --no-deps
```

根目录 `requirements.txt` 是 Linux/CUDA 训练清单。macOS 不支持其中的 vLLM、
xFormers 和 flash-attn；macOS 环境只适合构建数据、调试 Judge/RewardManager 和运行单测，
不能执行下面的 validate/smoke/pilot/train trainer 模式。

## 2. 环境变量

下面是一套 A800 主机示例。路径可以按机器实际情况修改：

```bash
conda activate robusttom

export RFT_MODEL_PATH=/root/autodl-tmp/runs/rft_train/20260820-qwen25-3b-k16/final
export RAW_DATA_DIR=/root/RobustToM-LLM/data/counterfactual_process_reward_v3
export NATURAL_SOURCE_DIR=/root/autodl-tmp/data/counterfactual_process_reward_v4_natural
export GRPO_DATA_DIR=/root/autodl-tmp/data/grpo/counterfactual_process_reward_v4_natural
export GRPO_OUTPUT_ROOT=/root/autodl-tmp/runs/grpo
export GRPO_LOG_DIR=/root/autodl-tmp/runs/grpo/logs

export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_DIR=/root/autodl-tmp/wandb
export WANDB_CACHE_DIR=/root/autodl-tmp/cache/wandb
export RAY_TMPDIR=/root/autodl-tmp/tmp/ray
export TMPDIR=/root/autodl-tmp/tmp
```

Judge 默认读取项目根目录 `.env`：

```dotenv
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

也可以直接导出 `DEEPSEEK_API_KEY`。不要把 key 写进 yaml、shell 脚本或训练日志。

如果不使用在线 W&B：

```bash
export WANDB_MODE=offline
```

## 3. 数据构建

执行：

```bash
bash grpo/run_grpo_natural.sh build
```

该命令会：

1. 从 `RAW_DATA_DIR` 读取 v3 train/val/test JSONL；
2. 删除旧 `process_response`、JSON schema、few-shot 和事件编号；
3. 为每个样本生成 `natural-cot-think-state-v2-exact-order` 多行 actor prompt，
   明确 ToM 阶数、人物层级和恰好 N 个空 `Think/State` 模板；
4. 生成 natural-CoT source JSONL 到 `NATURAL_SOURCE_DIR`；
5. 生成 verl parquet 到 `GRPO_DATA_DIR`；
6. 在 parquet 中显式保存 `judge_prompt` 和隐藏的结构化 process target；
7. 使用 `RFT_MODEL_PATH` 的 tokenizer 审计 prompt 长度。

仓库当前已包含一份构建结果：

```text
data/counterfactual_process_reward_v4_natural/
data/grpo/counterfactual_process_reward_v4_natural/
```

## 4. 推荐执行顺序

```bash
bash grpo/run_grpo_natural.sh validate
bash grpo/run_grpo_natural.sh smoke
bash grpo/run_grpo_natural.sh pilot
bash grpo/run_grpo_natural.sh train
```

### 4.1 validate：规则验证，不请求 Judge

```bash
bash grpo/run_grpo_natural.sh validate
```

行为：

- `trainer.val_only=true`；
- 对整个 val split 做 deterministic generation；
- 只计算 Think 结构、State、Answer、EOS、shortcut conflict 等本地指标；
- 不调用 DeepSeek；
- 不计算梯度，不执行 optimizer update，不保存 checkpoint；
- logger 强制为 console。

主要关注：

```text
val/overall/structure_valid_rate
val/overall/state_step_accuracy
val/overall/all_states_correct_rate
val/overall/answer_accuracy
val/overall/answer_correct_state_trace_wrong_rate
```

validate 的意义是确认 checkpoint 能在新 prompt 下产生可解析的 natural-CoT。如果
`structure_valid_rate` 很低，应先处理格式迁移，不要直接进入完整 GRPO。

### 4.2 smoke：一次真实 Judge + optimizer update

```bash
bash grpo/run_grpo_natural.sh smoke
```

固定缩小为：

```text
1 prompt × 2 rollouts × 1 optimizer step
1 个并发 Judge worker
无 validation、无 checkpoint、仅 console logger
```

smoke 会真实执行：模型采样、packed Judge 请求、本地规则门控、reward 回填、GRPO
advantage、actor update。启动前 preflight 会验证：

- `DEEPSEEK_API_KEY`；
- `https://api.deepseek.com/chat/completions`；
- `deepseek-v4-flash` 是否位于 `/models` 返回列表；
- Judge cache 目录是否可写。

smoke 通过标准是进程完成一个 optimizer step，并出现非空的 reward/Judge 指标。

### 4.3 pilot：短程生产形状实验

```bash
bash grpo/run_grpo_natural.sh pilot
```

pilot 使用与正式训练相同的 `8 prompts × 16 rollouts`，但只跑 50 个 optimizer
steps。每 10 步执行 rule-only validation，每 25 步保存 checkpoint，并在结束时保存和验证。

建议重点检查：

```text
reward/structure_valid_rate
reward/state_step_accuracy
reward/judge_reasoning_step_mean
reward/answer_correct_process_imperfect_rate
reward/judge/latency_mean_seconds
reward/judge/latency_max_seconds
reward/judge/total_tokens
grpo/group_reward_std_mean
grpo/zero_variance_group_rate
```

进入正式训练前建议满足：

- response 格式可解析率稳定；
- 大部分 group 的 reward 不是零方差；
- Judge latency 和 token 成本可接受；
- 没有持续的 schema normalization、认证失败或限流重试；
- `answer correct / process imperfect` 没有持续恶化。

### 4.4 train：完整训练

```bash
bash grpo/run_grpo_natural.sh train
```

默认配置位于 `verl/trainer/config/robust_tom_natural_grpo.yaml`：

| 参数 | 默认值 |
| --- | ---: |
| train batch | 8 prompts |
| rollout.n | 16 |
| 每步 response | 128 |
| Judge 并发 | 8 |
| max response length | 384 |
| optimizer steps | 800 |
| epochs | 2 |
| learning rate | 5e-7 |
| save frequency | 400 |
| rule-only validation frequency | 50 |

Judge 或网络错误在重试耗尽后会让当前训练安全失败，不会把基础设施故障转换成
全零 reward 后继续更新。

## 5. Hydra 覆盖与恢复训练

所有额外参数都会原样转发给 Hydra。例如只跑 100 步：

```bash
bash grpo/run_grpo_natural.sh train \
  trainer.total_epochs=1 \
  trainer.total_training_steps=100 \
  trainer.save_freq=50
```

从 actor checkpoint 恢复：

```bash
bash grpo/run_grpo_natural.sh train \
  trainer.resume_from_path=/root/autodl-tmp/runs/grpo/qwen25_3b_natural_cot_judge_n16_seed2026/actor/global_step_400
```

`RFT_MODEL_PATH` 仍然作为 frozen reference；`resume_from_path` 只替换待继续训练的 actor。

查看脚本帮助：

```bash
bash grpo/run_grpo_natural.sh help
```

## 6. 输出目录

默认输出：

```text
runs/grpo/<experiment_name>/
runs/grpo/logs/<experiment_name>.log
runs/grpo/<experiment_name>/judge_cache/
```

Judge cache key 包含 endpoint、模型、rubric、prompt、target 和全部候选 response，
但不包含训练 step uid。因此相同请求跨恢复或重跑时可以复用，cache hit 不计入当次
billable token 指标。

---

## Legacy：2026-08-21 JSON RFT 与 GRPO 评测结论

评测产物位于
[`runs/20260821-qwen25-3b-k16`](../runs/20260821-qwen25-3b-k16)。RFT 和
GRPO checkpoint 都分别在 dev、Hi-ToM_4-order_OOD 上完成了评测。同一 split 内，两组预测的
样本 ID、`process_prompt`、套用 chat template 后的 prompt hash 以及
`process_target` 均完全一致，因此模型间的逐样本配对比较有效。

两个 split 的数据分布并不相同：

- `dev` 有 400 条 1-3 阶样本，属于训练阶数范围内的数据。
- `Hi-ToM_4-order_OOD` 有 600 条纯 4 阶样本。训练集、验证集以及固定 few-shot 均只覆盖
  1-3 阶，因此 Hi-ToM_4-order_OOD 衡量的是 ToM 阶数外推，而不是普通的同分布泛化。

评测命令如下：

* ```
  python -m rft.generate \
    --data /root/autodl-tmp/data/grpo/counterfactual_process_reward_v3_fewshot/Hi-ToM_4-order_OOD.jsonl \
    --model /root/autodl-tmp/runs/grpo/qwen25_3b_rft_grpo_n16_seed2026/final \
    --output /root/autodl-tmp/runs/grpo_eval/${RUN_ID}/Hi-ToM_4-order_OOD_predictions.jsonl
  ```

* ```
  python -m rft.evaluate \
    --predictions /root/autodl-tmp/runs/grpo_eval/${RUN_ID}/Hi-ToM_4-order_OOD_predictions.jsonl \
    --data /root/autodl-tmp/data/grpo/counterfactual_process_reward_v3_fewshot/Hi-ToM_4-order_OOD.jsonl \
    --output /root/autodl-tmp/runs/grpo_eval/${RUN_ID}/Hi-ToM_4-order_OOD_metrics.json
  ```

* Derived_fewshot.jsonl的评测命令与上述两条命令类似。

### 总体结果

| 数据 | 指标 | RFT | GRPO | 绝对变化 |
| --- | --- | ---: | ---: | ---: |
| dev（1-3 阶） | 平均 process reward | 0.327 | 0.812 | +0.486 |
| dev（1-3 阶） | 最终答案正确率 | 10.3% | 66.5% | +56.3 pp |
| dev（1-3 阶） | 完整 belief trace 正确率 | 2.3% | 64.5% | +62.3 pp |
| dev（1-3 阶） | 满分率 | 1.5% | 61.0% | +59.5 pp |
| dev（1-3 阶） | pair 两条全对率 | 1.5% | 40.5% | +39.0 pp |
| Hi-ToM_4-order_OOD（纯 4 阶） | 平均 process reward | 0.256 | 0.634 | +0.378 |
| Hi-ToM_4-order_OOD（纯 4 阶） | 最终答案正确率 | 10.3% | 47.7% | +37.3 pp |
| Hi-ToM_4-order_OOD（纯 4 阶） | 完整 belief trace 正确率 | 0.3% | 27.0% | +26.7 pp |
| Hi-ToM_4-order_OOD（纯 4 阶） | 满分率 | 0.0% | 15.8% | +15.8 pp |
| Hi-ToM_4-order_OOD（纯 4 阶） | pair 两条全对率 | 0.3% | 16.7% | +16.3 pp |

提升不是由少数样本拉动的。GRPO 在 362/400 条 dev 样本和 549/600 条 Hi-ToM_4-order_OOD
样本上提高了逐样本 reward，分别只有 6 条和 27 条出现下降。Hi-ToM_4-order_OOD 的配对 reward
平均提升为 `+0.378`，近似 95% 置信区间为 `[+0.356, +0.400]`。

### 已取得的改进

GRPO 学到的不只是 JSON 格式。在纯 4 阶 Hi-ToM_4-order_OOD 上，把解析失败也计为错误时，
belief trace 各级位置正确率如下：

| belief trace 层级 | RFT | GRPO |
| --- | ---: | ---: |
| 第 1 级，最内层 belief | 33.2% | 96.5% |
| 第 2 级 | 14.0% | 71.2% |
| 第 3 级 | 8.3% | 51.8% |
| 第 4 级，完整查询链 | 8.8% | 40.3% |

Hi-ToM_4-order_OOD 上平均 trace step 正确比例约从 16.1% 提升到 65.0%，dev 上约从 23.3%
提升到 80.9%。这说明 GRPO 确实学到了由内向外更新 belief 的目标过程，但错误
仍会随嵌套层级加深而累积。

输出格式和稳定性也明显改善：

- dev 解析率从 95.0% 提升到 100%，Hi-ToM_4-order_OOD 从 97.3% 提升到 99.0%。
- dev response 长度 P95 从约 255 降至 89 tokens，Hi-ToM_4-order_OOD 从约 199 降至
  132 tokens。
- 两个 split 上的精确 shortcut-copy rate 均降至 0%。
- last-mention copy rate 在 dev 和 Hi-ToM_4-order_OOD 上分别降至 0.25% 和 0.17%。
- GRPO 的全部 dev 输出都正常到达 EOS；Hi-ToM_4-order_OOD 只有 6/600 条达到 256-token
  上限。RFT 在 dev 和 Hi-ToM_4-order_OOD 上分别有 20/400 和 16/600 条达到上限。

### 仍然存在的问题

4 阶任务仍未解决。平均 reward 不能直接当作完整任务准确率，因为权重为 0.55
的 belief trace 会按正确步骤比例给分。Hi-ToM_4-order_OOD reward 达到 0.634 的同时，完整
trace 正确率只有 27.0%，满分率只有 15.8%。

`tom_order` 是目前的重要瓶颈。GRPO 在 dev 上的 `tom_order` 正确率为 94.3%，
但在纯 4 阶 Hi-ToM_4-order_OOD 上只有 52.0%。Hi-ToM_4-order_OOD 中常见的错误值为 5（134 条）、3（82 条）
和 6（31 条）。其中有 67 条样本的 reward 为 0.95，除 `tom_order` 外的所有
评分内容均正确。如果只对该字段做 oracle 修正，满分样本会从 95 条增加到
162 条，即满分率从 15.8% 上升到 27.0%。

最终答案表现明显好于过程轨迹。GRPO 在 Hi-ToM_4-order_OOD 上有 286 条最终答案正确，但只有
162 条完整 trace 正确，即有 124 条属于“答案正确，但中间轨迹不完整或错误”。
当前 reward 只有在完整轨迹正确时才发放 0.20 的 answer 分，可以避免这些样本
仅凭最终猜测获得该部分奖励。

反事实 pair 行为仍然有限。Hi-ToM_4-order_OOD pair 两条全对率从 0.3% 提升到 16.7%，但
intervention sensitivity 只从 41.0% 提升到 42.3%。在 GRPO 只答对 pair 一侧的
186 对样本中，有 149 对在 hidden 和 observed 版本上给出了相同答案。因此，
不少单样本正确仍不能说明模型可靠地追踪了反事实干预。

hidden 干预在过程层面更难。纯 4 阶 Hi-ToM_4-order_OOD 上，hidden 和 observed 的最终答案
正确率接近，分别为 48.3% 和 47.0%；但完整 trace 正确率分别只有 20.3% 和
33.7%，满分率分别为 11.7% 和 20.0%。模型有时能得到正确最终答案，却没有在
完整 trace 中正确传播未观察事件对应的知识状态。

### 评测限制与后续建议

当前 1,000 条评测 prompt 均不包含 GRPO 训练时使用的消歧说明：

```text
tom_order is exactly the number of names in belief_chain, not the number of story events. belief_trace contains exactly tom_order entries.
```

这不影响当前 RFT 与 GRPO 的公平对比，因为两个模型看到的是完全相同的评测
prompt；但当前评测与 GRPO 训练 prompt 并不完全一致，可能低估 `tom_order` 和
满分率，尤其是未见过的 4 阶任务。最高优先级的后续工作是使用完整的
`build_grpo_prompt` 输出，在 dev 和 Hi-ToM_4-order_OOD 上重新评测两个 checkpoint。复评时可将
`max_new_tokens` 提高到 384，以区分剩余 6 条 GRPO runaway 输出究竟是截断问题
还是格式/推理问题；这最多影响当前 Hi-ToM_4-order_OOD 约 1 个百分点。

现有预测产物保存了 prompt、formatted prompt hash、target、response、token
数量和 finish reason，但没有记录实际 checkpoint 路径及完整生成命令。后续评测
应额外保存 manifest，以便独立复现模型和解码配置。
