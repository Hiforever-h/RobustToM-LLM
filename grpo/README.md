# RobustToM v3 GRPO

This directory contains the task-specific adapter and reward integration for
training `runs/final` with the bundled verl implementation. The scorer is
imported directly from `rft.reward`; the RFT implementation is not modified.

Create the independent RL environment on the A800 host:

```bash
conda create -n robusttom-grpo python=3.10 -y
conda activate robusttom-grpo
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.0
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.0.post2 --no-build-isolation
python -m pip install -e . --no-deps
wandb login
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export TMPDIR=/root/autodl-tmp/tmp
export HF_ENDPOINT=https://hf-mirror.com
export RFT_MODEL_PATH=/root/autodl-tmp/runs/rft_train/20260820-qwen25-3b-k16/final
export RAW_DATA_DIR=/root/RobustToM-RL/data/counterfactual_process_reward_v3

export GRPO_RUN_DIR=/root/autodl-tmp/runs/grpo/qwen25_3b_rft_grpo_n16_seed2026
export GRPO_DATA_DIR=/root/autodl-tmp/data/grpo/counterfactual_process_reward_v3_fewshot
export GRPO_LOG_DIR="${GRPO_RUN_DIR}/logs"

export WANDB_DIR="${GRPO_RUN_DIR}/wandb"
export WANDB_CACHE_DIR=/root/autodl-tmp/cache/wandb
export RAY_TMPDIR=/root/autodl-tmp/tmp/ray
export TMPDIR=/root/autodl-tmp/tmp
RUN_ID=20260822-qwen25-3b-k16
MODEL=Qwen/Qwen2.5-3B-Instruct
```

The main configuration logs to both the terminal and Weights & Biases under
project `robust_tom_grpo_v3`. Set `WANDB_API_KEY` non-interactively on a remote
host, or use `WANDB_MODE=offline` when outbound network access is unavailable.

Build the audited parquet files:

```bash
bash grpo/run_grpo_v3.sh build
```

Run the A800 checks in order:

```bash
bash grpo/run_grpo_v3.sh validate
bash grpo/run_grpo_v3.sh smoke
bash grpo/run_grpo_v3.sh pilot
bash grpo/run_grpo_v3.sh train
```

The main run uses 8 prompts per step, 16 rollouts per prompt, 800 optimizer
steps, two epochs, a learning rate of `5e-7`, temperature `1.0`, and asymmetric
PPO clipping with ratio bounds `[0.8, 1.3]`.

Resume from an actor checkpoint without changing the frozen reference:

```bash
bash grpo/run_grpo_v3.sh train \
  trainer.resume_from_path=runs/grpo/qwen25_3b_rft_grpo_n16_seed2026/actor/global_step_400
```

## 2026-08-21 RFT 与 GRPO 评测结论

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
