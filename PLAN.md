# RobustToM-RL 实施计划

> 目标模型：`Qwen/Qwen2.5-3B-Instruct`
>
> 目标硬件：单张 NVIDIA A800 80GB
>
> 固定顺序：独立拒绝采样微调（RFT）-> 项目内 `verl` GRPO
>
> 计划基线日期：2026-08-19

## 1. 项目目标与成功标准

RobustToM-RL 的核心目标不是单纯提高 ToM 数据集上的最终答案准确率，而是缓解原 ToM-RL 中 Qwen2.5-3B 的两类失效：

1. **shortcut**：模型在反事实干预后仍沿用位置、提及顺序或角色退出位置等表面线索。
2. **reasoning collapse**：模型可能给出正确答案，但无法稳定地产生与答案一致、可机器验证的过程状态；或者 RL 后只剩格式外壳、重复文本、错误嵌套信念。

项目成功必须同时满足以下条件：

- 输出是严格、唯一、无 Markdown 包裹的 JSON 对象，并符合 `process_target_version=1.0`。
- `answer` 与 `nested_belief` 或 `world_state` 一致，不允许只猜中末尾答案。
- 在 observed/hidden 反事实对上，模型能随干预改变判断，而不是沿用同一 shortcut。
- 在未参与训练和选模的 order-4 OOD 测试集上保持上述能力。
- GRPO 相比 RFT checkpoint 带来内容推理收益，而不以格式率、外部 ToM 能力或训练稳定性为代价。

### 1.1 本项目对 RFT 的固定定义

为避免术语歧义，本计划中的 **RFT** 固定指 **Rejection Sampling Fine-Tuning（拒绝采样微调）**：从当前 policy 对每个 `process_prompt` 采样多个候选响应，用确定性 `process_reward` 拒绝错误响应，只对通过验收的模型自生成响应进行 response-only supervised fine-tuning，作为 GRPO 前的 policy 初始化。

- RFT 不使用 `verl`，采样、评分、数据构建和微调均在根目录 `rft/` 中独立实现。
- `process_target` / canonical `process_response` 只提供评分目标和审计基准；canonical `process_response` 不直接作为 RFT 训练 completion。
- RFT 前不增加 deterministic SFT bootstrap，也不为零通过样本回填金标响应。
- 若拒绝采样覆盖率不足，先增加采样数或检查 prompt、tokenization 和生成参数；仍达不到覆盖门槛则停止 RFT，不用部分错误响应降低验收标准。

## 2. 当前仓库事实

以下内容在制定本计划时已经存在：

- `data/counterfactual_process_reward/{train,val,test}.jsonl`
- `process_target`、`process_prompt` 和 canonical `process_response`
- `scripts/process_reward.py` 中的确定性 `[0,1]` process reward
- process target 和 reward 的单元测试
- 项目内 `verl` 0.1、vLLM 0.6.3 路径和旧版 `tom_grpo.sh`

当前数据规模：

| split | 记录数 | 内容 |
| --- | ---: | --- |
| train | 3,200 | Hi-ToM 2,000 + ExploreToM 1,200，order 0-3 |
| val | 400 | Hi-ToM，order 0-3，按 base story 隔离 |
| test | 600 | Hi-ToM order-4 OOD，base story 与低阶数据隔离 |

全部 4,200 条记录组成 2,100 个 observed/hidden pair。训练集中的 1,350 个 belief pair 的 observed 与 hidden 金标答案全部不同，因此 pair accuracy 和 intervention sensitivity 是有效指标。

需要牢记的现状限制：

- `data/` 被 `.gitignore` 忽略，不能把本地数据文件本身当作长期可复现记录。
- 当前 JSONL 尚未经过 RFT 候选采样/过滤，也未转换成 `verl` 所需 parquet。
- 当前 `process_reward.py` 尚未接入 `verl.trainer.main_ppo.RewardManager`。
- 当前 GRPO 脚本为多模型、多 GPU 配置，不适合单卡 A800，也没有使用 RFT checkpoint。
- 当前本机是 macOS，无 NVIDIA GPU，训练和 GPU smoke test 必须在 A800 环境执行。

## 3. 不可变实验约束

这些约束除非有新的实验证据，不应在实现过程中随意改变：

1. base model 固定为 `Qwen/Qwen2.5-3B-Instruct`。
2. RFT 候选采样、RFT 微调和 GRPO 使用完全相同的用户提示语义和 JSON 输出 schema。
3. prompt 只应用一次 Qwen chat template；禁止 RFT 与 GRPO 的 tokenization 路径不一致。
4. RFT supervised loss 只覆盖被验收的模型自生成 response 和 EOS，prompt token label 必须为 `-100`；canonical `process_response` 不进入训练 labels。
5. 不得右截断 story，因为反事实干预通常在 story 尾部。超长时提高长度上限或减少 batch，而不是截掉末尾。
6. observed/hidden pair、Hi-ToM base story、ExploreToM `base_scenario_id` 不得跨 train/dev/test 泄漏。
7. order-4 OOD test 不参与超参数选择、early stopping 或 reward 权重设计。
8. GRPO 的 actor 初始 checkpoint 和 reference policy 必须指向同一个冻结的 RFT checkpoint；resume 后 reference 也不得漂移。
9. 第一版只使用逐样本 process reward。pair 指标用于评估，不在尚未验证的情况下把跨 prompt pair bonus 混入 GRPO advantage。
10. 每个正式 checkpoint 必须记录代码 commit、配置、模型/数据 hash、随机种子和完整评估结果。
11. 第一版 RFT 只接收 `process_reward == 1.0` 且通过严格 schema 检查的候选，不使用部分奖励阈值。

## 4. 数据协议与评估协议

### 4.1 派生采样/评估 split

原始 JSONL 不修改。由数据准备脚本产生固定的派生 split：

- 从 ExploreToM train 的 600 个 pair 中，按 `base_scenario_id` 和 seed `2026` 做稳定 hash，取 60 个 pair（120 条）加入 dev。
- 派生 train：原 train 去掉上述 120 条，共 **3,080** 条。
- 派生 dev：现有 Hi-ToM val 400 条 + ExploreToM holdout 120 条，共 **520** 条。
- sealed test：现有 order-4 OOD 600 条，不变。

这样 dev 同时覆盖 Hi-ToM 与 ExploreToM。RFT 只能从派生 train 采样和构建训练集，RFT 与 GRPO 必须共用同一份派生 train/dev/test 协议。数据准备脚本必须验证：

- 全部 pair 完整且不跨 split。
- base story/scenario 不跨 split。
- `json.loads(process_response) == process_target`。
- 所有金标经过 `score_process_output` 得分均为 `1.0`。
- source、order、intervention_type 的计数符合 manifest。
- 输出 manifest 包含输入文件 SHA256、派生记录 SHA256、seed、tokenizer revision 和各类统计。

### 4.2 防止“格式正确等于推理正确”

所有生成评估至少报告以下互不替代的指标：

| 指标 | 定义 |
| --- | --- |
| `parse_rate` | 能解析为 JSON object |
| `strict_format_rate` | 无 Markdown/前后缀，且字段集合和类型完全正确 |
| `mean_process_reward` | 当前分解 reward 的平均值 |
| `full_reward_rate` | reward 精确为 1.0 的比例 |
| `answer_accuracy` | JSON `answer` 与金标相同 |
| `core_state_accuracy` | belief 的 visibility + nested belief，或 order-0 的 world state 正确 |
| `pair_accuracy` | observed/hidden 两条均答对的 pair 比例 |
| `intervention_sensitivity` | 金标应变化时，模型预测是否也发生变化 |
| `shortcut_copy_rate` | shortcut-conflict 子集上仍输出 shortcut prediction 的比例，越低越好 |
| `last_mention_copy_rate` | last-mention-conflict 子集上复制最后提及位置的比例，越低越好 |
| `answer_state_consistency` | `answer == nested_belief/world_state` 的比例 |
| `eos_rate` / `length_p95` | 是否正常停止及响应长度尾部情况 |

所有指标需按以下维度给出总体值和分桶值：

- `source_dataset`
- `question_order`
- `intervention_type`
- observed/hidden pair
- Hi-ToM shortcut-conflict 与 last-mention-conflict 子集

### 4.3 固定评估节点

依次评估并保存以下模型的同格式结果：

1. 原始 Qwen2.5-3B-Instruct。
2. 选中的 RFT checkpoint。
3. 每个 GRPO pilot checkpoint。
4. 最终 GRPO checkpoint。

dev 用于全部选模。sealed order-4 OOD test 在 GRPO 配置、step 数和 checkpoint 选择规则全部锁定后才解封，一次性评估 M0-M3；看到 test 结果后不得回头调整训练。`eval_tom/tom_eval_datasets.csv` 用作外部报告集，至少报告其各数据源 final-answer accuracy；如果没有完整 process target，不伪造 process 指标。

生成参数分两套：

- **确定性评估**：`do_sample=false`、temperature 0、单响应，用于模型比较。
- **探索审计**：temperature 与 GRPO rollout 一致，每题 8 个响应，用于测 reward 方差和探索空间，不用于最终 accuracy 报告。

## 5. 阶段 A：独立 RFT

### A0. 新建独立目录

RFT 所有实现放在项目根目录 `rft/`，不得调用 `verl` trainer。建议最终结构：

```text
rft/
  README.md
  requirements.txt
  configs/
    qwen2_5_3b_sample.yaml
    qwen2_5_3b_train.yaml
    smoke.yaml
  prepare_data.py
  sample.py
  score_candidates.py
  build_dataset.py
  dataset.py
  train.py
  generate.py
  evaluate.py
  run_rft.sh
  tests/
    test_rejection.py
    test_dataset.py
    test_prompt_parity.py
```

共享的 schema、reward 和聚合指标应放入可安装的 `robusttom/` Python package。现有 `scripts/process_reward.py` 保留为薄 CLI wrapper，避免 RFT 与 `verl` 各复制一份 reward 逻辑后发生漂移。

### A1. 拒绝采样协议

候选必须由原始 `Qwen/Qwen2.5-3B-Instruct` 直接生成。第一版 RFT 只做一轮“base 采样 -> 严格过滤 -> 微调”，不在采样前进行任何监督微调，也不默认用 RFT checkpoint 再滚动生成第二轮数据。

采样实现使用独立 vLLM 或 Hugging Face generation，不经过项目内 `verl`。单卡 A800 的默认采样配置：

| 参数 | 默认值 |
| --- | --- |
| model | `Qwen/Qwen2.5-3B-Instruct`，固定 revision |
| initial samples per prompt | `K=16` |
| maximum cumulative samples | `K=64`，只补采未通过 prompt |
| temperature | 0.8 |
| top-p | 0.95 |
| top-k | -1 |
| max new tokens | 256 |
| stop | tokenizer EOS |
| seed | 2026 |
| tensor parallel | 1 |

每个候选必须保存而不是边生成边丢弃，原始采样文件至少记录：

- `global_sample_id`、`global_pair_id` 和 prompt SHA256
- model id、model revision、tokenizer revision
- sampling round、candidate index、seed 和完整生成参数
- raw response、response token ids、token count、EOS/截断状态
- source/order/intervention/shortcut 等评估元数据

采样按以下预算递增：

1. 先在按 source/order/intervention 分层的 64 个 prompt 上运行 `K=4` smoke test，验证 decode、EOS、持久化和 scorer 链路。
2. 再在分层抽取的 256 个 train prompt 上运行 `K=8` coverage audit，估计各分桶的 full-reward 概率和主采样成本。
3. 对全部 3,080 个派生 train prompt 运行 `K=16`。
4. 对仍无满分候选的 prompt 补采至累计 `K=32`；只有覆盖门槛仍未达到时才补采至累计 `K=64`。
5. dev/test 只用于评估，禁止把它们的候选加入 RFT 训练集。

每轮补采使用可复现但互不重复的 seed。prompt 文本、chat template、模型 revision 或生成配置发生变化时视为新实验，不得与旧候选混合。

### A2. 拒绝、去重与训练集构建

每个 candidate 只对 raw response 本身评分，禁止把 prompt 一起传给 JSON parser。第一版接受规则固定为：

```text
score_process_output(candidate, process_target).reward == 1.0
and checks.parseable_json == true
and checks.format == true
and generation_reached_eos == true
```

这意味着候选必须是唯一的严格 JSON object，schema、字段类型和所有 process 内容均正确。以下候选全部拒绝：

- reward 小于 1.0 的部分正确响应
- Markdown fence、解释前后缀或多余字段
- 虽然 final answer 正确但 visibility/nested belief/world state 错误
- 达到 `max_new_tokens` 仍未 EOS、空响应或重复循环

训练集构建规则：

1. 对通过候选按解析后的 JSON 做语义去重，避免仅空格或 key 顺序不同造成重复计数。
2. 每个 prompt 最多保留 1 个候选；多个满分候选时选择 token 数最短者，再以 candidate index 做确定性 tie-break。
3. 只保留 observed 与 hidden 两侧都至少有一个满分候选的完整 `global_pair_id`，防止 RFT 训练集向容易的一侧倾斜。
4. RFT 训练 completion 保留选中候选的原始文本，不替换为 canonical `process_response`。
5. 不复制低频分桶样本来伪造覆盖；通过 pair-level sampler 在训练时平衡 source/order，必要时使用显式 sample weight。

构建后的 manifest 必须报告：

- candidate 总数、整体 acceptance rate、prompt coverage、完整 pair coverage
- 按 source/order/intervention 的 acceptance 和 coverage
- 各 prompt 的首次成功 round、候选 reward 分布、EOS/长度统计
- 因 pair 不完整而丢弃的 prompt/pair 数量
- 原始候选文件和最终训练集 SHA256

进入微调的最低覆盖门槛：

- 总 prompt coverage 至少 70%。
- 完整 pair coverage 至少 60%。
- 每个 source x order 分桶的 prompt coverage 至少 50%。
- observed 与 hidden 的 coverage 绝对差不超过 10 个百分点。

若累计 `K=64` 后仍未达到门槛，应停止并分析 base policy 的失败类型；不得加入 canonical 金标、降低到部分 reward 或临时做 SFT bootstrap 来绕过门槛。覆盖门槛可以在 coverage audit 后基于实测重新讨论，但任何修改必须在正式全量采样前写入变更记录。

### A3. RFT 微调数据与训练栈

每条最终训练样本只保留：

- `global_sample_id`、`global_pair_id`
- `process_prompt`
- 被验收的模型自生成 `accepted_response`
- candidate provenance、reward 和 source/order/intervention/shortcut 元数据

编码流程固定为：

1. `tokenizer.apply_chat_template([{"role":"user","content":process_prompt}], add_generation_prompt=True)` 得到 prompt ids。
2. `accepted_response` 不加特殊起始 token，末尾确保恰好一个 EOS。
3. 拼接 prompt ids 和 response ids。
4. prompt 对应 labels 全部设为 `-100`，只学习 accepted response 和 EOS。
5. 动态右 padding，padding labels 为 `-100`；`packing=false`，避免 pair/样本边界混淆。

开训前必须用真实 Qwen tokenizer 生成长度报告：min、p50、p90、p95、p99、max 和超限样本 ID。默认从 `max_seq_length=2048` 开始；只要有样本超限，就提高到能完整容纳全部样本的下一个合理档位，不得静默截断。

微调首选 Hugging Face `transformers.Trainer` + 自定义 response-only dataset/collator：

- 使用单独的 RFT requirements/lock 文件，但版本必须与后续 `verl` 可加载范围兼容。
- 不依赖 TRL 的隐式 masking；单元测试逐 token 验证 prompt label 全为 `-100`。
- 模型以标准 Hugging Face safetensors 目录导出，包含 tokenizer、generation config、采样 manifest 和训练配置。
- 单卡 3B 首选全参数 BF16 + gradient checkpointing，不使用 DeepSpeed；实际 OOM 后再考虑 LoRA 回退。

默认微调配置：

| 参数 | 默认值 |
| --- | --- |
| initialization | 与采样完全相同的原始 Qwen base revision |
| precision | BF16，允许 TF32 |
| full fine-tuning | 是 |
| flash attention | `flash_attention_2` |
| gradient checkpointing | 开 |
| max sequence length | token audit 后取 2048 或更高，零截断 |
| per-device batch | 2；OOM 时降为 1 |
| gradient accumulation | 16；batch=1 时改为 32 |
| effective batch | 32 samples |
| epochs | 最多 3 |
| learning rate | `1e-5` |
| scheduler | cosine |
| warmup ratio | 0.03 |
| weight decay | 0.01 |
| max grad norm | 1.0 |
| seed | 2026 |
| save/eval | 每个 epoch；保留最近和最佳 checkpoint |

若使用 LoRA，进入 GRPO 前必须 merge，并做 merge 前后 greedy 输出一致性测试。不要一开始做大规模 LR grid；若 `1e-5` 出现能力下降，再运行唯一备选 `5e-6`。

### A4. RFT 运行顺序

1. 对原始 base model 在派生 dev 上做 deterministic M0 baseline。
2. 完成 `K=4` 链路 smoke 和 `K=8` coverage audit。
3. 完成全量 `K=16` 采样，并按需仅对失败 prompt 补采至 `K=32/64`。
4. 严格评分、语义去重、完整 pair 过滤，冻结 RFT 训练 manifest。
5. 在 32 条 accepted sample 上训练 5-10 step，验证 loss、label mask、保存和重载。
6. 运行 1 epoch 全参数 RFT 并做完整 dev 生成评估；仅在 dev 继续改善时运行第 2/3 epoch。
7. 不按 validation loss 单独选模。先过格式硬门槛，再按 pair accuracy、core state accuracy、mean process reward 的顺序选 checkpoint。
8. 冻结最佳 RFT checkpoint 及目录 hash，作为后续 GRPO actor init 和 reference model。

第一版不自动进行第二轮 rejection sampling。若未来要用 RFT checkpoint 重采样，必须作为独立 `RFT-iter2` 消融，重新记录 policy、候选和数据 hash，不能覆盖第一轮数据。

### A5. RFT 验收门槛

进入 GRPO 前必须同时满足：

- A2 的 acceptance/coverage 门槛全部通过，训练集只含模型自生成的满分候选。
- 派生 dev `parse_rate >= 99%`。
- 派生 dev `strict_format_rate >= 98%`。
- `answer_state_consistency >= 99%`。
- dev final-answer accuracy、pair accuracy 均不得低于 base model；如果 base 不输出 JSON，则使用对所有模型一致的任务答案抽取与归一化规则后比较。
- 520 条 dev 全部生成过程无 prompt 泄漏、无 Markdown fence、无无限重复。
- 8-sample 探索审计中仍有足够非零组内 reward 方差；若超过 80% prompt 的 8 个响应 reward 完全相同，先判断是任务已经饱和还是 RFT 过强，再决定是否进入 GRPO。

如果 RFT 只改善格式但内容指标下降，优先选择更早 checkpoint、降低 LR 或停止进入 GRPO；不得用 canonical 金标补训来修复。

## 6. 阶段 B：`verl` GRPO

### B0. 开训前工程硬门槛

当前 `verl` 路径不能直接用于本项目。正式训练前至少完成以下修改和测试：

1. **GRPO parquet adapter**：从派生 split 输出 `data_source`、chat `prompt`、`reward_model.ground_truth` 和 `extra_info`。ground truth 使用 canonical JSON string，避免 parquet union struct 为两种 schema 自动补 `None` 字段。
2. **prompt parity**：给 `RLHFDataset` 增加默认关闭的 chat-template 选项，process 数据启用它；测试同一 `process_prompt` 在 RFT 和 GRPO 中产生完全相同的 prompt token ids。
3. **process reward 接线**：新增专用 `main_robust_tom_grpo.py` 或等价入口。只 decode response ids，`skip_special_tokens=True`，不得把 prompt 一起传给严格 JSON parser。
4. **分解指标接线**：训练和验证日志报告 reward components、格式率、full reward、source/order/intervention 分桶、pair 指标和 group reward std，而不只记录一个 scalar。
5. **修复验证逻辑**：当前 `_validate()` 用 `reward == 3` 作为正确条件，不适用于 `[0,1]` process reward；改为通用 mean/full-reward/accuracy 指标。
6. **验证分批**：当前 validation loader 一次装入整个验证集且忽略 `data.val_batch_size`；改为真正按配置分批，避免单卡一次生成 520 条。
7. **稳健 GRPO 归一化**：用 `torch.stack` 在正确 device 上计算每个 uid 组的 mean/std，使用 population std，并显式记录 zero-variance group；移除逐 step 打印完整 reward matrix。
8. **batch 语义断言**：旧 worker 会把配置中的 PPO mini/micro batch 再乘 `rollout.n`。启动时打印归一化前后值，并断言每个 uid 恰有 `n` 个响应、effective mini batch 不大于本 step 的响应数。
9. **随机种子**：driver、Python、NumPy、Torch、CUDA、DataLoader 和 vLLM rollout 都接受并记录 seed，不能继续依赖代码内固定的 `1000`。
10. **路径清理**：移除 `main_ppo.py` 中硬编码的 `/tmp/ray/zhangchunhui`，所有模型、数据、日志、checkpoint 路径由配置或环境变量显式提供。
11. **checkpoint 可恢复性**：至少保存 actor、optimizer、scheduler、global step 和 RNG state。reference model path 单独持久化并始终指向冻结 RFT checkpoint。恢复测试需证明一次中断后可继续至少 2 step。
12. **reward 回归测试**：gold response=1.0、非法 JSON=0、Markdown fence=0.95、错误 visibility 的 gating、两种 schema、空响应和 EOS 边界都要通过。

优先在当前仓库固定的 `verl`/vLLM 版本上做最小修复，不在同一实验中顺便升级大版本。只有确认 Qwen2.5-3B、CUDA 或 vLLM 在 A800 上存在无法修复的兼容问题时，才单独开升级分支并重跑全部 smoke tests。

### B1. GRPO 数据格式

每条 parquet row 至少包含：

```text
data_source = "robust_tom_process"
prompt = [{"role": "user", "content": process_prompt}]
reward_model = {
  "style": "rule",
  "ground_truth": canonical_process_target_json
}
extra_info = {
  "index", "global_sample_id", "global_pair_id", "source_dataset",
  "question_order", "intervention_type", shortcut metadata...
}
```

`index` 必须对每个原始 prompt 唯一。GRPO 组仍由同一 prompt 的 `n` 个 rollout 组成，不能把 observed/hidden 两个不同 prompt 错当成同一 GRPO group。

### B2. 单卡起始配置

正式参数以 A800 smoke test 的 peak memory 和 token audit 为准。安全起始值如下：

| 参数 | smoke | pilot/main 起点 |
| --- | ---: | ---: |
| `trainer.n_gpus_per_node` | 1 | 1 |
| `data.train_batch_size`（prompt 数） | 2 | 8 |
| `data.val_batch_size` | 8 | 32 |
| `rollout.n` | 4 | 8 |
| max prompt length | token audit 值 | 同左，预计 2048 左右 |
| max response length | 256 | 256 |
| rollout temperature | 0.8 | 0.8 |
| top-p | 0.95 | 0.95 |
| actor LR | `3e-7` | `3e-7` |
| PPO mini batch（乘 n 前） | 2 | 8 |
| PPO micro batch（乘 n 前） | 1 | 1 |
| PPO epochs | 1 | 1 |
| clip ratio | 0.2 | 0.2 |
| entropy coefficient | 0.001 | 0.001 |
| actor KL loss | 开 | 开 |
| KL coefficient | 0.001 | 0.001，按 pilot 调整 |
| tensor model parallel | 1 | 1 |
| gradient checkpointing | 开 | 开 |
| remove padding | 开 | 开 |
| vLLM GPU utilization | 0.35 | 0.35-0.45 |
| seed | 2026 | 2026 |

单卡初始采用 actor parameter/gradient/optimizer offload 和 ref parameter offload，以稳定跑通为优先。smoke 后根据 peak memory 逐项关闭不必要的 offload 来提速，每次只改一项并重新做 2-step test。

`max_response_length` 不沿用旧脚本的 2048。canonical JSON 很短，256 应足够；如果探索审计的 `length_p99` 或 EOS 统计证明不足，再提高。`rollout.n` 也不沿用旧脚本的 16，先用 8 平衡组内方差、显存和吞吐。

offload 会把压力从 GPU 转移到主机。A800 机器在正式 GRPO 前还应确认：

- CPU RAM 建议至少 96GB，优先 128GB；不足时不能默认启用全部 offload 后直接长跑。
- checkpoint、Ray 临时文件和 vLLM cache 使用本地 NVMe，预留至少 150GB。
- 基线兼容矩阵为 CUDA 12.1、PyTorch 2.4.0、Transformers `<4.48`、vLLM 0.6.3、tensordict `<0.6`；以实际通过 smoke 的精确 patch version 形成 lock。
- 50-step pilot 记录 wall time，并以实测线性外推 200/300 step 预算；在没有 A800 实测前不承诺训练时长。

### B3. 分阶段运行

#### B3.1 CPU/单元测试

- 数据 adapter 计数、hash、schema 和 pair 隔离测试。
- RFT/GRPO prompt token parity 测试。
- process reward 与 `RewardManager` 集成测试。
- GRPO uid 分组与标准化的合成 tensor 测试，包括全同 reward。
- validation 聚合测试，确保 `[0,1]` reward 不再使用旧的 `==3` 判定。

#### B3.2 A800 初始化 smoke

- 只加载冻结 RFT checkpoint，构建 actor、ref 和 vLLM。
- 运行 validation-only 8 条，确认生成、decode、reward 和指标链路完整。
- 记录各阶段 `allocated/reserved/peak` GPU memory 和 CPU RAM。
- 保存并重载 checkpoint，确认 greedy 输出一致。

#### B3.3 2-step 训练 smoke

- 2 prompts x 4 rollouts，运行 2 个 optimizer step。
- 确认无 OOM、NaN/Inf、空 response、uid 丢失或 weight sync 错误。
- 确认 actor 参数发生变化，reference 参数 hash 不变。
- 确认 gold/malformed 手工样本的 reward 与独立 scorer 完全相同。
- 模拟中断并恢复 2 step，验证 resume。

#### B3.4 50-step pilot

先只运行 `lr=3e-7, n=8, seed=2026`。每 10 step 做小 dev slice，每 25 step 做完整 dev，每 25 step 保存 checkpoint。pilot 期间重点观察：

- mean/full process reward 是否上升。
- pair accuracy 和 shortcut-copy rate 是否改善。
- strict format 是否维持在 98% 以上。
- zero-variance group 占比和 group reward std。
- token-level KL、entropy、clip fraction、response length 和 EOS rate。
- generation/ref/update 三阶段耗时与峰值显存。

只有出现“训练稳定但无学习信号”时才做一个受控调整：

- reward 几乎全同且未饱和：先把 temperature 提到 1.0；仍无方差再考虑 `n=12`。
- KL 过大或 dev 回退：LR 降到 `1e-7`，或 KL coefficient 提到 `0.003`。
- KL 极小且 reward 不动但有方差：LR 提到 `5e-7`，只做 25-step 对照。
- 格式率下降：不增加 format reward 权重，先降低 LR/temperature 并检查 prompt parity 与截断。

一次只改一个变量，所有 pilot 复用相同 dev 和评估脚本。

#### B3.5 主训练

- 从冻结 RFT checkpoint 重新开始，不从被淘汰 pilot 延长训练。
- 先设 `total_training_steps=200`，每 25 step eval、每 50 step checkpoint。
- 200 step 仍有稳定 dev 增益且无 guardrail 触发时，最多延长到 300 step；不默认跑 2 个完整 epoch。
- checkpoint 选择先过格式和回归硬门槛，再最大化 belief-pair accuracy，其次是 core-state accuracy 和 mean reward。
- 选定 M2/M3 checkpoint 且冻结所有配置后，才对 M0-M3 运行 sealed order-4 OOD 和完整外部 ToM eval。

### B4. 停止与回退条件

发生以下任一情况立即暂停，不继续消耗 GPU 等待“自行恢复”：

- NaN/Inf、连续 OOM、reward 与独立 scorer 不一致。
- dev strict format 连续两次低于 98%，或相对 RFT 下降超过 2 个百分点。
- pair accuracy 或 final-answer accuracy 相对 RFT 连续两次绝对下降超过 3 个百分点。
- response `length_p95` 持续触顶，EOS rate 低于 98%，或出现明显重复循环。
- token-level KL 连续三个窗口高于 0.1；该阈值在 pilot 后可基于实际量级修订，但修订必须记录理由。
- 超过 80% group 长期 zero variance 且模型并未达到 dev 饱和，此时 GRPO 没有可用的相对信号。

回退顺序固定为：恢复最佳 checkpoint -> 降 LR -> 提高 KL -> 调整 temperature -> 最后才调整 `rollout.n` 或 reward。不要同时改变数据、reward 权重和优化器参数。

## 7. 必要对照与消融

为了证明改善来自 process supervision，而不是仅来自更多训练，至少保留以下模型：

| ID | 模型 | 用途 |
| --- | --- | --- |
| M0 | base Qwen2.5-3B-Instruct | 原始基线 |
| M1 | RFT | 拒绝采样微调效果 |
| M2 | RFT + answer-only GRPO | 等训练预算的 outcome reward 对照 |
| M3 | RFT + process-reward GRPO | RobustToM-RL 主模型 |

answer-only 对照仍要求模型输出相同 JSON，但 reward 只按最终 `answer` 正确性计分。M2 和 M3 必须共用 RFT init、数据、rollout 参数、step 数和 seed，唯一差异是 reward。

资源有限时的执行优先级：M0 -> M1 -> M3；M2 先做 50-step pilot。准备发布结论时，最佳 M2/M3 配置至少各补 seed 2026、2027、2028 三次，报告均值和标准差。开发期间不为每个超参数跑三 seed。

## 8. 结果判定

M3 只有在以下条件同时成立时才能称为“缓解 shortcut/reasoning collapse”：

1. 相比 M1，dev belief-pair accuracy、core-state accuracy 或 shortcut-conflict accuracy 有明确提升。
2. 相比等预算 M2，M3 的 pair accuracy/shortcut-copy rate 更优，而不只是 mean scalar reward 更高。
3. order-4 OOD 上方向一致，不要求每个小分桶都提升，但不能用显著 OOD 回退换取 ID 增益。
4. strict format、answer-state consistency、EOS 和外部 ToM accuracy 均通过 guardrail。
5. 失败案例人工抽查能对应到 scorer components，未发现通过字段复制、冗余前后缀、异常长度或 parser 漏洞刷分。

建议最终汇总表至少包含：

```text
model | split/source/order | answer_acc | pair_acc | core_state_acc |
full_reward | shortcut_copy | strict_format | mean_reward | length_p95
```

同时固定抽查以下四类样本：observed 正确/hidden 错误、hidden 正确/observed 错误、两者都错且复制 shortcut、格式正确但 nested belief 错。

## 9. 交付物与里程碑

### Milestone 0：协议冻结

- [ ] `PLAN.md` 成为主计划，后续重要决策写入“变更记录”。
- [ ] 固定 tokenizer/model revision、环境 lock 和数据 manifest/hash。
- [ ] 建立统一评估脚本并完成 M0 baseline。

### Milestone 1：RFT 可复现

- [ ] 创建并测试独立 `rft/`。
- [ ] 生成 3,080/520 派生 split，零截断、零泄漏。
- [ ] 完成 `K=4` smoke、`K=8` coverage audit 和全量 `K=16-64` 候选采样。
- [ ] 满分过滤、语义去重和完整 pair coverage 通过 A2 门槛。
- [ ] 完成 10-step 微调 smoke 和 1-3 epoch 正式 RFT。
- [ ] 选出并冻结通过 A5 门槛的 RFT checkpoint。

### Milestone 2：GRPO 链路可信

- [ ] 完成 B0 的 12 项工程硬门槛。
- [ ] CPU tests、validation-only、2-step、resume smoke 全通过。
- [ ] 单卡配置和显存/吞吐记录落盘。

### Milestone 3：GRPO pilot 与主训练

- [ ] 完成 50-step process-reward pilot。
- [ ] 锁定唯一主配置。
- [ ] 完成 200-300 step 主训练和 checkpoint 选择。
- [ ] 完成 answer-only 等预算对照。

### Milestone 4：结论与复现

- [ ] 完成 M0-M3 的统一 dev/OOD/外部评估。
- [ ] 完成关键配置三 seed 复验。
- [ ] 输出结果表、失败案例、环境和一键运行说明。
- [ ] README 更新 RobustToM-RL 方法、数据和复现实验入口。

## 10. 运行记录规范

每次 RFT/GRPO run 创建独立目录，至少包含：

```text
runs/<stage>/<run_id>/
  config.yaml
  command.txt
  env.txt
  git_commit.txt
  data_manifest.json
  train.log
  metrics.jsonl
  eval/
  checkpoints/
  summary.md
```

RFT sampling run 还必须保存 `candidate_manifest.json` 和 `acceptance_metrics.json`；原始候选 JSONL 体积较大，可以继续忽略，但其 SHA256 和存储位置必须写入 manifest。

`run_id` 建议为 `YYYYMMDD-HHMM_model_stage_seed_shorttag`。W&B 只作为可视化副本，本地 JSONL 和 config 才是可复现真源。模型大文件和数据继续忽略，但轻量 manifest、配置、评估摘要和代码必须进入 git。

## 11. 风险登记

| 风险 | 早期信号 | 处理 |
| --- | --- | --- |
| prompt/chat template 不一致 | RFT 格式很好，GRPO 首次验证突然下降 | token parity 测试设为硬门槛 |
| story 尾部被截断 | hidden/observed 同答、长度上限样本集中失败 | 零截断审计，提高 max length |
| base 满分采样率过低 | `K=32/64` 后仍有大量零通过 prompt | 按分桶检查失败；不降阈值、不回填金标、不增加 SFT bootstrap |
| 拒绝采样选择偏差 | 低阶/observed 覆盖显著高于高阶/hidden | 完整 pair 过滤、分桶 coverage 门槛和训练 sampler 平衡 |
| RFT 过强导致 GRPO 无方差 | 大量 group std=0 | 提前停止 RFT、提高 rollout temperature；先判断是否已饱和 |
| process reward 被字段复制利用 | 格式/浅字段高，core state 和 pair accuracy 低 | 监控 gated components 和 pair 指标，不提高浅字段权重 |
| 单卡显存不足 | actor/ref/vLLM 初始化或 update OOM | micro batch 1、offload、降低 vLLM utilization，再考虑 LoRA RFT |
| 旧 `verl` 指标错误 | reward 看似为 0 或 accuracy 恒为 0 | 修复 `==3` 假设并与独立 scorer 对拍 |
| parquet schema 污染 target | target 出现额外 `None` 字段，strict schema 全失败 | ground truth 存 canonical JSON string |
| resume 改变 reference | 恢复后 KL 突变 | reference path/hash 独立保存并校验 |
| 只对 ID 有效 | dev 提升但 order-4 OOD 回退 | 配置锁定后一次性 sealed OOD 评估，禁止用 test 调参 |
| 数据本地存在但不可复现 | 换机器后 `data/` 缺失或版本未知 | 生成脚本 + 输入/output hash + manifest 入库 |

## 12. 近期执行顺序

接下来的实现严格按此顺序进行：

1. 创建 `robusttom/` 共享 schema/reward/metrics，并保持现有 CLI 兼容。
2. 创建独立 `rft/`，完成派生 split、候选采样、严格拒绝、pair coverage、dataset mask 和 prompt parity 测试。
3. 在 A800 上完成 M0 baseline、RFT sampling audit 和拒绝采样微调，冻结 M1。
4. 完成 `verl` 数据 adapter、专用 reward manager、验证指标和 GRPO 标准化修复。
5. 在 A800 上按 validation-only -> 2-step -> resume -> 50-step 的顺序验收。
6. 锁定参数后运行 M3 主训练，再补 M2 answer-only 对照。
7. 先完成锁定配置的多 seed 复验，最后一次性运行 sealed OOD 和外部 eval。

## 13. 变更记录

- **2026-08-19**：建立初版计划。确认 RFT 独立于 `verl`，在根目录新增 `rft/`；GRPO 使用项目内 `verl`；目标资源为单张 A800 80GB。
- **2026-08-19**：按项目原意将 RFT 明确定义为 Rejection Sampling Fine-Tuning；删除原 Reasoning/Response Format Tuning 定义，确定不增加 SFT bootstrap，训练只使用 base model 自生成且 `process_reward == 1.0` 的候选。
