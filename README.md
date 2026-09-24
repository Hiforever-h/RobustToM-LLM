# RobustToM-LLM

RobustToM-LLM是一个面向高阶Theory of Mind（ToM）推理的训练项目。项目以`Qwen2.5-3B-Instruct`为基础模型，通过反事实数据、Rejection Sampling Fine-Tuning（RFT）和Group Relative Policy Optimization（GRPO），训练模型显式追踪多层嵌套信念，
并降低对world state、last mention等shortcut的依赖。

当前主链路采用自然语言CoT输出，不再要求模型生成JSON。模型需要从问题本身判断ToM阶数`N`，输出恰好`N`个`Think/State`推理块，并让最终`Answer`重复最后一个`State`：

```text
Think 1:
Henry privately observed that the flashlight moved to the blue_canvas_bag.
State: blue_canvas_bag
Think 2:
Thomas did not observe Henry's private update, so Thomas still believes that
Henry thinks the flashlight is in the ceramic_jar.
State: ceramic_jar
Answer: ceramic_jar
```

### Reward计算公式

reward为：

$$
R=0.75\cdot\frac{1}{N}\sum_{i=1}^{N}s_i(0.40+0.60j_i)
 +0.05\left(0.70\frac{L}{N}+0.30F\right)
 +0.20\,A\prod_{i=1}^{N}s_i.
$$

其中，`N`是目标ToM阶数；`s_i∈{0,1}`表示第`i`个`State`是否与标准信念状态一致；`j_i∈[0,1]`是LLM Judge对该步自然语言推理的评分，推理缺失时记为`0`。`L`是从`Think 1`开始、编号连续、各自包含非空推理和恰好一个非空`State`且`State`后没有多余内容的有效前缀长度；`F∈{0,1}`表示完整输出满足严格的`Think/State/Answer`结构；`A∈{0,1}`表示最终`Answer`正确。只有所有`State`都正确，才发放`0.20`的答案奖励。

## 主要结果

以下结果来自同一组deterministic compact-prompt评测。Base、RFT和GRPO使用完全相同的
样本ID、prompt哈希和process target；生成参数为`temperature=0`、`n=1`、
`max_new_tokens=384`。Answer Accuracy通过本地规则比较最终`Answer:`和标准答案，评测过程
不调用LLM Judge。

### Answer Accuracy

|数据集|样本数|Base|RFT|GRPO|
|---|---:|---:|---:|---:|
|测试集|400|0.75%（3/400）|12.50%（50/400）|**59.50%（238/400）**|
|Hi-ToM 4阶OOD数据集|600|0.00%（0/600）|4.67%（28/600）|**28.50%（171/600）**|

测试集上的分阶结果如下：

|ToM阶数|样本数|Base|RFT|GRPO|
|---:|---:|---:|---:|---:|
|1阶|100|3.00%|31.00%|**82.00%**|
|2阶|200|0.00%|6.50%|**55.00%**|
|3阶|100|0.00%|6.00%|**46.00%**|

相较RFT，GRPO在测试集上提升`47.00`个百分点，在Hi-ToM 4阶OOD数据集上提升`23.83`个百分点。训练阶数范围内的1阶、2阶和3阶任务均有明显提升；在训练阶段未出现的4阶任务上，GRPO仍达到`28.50%`，但与测试集之间仍存在明显的高阶外推差距。

当前结果来自单次训练和单个checkpoint，尚未报告多seed均值与方差。

### Process Reward与格式率

下表中的process reward对应评测文件里的`mean_process_reward`。这是不调用LLM Judge的rule-only分数：Judge reasoning分固定为`0`，本地规则只根据每步`State`和最终`Answer`计分，
完全正确输出的理论最高分为`0.52`。因此该分数只能在本表的相同评测协议内比较，不能直接
等同于GRPO训练时包含Judge reasoning的连续reward。

|数据集|模型|Rule-only process reward|Strict format rate|
|---|---|---:|---:|
|测试集|Base|0.0016|4.00%|
|测试集|RFT|0.0774|25.00%|
|测试集|GRPO|**0.3406**|**99.75%**|
|Hi-ToM 4阶OOD数据集|Base|0.0013|3.33%|
|Hi-ToM 4阶OOD数据集|RFT|0.0136|0.00%|
|Hi-ToM 4阶OOD数据集|GRPO|**0.1359**|**90.67%**|

GRPO的提升不只体现在最终答案上：测试集的strict format rate从RFT的`25.00%`提升到`99.75%`，Hi-ToM 4阶OOD数据集从`0.00%`提升到`90.67%`；rule-only process reward也在
两个数据集上同步提高。四阶结果仍明显低于测试集，说明格式泛化已经较稳定，但高阶信念状态
传播仍是主要瓶颈。

评测产物：

- [Base评测](runs/base_eval/20260828-qwen25-3b-natural-v2-k16)
- [RFT评测](runs/rft_eval/20260828-qwen25-3b-natural-v2-k16)
- [GRPO评测](runs/grpo_eval/20260828-qwen25-3b-natural-v2-k16)

## 数据

当前训练与评测数据采用observed/hidden反事实pair。每个pair共享故事和查询，只改变关键事件
是否被目标角色观察，用于检验模型是否真正追踪不同角色的知识状态。

|数据集|样本数|Pair数|ToM阶数|用途|
|---|---:|---:|---|---|
|训练集|3,200|1,600|1–3阶|RFT候选采样与GRPO训练|
|测试集|400|200|1–3阶|训练阶数范围内评测|
|Hi-ToM 4阶OOD数据集|600|300|4阶|未见阶数外推评测|

主要数据目录：

- `data/counterfactual_process_reward_v4_natural/`：自然语言CoT源数据，保留可审计的`judge_prompt`和结构化`process_target`，不包含`process_response`。
- `data/counterfactual_process_reward_v4_natural_compact/`：物化后的compact-prompt JSONL。
- `data/grpo/counterfactual_process_reward_v4_natural_compact/`：verl读取的Parquet数据。
- `data/rft/derived_v3_fewshot/`：RFT使用的固定train/dev/test split。

compact数据的manifest记录样本数、pair数、阶数分布、prompt版本和文件SHA256。训练集只包含1–3阶样本；Hi-ToM 4阶OOD数据集全部为4阶样本。

## 训练流程

```text
反事实数据构建
      ↓
Base模型K次采样
      ↓
本地State+Answer严格筛选
      ↓
RFT response-only训练
      ↓
compact-prompt数据构建
      ↓
RFT actor + frozen RFT reference
      ↓
LLM-as-a-Judge GRPO
      ↓
deterministic评测
```

### RFT

RFT对基础模型采样多个候选，只接受同时满足以下条件的轨迹：

- `Think`数量和编号与目标ToM阶数一致；
- 每个`State`都正确；
- 最终`Answer`正确；
- 输出结构完整并正常到达EOS。

默认筛选完全由本地规则完成，不调用LLM Judge。训练采用response-only loss，prompt token全部mask，只优化通过筛选的模型自生成response和EOS，不使用gold`process_response`回填。

RFT详细命令、采样审计和dataset builder说明见[rft/README.md](rft/README.md)。

### GRPO

GRPO从RFT checkpoint开始，actor和frozen reference在启动时指向同一个RFT模型。每个训练prompt采样16条response，由本地规则和`deepseek-v4-flash`共同评分：

- 本地规则检查`Think`数量、渐进式结构、每步`State`和最终`Answer`；
- LLM Judge只评价每步自然语言reasoning，默认关闭thinking模式；
- 一个prompt及其16条response打包为一次Judge请求；
- GRPO在同一prompt的16条response内计算相对优势。

当前默认reward权重为过程质量`0.75`、渐进式结构`0.05`、最终答案`0.20`。过程质量中每步`State`占`0.40`，Judge reasoning占`0.60`。只有所有`State`与最终`Answer`同时正确时，
才发放完整答案奖励。

GRPO详细配置、恢复训练和指标说明见[grpo/README.md](grpo/README.md)。

## 环境安装

正式GRPO训练要求Linux、NVIDIA GPU和CUDA 12.1。推荐Python 3.10与A800 80GB或同等级GPU。

```bash
conda create -n robusttom python=3.10 pip -y
conda activate robusttom

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.0.post2 --no-build-isolation
python -m pip install -e . --no-deps
```

RFT可以使用独立环境：

```bash
conda create -n robusttom-rft python=3.10 -y
conda activate robusttom-rft
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r rft/requirements.txt
python -m pip install flash-attn==2.6.3 --no-build-isolation
```

macOS可用于数据构建、本地reward测试和结果分析，但不能运行vLLM/flash-attn训练链路。

## RFT复现

以下命令展示compact-prompt RFT主流程。完整训练集采样成本为`3,200×16=51,200`条response，
建议先使用`data/rft/pilot_v2/train.jsonl`完成链路验证。

```bash
export RUN_ID=20260828-qwen25-3b-natural-v2-k16
export MODEL=Qwen/Qwen2.5-3B-Instruct

python -m rft.sample \
  --data data/rft/pilot_v2/train.jsonl \
  --model "$MODEL" \
  --output "runs/sampling/$RUN_ID/candidates.jsonl" \
  --num-samples 16 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 384 \
  --gpu-memory-utilization 0.85 \
  --seed 2026

python -m rft.score_candidates \
  --candidates "runs/sampling/$RUN_ID/candidates.jsonl" \
  --data data/rft/pilot_v2/train.jsonl \
  --output "runs/sampling/$RUN_ID/scored.jsonl"

python -m rft.build_dataset \
  --scored "runs/sampling/$RUN_ID/scored.jsonl" \
  --output "data/rft/accepted/$RUN_ID/train.jsonl" \
  --min-samples 0 \
  --max-samples 3000 \
  --seed 2026 \
  --compact-prompt

CUDA_VISIBLE_DEVICES=0 python -m rft.train \
  --model "$MODEL" \
  --train-file "data/rft/accepted/$RUN_ID/train.jsonl" \
  --output-dir "runs/rft_train/$RUN_ID" \
  --logging-dir "runs/rft_train/$RUN_ID/tensorboard" \
  --max-seq-length 2048 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 1 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.03 \
  --weight-decay 0.01 \
  --seed 2026
```

## GRPO复现

首先配置RFT checkpoint、数据目录、输出目录和Judge API：

```bash
export RFT_MODEL_PATH=/root/autodl-tmp/runs/rft_train/20260828-qwen25-3b-natural-v2-k16/final
export RAW_DATA_DIR=/root/RobustToM-LLM/data/counterfactual_process_reward_v3
export NATURAL_SOURCE_DIR=/root/autodl-tmp/data/counterfactual_process_reward_v4_natural_compact
export GRPO_DATA_DIR=/root/autodl-tmp/data/grpo/counterfactual_process_reward_v4_natural_compact
export GRPO_OUTPUT_ROOT=/root/autodl-tmp/runs/grpo
export GRPO_LOG_DIR=/root/autodl-tmp/runs/grpo/logs

export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_DIR=/root/autodl-tmp/wandb
export WANDB_CACHE_DIR=/root/autodl-tmp/cache/wandb
export RAY_TMPDIR=/root/autodl-tmp/tmp/ray
export TMPDIR=/root/autodl-tmp/tmp

export DEEPSEEK_API_KEY=your-key
```

按顺序执行：

```bash
bash grpo/run_grpo_natural.sh build
bash grpo/run_grpo_natural.sh validate
bash grpo/run_grpo_natural.sh smoke
bash grpo/run_grpo_natural.sh pilot
bash grpo/run_grpo_natural.sh train
```

默认正式训练配置：

|参数|值|
|---|---:|
|训练batch|8个prompt|
|每个prompt的rollout数|16|
|每步response数|128|
|训练步数|800|
|Epoch|2|
|Actor learning rate|`5e-7`|
|最大response长度|384|
|Judge并发数|8|
|Checkpoint间隔|400步|
|本地规则验证间隔|50步|

配置文件位于`verl/trainer/config/robust_tom_natural_grpo.yaml`。训练日志默认同时写入console和Weights & Biases。将`WANDB_MODE=offline`导出到环境变量可使用离线模式。

## OPSD复现

OPSD使用独立的TRL GOLD环境，避免与项目内`verl`固定的旧版vLLM依赖冲突。准备好完整的GRPO actor checkpoint后执行：

```bash
export OPSD_MODEL_PATH=/path/to/grpo/actor/global_step_800
bash opsd/run_opsd.sh train
```

默认训练恰好100个optimizer step，覆盖3200条训练样本一轮；每题只生成一条student rollout，固定teacher读取答案和逐层support events。终端显示tqdm进度条，并逐step记录到Weights & Biases。完整环境、配置、数据构造和LoRA合并方式见[`opsd/README.md`](opsd/README.md)。

## 评测复现

下面以任意一个Base、RFT或GRPO Hugging Face checkpoint为例。由于训练使用compact prompt，
生成和评测都必须传入`--compact-prompt`。

```bash
export MODEL=/path/to/model
export EVAL_DIR=runs/eval/example

python -m rft.generate \
  --data data/counterfactual_process_reward_v4_natural/val.jsonl \
  --model "$MODEL" \
  --output "$EVAL_DIR/dev_predictions.jsonl" \
  --backend vllm \
  --max-new-tokens 384 \
  --seed 2026 \
  --compact-prompt

python -m rft.evaluate \
  --predictions "$EVAL_DIR/dev_predictions.jsonl" \
  --data data/counterfactual_process_reward_v4_natural/val.jsonl \
  --output "$EVAL_DIR/dev_rule_metrics.json" \
  --compact-prompt

python -m rft.generate \
  --data data/counterfactual_process_reward_v4_natural/test.jsonl \
  --model "$MODEL" \
  --output "$EVAL_DIR/test_predictions.jsonl" \
  --backend vllm \
  --max-new-tokens 384 \
  --seed 2026 \
  --compact-prompt

python -m rft.evaluate \
  --predictions "$EVAL_DIR/test_predictions.jsonl" \
  --data data/counterfactual_process_reward_v4_natural/test.jsonl \
  --output "$EVAL_DIR/test_rule_metrics.json" \
  --compact-prompt
```

## 项目结构

```text
RobustToM-LLM/
├── data/       #反事实数据、compact JSONL、RFT split与GRPO Parquet
├── grpo/       #GRPO数据适配、RewardManager、指标与运行脚本
├── opsd/       #privileged-context OPSD训练、数据与LoRA合并
├── rft/        #采样、筛选、response-only训练与deterministic评测
├── scripts/    #反事实数据生成、compact数据和LLM Judge reward
├── verl/       #项目内适配的verl训练代码
├── runs/       #采样与评测产物
└── tests/      #回归测试
```

## 测试

```bash
pytest -q rft/tests tests
```

## 数据与许可证

- 基础模型：[Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- 数据设计参考：[Hi-ToM](https://github.com/ying-hui-he/Hi-ToM_dataset)
- 数据设计参考：[ExploreToM](https://github.com/facebookresearch/ExploreToM)
- RL训练框架：[verl](https://github.com/volcengine/verl)

项目代码许可证见[LICENSE](LICENSE)。第三方模型、框架和数据继续受各自许可证约束。
